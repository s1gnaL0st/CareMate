"""Conservative recent-health-event memory.

Events are contextual observations, not diagnoses. Only explicit recovery
language resolves an event; inferred events remain clearly labelled.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import re
from typing import Any, Mapping
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

RESPIRATORY = ("感冒", "咳嗽", "流鼻涕", "鼻塞", "咽痛", "发热", "发烧")
PAIN = ("疼", "痛", "头晕", "恶心", "呕吐", "腹泻")
RESOLVED = ("好了", "恢复了", "痊愈", "康复", "不咳了", "已经没事", "症状消失")
THIRD_PERSON = ("我家人", "我父母", "我爸", "我妈", "孩子", "老人家")


@dataclass(frozen=True)
class HealthObservation:
    category: str
    display_name: str
    status: str
    confidence: float
    confirmation: str
    summary: str


def infer_health_event_observation(text: str, now: datetime | None = None) -> HealthObservation | None:
    """Classify only explicit symptom/recovery signals with bounded rules."""
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not text or any(marker in text for marker in THIRD_PERSON):
        return None
    is_resolved = any(marker in text for marker in RESOLVED)
    if not any(marker in text for marker in RESPIRATORY + PAIN) and not is_resolved:
        return None
    category = "respiratory_symptoms" if any(marker in text for marker in RESPIRATORY) else "symptom_event"
    display_name = "近期呼吸道症状" if category == "respiratory_symptoms" else "近期症状"
    explicit = bool(re.search(r"(我|本人).*(这几天|最近|现在|还是|一直|继续)", text))
    status = "resolved" if is_resolved else ("active" if explicit else "possible")
    confidence = 0.95 if explicit or is_resolved else 0.55
    confirmation = "user_reported" if explicit or is_resolved else "inferred"
    return HealthObservation(category, display_name, status, confidence, confirmation, text[:240])


def event_is_relevant(event: Mapping[str, Any], query: str) -> bool:
    if event.get("status") not in {"candidate", "possible", "active", "stale"}:
        return False
    observation = infer_health_event_observation(query)
    return observation is None or observation.category == event.get("category")


def should_mark_stale(last_observed_at: datetime, now: datetime | None = None, days: int = 14) -> bool:
    now = now or datetime.now(timezone.utc)
    if last_observed_at.tzinfo is None:
        last_observed_at = last_observed_at.replace(tzinfo=timezone.utc)
    return now - last_observed_at > timedelta(days=days)


def health_context(events: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Serialize a minimal prompt-safe view; resolved history is excluded."""
    return {"schema_version": "health_memory_v1", "events": [
        {"id": str(e.get("id", "")), "category": e.get("category", ""),
         "display_name": e.get("display_name", ""), "status": e.get("status", ""),
         "confidence": float(e.get("confidence", 0) or 0), "summary": str(e.get("summary", ""))[:240]}
        for e in events if e.get("status") in {"candidate", "possible", "active"}
    ]}


async def load_relevant_health_events(db: AsyncSession, *, user_id: str, query: str, limit: int = 4) -> list[dict[str, Any]]:
    from models import HealthEvent
    rows = list(await db.scalars(select(HealthEvent).where(
        HealthEvent.user_id == user_id,
        HealthEvent.status.in_(["candidate", "possible", "active"]),
    ).order_by(HealthEvent.last_observed_at.desc()).limit(limit)))
    return [{"id": row.id, "category": row.category, "display_name": row.display_name,
             "status": row.status, "confidence": row.confidence, "summary": row.summary}
            for row in rows if event_is_relevant({"status": row.status, "category": row.category}, query)]


async def observe_health_message(db: AsyncSession, *, user_id: str, text: str,
                                conversation_id: str | None = None, message_id: str | None = None) -> HealthEvent | None:
    from models import HealthEvent, HealthEventEvidence, uuid_str
    observation = infer_health_event_observation(text)
    if observation is None:
        return None
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    existing = await db.scalar(select(HealthEvent).where(
        HealthEvent.user_id == user_id, HealthEvent.category == observation.category,
        HealthEvent.status.in_(["candidate", "possible", "active"]),
    ).order_by(HealthEvent.last_observed_at.desc()))
    if observation.status == "resolved":
        if existing is not None:
            existing.status = "resolved"; existing.resolved_at = now
            existing.resolution_source = "user_reported"; existing.last_observed_at = now
        return existing
    if existing is None:
        existing = HealthEvent(user_id=user_id, category=observation.category,
            display_name=observation.display_name, status=observation.status,
            confidence=observation.confidence, confirmation=observation.confirmation,
            summary=observation.summary, started_at=now, last_observed_at=now)
        db.add(existing); await db.flush()
    else:
        existing.status = "active" if observation.status == "active" else existing.status
        existing.confidence = max(existing.confidence, observation.confidence)
        existing.last_observed_at = now; existing.summary = observation.summary
    db.add(HealthEventEvidence(id=uuid_str(), event_id=existing.id,
        conversation_id=conversation_id, message_id=message_id,
        source_type=observation.confirmation, summary=observation.summary, observed_at=now))
    return existing
