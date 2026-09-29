"""User-scoped memory evolution and Hermes-style Markdown projection.

The database remains the source of truth. Markdown files are a bounded,
human-readable projection for audit and retrieval; they never become prompt
instructions and never store inferred diagnoses.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import os
import re
from typing import Any

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from config import get_settings
from models import Conversation, HealthEvent, MemoryCandidate, Message
from agents.safety import UnsafePromptError, validate_untrusted_text


SHADOW_INTERVAL = 10
_PREFERENCE_RULES = (
    ("prefer_plain_language", ("说简单一点", "用大白话", "通俗一点", "少用专业术语"), "回答使用通俗、少术语的表达"),
    ("prefer_step_by_step", ("一步一步", "分步骤", "按步骤说", "按步骤讲", "详细步骤"), "回答按步骤组织"),
    ("prefer_conclusion_first", ("先说结论", "先给结论", "直接告诉我结论"), "回答先给结论再解释"),
    ("prefer_professional_language", ("专业一点", "多用专业术语", "按专业人士说"), "回答可使用必要的专业术语"),
)

_PROFILE_RULES = (
    ("profile_programmer", ("我是程序员", "我是一名程序员", "我的职业是程序员", "做程序员"), "用户职业：程序员"),
    ("profile_doctor", ("我是医生", "我是一名医生", "我的职业是医生"), "用户职业：医生"),
    ("profile_nurse", ("我是护士", "我是一名护士", "我的职业是护士"), "用户职业：护士"),
    ("profile_pharmacist", ("我是药师", "我是一名药师", "我的职业是药师"), "用户职业：药师"),
)
_OCCUPATION_RE = re.compile(
    r"(?:我(?:是|是一名|叫做)|我的职业是|我从事)(?P<occupation>[\u4e00-\u9fffA-Za-z0-9·]{2,20})"
    r"(?:司机|工程师|教师|老师|学生|研究生|医生|护士|药师|程序员|设计师|律师|会计|工人|销售|管理|技师|厨师|农民)(?=[，,。！？!?；;]|$)"
)


def _projection_root() -> Path:
    configured = get_settings().memory_projection_root
    root = Path(configured)
    if not root.is_absolute():
        root = Path(__file__).resolve().parent / root
    return root


def _safe_text(value: Any, limit: int = 160) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:limit]


def extract_explicit_preferences(text: str) -> list[dict[str, Any]]:
    """Extract only explicit, low-risk communication preferences."""
    source = _safe_text(text, 1000)
    found: list[dict[str, Any]] = []
    for key, markers, value in _PREFERENCE_RULES:
        if any(marker in source for marker in markers):
            found.append({"key": key, "value": value, "source": "user_explicit", "confidence": 1.0})
    for key, markers, value in _PROFILE_RULES:
        if any(marker in source for marker in markers):
            found.append({"key": key, "value": value, "source": "user_explicit_profile", "confidence": 1.0})
    # Open-vocabulary occupation extraction: keep the trigger explicit and
    # constrain the suffix to a small occupation lexicon. This handles
    # ``我是货车司机`` without maintaining a brittle list of every job title.
    occupation_match = _OCCUPATION_RE.search(source)
    if occupation_match:
        occupation = occupation_match.group("occupation")
        value = f"用户职业：{occupation}"
        if not any(item["value"] == value for item in found):
            found.append({"key": "profile_occupation", "value": value, "source": "user_explicit_profile", "confidence": 1.0})
    return found


async def _upsert_preferences(db: AsyncSession, *, user_id: str, text: str, conversation_id: str | None) -> int:
    count = 0
    for item in extract_explicit_preferences(text):
        try:
            validate_untrusted_text(item["value"])
        except UnsafePromptError:
            continue
        existing = await db.scalar(select(MemoryCandidate).where(
            MemoryCandidate.user_id == user_id,
            MemoryCandidate.kind == "preference",
            MemoryCandidate.value == item["value"],
            MemoryCandidate.status.in_(("pending", "accepted")),
        ).limit(1))
        if existing is not None:
            continue
        db.add(MemoryCandidate(
            user_id=user_id,
            kind="preference",
            value=item["value"],
            source=item["source"],
            confidence=item["confidence"],
            status="accepted",
            provenance={"extractor": "shadow_memory_agent", "conversation_id": conversation_id, "key": item["key"]},
        ))
        count += 1
    if count:
        await db.flush()
    return count


async def _user_message_count(db: AsyncSession, user_id: str) -> int:
    value = await db.scalar(
        select(func.count(Message.id)).join(Conversation, Conversation.id == Message.conversation_id).where(
            Conversation.user_id == user_id, Message.role == "user"
        )
    )
    return int(value or 0)


async def project_user_memory(db: AsyncSession, *, user_id: str) -> dict[str, Any]:
    """Write bounded Markdown projections atomically from accepted DB state."""
    preferences = list(await db.scalars(select(MemoryCandidate).where(
        MemoryCandidate.user_id == user_id,
        MemoryCandidate.kind.in_(("preference", "service_preference")),
        MemoryCandidate.status == "accepted",
    ).order_by(MemoryCandidate.created_at.desc()).limit(20)))
    events = list(await db.scalars(select(HealthEvent).where(
        HealthEvent.user_id == user_id,
        HealthEvent.status.in_(("candidate", "possible", "active", "resolved", "archived")),
    ).order_by(HealthEvent.last_observed_at.desc()).limit(30)))
    root = _projection_root() / _safe_text(user_id, 100)
    root.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    profile = "# User Profile\n\n- user_id: `" + _safe_text(user_id, 100) + "`\n- generated_at: " + now + "\n"
    preference_lines = ["# Communication Preferences", "", "<!-- Generated by Shadow Memory Agent. DB is the source of truth. -->"]
    for row in preferences:
        preference_lines.append(f"- { _safe_text(row.value) } (confidence: {float(row.confidence or 0):.2f})")
    active_lines = ["# Active Health Context", "", "<!-- User-reported context only; not a diagnosis. -->"]
    history_lines = ["# Health History", "", "<!-- Resolved events remain available for explicit history queries. -->"]
    for row in events:
        line = f"- { _safe_text(row.display_name) } | status={_safe_text(row.status)} | last_observed={row.last_observed_at.isoformat()} | source={_safe_text(row.confirmation)}"
        (active_lines if row.status in {"candidate", "possible", "active"} else history_lines).append(line)
    files = {
        "profile.md": profile,
        "preferences.md": "\n".join(preference_lines) + "\n",
        "active_health_context.md": "\n".join(active_lines) + "\n",
        "health_history.md": "\n".join(history_lines) + "\n",
    }
    for name, content in files.items():
        target = root / name
        temp = target.with_suffix(target.suffix + ".tmp")
        temp.write_text(content, encoding="utf-8")
        os.replace(temp, target)
    return {"user_id": user_id, "files": sorted(files), "preference_count": len(preferences), "health_event_count": len(events), "generated_at": now}


async def maybe_run_shadow_memory(
    db: AsyncSession,
    *,
    user_id: str,
    latest_user_text: str,
    conversation_id: str | None = None,
) -> dict[str, Any]:
    """Run after a turn; explicit preferences are immediate, otherwise every 10 user turns."""
    extracted = await _upsert_preferences(db, user_id=user_id, text=latest_user_text, conversation_id=conversation_id)
    turns = await _user_message_count(db, user_id)
    triggered = bool(extracted or (turns > 0 and turns % SHADOW_INTERVAL == 0))
    if not triggered:
        return {"triggered": False, "turns": turns, "extracted": extracted}
    projection = await project_user_memory(db, user_id=user_id)
    return {"triggered": True, "turns": turns, "extracted": extracted, "projection": projection}


async def reset_user_memory_projection(*, user_id: str) -> bool:
    """Remove only the Markdown projection; database/audit records remain intact."""
    root = _projection_root() / _safe_text(user_id, 100)
    if not root.exists():
        return False
    for child in root.glob("*.md"):
        child.unlink(missing_ok=True)
    try:
        root.rmdir()
    except OSError:
        pass
    return True
