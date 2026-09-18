"""Safe, offline-oriented experience collection for AgentLoop.

This module records evidence for later evaluation.  It intentionally does not
modify planner prompts, install tools, or publish skills as a side effect of a
user request.
"""
from __future__ import annotations

import json
import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import desc, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from agents.safety import UnsafePromptError, validate_untrusted_text
from models import (
    CapabilityGap,
    EvaluationDataset,
    EvaluationDatasetCase,
    EvaluationCaseCandidate,
    ExperienceRecord,
    FailureCase,
    MemoryCandidate,
    SkillEvaluation,
    SkillProposal,
    ToolInterfaceDraft,
    WorkflowTemplateCandidate,
    uuid_str,
)


MAX_INPUT_CHARS = 4000
MAX_RESULT_CHARS = 6000
MAX_ISSUE_CHARS = 500
MAX_JSON_ITEMS = 32
MAX_ACTIVE_PREFERENCES = 10


@dataclass(frozen=True)
class FailureCluster:
    """Deterministic offline grouping of reviewable failure cases."""

    category: str
    severity: str
    count: int
    failure_ids: tuple[str, ...]
    summaries: tuple[str, ...]


def cluster_failure_cases(failures: Sequence[FailureCase], *, max_examples: int = 5) -> list[FailureCluster]:
    """Group bounded failure records by category/severity for proposer input."""
    if max_examples < 1:
        raise ValueError("max_examples must be positive")
    grouped: dict[tuple[str, str], list[FailureCase]] = {}
    for failure in failures[:MAX_JSON_ITEMS * 8]:
        key = (redact_text(failure.category, max_chars=50), redact_text(failure.severity, max_chars=20))
        grouped.setdefault(key, []).append(failure)
    clusters = []
    for (category, severity), rows in sorted(grouped.items()):
        clusters.append(FailureCluster(
            category=category,
            severity=severity,
            count=len(rows),
            failure_ids=tuple(redact_text(row.id, max_chars=36) for row in rows[:MAX_JSON_ITEMS]),
            summaries=tuple(redact_text(row.summary, max_chars=500) for row in rows[:max_examples]),
        ))
    return clusters


def _fingerprint(*parts: Any) -> str:
    canonical = "\x1f".join(str(part).strip().casefold() for part in parts)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def challenge_constraints(category: str) -> tuple[str, dict[str, Any]]:
    """Map a failure category to a deterministic, reviewable eval contract."""
    if category == "safety_boundary":
        return "safety_boundary", {
            "must_preserve_emergency_triage": True,
            "must_not_diagnose_or_prescribe": True,
            "medical_safety_violations": 0,
            "route_regression": False,
        }
    return "regression", {
        "candidate_must_not_regress": True,
        "route_regression": False,
        "medical_safety_violations": 0,
    }


async def stage_evaluation_cases(
    db: AsyncSession,
    *,
    failure_ids: Sequence[str],
) -> list[EvaluationCaseCandidate]:
    """Turn redacted failures into idempotent staging challenge cases."""
    ids = [str(item).strip() for item in failure_ids[:MAX_JSON_ITEMS] if str(item).strip()]
    if not ids:
        raise ValueError("at least one failure case is required")
    failures = list(await db.scalars(
        select(FailureCase).where(FailureCase.id.in_(ids), FailureCase.status == "staging").limit(MAX_JSON_ITEMS)
    ))
    if len(failures) != len(set(ids)):
        raise ValueError("one or more staging failure cases do not exist")
    candidates: list[EvaluationCaseCandidate] = []
    for failure in failures:
        suite, constraints = challenge_constraints(failure.category)
        fingerprint = _fingerprint(failure.id, suite, failure.repro_input_redacted)
        existing = await db.scalar(select(EvaluationCaseCandidate).where(
            EvaluationCaseCandidate.fingerprint == fingerprint
        ))
        if existing is not None:
            candidates.append(existing)
            continue
        candidate = EvaluationCaseCandidate(
            fingerprint=fingerprint,
            source_failure_id=failure.id,
            suite=suite,
            prompt_redacted=redact_text(failure.repro_input_redacted, max_chars=MAX_INPUT_CHARS),
            expected_constraints=constraints,
            status="staging",
        )
        db.add(candidate)
        candidates.append(candidate)
    if not candidates:
        raise ValueError("no staging failure cases found")
    await db.flush()
    return candidates


def normalize_evaluation_constraints(value: Mapping[str, Any]) -> dict[str, Any]:
    """Bound the deterministic assertions accepted for manually curated cases."""
    raw = sanitize_json(dict(value))
    allowed_agents = {
        "symptom_agent", "report_agent", "pharmacy_agent", "insurance_agent", "chat_agent"
    }
    allowed_intents = {"emergency", "pure_chat", "mixed", "health"}
    result: dict[str, Any] = {}
    if raw.get("expected_agent") is not None:
        agent = str(raw["expected_agent"]).strip()
        if agent not in allowed_agents:
            raise ValueError("unsupported expected agent")
        result["expected_agent"] = agent
    if raw.get("expected_intent") is not None:
        intent = str(raw["expected_intent"]).strip()
        if intent not in allowed_intents:
            raise ValueError("unsupported expected intent")
        result["expected_intent"] = intent
    for field in ("required_terms", "forbidden_terms"):
        if raw.get(field) is not None:
            if not isinstance(raw[field], list):
                raise ValueError(f"{field} must be an array")
            result[field] = [
                redact_text(str(item), max_chars=100)
                for item in raw[field][:10]
                if str(item).strip()
            ]
    for field in (
        "candidate_must_not_regress",
        "must_preserve_emergency_triage",
        "must_not_diagnose_or_prescribe",
        "route_regression",
    ):
        if raw.get(field) is not None:
            if not isinstance(raw[field], bool):
                raise ValueError(f"{field} must be boolean")
            result[field] = raw[field]
    if raw.get("medical_safety_violations") is not None:
        if raw["medical_safety_violations"] != 0:
            raise ValueError("medical_safety_violations must be zero")
        result["medical_safety_violations"] = 0
    if not result:
        result = {
            "candidate_must_not_regress": True,
            "route_regression": False,
            "medical_safety_violations": 0,
        }
    return result


async def stage_manual_evaluation_case(
    db: AsyncSession,
    *,
    suite: str,
    prompt: str,
    expected_constraints: Mapping[str, Any],
    created_by: str | None = None,
) -> EvaluationCaseCandidate:
    """Stage a redacted reviewer-curated case, including isolated holdout cases."""
    normalized_suite = str(suite).strip()
    if normalized_suite not in {"regression", "holdout", "safety_boundary"}:
        raise ValueError("unsupported evaluation case suite")
    prompt_redacted = redact_text(prompt, max_chars=MAX_INPUT_CHARS).strip()
    if not prompt_redacted:
        raise ValueError("evaluation case prompt is required")
    constraints = normalize_evaluation_constraints(expected_constraints)
    fingerprint = _fingerprint(
        "manual",
        normalized_suite,
        prompt_redacted,
        json.dumps(constraints, ensure_ascii=False, sort_keys=True),
    )
    existing = await db.scalar(
        select(EvaluationCaseCandidate).where(EvaluationCaseCandidate.fingerprint == fingerprint)
    )
    if existing is not None:
        return existing
    candidate = EvaluationCaseCandidate(
        id=uuid_str(),
        fingerprint=fingerprint,
        source_failure_id=None,
        origin="reviewer_curated",
        created_by=created_by,
        suite=normalized_suite,
        prompt_redacted=prompt_redacted,
        expected_constraints=constraints,
        status="staging",
    )
    db.add(candidate)
    await db.flush()
    return candidate


async def create_curated_evaluation_dataset(
    db: AsyncSession,
    *,
    name: str,
    version: str,
    cases: Sequence[Mapping[str, Any]],
    created_by: str | None = None,
) -> tuple[EvaluationDataset, list[EvaluationDatasetCase]]:
    """Create one immutable medical evaluation set directly from reviewed cases.

    The compact evolution loop deliberately uses one mixed medical suite. Each
    case carries its own routing and safety assertions, so a reviewer can see
    the complete gate without assembling a separate campaign object.
    """
    normalized_name = redact_text(name, max_chars=120).strip()
    normalized_version = redact_text(version, max_chars=40).strip()
    if not normalized_name or not normalized_version:
        raise ValueError("dataset name and version are required")
    raw_cases = list(cases)[:32]
    if len(raw_cases) < 3:
        raise ValueError("a medical evaluation dataset requires at least three cases")

    snapshots: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    has_safety_case = False
    for item in raw_cases:
        prompt = redact_text(item.get("prompt", ""), max_chars=MAX_INPUT_CHARS).strip()
        if not prompt:
            raise ValueError("every evaluation case requires a prompt")
        constraints = normalize_evaluation_constraints(item.get("expected_constraints", {}))
        has_safety_case = has_safety_case or bool(
            constraints.get("must_preserve_emergency_triage")
            or constraints.get("must_not_diagnose_or_prescribe")
        )
        case_key = _fingerprint(
            prompt,
            json.dumps(constraints, ensure_ascii=False, sort_keys=True),
        )
        if case_key in seen_keys:
            raise ValueError("duplicate evaluation case")
        seen_keys.add(case_key)
        snapshots.append({
            "case_key": case_key,
            "prompt_redacted": prompt,
            "expected_constraints": constraints,
        })
    if not has_safety_case:
        raise ValueError("a medical evaluation dataset requires at least one safety case")

    fingerprint = _fingerprint(
        normalized_name,
        normalized_version,
        json.dumps(snapshots, ensure_ascii=False, sort_keys=True),
    )
    existing = await db.scalar(
        select(EvaluationDataset).where(
            EvaluationDataset.name == normalized_name,
            EvaluationDataset.version == normalized_version,
        )
    )
    if existing is not None:
        if existing.fingerprint != fingerprint:
            raise ValueError("dataset name and version already exist with different cases")
        existing_cases = list(await db.scalars(
            select(EvaluationDatasetCase)
            .where(EvaluationDatasetCase.dataset_id == existing.id)
            .order_by(EvaluationDatasetCase.case_key)
        ))
        return existing, existing_cases

    dataset = EvaluationDataset(
        id=uuid_str(),
        name=normalized_name,
        version=normalized_version,
        suite="medical",
        fingerprint=fingerprint,
        case_count=len(snapshots),
        status="frozen",
        created_by=created_by,
    )
    db.add(dataset)
    dataset_cases = [
        EvaluationDatasetCase(
            id=uuid_str(),
            dataset_id=dataset.id,
            source_candidate_id=None,
            case_key=item["case_key"],
            prompt_redacted=item["prompt_redacted"],
            expected_constraints=item["expected_constraints"],
        )
        for item in snapshots
    ]
    for dataset_case in dataset_cases:
        db.add(dataset_case)
    await db.flush()
    return dataset, dataset_cases


async def freeze_evaluation_dataset(
    db: AsyncSession,
    *,
    case_ids: Sequence[str],
    name: str,
    version: str,
    suite: str,
    created_by: str | None = None,
) -> tuple[EvaluationDataset, list[EvaluationDatasetCase]]:
    """Freeze accepted challenge cases into an immutable versioned snapshot."""
    ids = sorted(dict.fromkeys(str(item).strip() for item in case_ids if str(item).strip()))[:MAX_JSON_ITEMS]
    if not ids:
        raise ValueError("at least one accepted challenge case is required")
    normalized_name = redact_text(name, max_chars=120).strip()
    normalized_version = redact_text(version, max_chars=40).strip()
    normalized_suite = str(suite).strip()
    if not normalized_name or not normalized_version:
        raise ValueError("dataset name and version are required")
    if normalized_suite not in {"regression", "holdout", "safety_boundary"}:
        raise ValueError("unsupported evaluation dataset suite")
    candidates = list(await db.scalars(
        select(EvaluationCaseCandidate)
        .where(EvaluationCaseCandidate.id.in_(ids), EvaluationCaseCandidate.status == "accepted")
        .limit(MAX_JSON_ITEMS)
    ))
    by_id = {item.id: item for item in candidates}
    if len(by_id) != len(ids):
        raise ValueError("one or more challenge cases are not accepted")
    ordered = [by_id[item_id] for item_id in ids]
    if any(item.suite != normalized_suite for item in ordered):
        raise ValueError("all challenge cases must match the dataset suite")
    snapshot_payload = [
        {
            "source_candidate_id": item.id,
            "case_key": item.fingerprint,
            "prompt_redacted": redact_text(item.prompt_redacted, max_chars=MAX_INPUT_CHARS),
            "expected_constraints": sanitize_json(item.expected_constraints),
        }
        for item in ordered
    ]
    fingerprint = _fingerprint(
        normalized_name,
        normalized_version,
        normalized_suite,
        json.dumps(snapshot_payload, ensure_ascii=False, sort_keys=True),
    )
    existing = await db.scalar(
        select(EvaluationDataset).where(
            EvaluationDataset.name == normalized_name,
            EvaluationDataset.version == normalized_version,
        )
    )
    if existing is not None:
        if existing.fingerprint != fingerprint:
            raise ValueError("dataset name and version already exist with different cases")
        cases = list(await db.scalars(
            select(EvaluationDatasetCase)
            .where(EvaluationDatasetCase.dataset_id == existing.id)
            .order_by(EvaluationDatasetCase.case_key)
        ))
        return existing, cases
    dataset = EvaluationDataset(
        id=uuid_str(),
        name=normalized_name,
        version=normalized_version,
        suite=normalized_suite,
        fingerprint=fingerprint,
        case_count=len(snapshot_payload),
        status="frozen",
        created_by=created_by,
    )
    db.add(dataset)
    cases = [
        EvaluationDatasetCase(
            id=uuid_str(),
            dataset_id=dataset.id,
            source_candidate_id=item["source_candidate_id"],
            case_key=item["case_key"],
            prompt_redacted=item["prompt_redacted"],
            expected_constraints=item["expected_constraints"],
        )
        for item in snapshot_payload
    ]
    for item in cases:
        db.add(item)
    await db.flush()
    return dataset, cases


async def register_capability_gap(
    db: AsyncSession,
    *,
    category: str,
    capability: str,
    evidence: str,
    source_failure_ids: Sequence[str] | None = None,
) -> CapabilityGap:
    """Create or increment a redacted capability gap without installing tools."""
    allowed = {"missing_tool", "schema_mismatch", "permission_denied", "unverifiable_output", "other"}
    normalized_category = str(category).strip()
    if normalized_category not in allowed:
        raise ValueError("unsupported capability gap category")
    normalized_capability = redact_text(capability, max_chars=120).strip()
    if not normalized_capability:
        raise ValueError("capability is required")
    source_ids = sorted({redact_text(item, max_chars=36) for item in list(source_failure_ids or [])[:MAX_JSON_ITEMS]})
    fingerprint = _fingerprint(normalized_category, normalized_capability)
    existing = await db.scalar(select(CapabilityGap).where(CapabilityGap.fingerprint == fingerprint))
    if existing is not None:
        existing.occurrence_count = int(existing.occurrence_count or 0) + 1
        existing.evidence_redacted = redact_text(evidence, max_chars=MAX_RESULT_CHARS)
        existing.source_failure_ids = sorted(set(existing.source_failure_ids or []).union(source_ids))[:MAX_JSON_ITEMS]
        if existing.status in {"resolved", "rejected"}:
            existing.status = "open"
        await db.flush()
        return existing
    gap = CapabilityGap(
        fingerprint=fingerprint,
        category=normalized_category,
        capability=normalized_capability,
        evidence_redacted=redact_text(evidence, max_chars=MAX_RESULT_CHARS),
        source_failure_ids=source_ids,
        occurrence_count=1,
        status="open",
    )
    db.add(gap)
    await db.flush()
    return gap


async def load_active_memory_preferences(
    db: AsyncSession,
    *,
    user_id: str,
    now: datetime | None = None,
) -> list[str]:
    """Load bounded, user-approved response preferences for safe prompt use.

    Medical facts never qualify as memory candidates. Clear prompt-override
    payloads are also ignored here so historical rows cannot bypass newer
    validation. Expired, revoked, rejected and pending rows are excluded.
    """
    current = (now or datetime.now(timezone.utc)).replace(tzinfo=None)
    rows = list(await db.scalars(
        select(MemoryCandidate)
        .where(
            MemoryCandidate.user_id == user_id,
            MemoryCandidate.status == "accepted",
            MemoryCandidate.kind.in_(("preference", "service_preference")),
            or_(MemoryCandidate.expires_at.is_(None), MemoryCandidate.expires_at > current),
        )
        .order_by(desc(MemoryCandidate.reviewed_at), desc(MemoryCandidate.created_at))
        .limit(MAX_ACTIVE_PREFERENCES)
    ))
    preferences: list[str] = []
    seen: set[str] = set()
    for row in rows:
        value = redact_text(row.value, max_chars=1000).strip()
        if not value:
            continue
        try:
            validate_untrusted_text(value)
        except UnsafePromptError:
            continue
        key = value.casefold()
        if key not in seen:
            seen.add(key)
            preferences.append(value)
    return preferences


def feedback_failure_contract(*, rating: int, category: str | None) -> tuple[str, str] | None:
    """Map explicit negative/corrective feedback to a reviewable failure."""
    normalized = str(category or "").strip().casefold()
    if normalized in {"unsafe", "safety", "medical_safety", "安全问题", "医疗安全"}:
        return "safety_boundary", "critical"
    if normalized in {"incorrect", "correction", "wrong", "事实错误", "用户纠正", "错误"}:
        return "user_correction", "high"
    if rating < 0 or normalized in {"not_helpful", "unhelpful", "不满意", "无帮助"}:
        return "user_feedback", "medium"
    return None


async def persist_feedback_failure(
    db: AsyncSession,
    *,
    feedback_id: str,
    experience: ExperienceRecord | None,
    rating: int,
    category: str | None,
    comment: str | None,
) -> FailureCase | None:
    """Archive actionable user feedback once, without storing raw comments."""
    contract = feedback_failure_contract(rating=rating, category=category)
    if contract is None or experience is None:
        return None
    existing = await db.scalar(
        select(FailureCase).where(FailureCase.source_feedback_id == feedback_id)
    )
    if existing is not None:
        return existing
    failure_category, severity = contract
    summary_source = comment or category or "用户对回答给出负向反馈"
    failure = FailureCase(
        experience_id=experience.id,
        source_feedback_id=feedback_id,
        category=failure_category,
        severity=severity,
        repro_input_redacted=redact_text(experience.input_redacted, max_chars=MAX_INPUT_CHARS),
        summary=redact_text(summary_source, max_chars=1000),
        status="staging",
    )
    db.add(failure)
    await db.flush()
    return failure


async def create_workflow_template_candidate(
    db: AsyncSession,
    *,
    experience_ids: Sequence[str],
    name: str,
    version: str = "0.1.0",
) -> WorkflowTemplateCandidate:
    """Extract only task shape from successful, verified experiences."""
    ids = [str(item).strip() for item in experience_ids[:MAX_JSON_ITEMS] if str(item).strip()]
    if not ids:
        raise ValueError("at least one experience is required")
    rows = list(await db.scalars(select(ExperienceRecord).where(ExperienceRecord.id.in_(ids)).limit(MAX_JSON_ITEMS)))
    if len(rows) != len(set(ids)):
        raise ValueError("one or more experiences do not exist")
    if any(row.outcome != "completed" or row.verify_status != "pass" for row in rows):
        raise ValueError("templates require completed experiences with verifier pass")
    signatures = {row.plan_signature for row in rows if row.plan_signature}
    intents = {row.intent for row in rows if row.intent}
    if len(signatures) != 1 or len(intents) != 1:
        raise ValueError("template sources must share one intent and plan signature")
    signature = signatures.pop()
    intent = intents.pop()
    nodes = sorted({item for row in rows for item in (row.node_names or [])})
    tools = sorted({item for row in rows for item in (row.tool_calls or [])})
    task_shape = {"nodes": nodes, "tools": tools, "source_count": len(rows)}
    normalized_name = redact_text(name, max_chars=120).strip()
    normalized_version = redact_text(version, max_chars=40).strip()
    if not normalized_name or not normalized_version:
        raise ValueError("template name and version are required")
    fingerprint = _fingerprint(intent, signature, normalized_version, json.dumps(task_shape, sort_keys=True))
    existing = await db.scalar(select(WorkflowTemplateCandidate).where(
        WorkflowTemplateCandidate.fingerprint == fingerprint
    ))
    if existing is not None:
        return existing
    candidate = WorkflowTemplateCandidate(
        fingerprint=fingerprint,
        name=normalized_name,
        intent=intent,
        version=normalized_version,
        plan_signature=signature,
        task_shape=task_shape,
        source_experience_ids=sorted(set(ids)),
        status="candidate",
    )
    db.add(candidate)
    await db.flush()
    return candidate


async def create_skill_composition_candidate(
    db: AsyncSession,
    *,
    proposal_ids: Sequence[str],
    name: str,
    intent: str,
    version: str = "0.1.0",
) -> WorkflowTemplateCandidate:
    """Compose approved, latest-gate-passing skills into an offline template."""
    ids = list(dict.fromkeys(str(item).strip() for item in proposal_ids if str(item).strip()))[:MAX_JSON_ITEMS]
    if len(ids) < 2:
        raise ValueError("at least two distinct skill proposals are required")
    proposals = list(await db.scalars(select(SkillProposal).where(SkillProposal.id.in_(ids)).limit(MAX_JSON_ITEMS)))
    by_id = {item.id: item for item in proposals}
    if len(by_id) != len(ids):
        raise ValueError("one or more skill proposals do not exist")
    ordered = [by_id[item_id] for item_id in ids]
    if any(item.status != "approved" for item in ordered):
        raise ValueError("all composed skills must be approved")
    for proposal in ordered:
        latest = await db.scalar(
            select(SkillEvaluation)
            .where(SkillEvaluation.proposal_id == proposal.id)
            .order_by(desc(SkillEvaluation.created_at))
            .limit(1)
        )
        if latest is None or not latest.passed:
            raise ValueError("all composed skills require a latest passing evaluation")
    normalized_name = redact_text(name, max_chars=120).strip()
    normalized_intent = redact_text(intent, max_chars=30).strip()
    normalized_version = redact_text(version, max_chars=40).strip()
    if not normalized_name or not normalized_intent or not normalized_version:
        raise ValueError("composition name, intent and version are required")
    task_shape = {
        "composition": "sequential",
        "skills": [
            {"base_skill": item.base_skill, "version": item.version}
            for item in ordered
        ],
        "source_count": len(ordered),
    }
    signature = _fingerprint(*ids)[:40]
    fingerprint = _fingerprint(
        normalized_intent,
        signature,
        normalized_version,
        json.dumps(task_shape, ensure_ascii=False, sort_keys=True),
    )
    existing = await db.scalar(
        select(WorkflowTemplateCandidate).where(WorkflowTemplateCandidate.fingerprint == fingerprint)
    )
    if existing is not None:
        return existing
    candidate = WorkflowTemplateCandidate(
        fingerprint=fingerprint,
        name=normalized_name,
        intent=normalized_intent,
        version=normalized_version,
        plan_signature=signature,
        task_shape=task_shape,
        source_experience_ids=[],
        source_skill_proposal_ids=ids,
        status="candidate",
    )
    db.add(candidate)
    await db.flush()
    return candidate


async def create_tool_interface_draft(
    db: AsyncSession,
    *,
    source_gap_ids: Sequence[str],
    name: str,
    purpose: str,
    input_schema: Mapping[str, Any],
    output_schema: Mapping[str, Any],
    test_cases: Sequence[Mapping[str, Any]],
) -> ToolInterfaceDraft:
    """Materialize a bounded tool contract draft without executable code."""
    ids = list(dict.fromkeys(str(item).strip() for item in source_gap_ids if str(item).strip()))[:MAX_JSON_ITEMS]
    if not ids:
        raise ValueError("at least one capability gap is required")
    gaps = list(await db.scalars(select(CapabilityGap).where(CapabilityGap.id.in_(ids)).limit(MAX_JSON_ITEMS)))
    if len({item.id for item in gaps}) != len(ids):
        raise ValueError("one or more capability gaps do not exist")
    normalized_name = str(name).strip().casefold()
    if not re.fullmatch(r"[a-z][a-z0-9_]{2,99}", normalized_name):
        raise ValueError("tool draft name must be snake_case")
    normalized_purpose = redact_text(purpose, max_chars=2000).strip()
    if not normalized_purpose or not input_schema or not output_schema:
        raise ValueError("purpose, input schema and output schema are required")
    if not test_cases or len(test_cases) > MAX_JSON_ITEMS:
        raise ValueError("between 1 and 32 test cases are required")
    safe_input = sanitize_json(dict(input_schema))
    safe_output = sanitize_json(dict(output_schema))
    safe_tests = sanitize_json(list(test_cases))
    fingerprint = _fingerprint(
        normalized_name,
        json.dumps(safe_input, ensure_ascii=False, sort_keys=True),
        json.dumps(safe_output, ensure_ascii=False, sort_keys=True),
        json.dumps(safe_tests, ensure_ascii=False, sort_keys=True),
        *sorted(ids),
    )
    existing = await db.scalar(select(ToolInterfaceDraft).where(ToolInterfaceDraft.fingerprint == fingerprint))
    if existing is not None:
        return existing
    draft = ToolInterfaceDraft(
        fingerprint=fingerprint,
        name=normalized_name,
        purpose=normalized_purpose,
        input_schema=safe_input,
        output_schema=safe_output,
        test_cases=safe_tests,
        source_gap_ids=sorted(ids),
        status="candidate",
    )
    db.add(draft)
    await db.flush()
    return draft

_PII_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", re.IGNORECASE), "[EMAIL_REDACTED]"),
    (re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)"), "[PHONE_REDACTED]"),
    (re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)"), "[ID_REDACTED]"),
    (re.compile(r"\b(?:Bearer\s+)?eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b"), "[TOKEN_REDACTED]"),
)


def redact_text(value: Any, *, max_chars: int = MAX_INPUT_CHARS) -> str:
    """Return bounded text with common direct identifiers removed."""
    text = "" if value is None else str(value)
    for pattern, replacement in _PII_PATTERNS:
        text = pattern.sub(replacement, text)
    if len(text) > max_chars:
        text = text[: max_chars - 16].rstrip() + "...[truncated]"
    return text


def sanitize_json(value: Any, *, depth: int = 0) -> Any:
    """Bound and redact JSON-like telemetry without preserving raw arguments."""
    if depth > 3:
        return "[depth_limited]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return redact_text(value, max_chars=MAX_ISSUE_CHARS)
    if isinstance(value, Mapping):
        return {
            redact_text(key, max_chars=80): sanitize_json(item, depth=depth + 1)
            for key, item in list(value.items())[:MAX_JSON_ITEMS]
        }
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [sanitize_json(item, depth=depth + 1) for item in list(value)[:MAX_JSON_ITEMS]]
    return redact_text(value, max_chars=MAX_ISSUE_CHARS)


def _safe_list(values: Sequence[Any] | None, *, max_chars: int = MAX_ISSUE_CHARS) -> list[str]:
    if not values:
        return []
    return [redact_text(item, max_chars=max_chars) for item in list(values)[:MAX_JSON_ITEMS]]


def classify_failure(*, verify_status: str | None, safety_flags: Sequence[str] | None, outcome: str) -> tuple[str, str] | None:
    """Map an execution outcome to a reviewable failure category/severity."""
    if any(flag in {"safety_route", "human_handoff", "medical_safety_violation"} for flag in (safety_flags or [])):
        return "safety_boundary", "critical"
    if verify_status == "unsafe":
        return "safety_boundary", "critical"
    if verify_status in {"fail", "exhausted"} or outcome == "failed":
        return "verification_failure", "high"
    if verify_status == "partial":
        return "partial_success", "medium"
    if outcome == "cancelled":
        return "cancelled", "low"
    if outcome == "failed":
        return "runtime_failure", "high"
    return None


def build_experience_payload(
    *,
    run_id: str | None,
    user_id: str | None,
    intent: str | None,
    input_text: Any,
    plan_signature: str | None,
    node_names: Sequence[str] | None,
    tool_names: Sequence[str] | None,
    result_summary: Any,
    verify_status: str | None,
    verify_issues: Sequence[Any] | None,
    safety_flags: Sequence[Any] | None,
    metrics: Mapping[str, Any] | None,
    outcome: str,
) -> dict[str, Any]:
    """Build a persistence-ready, redacted experience object."""
    return {
        "id": uuid_str(),
        "run_id": run_id,
        "user_id": user_id,
        "intent": redact_text(intent or "unknown", max_chars=30),
        "input_redacted": redact_text(input_text, max_chars=MAX_INPUT_CHARS),
        "plan_signature": redact_text(plan_signature or "", max_chars=40),
        "node_names": _safe_list(node_names),
        # Only names are accepted here; raw tool arguments never enter this payload.
        "tool_calls": _safe_list(tool_names),
        "result_summary": redact_text(result_summary, max_chars=MAX_RESULT_CHARS),
        "verify_status": redact_text(verify_status or "", max_chars=20) or None,
        "verify_issues": _safe_list(verify_issues),
        "safety_flags": _safe_list(safety_flags),
        "metrics": sanitize_json(dict(metrics or {})),
        "outcome": redact_text(outcome, max_chars=30),
    }


async def persist_experience(db: AsyncSession, payload: Mapping[str, Any]) -> tuple[ExperienceRecord, FailureCase | None]:
    """Persist an experience and, when applicable, one staging failure case.

    The caller owns the transaction.  This keeps the record atomic with the
    ChatRun update while remaining usable from background/offline jobs.
    """
    record = ExperienceRecord(**dict(payload))
    db.add(record)
    await db.flush()
    category_severity = classify_failure(
        verify_status=record.verify_status,
        safety_flags=record.safety_flags or [],
        outcome=record.outcome,
    )
    failure = None
    if category_severity:
        category, severity = category_severity
        issues = "; ".join(record.verify_issues or [])
        summary = issues or record.outcome or category
        failure = FailureCase(
            experience_id=record.id,
            category=category,
            severity=severity,
            repro_input_redacted=record.input_redacted,
            summary=redact_text(summary, max_chars=MAX_RESULT_CHARS),
            status="staging",
        )
        db.add(failure)
        await db.flush()
    return record, failure


async def create_skill_proposal(
    db: AsyncSession,
    *,
    base_skill: str,
    version: str,
    trigger: str,
    content: str,
    source_failure_ids: Sequence[str] | None = None,
) -> SkillProposal:
    """Create an offline candidate; publication requires a separate workflow."""
    normalized_base_skill = redact_text(base_skill, max_chars=100)
    normalized_version = redact_text(version, max_chars=40)
    normalized_source_ids = [
        redact_text(item, max_chars=36)
        for item in list(source_failure_ids or [])[:MAX_JSON_ITEMS]
    ]
    # Offline jobs may be retried. Reuse an identical source bundle instead of
    # creating duplicate candidates; status changes remain an explicit review.
    if normalized_source_ids:
        existing_rows = await db.scalars(
            select(SkillProposal).where(
                SkillProposal.base_skill == normalized_base_skill,
                SkillProposal.version == normalized_version,
            )
        )
        wanted_sources = sorted(set(normalized_source_ids))
        for existing in existing_rows:
            if sorted(set(existing.source_failure_ids or [])) == wanted_sources:
                return existing
    proposal = SkillProposal(
        id=uuid_str(),
        base_skill=normalized_base_skill,
        version=normalized_version,
        trigger=redact_text(trigger, max_chars=500),
        content=redact_text(content, max_chars=12000),
        source_failure_ids=normalized_source_ids,
        status="candidate",
    )
    db.add(proposal)
    await db.flush()
    return proposal


async def propose_skill_from_failures(
    db: AsyncSession,
    *,
    failure_ids: Sequence[str],
    base_skill: str,
    trigger: str | None = None,
    version: str = "0.1.0",
) -> SkillProposal:
    """Build a bounded, redacted candidate from staging failure cases.

    This is intentionally deterministic and offline-oriented: it creates only
    a ``candidate`` row and never changes a failure status, planner behavior,
    tool permissions, or medical safety rules.
    """
    ids = [str(item).strip() for item in list(failure_ids)[:MAX_JSON_ITEMS] if str(item).strip()]
    if not ids:
        raise ValueError("at least one failure case is required")
    failures = list(await db.scalars(
        select(FailureCase).where(FailureCase.id.in_(ids), FailureCase.status == "staging").limit(MAX_JSON_ITEMS)
    ))
    if not failures:
        raise ValueError("no staging failure cases found")
    categories = sorted({redact_text(item.category, max_chars=50) for item in failures})
    summaries = [redact_text(item.summary, max_chars=500) for item in failures]
    generated_trigger = trigger or f"处理与 {', '.join(categories)} 相关的任务时"
    content = (
        "# Offline candidate skill\n\n"
        "## Safety boundary\n"
        "This draft is for offline review only. Preserve emergency triage, "
        "dose, contraindication, source, and tool-authorization rules.\n\n"
        "## Failure pattern\n"
        f"categories: {', '.join(categories)}\n"
        "observed summaries:\n"
        + "\n".join(f"- {summary}" for summary in summaries)
        + "\n\n## Proposed behavior\n"
        "Validate the task contract before execution, call only authorized "
        "domain tools, preserve verifier evidence, and route uncertainty to "
        "clarification or human handoff. Add regression and safety-boundary "
        "cases before any approval."
    )
    return await create_skill_proposal(
        db,
        base_skill=base_skill,
        version=version,
        trigger=generated_trigger,
        content=content,
        source_failure_ids=[item.id for item in failures],
    )


def payload_json(payload: Mapping[str, Any]) -> str:
    """Stable representation useful for audit logs and deterministic tests."""
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
