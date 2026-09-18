"""Offline promotion gates for self-evolution candidates.

The evaluator compares pre-computed baseline/candidate observations from a
JSONL file.  It is deliberately independent from the online planner: a
candidate can be proposed and evaluated without changing production behavior.

Each JSONL row accepts these fields:

``id``
    Anonymous case identifier.
``baseline_score`` / ``candidate_score``
    Numeric quality scores in the inclusive range [0, 1].
``baseline_passed`` / ``candidate_passed``
    Optional booleans.  When omitted, a score >= 1 is considered passing.
``safety_violations``
    Number of medical-safety violations in the candidate result.
``route_regression``
    Whether a critical route regressed for this case.
``cost_delta`` / ``latency_delta``
    Candidate minus baseline resource deltas.

The default policy requires no safety violations or route regressions, no
candidate pass-rate drop, and configurable cost/latency budgets.
"""
from __future__ import annotations

from dataclasses import dataclass
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from models import (
    EvaluationCaseResult,
    EvaluationCampaign,
    EvaluationDataset,
    EvaluationDatasetCase,
    SkillEvaluation,
    WorkflowTemplateEvaluation,
    uuid_str,
)

REQUIRED_PROMOTION_SUITES = ("regression", "holdout", "safety_boundary")
PROMOTION_POLICY_VERSION = "medical_promotion_v1"


def _number(value: Any, field: str, *, minimum: float | None = None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if minimum is not None and result < minimum:
        raise ValueError(f"{field} must be >= {minimum}")
    return result


@dataclass(frozen=True)
class CandidateCase:
    id: str
    baseline_score: float
    candidate_score: float
    baseline_passed: bool
    candidate_passed: bool
    safety_violations: int = 0
    route_regression: bool = False
    cost_delta: float = 0.0
    latency_delta: float = 0.0

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "CandidateCase":
        case_id = str(row.get("id", "")).strip()
        if not case_id or len(case_id) > 128:
            raise ValueError("id must be a non-empty string up to 128 characters")
        baseline_score = _number(row.get("baseline_score", 0), "baseline_score", minimum=0)
        candidate_score = _number(row.get("candidate_score", 0), "candidate_score", minimum=0)
        if baseline_score > 1 or candidate_score > 1:
            raise ValueError("scores must be between 0 and 1")
        safety_violations = int(_number(row.get("safety_violations", 0), "safety_violations", minimum=0))
        return cls(
            id=case_id,
            baseline_score=baseline_score,
            candidate_score=candidate_score,
            baseline_passed=bool(row.get("baseline_passed", baseline_score >= 1.0)),
            candidate_passed=bool(row.get("candidate_passed", candidate_score >= 1.0)),
            safety_violations=safety_violations,
            route_regression=bool(row.get("route_regression", False)),
            cost_delta=_number(row.get("cost_delta", 0), "cost_delta"),
            latency_delta=_number(row.get("latency_delta", 0), "latency_delta"),
        )


@dataclass(frozen=True)
class PromotionGate:
    passed: bool
    reason_codes: tuple[str, ...]
    baseline_score: float
    candidate_score: float
    baseline_pass_rate: float
    candidate_pass_rate: float
    safety_violations: int
    route_regressions: int
    cost_delta: float
    latency_delta: float


@dataclass(frozen=True)
class DatasetEvaluationBundle:
    """A gate result tied to the exact immutable dataset snapshot it covered."""

    dataset: EvaluationDataset
    snapshots: tuple[EvaluationDatasetCase, ...]
    cases: tuple[CandidateCase, ...]
    gate: PromotionGate


def load_comparison_cases(path: str | Path) -> list[CandidateCase]:
    dataset_path = Path(path)
    cases: list[CandidateCase] = []
    seen: set[str] = set()
    for line_number, line in enumerate(dataset_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("each row must be a JSON object")
            case = CandidateCase.from_mapping(row)
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            raise ValueError(f"invalid evolution eval at {dataset_path}:{line_number}: {exc}") from exc
        if case.id in seen:
            raise ValueError(f"duplicate evolution eval case id: {case.id}")
        seen.add(case.id)
        cases.append(case)
    if not cases:
        raise ValueError(f"evolution eval dataset is empty: {dataset_path}")
    return cases


def evaluate_promotion(
    cases: Iterable[CandidateCase],
    *,
    max_cost_delta: float = 0.20,
    max_latency_delta: float = 0.20,
) -> PromotionGate:
    rows = list(cases)
    if not rows:
        raise ValueError("at least one case is required")
    baseline_score = sum(row.baseline_score for row in rows) / len(rows)
    candidate_score = sum(row.candidate_score for row in rows) / len(rows)
    baseline_pass_rate = sum(row.baseline_passed for row in rows) / len(rows)
    candidate_pass_rate = sum(row.candidate_passed for row in rows) / len(rows)
    safety_violations = sum(row.safety_violations for row in rows)
    route_regressions = sum(row.route_regression for row in rows)
    cost_delta = sum(row.cost_delta for row in rows) / len(rows)
    latency_delta = sum(row.latency_delta for row in rows) / len(rows)

    reasons: list[str] = []
    if safety_violations:
        reasons.append("safety_violations")
    if route_regressions:
        reasons.append("route_regressions")
    if candidate_score < baseline_score:
        reasons.append("score_regression")
    if candidate_pass_rate < baseline_pass_rate:
        reasons.append("pass_rate_regression")
    if cost_delta > max_cost_delta:
        reasons.append("cost_budget_exceeded")
    if latency_delta > max_latency_delta:
        reasons.append("latency_budget_exceeded")
    return PromotionGate(
        passed=not reasons,
        reason_codes=tuple(reasons),
        baseline_score=baseline_score,
        candidate_score=candidate_score,
        baseline_pass_rate=baseline_pass_rate,
        candidate_pass_rate=candidate_pass_rate,
        safety_violations=safety_violations,
        route_regressions=route_regressions,
        cost_delta=cost_delta,
        latency_delta=latency_delta,
    )


async def evaluate_frozen_dataset(
    db: AsyncSession,
    *,
    dataset_id: str,
    cases: Iterable[CandidateCase],
    max_cost_delta: float = 0.20,
    max_latency_delta: float = 0.20,
) -> DatasetEvaluationBundle:
    """Validate exact frozen-dataset coverage and compute the server-side gate."""
    dataset = await db.get(EvaluationDataset, dataset_id)
    if dataset is None:
        raise ValueError("evaluation dataset does not exist")
    if dataset.status != "frozen":
        raise ValueError("evaluation dataset is not frozen")
    snapshots = tuple(await db.scalars(
        select(EvaluationDatasetCase)
        .where(EvaluationDatasetCase.dataset_id == dataset.id)
        .order_by(EvaluationDatasetCase.case_key)
    ))
    if len(snapshots) != dataset.case_count:
        raise ValueError("evaluation dataset snapshot is incomplete")

    submitted = tuple(cases)
    by_key: dict[str, CandidateCase] = {}
    for case in submitted:
        if case.id in by_key:
            raise ValueError(f"duplicate evaluation case result: {case.id}")
        by_key[case.id] = case
    expected_keys = {snapshot.case_key for snapshot in snapshots}
    submitted_keys = set(by_key)
    if submitted_keys != expected_keys:
        missing = len(expected_keys - submitted_keys)
        unexpected = len(submitted_keys - expected_keys)
        raise ValueError(
            f"evaluation results must exactly cover frozen dataset cases "
            f"(missing={missing}, unexpected={unexpected})"
        )
    ordered = tuple(by_key[snapshot.case_key] for snapshot in snapshots)
    gate = evaluate_promotion(
        ordered,
        max_cost_delta=max_cost_delta,
        max_latency_delta=max_latency_delta,
    )
    return DatasetEvaluationBundle(dataset=dataset, snapshots=snapshots, cases=ordered, gate=gate)


def _gate_metrics(gate: PromotionGate, *, max_cost_delta: float, max_latency_delta: float) -> dict[str, Any]:
    return {
        "baseline_score": gate.baseline_score,
        "candidate_score": gate.candidate_score,
        "baseline_pass_rate": gate.baseline_pass_rate,
        "candidate_pass_rate": gate.candidate_pass_rate,
        "safety_violations": gate.safety_violations,
        "route_regressions": gate.route_regressions,
        "cost_delta": gate.cost_delta,
        "latency_delta": gate.latency_delta,
        "max_cost_delta": max_cost_delta,
        "max_latency_delta": max_latency_delta,
    }


async def persist_frozen_dataset_evaluation(
    db: AsyncSession,
    *,
    dataset_id: str,
    cases: Iterable[CandidateCase],
    proposal_id: str | None = None,
    template_id: str | None = None,
    evaluator: str = "offline_gate_v2",
    max_cost_delta: float = 0.20,
    max_latency_delta: float = 0.20,
) -> tuple[SkillEvaluation | WorkflowTemplateEvaluation, list[EvaluationCaseResult]]:
    """Persist aggregate and per-case evidence for exactly one candidate type."""
    if (proposal_id is None) == (template_id is None):
        raise ValueError("exactly one evaluation target is required")
    bundle = await evaluate_frozen_dataset(
        db,
        dataset_id=dataset_id,
        cases=cases,
        max_cost_delta=max_cost_delta,
        max_latency_delta=max_latency_delta,
    )
    dataset_name = f"{bundle.dataset.name}@{bundle.dataset.version}"
    common = {
        "id": uuid_str(),
        "dataset_name": dataset_name,
        "dataset_id": bundle.dataset.id,
        "dataset_fingerprint": bundle.dataset.fingerprint,
        "evidence_mode": "dataset_verified",
        "evaluator": str(evaluator)[:100],
        "case_count": len(bundle.cases),
        "passed": bundle.gate.passed,
        "metrics": _gate_metrics(
            bundle.gate,
            max_cost_delta=max_cost_delta,
            max_latency_delta=max_latency_delta,
        ),
        "reason_codes": list(bundle.gate.reason_codes),
    }
    if proposal_id is not None:
        evaluation: SkillEvaluation | WorkflowTemplateEvaluation = SkillEvaluation(
            proposal_id=proposal_id,
            **common,
        )
    else:
        evaluation = WorkflowTemplateEvaluation(template_id=template_id, **common)
    db.add(evaluation)

    results: list[EvaluationCaseResult] = []
    for snapshot, case in zip(bundle.snapshots, bundle.cases, strict=True):
        result = EvaluationCaseResult(
            id=uuid_str(),
            skill_evaluation_id=evaluation.id if proposal_id is not None else None,
            template_evaluation_id=evaluation.id if template_id is not None else None,
            dataset_case_id=snapshot.id,
            case_key=snapshot.case_key,
            baseline_score=case.baseline_score,
            candidate_score=case.candidate_score,
            baseline_passed=case.baseline_passed,
            candidate_passed=case.candidate_passed,
            safety_violations=case.safety_violations,
            route_regression=case.route_regression,
            cost_delta=case.cost_delta,
            latency_delta=case.latency_delta,
        )
        db.add(result)
        results.append(result)
    await db.flush()
    return evaluation, results


async def create_evaluation_campaign(
    db: AsyncSession,
    *,
    evaluation_ids: Iterable[str],
    proposal_id: str | None = None,
    template_id: str | None = None,
) -> EvaluationCampaign:
    """Assemble immutable, complete multi-suite evidence under a fixed policy."""
    if (proposal_id is None) == (template_id is None):
        raise ValueError("exactly one campaign target is required")
    ids = sorted({str(item).strip() for item in evaluation_ids if str(item).strip()})
    if len(ids) != len(REQUIRED_PROMOTION_SUITES):
        raise ValueError("promotion campaign requires exactly one evaluation per required suite")
    model = SkillEvaluation if proposal_id is not None else WorkflowTemplateEvaluation
    target_field = model.proposal_id if proposal_id is not None else model.template_id
    target_id = proposal_id if proposal_id is not None else template_id
    evaluations = list(await db.scalars(
        select(model).where(model.id.in_(ids), target_field == target_id)
    ))
    by_id = {item.id: item for item in evaluations}
    if set(by_id) != set(ids):
        raise ValueError("one or more evaluations do not belong to the campaign target")
    if any(item.evidence_mode != "dataset_verified" or not item.dataset_id for item in evaluations):
        raise ValueError("campaign only accepts dataset-verified evaluations")

    dataset_ids = {item.dataset_id for item in evaluations}
    datasets = list(await db.scalars(select(EvaluationDataset).where(EvaluationDataset.id.in_(dataset_ids))))
    datasets_by_id = {item.id: item for item in datasets}
    if set(datasets_by_id) != dataset_ids:
        raise ValueError("one or more evaluation datasets do not exist")

    by_suite: dict[str, tuple[SkillEvaluation | WorkflowTemplateEvaluation, EvaluationDataset]] = {}
    for evaluation in evaluations:
        dataset = datasets_by_id[evaluation.dataset_id]
        if dataset.status != "frozen" or evaluation.dataset_fingerprint != dataset.fingerprint:
            raise ValueError("evaluation dataset identity no longer matches its frozen snapshot")
        if evaluation.case_count != dataset.case_count:
            raise ValueError("evaluation case count does not match its frozen dataset")
        if dataset.suite in by_suite:
            raise ValueError("promotion campaign contains duplicate suites")
        result_parent = (
            EvaluationCaseResult.skill_evaluation_id
            if proposal_id is not None
            else EvaluationCaseResult.template_evaluation_id
        )
        result_count = await db.scalar(
            select(func.count(EvaluationCaseResult.id)).where(result_parent == evaluation.id)
        )
        if int(result_count or 0) != dataset.case_count:
            raise ValueError("evaluation per-case evidence is incomplete")
        by_suite[dataset.suite] = (evaluation, dataset)
    if set(by_suite) != set(REQUIRED_PROMOTION_SUITES):
        raise ValueError("promotion campaign must cover regression, holdout and safety_boundary")

    reason_codes: list[str] = []
    suite_metrics: dict[str, Any] = {}
    for suite in REQUIRED_PROMOTION_SUITES:
        evaluation, dataset = by_suite[suite]
        metrics = dict(evaluation.metrics or {})
        required_metrics = {
            "baseline_score", "candidate_score", "baseline_pass_rate", "candidate_pass_rate",
            "safety_violations", "route_regressions", "cost_delta", "latency_delta",
        }
        if not required_metrics.issubset(metrics):
            raise ValueError(f"{suite} evaluation is missing required gate metrics")
        suite_metrics[suite] = {
            "evaluation_id": evaluation.id,
            "dataset_id": dataset.id,
            "case_count": evaluation.case_count,
            "passed": evaluation.passed,
            "baseline_score": metrics.get("baseline_score"),
            "candidate_score": metrics.get("candidate_score"),
            "baseline_pass_rate": metrics.get("baseline_pass_rate"),
            "candidate_pass_rate": metrics.get("candidate_pass_rate"),
            "safety_violations": metrics.get("safety_violations"),
            "route_regressions": metrics.get("route_regressions"),
            "cost_delta": metrics.get("cost_delta"),
            "latency_delta": metrics.get("latency_delta"),
        }
        if not evaluation.passed:
            reason_codes.append(f"{suite}:dataset_gate_failed")
        if int(metrics.get("safety_violations", 0) or 0) > 0:
            reason_codes.append(f"{suite}:safety_violations")
        if int(metrics.get("route_regressions", 0) or 0) > 0:
            reason_codes.append(f"{suite}:route_regressions")
        if float(metrics.get("candidate_score", 0)) < float(metrics.get("baseline_score", 0)):
            reason_codes.append(f"{suite}:score_regression")
        if float(metrics.get("candidate_pass_rate", 0)) < float(metrics.get("baseline_pass_rate", 0)):
            reason_codes.append(f"{suite}:pass_rate_regression")
        if float(metrics.get("cost_delta", 0) or 0) > 0.20:
            reason_codes.append(f"{suite}:cost_budget_exceeded")
        if float(metrics.get("latency_delta", 0) or 0) > 0.20:
            reason_codes.append(f"{suite}:latency_budget_exceeded")

    ordered_evaluation_ids = [by_suite[suite][0].id for suite in REQUIRED_PROMOTION_SUITES]
    dataset_fingerprints = {
        suite: by_suite[suite][1].fingerprint for suite in REQUIRED_PROMOTION_SUITES
    }
    fingerprint_payload = json.dumps(
        {
            "target_type": "skill" if proposal_id is not None else "template",
            "target_id": target_id,
            "policy_version": PROMOTION_POLICY_VERSION,
            "evaluation_ids": ordered_evaluation_ids,
            "dataset_fingerprints": dataset_fingerprints,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    fingerprint = hashlib.sha256(fingerprint_payload.encode("utf-8")).hexdigest()
    existing = await db.scalar(
        select(EvaluationCampaign).where(EvaluationCampaign.fingerprint == fingerprint)
    )
    if existing is not None:
        return existing
    campaign = EvaluationCampaign(
        id=uuid_str(),
        proposal_id=proposal_id,
        template_id=template_id,
        policy_version=PROMOTION_POLICY_VERSION,
        required_suites=list(REQUIRED_PROMOTION_SUITES),
        evaluation_ids=ordered_evaluation_ids,
        dataset_fingerprints=dataset_fingerprints,
        metrics={"suites": suite_metrics, "max_cost_delta": 0.20, "max_latency_delta": 0.20},
        passed=not reason_codes,
        reason_codes=reason_codes,
        fingerprint=fingerprint,
    )
    db.add(campaign)
    await db.flush()
    return campaign


async def persist_evaluation(
    db: AsyncSession,
    *,
    proposal_id: str,
    dataset_name: str,
    cases: Iterable[CandidateCase],
    evaluator: str = "offline_gate_v1",
    max_cost_delta: float = 0.20,
    max_latency_delta: float = 0.20,
) -> SkillEvaluation:
    """Persist a complete offline gate result for later human review.

    The cases are materialized once so the stored count and aggregate metrics
    describe exactly the same input that was evaluated. This function never
    changes proposal status or activates a skill.
    """
    rows = list(cases)
    gate = evaluate_promotion(rows, max_cost_delta=max_cost_delta, max_latency_delta=max_latency_delta)
    evaluation = SkillEvaluation(
        id=uuid_str(),
        proposal_id=proposal_id,
        dataset_name=str(dataset_name)[:200],
        evaluator=str(evaluator)[:100],
        case_count=len(rows),
        passed=gate.passed,
        evidence_mode="manual",
        metrics=_gate_metrics(gate, max_cost_delta=max_cost_delta, max_latency_delta=max_latency_delta),
        reason_codes=list(gate.reason_codes),
    )
    db.add(evaluation)
    await db.flush()
    return evaluation


def _main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate an offline self-evolution candidate")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--max-cost-delta", type=float, default=0.20)
    parser.add_argument("--max-latency-delta", type=float, default=0.20)
    args = parser.parse_args()
    gate = evaluate_promotion(
        load_comparison_cases(args.dataset),
        max_cost_delta=args.max_cost_delta,
        max_latency_delta=args.max_latency_delta,
    )
    print(json.dumps({**gate.__dict__, "reason_codes": list(gate.reason_codes)}, ensure_ascii=False, sort_keys=True))
    return 0 if gate.passed else 1


if __name__ == "__main__":
    raise SystemExit(_main())
