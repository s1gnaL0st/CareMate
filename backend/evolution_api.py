"""Compact, audited self-evolution API for offline medical skill candidates."""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from agents.safety import UnsafePromptError
from auth import get_current_user
from config import get_settings
from db import get_db
from evolution import (
    cluster_failure_cases,
    create_curated_evaluation_dataset,
    propose_skill_from_failures,
    redact_text,
)
from evolution_proposer import propose_skill_with_llm
from models import (
    EvaluationCaseResult,
    EvaluationDataset,
    EvaluationDatasetCase,
    FailureCase,
    OfflineEvaluationRun,
    PromotionDecision,
    SkillEvaluation,
    SkillProposal,
    SkillDeployment,
    User,
)
from models import uuid_str
from tasks import enqueue_evolution_evaluation

router = APIRouter(prefix="/api/v1/evolution", tags=["evolution"])
MAX_CLUSTER_ROWS = 2000


class ReviewRequest(BaseModel):
    decision: str = Field(pattern="^(approved|rejected|revoked)$")
    reason: str = Field(min_length=3, max_length=2000)
    evaluation_id: str | None = Field(default=None, max_length=36)


class DeploymentRequest(BaseModel):
    reason: str = Field(min_length=3, max_length=2000)


class ProposalFromFailuresRequest(BaseModel):
    failure_ids: list[str] = Field(min_length=1, max_length=32)
    base_skill: str = Field(min_length=1, max_length=100)
    trigger: str | None = Field(default=None, max_length=500)
    version: str = Field(default="0.1.0", min_length=1, max_length=40)


class LLMProposalFromFailuresRequest(BaseModel):
    failure_ids: list[str] = Field(min_length=1, max_length=32)
    base_skill: str = Field(min_length=1, max_length=100)
    version: str = Field(default="0.1.0", min_length=1, max_length=40)
    parent_proposal_id: str | None = Field(default=None, max_length=36)


class EvaluationDatasetCaseRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=6000)
    expected_constraints: dict = Field(default_factory=dict)


class EvaluationDatasetRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    version: str = Field(min_length=1, max_length=40)
    cases: list[EvaluationDatasetCaseRequest] = Field(min_length=3, max_length=32)


class EvaluationRunRequest(BaseModel):
    dataset_id: str = Field(min_length=1, max_length=36)
    idempotency_key: str = Field(
        min_length=8,
        max_length=100,
        pattern=r"^[A-Za-z0-9_.:-]+$",
    )


def _require_reviewer(user: User) -> None:
    settings = get_settings()
    allowlist = settings.evolution_reviewer_email_list
    if settings.environment.lower() in {"prod", "production"} and user.email.lower() not in allowlist:
        raise HTTPException(status_code=403, detail="没有自进化候选审核权限")


@router.get("/failures")
async def list_failure_cases(
    status: str = "staging",
    category: str | None = None,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List redacted failures that may be used by the offline proposer."""
    _require_reviewer(user)
    if status not in {"staging", "archived", "resolved"}:
        raise HTTPException(status_code=400, detail="无效的失败案例状态")
    query = select(FailureCase).where(FailureCase.status == status).order_by(desc(FailureCase.created_at))
    if category:
        query = query.where(FailureCase.category == category[:50])
    rows = list(await db.scalars(query.limit(200)))
    return [
        {
            "id": row.id,
            "experience_id": row.experience_id,
            "category": row.category,
            "severity": row.severity,
            "summary": row.summary,
            "status": row.status,
            "created_at": row.created_at,
        }
        for row in rows
    ]


@router.get("/failures/clusters")
async def list_failure_clusters(
    status: str = "staging",
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Group failures deterministically by category and severity."""
    _require_reviewer(user)
    if status not in {"staging", "archived", "resolved"}:
        raise HTTPException(status_code=400, detail="无效的失败案例状态")
    rows = list(await db.scalars(
        select(FailureCase)
        .where(FailureCase.status == status)
        .order_by(desc(FailureCase.created_at))
        .limit(MAX_CLUSTER_ROWS)
    ))
    return [cluster.__dict__ for cluster in cluster_failure_cases(rows)]


@router.get("/proposals")
async def list_proposals(
    status: str | None = None,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    _require_reviewer(user)
    query = select(SkillProposal).order_by(desc(SkillProposal.created_at))
    if status:
        if status not in {"candidate", "evaluating", "approved", "rejected", "revoked"}:
            raise HTTPException(status_code=400, detail="无效的候选状态")
        query = query.where(SkillProposal.status == status)
    return list(await db.scalars(query.limit(100)))


@router.post("/proposals/from-failures", status_code=201)
async def create_proposal_from_failures(
    payload: ProposalFromFailuresRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    _require_reviewer(user)
    try:
        proposal = await propose_skill_from_failures(
            db,
            failure_ids=payload.failure_ids,
            base_skill=payload.base_skill,
            trigger=payload.trigger,
            version=payload.version,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    await db.commit()
    await db.refresh(proposal)
    return {"proposal": proposal, "published": False}


@router.post("/proposals/from-failures/llm", status_code=201)
async def create_llm_proposal_from_failures(
    payload: LLMProposalFromFailuresRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Generate a structured offline proposal without changing the live graph."""
    _require_reviewer(user)
    try:
        proposal = await propose_skill_with_llm(db, **payload.model_dump())
    except UnsafePromptError as exc:
        raise HTTPException(status_code=400, detail="模型生成的候选包含不安全指令") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    await db.commit()
    await db.refresh(proposal)
    return {
        "proposal": proposal,
        "generation_mode": "llm_structured_offline",
        "published": False,
        "injected_into_planner": False,
    }


@router.get("/proposals/{proposal_id}")
async def proposal_detail(
    proposal_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    _require_reviewer(user)
    proposal = await db.get(SkillProposal, proposal_id)
    if proposal is None:
        raise HTTPException(status_code=404, detail="候选技能不存在")
    evaluations = list(await db.scalars(
        select(SkillEvaluation)
        .where(SkillEvaluation.proposal_id == proposal_id)
        .order_by(desc(SkillEvaluation.created_at))
    ))
    runs = list(await db.scalars(
        select(OfflineEvaluationRun)
        .where(OfflineEvaluationRun.proposal_id == proposal_id)
        .order_by(desc(OfflineEvaluationRun.created_at))
    ))
    decisions = list(await db.scalars(
        select(PromotionDecision)
        .where(PromotionDecision.proposal_id == proposal_id)
        .order_by(desc(PromotionDecision.created_at))
    ))
    return {"proposal": proposal, "evaluation_runs": runs, "evaluations": evaluations, "decisions": decisions}


@router.post("/evaluation-datasets", status_code=201)
async def create_evaluation_dataset(
    payload: EvaluationDatasetRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Freeze one reviewer-curated mixed medical evaluation set."""
    _require_reviewer(user)
    try:
        dataset, cases = await create_curated_evaluation_dataset(
            db,
            name=payload.name,
            version=payload.version,
            cases=[item.model_dump() for item in payload.cases],
            created_by=user.id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    await db.commit()
    return {"dataset": dataset, "cases": cases, "immutable": True, "used_online": False}


@router.get("/evaluation-datasets")
async def list_evaluation_datasets(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    _require_reviewer(user)
    return list(await db.scalars(
        select(EvaluationDataset)
        .where(EvaluationDataset.suite == "medical")
        .order_by(desc(EvaluationDataset.created_at))
        .limit(100)
    ))


@router.get("/evaluation-datasets/{dataset_id}")
async def evaluation_dataset_detail(
    dataset_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    _require_reviewer(user)
    dataset = await db.get(EvaluationDataset, dataset_id)
    if dataset is None or dataset.suite != "medical":
        raise HTTPException(status_code=404, detail="评测数据集不存在")
    cases = list(await db.scalars(
        select(EvaluationDatasetCase)
        .where(EvaluationDatasetCase.dataset_id == dataset.id)
        .order_by(EvaluationDatasetCase.case_key)
    ))
    return {"dataset": dataset, "cases": cases, "immutable": True, "used_online": False}


@router.post("/proposals/{proposal_id}/evaluation-runs", status_code=202)
async def create_evaluation_run(
    proposal_id: str,
    payload: EvaluationRunRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Queue a durable baseline/candidate comparison through the real Agent Graph."""
    _require_reviewer(user)
    existing = await db.scalar(
        select(OfflineEvaluationRun).where(OfflineEvaluationRun.idempotency_key == payload.idempotency_key)
    )
    if existing is not None:
        if existing.proposal_id != proposal_id or existing.dataset_id != payload.dataset_id:
            raise HTTPException(status_code=409, detail="幂等键已用于其他评测任务")
        return {"run": existing, "queued": existing.status in {"queued", "running", "retrying"}}

    proposal = await db.get(SkillProposal, proposal_id)
    if proposal is None:
        raise HTTPException(status_code=404, detail="候选技能不存在")
    if proposal.status not in {"candidate", "evaluating"}:
        raise HTTPException(status_code=409, detail="当前候选状态不能评测")
    dataset = await db.get(EvaluationDataset, payload.dataset_id)
    if dataset is None or dataset.status != "frozen" or dataset.suite != "medical":
        raise HTTPException(status_code=400, detail="必须选择冻结的医疗评测集")

    run = OfflineEvaluationRun(
        idempotency_key=payload.idempotency_key,
        proposal_id=proposal.id,
        dataset_id=dataset.id,
        status="queued",
        case_count=dataset.case_count,
        created_by=user.id,
    )
    db.add(run)
    await db.commit()
    await db.refresh(run)
    try:
        queued = await enqueue_evolution_evaluation(run.id)
        if not queued:
            raise RuntimeError("task queue did not accept evaluation run")
    except Exception as exc:
        run.status = "queue_failed"
        run.error_message = redact_text(str(exc), max_chars=2000)
        run.finished_at = datetime.now(timezone.utc).replace(tzinfo=None)
        await db.commit()
        raise HTTPException(status_code=503, detail="离线评测任务入队失败，可使用同一幂等键查询状态") from exc
    return {"run": run, "queued": True, "published": False}


@router.get("/evaluation-runs/{run_id}")
async def evaluation_run_detail(
    run_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    _require_reviewer(user)
    run = await db.get(OfflineEvaluationRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="离线评测任务不存在")
    evaluation = None
    case_results = []
    if run.skill_evaluation_id:
        evaluation = await db.get(SkillEvaluation, run.skill_evaluation_id)
        case_results = list(await db.scalars(
            select(EvaluationCaseResult)
            .where(EvaluationCaseResult.skill_evaluation_id == run.skill_evaluation_id)
            .order_by(EvaluationCaseResult.case_key)
        ))
    return {"run": run, "evaluation": evaluation, "case_results": case_results, "published": False}


@router.post("/proposals/{proposal_id}/review")
async def review_proposal(
    proposal_id: str,
    payload: ReviewRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Record a human decision; approval never publishes the candidate online."""
    _require_reviewer(user)
    proposal = await db.get(SkillProposal, proposal_id)
    if proposal is None:
        raise HTTPException(status_code=404, detail="候选技能不存在")
    if payload.decision == "approved" and proposal.status != "evaluating":
        raise HTTPException(status_code=409, detail="只有 evaluating 状态可以批准")
    if payload.decision == "rejected" and proposal.status not in {"candidate", "evaluating"}:
        raise HTTPException(status_code=409, detail="当前状态不能拒绝")
    if payload.decision == "revoked" and proposal.status != "approved":
        raise HTTPException(status_code=409, detail="只有 approved 状态可以撤销")

    evaluation = None
    if payload.evaluation_id:
        evaluation = await db.get(SkillEvaluation, payload.evaluation_id)
        if evaluation is None or evaluation.proposal_id != proposal_id:
            raise HTTPException(status_code=400, detail="评测记录与候选技能不匹配")
    if payload.decision == "approved":
        if evaluation is None:
            raise HTTPException(status_code=400, detail="批准必须关联该候选最新的离线评测")
        latest_evaluation = await db.scalar(
            select(SkillEvaluation)
            .where(SkillEvaluation.proposal_id == proposal_id)
            .order_by(desc(SkillEvaluation.created_at), desc(SkillEvaluation.id))
            .limit(1)
        )
        if latest_evaluation is None or latest_evaluation.id != evaluation.id:
            raise HTTPException(status_code=409, detail="批准必须关联该候选最新的离线评测")
        if not evaluation.passed:
            raise HTTPException(status_code=409, detail="医疗安全与质量门禁未通过")
        if evaluation.evidence_mode != "dataset_verified" or not evaluation.dataset_id:
            raise HTTPException(status_code=409, detail="批准必须关联真实运行产生的数据集评测证据")
        dataset = await db.get(EvaluationDataset, evaluation.dataset_id)
        if (
            dataset is None
            or dataset.status != "frozen"
            or dataset.fingerprint != evaluation.dataset_fingerprint
            or dataset.case_count != evaluation.case_count
        ):
            raise HTTPException(status_code=409, detail="评测证据与冻结数据集不一致")
        result_count = await db.scalar(
            select(func.count(EvaluationCaseResult.id)).where(
                EvaluationCaseResult.skill_evaluation_id == evaluation.id
            )
        )
        if int(result_count or 0) != dataset.case_count:
            raise HTTPException(status_code=409, detail="逐病例评测证据不完整")

    previous_status = proposal.status
    proposal.status = payload.decision
    decision = PromotionDecision(
        proposal_id=proposal.id,
        evaluation_id=evaluation.id if evaluation else None,
        campaign_id=None,
        reviewer_id=user.id,
        decision=payload.decision,
        previous_status=previous_status,
        reason=redact_text(payload.reason, max_chars=2000),
    )
    db.add(decision)
    await db.commit()
    await db.refresh(proposal)
    return {
        "proposal": proposal,
        "decision": decision,
        "published": False,
        "injected_into_planner": False,
    }


@router.get("/deployments")
async def list_deployments(
    user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db),
):
    _require_reviewer(user)
    rows = list(await db.scalars(select(SkillDeployment).order_by(desc(SkillDeployment.updated_at))))
    return rows


@router.post("/proposals/{proposal_id}/deploy")
async def deploy_proposal(
    proposal_id: str, payload: DeploymentRequest,
    user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db),
):
    """Move the online pointer to an approved, evaluated proposal."""
    _require_reviewer(user)
    proposal = await db.get(SkillProposal, proposal_id)
    if proposal is None:
        raise HTTPException(status_code=404, detail="候选技能不存在")
    if proposal.status != "approved":
        raise HTTPException(status_code=409, detail="只有 approved 候选可以发布")
    previous = await db.scalar(select(SkillDeployment).where(SkillDeployment.base_skill == proposal.base_skill))
    if previous is None:
        deployment = SkillDeployment(id=uuid_str(), base_skill=proposal.base_skill,
            proposal_id=proposal.id, deployed_by=user.id, reason=redact_text(payload.reason, max_chars=2000))
        db.add(deployment)
    else:
        previous.previous_proposal_id = previous.proposal_id
        previous.proposal_id = proposal.id
        previous.status = "active"
        previous.deployed_by = user.id
        previous.reason = redact_text(payload.reason, max_chars=2000)
        deployment = previous
    await db.commit(); await db.refresh(deployment)
    return {"deployment": deployment, "published": True, "rollback_supported": True}


@router.post("/deployments/{base_skill}/rollback")
async def rollback_deployment(
    base_skill: str, payload: DeploymentRequest,
    user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db),
):
    """Restore the immediately previous version without deleting audit data."""
    _require_reviewer(user)
    deployment = await db.scalar(select(SkillDeployment).where(
        SkillDeployment.base_skill == base_skill, SkillDeployment.status == "active"))
    if deployment is None or deployment.previous_proposal_id is None:
        raise HTTPException(status_code=409, detail="没有可回滚的上一版本")
    current = deployment.proposal_id
    deployment.proposal_id = deployment.previous_proposal_id
    deployment.previous_proposal_id = current
    deployment.deployed_by = user.id
    deployment.reason = redact_text("rollback: " + payload.reason, max_chars=2000)
    await db.commit(); await db.refresh(deployment)
    return {"deployment": deployment, "rolled_back": True}


@router.post("/deployments/{base_skill}/reset")
async def reset_deployment(
    base_skill: str, payload: DeploymentRequest,
    user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db),
):
    """Disable the online pointer; proposals and evidence remain intact."""
    _require_reviewer(user)
    deployment = await db.scalar(select(SkillDeployment).where(SkillDeployment.base_skill == base_skill))
    if deployment is None:
        raise HTTPException(status_code=404, detail="技能没有部署记录")
    deployment.status = "reset"
    deployment.deployed_by = user.id
    deployment.reason = redact_text("reset: " + payload.reason, max_chars=2000)
    await db.commit(); await db.refresh(deployment)
    return {"deployment": deployment, "reset": True, "data_deleted": False}
