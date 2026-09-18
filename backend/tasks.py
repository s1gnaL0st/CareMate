"""Arq queue integration and the report-analysis worker entry point."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json

from arq import Retry, create_pool
from arq.connections import RedisSettings
from arq.cron import cron
from sqlalchemy import select

from config import get_settings
from db import SessionLocal
from models import MedicalReport, OfflineEvaluationRun, ReportAnalysis, SkillProposal
from object_storage import delete_object, get_object, put_object
from agents.vision import compose_agent_message, normalize_scan_type
from cache import distributed_lock
from document_parser import parse_document
from metrics import record_task_event


MAX_ANALYSIS_ATTEMPTS = 3
MAX_EVOLUTION_EVAL_ATTEMPTS = 2


def analysis_failure_status(attempt: int) -> str:
    """Return the durable status for a failed attempt."""
    return "dead_letter" if attempt >= MAX_ANALYSIS_ATTEMPTS else "retrying"


async def create_task_pool():
    settings = get_settings()
    return await create_pool(RedisSettings.from_dsn(settings.redis_url))


async def enqueue_report_analysis(report_id: str, analysis_id: str, scan_type: str) -> bool:
    pool = await create_task_pool()
    try:
        await pool.enqueue_job("analyze_report", report_id, analysis_id, scan_type, _job_id=f"report:{analysis_id}")
        return True
    finally:
        await pool.close()


async def enqueue_evolution_evaluation(run_id: str) -> bool:
    pool = await create_task_pool()
    try:
        await pool.enqueue_job(
            "execute_evolution_evaluation",
            run_id,
            _job_id=f"evolution-eval:{run_id}",
        )
        return True
    finally:
        await pool.close()


async def execute_evolution_evaluation(ctx: dict, run_id: str) -> None:
    """Run an idempotent baseline/candidate comparison through the real graph."""
    from evolution import redact_text
    from evolution_runner import run_frozen_dataset_comparison

    async with SessionLocal() as db:
        run = await db.get(OfflineEvaluationRun, run_id)
        if run is None or run.status == "completed":
            return
        attempt = int(ctx.get("job_try", 1))
        run.attempt_count = attempt
        run.status = "running"
        run.started_at = run.started_at or datetime.now(timezone.utc).replace(tzinfo=None)
        run.error_message = None
        await db.commit()
        try:
            outcome = await run_frozen_dataset_comparison(
                db,
                dataset_id=run.dataset_id,
                proposal_id=run.proposal_id,
            )
            proposal = await db.get(SkillProposal, run.proposal_id)
            if proposal is not None and proposal.status == "candidate":
                proposal.status = "evaluating"
            run.skill_evaluation_id = outcome.evaluation.id
            run.status = "completed"
            run.case_count = int(outcome.metrics["case_count"])
            run.completed_case_count = len(outcome.case_results)
            run.metrics = outcome.metrics
            run.finished_at = datetime.now(timezone.utc).replace(tzinfo=None)
            await db.commit()
        except Exception as exc:
            run.error_message = redact_text(str(exc), max_chars=2000)
            run.finished_at = datetime.now(timezone.utc).replace(tzinfo=None)
            if attempt < MAX_EVOLUTION_EVAL_ATTEMPTS:
                run.status = "retrying"
                await db.commit()
                raise Retry(defer=2 ** (attempt - 1)) from exc
            run.status = "dead_letter"
            await db.commit()
            record_task_event("dead_letter")
            raise


async def analyze_report(ctx: dict, report_id: str, analysis_id: str, scan_type: str) -> None:
    """Idempotently analyse a report with exponential retry and durable status."""
    async with distributed_lock(f"report:{report_id}", timeout=300, blocking_timeout=0) as acquired:
        if not acquired:
            raise Retry(defer=1)
        async with SessionLocal() as db:
            analysis = await db.scalar(select(ReportAnalysis).where(ReportAnalysis.id == analysis_id))
            report = await db.scalar(select(MedicalReport).where(MedicalReport.id == report_id))
            if analysis is None or report is None or analysis.status in {"completed", "cancelled"}:
                return
            attempt = int(ctx.get("job_try", 1))
            analysis.attempt_count = attempt
            analysis.status = "running"
            report.status = "processing"
            await db.commit()
            try:
                document = await get_object(report.object_key)
                # Cancellation is cooperative: the API marks the durable row,
                # and the worker checks it before invoking OCR/vision.
                await db.refresh(analysis)
                if analysis.status == "cancelled":
                    report.status = "cancelled"
                    await db.commit()
                    return
                normalized = normalize_scan_type(scan_type)
                parsed = await parse_document(document, report.content_type, normalized)
                text = parsed.text
                analysis.status = "completed"
                result = {
                    "scan_type": normalized,
                    "text": text,
                    "message": compose_agent_message(normalized, text),
                    "document": parsed.as_dict(),
                }
                result_key = f"reports/{report.user_id}/{report.sha256 or report.id}/analyses/{analysis.id}.json"
                await put_object(result_key, json.dumps(result, ensure_ascii=False).encode("utf-8"), "application/json")
                analysis.result_object_key = result_key
                analysis.result = result
                report.status = "completed"
                report.analysis_result = analysis.result
            except Exception as exc:
                analysis.error_message = str(exc)[:2000]
                is_final_attempt = attempt >= MAX_ANALYSIS_ATTEMPTS
                analysis.status = analysis_failure_status(attempt)
                report.status = "failed" if is_final_attempt else "retrying"
                await db.commit()
                if not is_final_attempt:
                    raise Retry(defer=2 ** (attempt - 1)) from exc
                record_task_event("dead_letter")
                raise
            finally:
                analysis.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
                report.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
                await db.commit()


async def purge_expired_reports(_ctx: dict) -> None:
    """Remove report records and blobs after the configured retention period."""
    cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=get_settings().report_retention_days)
    async with SessionLocal() as db:
        reports = list(await db.scalars(select(MedicalReport).where(MedicalReport.created_at < cutoff)))
        for report in reports:
            try:
                await delete_object(report.object_key)
            except Exception:
                # Keep the database row so the next scheduled run can retry cleanup.
                continue
            await db.delete(report)
        await db.commit()


class WorkerSettings:
    functions = [analyze_report, execute_evolution_evaluation, purge_expired_reports]
    max_jobs = 4
    job_timeout = 300
    max_tries = max(MAX_ANALYSIS_ATTEMPTS, MAX_EVOLUTION_EVAL_ATTEMPTS)
    keep_result = 86400
    cron_jobs = [cron(purge_expired_reports, hour=3, minute=15)]
