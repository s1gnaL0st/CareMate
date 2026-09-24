"""Database-backed AgentLoop run and task persistence.

This module deliberately does not use Redis.  The database is the source of
truth for task state; the graph remains responsible for orchestration.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import or_, select, update

from config import get_settings
from db import SessionLocal
from models import AgentRun, AgentTask


def _now() -> datetime:
    # MySQL DateTime columns in this project are timezone-naive.
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _lease_until() -> datetime:
    return _now() + timedelta(seconds=get_settings().agent_task_lease_seconds)


async def create_agent_run(
    *,
    run_id: str,
    conversation_id: str,
    user_id: str,
    requested_mode: str,
    initial_state: dict[str, Any],
) -> None:
    serializable_state = dict(initial_state)
    serializable_state["messages"] = [
        {
            "role": "assistant" if getattr(message, "type", "") == "ai" else "user",
            "content": str(getattr(message, "content", "")),
        }
        for message in initial_state.get("messages", [])
    ]
    async with SessionLocal() as db:
        db.add(AgentRun(
            id=run_id,
            conversation_id=conversation_id,
            user_id=user_id,
            requested_mode=requested_mode,
            initial_state=serializable_state,
            status="running",
            lease_until=_lease_until(),
        ))
        await db.commit()


async def persist_plan(run_id: str, tasks: list[Any], signature: str, replan_count: int) -> None:
    """Upsert the current plan without erasing completed task evidence."""
    async with SessionLocal() as db:
        run = await db.get(AgentRun, run_id)
        if run is None:
            return
        current_keys = set()
        for task in tasks:
            current_keys.add(task.id)
            row = await db.scalar(select(AgentTask).where(
                AgentTask.run_id == run_id,
                AgentTask.task_key == task.id,
            ))
            if row is None:
                db.add(AgentTask(
                    run_id=run_id,
                    task_key=task.id,
                    agent=task.agent,
                    objective=task.objective,
                    input_slice=task.input_slice,
                    depends_on=list(task.depends_on),
                    status="pending",
                ))
            elif row.status not in {"completed", "running"}:
                row.agent = task.agent
                row.objective = task.objective
                row.input_slice = task.input_slice
                row.depends_on = list(task.depends_on)
                row.status = "pending"
                row.error_message = None

        # Tasks omitted by a newer plan must never be resumed accidentally.
        old_rows = await db.scalars(select(AgentTask).where(AgentTask.run_id == run_id))
        for row in old_rows:
            if row.task_key not in current_keys and row.status in {"pending", "running"}:
                row.status = "obsolete"
                row.error_message = "removed_by_replan"

        run.plan_signature = signature
        run.replan_count = replan_count
        run.status = "running"
        run.lease_until = _lease_until()
        await db.commit()


async def request_pause(run_id: str, user_id: str) -> bool:
    async with SessionLocal() as db:
        run = await db.scalar(select(AgentRun).where(AgentRun.id == run_id, AgentRun.user_id == user_id))
        if run is None or run.status in {"completed", "failed", "cancelled"}:
            return False
        run.pause_requested = True
        run.status = "pause_requested"
        await db.commit()
        return True


async def get_run(run_id: str, user_id: str) -> AgentRun | None:
    async with SessionLocal() as db:
        return await db.scalar(select(AgentRun).where(AgentRun.id == run_id, AgentRun.user_id == user_id))


async def get_run_snapshot(run_id: str, user_id: str) -> dict[str, Any] | None:
    async with SessionLocal() as db:
        run = await db.scalar(select(AgentRun).where(AgentRun.id == run_id, AgentRun.user_id == user_id))
        if run is None:
            return None
        tasks = list(await db.scalars(select(AgentTask).where(AgentTask.run_id == run_id).order_by(AgentTask.created_at)))
        return {
            "run_id": run.id,
            "status": run.status,
            "pause_requested": run.pause_requested,
            "plan_signature": run.plan_signature,
            "replan_count": run.replan_count,
            "tasks": [
                {
                    "task_id": task.task_key,
                    "agent": task.agent,
                    "objective": task.objective,
                    "depends_on": task.depends_on or [],
                    "status": task.status,
                    "attempt_count": task.attempt_count,
                    "result": task.result,
                    "error": task.error_message,
                }
                for task in tasks
            ],
        }


async def mark_run_status(run_id: str, status: str, *, error_message: str | None = None) -> None:
    async with SessionLocal() as db:
        run = await db.get(AgentRun, run_id)
        if run is None:
            return
        run.status = status
        run.error_message = error_message
        run.lease_until = _lease_until() if status in {"running", "pause_requested"} else None
        await db.commit()


async def resume_run(run_id: str, user_id: str) -> bool:
    """Claim a paused run, or a run whose process lease expired after a crash."""
    now = _now()
    async with SessionLocal() as db:
        result = await db.execute(update(AgentRun).where(
            AgentRun.id == run_id,
            AgentRun.user_id == user_id,
            or_(
                AgentRun.status == "paused",
                (AgentRun.status == "running") & (AgentRun.lease_until.is_not(None)) & (AgentRun.lease_until < now),
            ),
        ).values(status="running", pause_requested=False, lease_until=_lease_until(), error_message=None))
        await db.commit()
        return bool(result.rowcount)


async def requeue_running_tasks(run_id: str) -> None:
    """Release claimed tasks after a known stopped stream before a later resume."""
    async with SessionLocal() as db:
        await db.execute(update(AgentTask).where(
            AgentTask.run_id == run_id,
            AgentTask.status == "running",
        ).values(status="pending", error_message="interrupted_before_completion", lease_until=None))
        await db.commit()


async def pause_requested(run_id: str) -> bool:
    async with SessionLocal() as db:
        run = await db.get(AgentRun, run_id)
        return bool(run and run.pause_requested)


async def recover_and_claim_task(run_id: str, task_key: str) -> bool:
    """Atomically claim a task, including a stale running task after a crash."""
    now = _now()
    async with SessionLocal() as db:
        await db.execute(update(AgentTask).where(
            AgentTask.run_id == run_id,
            AgentTask.status == "running",
            AgentTask.lease_until.is_not(None),
            AgentTask.lease_until < now,
        ).values(status="pending", error_message="recovered_after_lease_expiry", lease_until=None))
        result = await db.execute(update(AgentTask).where(
            AgentTask.run_id == run_id,
            AgentTask.task_key == task_key,
            or_(AgentTask.status == "pending", AgentTask.status == "failed"),
        ).values(
            status="running",
            attempt_count=AgentTask.attempt_count + 1,
            started_at=now,
            lease_until=_lease_until(),
            error_message=None,
        ))
        await db.commit()
        return bool(result.rowcount)


async def persist_task_result(run_id: str, task_key: str, result: dict[str, Any]) -> None:
    async with SessionLocal() as db:
        row = await db.scalar(select(AgentTask).where(
            AgentTask.run_id == run_id,
            AgentTask.task_key == task_key,
        ))
        if row is None:
            return
        row.status = result.get("status", "failed")
        row.result = result
        # The executor may perform bounded in-process retries before writing
        # the final result. Preserve that durable count for operators and
        # future resume decisions.
        try:
            row.attempt_count = max(int(row.attempt_count or 0), int(result.get("attempt_count", 0) or 0))
        except (TypeError, ValueError):
            pass
        row.error_message = result.get("error")
        row.completed_at = _now()
        row.lease_until = None
        await db.commit()


async def mark_tasks_stale(run_id: str, task_keys: set[str]) -> None:
    """Invalidate targeted tasks and downstream descendants before repair."""
    if not task_keys:
        return
    async with SessionLocal() as db:
        await db.execute(update(AgentTask).where(
            AgentTask.run_id == run_id,
            AgentTask.task_key.in_(task_keys),
            AgentTask.status == "completed",
        ).values(
            status="pending",
            result=None,
            error_message="stale_after_targeted_repair",
            completed_at=None,
            lease_until=None,
        ))
        await db.commit()


async def load_resume_state(run_id: str, user_id: str) -> dict[str, Any] | None:
    """Rebuild the graph input from the durable run and task ledger."""
    async with SessionLocal() as db:
        run = await db.scalar(select(AgentRun).where(AgentRun.id == run_id, AgentRun.user_id == user_id))
        if run is None:
            return None
        state = dict(run.initial_state or {})
        from langchain_core.messages import AIMessage, HumanMessage
        state["messages"] = [
            AIMessage(content=item.get("content", "")) if item.get("role") == "assistant"
            else HumanMessage(content=item.get("content", ""))
            for item in state.get("messages", [])
            if isinstance(item, dict)
        ]
        rows = list(await db.scalars(select(AgentTask).where(
            AgentTask.run_id == run_id,
            AgentTask.status != "obsolete",
        ).order_by(AgentTask.created_at)))
        state.update({
            "agent_run_id": run_id,
            "resume_from_executor": True,
            "task_queue": [],
            "task_results": {},
            "replan_count": run.replan_count,
            "plan_signature": run.plan_signature,
        })
        from agents.graph_new import PlannedTask
        for row in rows:
            state["task_queue"].append(PlannedTask(
                id=row.task_key,
                agent=row.agent,
                objective=row.objective,
                input_slice=row.input_slice,
                depends_on=list(row.depends_on or []),
            ))
            if row.result and row.status == "completed":
                state["task_results"][row.task_key] = row.result
        return state
