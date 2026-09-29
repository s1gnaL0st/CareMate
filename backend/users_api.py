"""Privacy endpoints for authenticated user-owned data."""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from auth import get_current_user
from db import get_db
from models import MedicalReport, ReportAnalysis, User
from memory_evolution import reset_user_memory_projection
from object_storage import delete_object


logger = logging.getLogger("smart_health.users")
router = APIRouter(prefix="/api/v1/users", tags=["users"])


@router.delete("/me", status_code=204)
async def delete_current_user(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> None:
    """Delete user-owned relational data and scheduled object-storage files."""
    report_object_keys = list(
        await db.scalars(select(MedicalReport.object_key).where(MedicalReport.user_id == user.id))
    )
    analysis_object_keys = list(
        await db.scalars(
            select(ReportAnalysis.result_object_key)
            .join(MedicalReport, ReportAnalysis.report_id == MedicalReport.id)
            .where(MedicalReport.user_id == user.id, ReportAnalysis.result_object_key.is_not(None))
        )
    )
    await db.delete(user)
    await db.commit()
    for object_key in [*report_object_keys, *analysis_object_keys]:
        try:
            await delete_object(object_key)
        except Exception as exc:
            logger.warning("user object deletion failed error=%s", type(exc).__name__)


@router.post("/me/memory/reset", status_code=200)
async def reset_current_user_memory_projection(user: User = Depends(get_current_user)) -> dict[str, bool]:
    """Reset the Hermes-style Markdown projection without deleting DB memory."""
    removed = await reset_user_memory_projection(user_id=user.id)
    return {"reset": True, "projection_removed": removed, "database_records_deleted": False}
