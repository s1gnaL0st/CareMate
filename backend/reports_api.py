"""Authenticated report upload and asynchronous analysis endpoints."""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agents.vision import VisionInputError, normalize_scan_type, validate_image_content
from config import get_settings
from auth import get_current_user
from db import get_db
from models import MedicalReport, ReportAnalysis, User
from object_storage import delete_object, put_object
from schemas import ReportListItem, ReportStatusResponse, ReportUploadResponse
from tasks import enqueue_report_analysis


logger = logging.getLogger("smart_health.reports")
router = APIRouter(prefix="/api/v1/reports", tags=["reports"])


@router.get("", response_model=list[ReportListItem])
async def list_reports(
    limit: int = 20,
    offset: int = 0,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[ReportListItem]:
    """List the authenticated user's reports and latest analysis state."""
    limit = min(max(limit, 1), 100)
    offset = max(offset, 0)
    reports = list(await db.scalars(
        select(MedicalReport)
        .where(MedicalReport.user_id == user.id)
        .order_by(MedicalReport.created_at.desc())
        .offset(offset).limit(limit)
    ))
    result: list[ReportListItem] = []
    for report in reports:
        analysis = await db.scalar(
            select(ReportAnalysis).where(ReportAnalysis.report_id == report.id)
            .order_by(ReportAnalysis.created_at.desc())
        )
        result.append(ReportListItem(
            report_id=report.id,
            original_filename=report.original_filename,
            content_type=report.content_type,
            size_bytes=report.size_bytes,
            status=report.status,
            analysis_status=analysis.status if analysis else None,
            created_at=report.created_at,
        ))
    return result


def _object_key(user_id: str, digest: str, filename: str | None) -> str:
    extension = Path(filename or "upload").suffix.lower().lstrip(".")
    safe_extension = extension if extension in {"jpg", "jpeg", "png", "webp", "pdf"} else "bin"
    return f"reports/{user_id}/{digest}.{safe_extension}"


def _validate_report_upload(content: bytes, content_type: str | None, scan_type: str) -> None:
    settings = get_settings()
    if content_type == "application/pdf":
        if scan_type != "report":
            raise VisionInputError("PDF uploads are only supported for medical reports")
        if not content or len(content) > settings.max_report_file_bytes or not content.startswith(b"%PDF-"):
            raise VisionInputError("Uploaded file is not a valid supported PDF report")
        return
    validate_image_content(content, content_type, settings.max_image_pixels)


@router.post("", response_model=ReportUploadResponse, status_code=202)
async def upload_report(
    file: UploadFile = File(...),
    scan_type: str = Form("report"),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ReportUploadResponse:
    """Store an upload once and enqueue its analysis without blocking the API."""
    normalized_scan_type = normalize_scan_type(scan_type)
    content = await file.read()
    try:
        _validate_report_upload(content, file.content_type, normalized_scan_type)
    except VisionInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    digest = hashlib.sha256(content).hexdigest()
    existing = await db.scalar(
        select(MedicalReport).where(
            MedicalReport.user_id == user.id,
            MedicalReport.sha256 == digest,
        )
    )
    if existing is not None:
        analysis = await db.scalar(
            select(ReportAnalysis)
            .where(ReportAnalysis.report_id == existing.id)
            .order_by(ReportAnalysis.created_at.desc())
        )
        return ReportUploadResponse(
            report_id=existing.id,
            analysis_id=analysis.id if analysis else "",
            status=analysis.status if analysis else existing.status,
            duplicate=True,
        )

    object_key = _object_key(user.id, digest, file.filename)
    try:
        await put_object(object_key, content, file.content_type or "application/octet-stream")
    except Exception as exc:
        logger.exception("report object upload failed user_id=%s", user.id)
        raise HTTPException(status_code=503, detail="报告存储暂时不可用") from exc

    report = MedicalReport(
        user_id=user.id,
        object_key=object_key,
        original_filename=(file.filename or "upload")[:255],
        content_type=file.content_type or "application/octet-stream",
        size_bytes=len(content),
        sha256=digest,
        status="queued",
    )
    try:
        db.add(report)
        await db.flush()
        analysis = ReportAnalysis(report_id=report.id, status="queued", engine="vision_llm")
        db.add(analysis)
        await db.commit()
    except Exception as exc:
        await db.rollback()
        await _delete_orphaned_object(object_key)
        raise HTTPException(status_code=500, detail="无法创建报告分析任务") from exc

    try:
        await enqueue_report_analysis(report.id, analysis.id, normalized_scan_type)
    except Exception as exc:
        logger.warning("report enqueue failed report_id=%s error=%s", report.id, type(exc).__name__)
        analysis.status = "queue_failed"
        analysis.error_message = "任务队列暂时不可用"
        report.status = "queue_failed"
        await db.commit()
        raise HTTPException(status_code=503, detail="分析队列暂时不可用") from exc

    return ReportUploadResponse(report_id=report.id, analysis_id=analysis.id, status="queued")


@router.get("/{report_id}", response_model=ReportStatusResponse)
async def get_report_status(
    report_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ReportStatusResponse:
    report = await db.scalar(
        select(MedicalReport).where(MedicalReport.id == report_id, MedicalReport.user_id == user.id)
    )
    if report is None:
        raise HTTPException(status_code=404, detail="报告不存在")
    analysis = await db.scalar(
        select(ReportAnalysis)
        .where(ReportAnalysis.report_id == report.id)
        .order_by(ReportAnalysis.created_at.desc())
    )
    return ReportStatusResponse(
        report_id=report.id,
        status=report.status,
        analysis_status=analysis.status if analysis else None,
        analysis=analysis.result if analysis and analysis.status == "completed" else None,
        error=analysis.error_message if analysis else None,
        attempt_count=analysis.attempt_count if analysis else 0,
    )


@router.post("/{report_id}/cancel", response_model=ReportStatusResponse)
async def cancel_report_analysis(
    report_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ReportStatusResponse:
    """Request cooperative cancellation of a queued or running analysis."""
    report = await db.scalar(
        select(MedicalReport).where(MedicalReport.id == report_id, MedicalReport.user_id == user.id)
    )
    if report is None:
        raise HTTPException(status_code=404, detail="报告不存在")
    analysis = await db.scalar(
        select(ReportAnalysis)
        .where(ReportAnalysis.report_id == report.id)
        .order_by(ReportAnalysis.created_at.desc())
    )
    if analysis is None:
        raise HTTPException(status_code=404, detail="分析任务不存在")
    if analysis.status in {"completed", "failed", "dead_letter", "cancelled"}:
        return ReportStatusResponse(
            report_id=report.id, status=report.status, analysis_status=analysis.status,
            analysis=analysis.result if analysis.status == "completed" else None,
            error=analysis.error_message, attempt_count=analysis.attempt_count,
        )
    analysis.status = "cancelled"
    analysis.error_message = "用户主动取消"
    report.status = "cancelled"
    await db.commit()
    return ReportStatusResponse(
        report_id=report.id, status=report.status, analysis_status=analysis.status,
        analysis=None, error=analysis.error_message, attempt_count=analysis.attempt_count,
    )


@router.delete("/{report_id}", status_code=204)
async def delete_report(
    report_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> None:
    report = await db.scalar(
        select(MedicalReport).where(MedicalReport.id == report_id, MedicalReport.user_id == user.id)
    )
    if report is None:
        raise HTTPException(status_code=404, detail="报告不存在")
    analysis_object_keys = list(
        await db.scalars(
            select(ReportAnalysis.result_object_key).where(
                ReportAnalysis.report_id == report.id,
                ReportAnalysis.result_object_key.is_not(None),
            )
        )
    )
    object_keys = [report.object_key, *analysis_object_keys]
    await db.delete(report)
    await db.commit()
    for object_key in object_keys:
        await _delete_orphaned_object(object_key)


async def _delete_orphaned_object(object_key: str) -> None:
    try:
        await delete_object(object_key)
    except Exception as exc:
        logger.warning("report object deletion failed object_key=%s error=%s", object_key, type(exc).__name__)
