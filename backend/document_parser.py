"""Worker-side document parsing contract for report images and PDFs.

External MinerU and PaddleOCR services use a deliberately small HTTP contract:
``POST /parse`` with a multipart ``file`` field returns ``text`` or
``markdown`` plus an optional ``pages`` list. Each page may include structured
blocks with ``type``, ``text``, ``bbox``, and ``confidence`` fields.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from agents.vision import VisionScanType, recognize_image
from config import get_settings


class DocumentParseError(RuntimeError):
    """Raised when the configured OCR backend cannot parse a document."""


@dataclass(frozen=True)
class ParsedDocument:
    text: str
    provider: str
    pages: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {"text": self.text, "provider": self.provider, "pages": self.pages}


def _normalize_pages(raw_pages: Any) -> list[dict[str, Any]]:
    """Normalize OCR page/block variants into one JSON contract."""
    if not isinstance(raw_pages, list):
        return []

    pages: list[dict[str, Any]] = []
    for index, raw_page in enumerate(raw_pages, start=1):
        if not isinstance(raw_page, dict):
            continue
        raw_blocks = raw_page.get("blocks")
        if not isinstance(raw_blocks, list):
            raw_blocks = []
        blocks: list[dict[str, Any]] = []
        for raw_block in raw_blocks:
            if not isinstance(raw_block, dict):
                continue
            text = str(raw_block.get("text") or "").strip()
            if not text and raw_block.get("content"):
                text = str(raw_block["content"]).strip()
            bbox = raw_block.get("bbox")
            if not isinstance(bbox, list):
                bbox = None
            confidence = raw_block.get("confidence")
            try:
                confidence = float(confidence) if confidence is not None else None
            except (TypeError, ValueError):
                confidence = None
            blocks.append({
                "type": str(raw_block.get("type") or "text"),
                "text": text,
                "bbox": bbox,
                "confidence": confidence,
            })
        pages.append({
            "page_number": raw_page.get("page_number", raw_page.get("page", index)),
            "text": str(raw_page.get("text") or "").strip(),
            "blocks": blocks,
        })
    return pages


async def parse_document(content: bytes, content_type: str, scan_type: VisionScanType) -> ParsedDocument:
    settings = get_settings()
    is_pdf = content_type == "application/pdf"
    provider = settings.ocr_provider.strip().lower()
    if is_pdf:
        if provider != "mineru":
            raise DocumentParseError("PDF report parsing requires OCR_PROVIDER=mineru")
        return await _parse_with_endpoint(content, content_type, settings.mineru_endpoint, "mineru")
    if provider == "paddleocr":
        return await _parse_with_endpoint(content, content_type, settings.paddleocr_endpoint, "paddleocr")
    if provider == "mineru":
        return await _parse_with_endpoint(content, content_type, settings.mineru_endpoint, "mineru")
    text = await recognize_image(content, content_type, scan_type)
    return ParsedDocument(text=text, provider="vision", pages=[])


async def _parse_with_endpoint(content: bytes, content_type: str, endpoint: str, provider: str) -> ParsedDocument:
    if not endpoint:
        raise DocumentParseError(f"{provider} endpoint is not configured")
    settings = get_settings()
    try:
        async with httpx.AsyncClient(timeout=settings.ocr_timeout_seconds) as client:
            response = await client.post(
                f"{endpoint.rstrip('/')}/parse",
                files={"file": ("report", content, content_type)},
            )
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise DocumentParseError(f"{provider} parsing service is unavailable") from exc
    if not isinstance(payload, dict):
        raise DocumentParseError(f"{provider} returned an invalid response")
    text = str(payload.get("markdown") or payload.get("text") or "").strip()
    pages = _normalize_pages(payload.get("pages", []))
    if not text:
        raise DocumentParseError(f"{provider} returned no extracted text")
    return ParsedDocument(text=text, provider=provider, pages=pages)
