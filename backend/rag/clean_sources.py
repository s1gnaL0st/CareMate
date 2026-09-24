"""Extract registered official HTML/PDF sources into traceable Markdown."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from bs4 import BeautifulSoup, NavigableString, Tag
import hashlib
import json
from pathlib import Path
import re
from typing import Any


DROP_TAGS = {"script", "style", "noscript", "svg", "nav", "header", "footer", "aside", "form"}
DROP_CLASSES = {"related", "related-items", "related-item", "multimedia", "sf-multimedia", "share", "breadcrumb"}
CHAPTER_LINE = re.compile(r"^(第[一二三四五六七八九十百0-9]+[章节部分条]|附录[一二三四五六七八九十0-9]+)")
NUMBERED_HEADING = re.compile(r"^(?:[一二三四五六七八九十]+、\s*\S+|\d+(?:\.\d+){1,3}(?:[.、]\s*|\s+)(?=[A-Za-z\u4e00-\u9fff“\"（(])|\d+[.、]\s*(?=[A-Za-z\u4e00-\u9fff“\"（(]))")
STANDARD_HEADING = re.compile(r"^\d{1,2}\s+[\u4e00-\u9fff]")
PAGE_NUMBER = re.compile(r"^\d{1,4}$")
REFERENCE_HEADING = re.compile(r"^(?:参考文献|References|Bibliography)\s*$", re.IGNORECASE)
REFERENCE_LINE = re.compile(r"^(?:\[?\d{1,4}\]?\s*[.、)]?\s*)?(?:DOI\s*:|10\.\d{4,9}/|[A-Za-z].{0,140}(?:\b\d{4}\b|\b(?:19|20)\d{2}\b).{0,100}(?:;|,|\d+\(|\d+:))", re.IGNORECASE)


def _node_lines(node: Tag) -> list[str]:
    """Render meaningful article blocks without markup or navigation chrome."""
    lines: list[str] = []

    def walk(item: Any) -> None:
        if isinstance(item, NavigableString):
            text = re.sub(r"\s+", " ", str(item)).strip()
            if text:
                lines.append(text)
            return
        if not isinstance(item, Tag):
            return
        name = item.name.lower()
        if name in DROP_TAGS:
            return
        classes = set(item.get("class", []))
        if classes.intersection(DROP_CLASSES) or any(str(item.get("id", "")).lower().startswith(x) for x in ("related", "news", "social-share")):
            return
        if name in {"h1", "h2", "h3", "h4"}:
            text = item.get_text(" ", strip=True)
            if text:
                level = int(name[1])
                lines.append(f"{'#' * level} {text}")
            return
        if name == "li":
            text = item.get_text(" ", strip=True)
            if text:
                lines.append(f"- {text}")
            return
        if name in {"p", "blockquote", "td", "th"}:
            text = item.get_text(" ", strip=True)
            if text:
                lines.append(text)
            return
        if name == "br":
            return
        for child in item.children:
            walk(child)

    walk(node)
    return lines


def _extract_html(raw: bytes, entry: dict[str, Any]) -> tuple[str, str, int, int]:
    soup = BeautifulSoup(raw, "html.parser")
    title_tag = soup.find("h1") or soup.title
    title = title_tag.get_text(" ", strip=True) if title_tag else str(entry.get("title", ""))
    for element in soup.find_all(list(DROP_TAGS)):
        element.decompose()
    root = soup.find("article") or soup.find("main") or soup.find(attrs={"role": "main"}) or soup.find(id="content") or soup.body
    if not root:
        raise ValueError("HTML has no body/content root")
    lines = _node_lines(root)
    return title, "\n\n".join(lines), 1, len(lines)


def _extract_pdf(raw: bytes, entry: dict[str, Any]) -> tuple[str, str, int, int]:
    import pymupdf

    pdf = pymupdf.open(stream=raw, filetype="pdf")
    blocks: list[str] = []
    line_count = 0
    title = str(entry.get("title") or pdf.metadata.get("title") or "")
    in_references = False
    for page_number, page in enumerate(pdf, 1):
        page_text = page.get_text("text", sort=True)
        page_lines = [re.sub(r"\s+", " ", line).strip() for line in page_text.splitlines()]
        page_lines = [line for line in page_lines if line]
        if not title and page_lines:
            title = page_lines[0]
        for line in page_lines:
            if PAGE_NUMBER.fullmatch(line):
                continue
            if REFERENCE_HEADING.fullmatch(line) or re.match(r"^\[\d{1,4}\]\s+", line):
                in_references = True
                continue
            if in_references:
                continue
            is_heading = (
                len(line) <= 52
                and not re.search(r"\.{3,}|…{2,}\s*\d*$", line)
                and not re.match(r"^\d{3,4}\.\s*DOI\b", line, re.IGNORECASE)
                and (
                    CHAPTER_LINE.match(line)
                    or NUMBERED_HEADING.match(line)
                    or (entry.get("document_type") == "health_standard" and STANDARD_HEADING.match(line))
                )
            )
            if is_heading:
                blocks.append(f"## {line}")
            else:
                blocks.append(line)
        line_count += len(page_lines)
    text = "\n\n".join(blocks).strip()
    return title, text, pdf.page_count, line_count


def clean_one(source: str, entry: dict[str, Any], snapshot: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    raw_path = Path(__file__).parent.parent / snapshot["local_path"].replace("\\", "/")
    raw = raw_path.read_bytes()
    file_type = snapshot.get("file_type") or ("pdf" if raw.startswith(b"%PDF") else "html")
    if file_type == "pdf" or raw.startswith(b"%PDF"):
        title, body, page_count, block_count = _extract_pdf(raw, entry)
    else:
        title, body, page_count, block_count = _extract_html(raw, entry)
    if not body or len(body) < 500:
        raise ValueError(f"cleaned body unexpectedly short for {source}: {len(body)} chars")
    title = title or str(entry.get("title", source))
    if not body.lstrip().startswith(("# ", "## ")):
        body = f"# {title}\n\n{body}"
    content_hash = hashlib.sha256(raw).hexdigest()
    cleaned_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()
    retrieved_at = snapshot.get("retrieved_at", "")
    frontmatter = "\n".join([
        "---",
        f"source: {source}",
        f"title: {title}",
        f"publisher: {entry.get('publisher', '')}",
        f"source_url: {snapshot.get('final_url') or entry.get('source_url', '')}",
        f"retrieved_at: {retrieved_at}",
        f"published_at: {entry.get('published_at', 'unknown')}",
        f"raw_sha256: {content_hash}",
        f"cleaned_sha256: {cleaned_hash}",
        f"language: {entry.get('language', 'zh-CN')}",
        f"document_type: {entry.get('document_type', 'medical_reference')}",
        f"jurisdiction: {entry.get('jurisdiction', 'China-mainland')}",
        "review_status: source_curated",
        "---",
        "",
    ])
    out_path = out_dir / f"{source}.md"
    out_path.write_text(frontmatter + body.strip() + "\n", encoding="utf-8")
    return {
        "source": source,
        "file_type": file_type,
        "raw_file": str(raw_path),
        "output_file": str(out_path),
        "raw_bytes": len(raw),
        "raw_sha256": content_hash,
        "cleaned_chars": len(body),
        "page_count": page_count,
        "block_count": block_count,
        "title": title,
        "warning": "" if len(body) >= 1000 else "short source",
    }


def clean(root: Path) -> dict[str, Any]:
    rag_dir = root / "rag"
    manifest = json.loads((rag_dir / "source_manifest.json").read_text(encoding="utf-8"))
    snapshots = json.loads((rag_dir / "source_snapshot_manifest.json").read_text(encoding="utf-8"))["snapshots"]
    by_source = {row["source"]: row for row in snapshots}
    out_dir = rag_dir / "sources" / "cleaned"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for source, entry in manifest.items():
        if entry.get("review_status") != "source_curated":
            continue
        snapshot = by_source.get(source)
        if not snapshot:
            raise ValueError(f"missing raw snapshot for {source}")
        rows.append(clean_one(source, entry, snapshot, out_dir))
    report = {
        "schema_version": "1.0",
        "status": "cleaned",
        "cleaned_at": datetime.now(timezone.utc).isoformat(),
        "source_count": len(rows),
        "total_cleaned_chars": sum(row["cleaned_chars"] for row in rows),
        "sources": rows,
        "policy": "Only source_curated and provenance-registered documents are cleaned for indexing.",
    }
    (rag_dir / "cleaning_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    report = clean(args.root)
    print(json.dumps({"source_count": report["source_count"], "cleaned_chars": report["total_cleaned_chars"]}, ensure_ascii=False))
    print(f"wrote {args.root / 'rag' / 'cleaning_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
