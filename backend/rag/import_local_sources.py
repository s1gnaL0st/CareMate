"""Import user-provided cards.json and legacy Word drug archives into RAG."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import tempfile
import zipfile
from typing import Any


ACTIVE_STATUSES = {"source_curated", "local_imported"}
SECTION_RE = re.compile(r"^【([^】]{1,40})】")


def _slug(text: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]+", "_", text).strip("_")
    return value[:70] or "unnamed"


def _frontmatter(source: str, title: str, source_url: str, content_hash: str, document_type: str) -> str:
    fields = [
        "---",
        f"source: {source}",
        f"title: {title}",
        "publisher: 用户提供的本地资料（未核验）",
        f"source_url: {source_url}",
        "retrieved_at: local-import",
        "published_at: unknown",
        f"raw_sha256: {content_hash}",
        f"cleaned_sha256: {content_hash}",
        "language: zh-CN",
        f"document_type: {document_type}",
        "jurisdiction: China-mainland",
        "review_status: local_imported",
        "version: unknown",
        "---",
        "",
    ]
    return "\n".join(fields)


def _format_text(text: str) -> str:
    text = text.replace("\x0c", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    output: list[str] = []
    for line in lines:
        match = SECTION_RE.match(line)
        if match:
            output.append(f"## {match.group(1).strip()}")
            rest = line[match.end():].strip()
            if rest:
                output.append(rest)
        else:
            output.append(line)
    return "\n\n".join(output).strip()


def _word_to_text(word: Any, doc_path: Path, txt_path: Path) -> str:
    document = word.Documents.Open(
        str(doc_path), ReadOnly=True, AddToRecentFiles=False, ConfirmConversions=False
    )
    try:
        document.SaveAs2(str(txt_path), FileFormat=2, Encoding=936)
    finally:
        document.Close(False)
    return txt_path.read_bytes().decode("gb18030", errors="replace")


def _doc_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("通用名："):
            return line.split("：", 1)[1].strip() or fallback
    for line in text.splitlines():
        line = line.strip("\u3000 ")
        if line and not line.startswith("【") and len(line) < 80:
            return line
    return fallback


def _import_drug_archives(rag_dir: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    import win32com.client

    output_dir = rag_dir / "sources" / "cleaned_local_drugs"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    word = win32com.client.DispatchEx("Word.Application")
    word.Visible = False
    word.DisplayAlerts = 0
    try:
        for archive in sorted(rag_dir.glob("*.zip")):
            with zipfile.ZipFile(archive) as package:
                members = [name for name in package.namelist() if name.lower().endswith(".doc")]
                for index, member in enumerate(members, 1):
                    raw = package.read(member)
                    digest = hashlib.sha256(raw).hexdigest()
                    source = f"local_drug_{archive.stem}_{digest[:12]}"
                    title_fallback = Path(member).stem
                    with tempfile.TemporaryDirectory(prefix="sha_drug_") as temp:
                        doc_path = Path(temp) / "source.doc"
                        txt_path = Path(temp) / "source.txt"
                        doc_path.write_bytes(raw)
                        try:
                            extracted = _word_to_text(word, doc_path, txt_path)
                        except Exception as exc:
                            rows.append({"source": source, "archive": archive.name, "member": member, "status": "conversion_failed", "error": type(exc).__name__})
                            continue
                    body = _format_text(extracted)
                    if len(body) < 100:
                        rows.append({"source": source, "archive": archive.name, "member": member, "status": "skipped_short", "chars": len(body)})
                        continue
                    title = _doc_title(extracted, title_fallback)
                    output = output_dir / f"{source}.md"
                    output.write_text(
                        _frontmatter(source, title, f"local://{archive.name}!{member}", digest, "drug_label_local")
                        + f"# {title}\n\n{body}\n",
                        encoding="utf-8",
                    )
                    manifest[source] = {
                        "title": title,
                        "publisher": "用户提供的本地资料（未核验）",
                        "source_url": f"local://{archive.name}!{member}",
                        "source_urls": [f"local://{archive.name}!{member}"],
                        "retrieved_at": "local-import",
                        "published_at": "unknown",
                        "jurisdiction": "China-mainland",
                        "language": "zh-CN",
                        "document_type": "drug_label_local",
                        "review_status": "local_imported",
                        "version": "unknown",
                        "raw_sha256": digest,
                        "archive": archive.name,
                        "archive_member": member,
                        "scope_note": "用户提供的药品资料，尚未与国家药监局批准说明书逐条核验；不得单独作为用药依据。",
                    }
                    rows.append({"source": source, "title": title, "archive": archive.name, "member": member, "status": "imported", "chars": len(body), "output": str(output)})
    finally:
        word.Quit(False)
    return rows


def _import_cards(rag_dir: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = [
        rag_dir / "sources" / "cards.jsonl",
        rag_dir / "sources" / "cards.json",
        rag_dir / "source" / "cards.json",
        rag_dir.parent.parent / "source" / "cards.json",
        rag_dir / "cards.json",
    ]
    cards_path = next((path for path in candidates if path.exists()), None)
    if cards_path is None:
        return [{"status": "not_found", "searched": [str(path) for path in candidates]}]
    if cards_path.suffix.lower() == ".jsonl":
        cards = [json.loads(line) for line in cards_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        payload = json.loads(cards_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            cards = payload
        elif isinstance(payload, dict):
            cards = payload.get("cards") or payload.get("items") or payload.get("data") or []
        else:
            cards = []
    if not isinstance(cards, list):
        raise ValueError("cards.json cards/items/data must be a list")
    output_dir = rag_dir / "sources" / "cleaned_local_cards"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, card in enumerate(cards, 1):
        if not isinstance(card, dict):
            continue
        title = str(card.get("title") or card.get("name") or f"医学知识卡 {index}")
        text = str(card.get("text") or card.get("content") or "").strip()
        tags = card.get("tags") or []
        body = "\n\n".join([
            text,
            f"标签：{', '.join(str(tag) for tag in tags)}" if tags else "",
            f"疾病类别：{card.get('condition') or card.get('family')}" if card.get("condition") or card.get("family") else "",
            f"原始来源：{card.get('attribution') or card.get('source_title') or card.get('source')}" if card.get("attribution") or card.get("source_title") or card.get("source") else "",
        ]).strip()
        if not body:
            body = json.dumps(card, ensure_ascii=False, indent=2)
        digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
        source = f"local_card_{digest[:12]}"
        output = output_dir / f"{source}.md"
        output.write_text(_frontmatter(source, title, f"local://{cards_path.name}#{index}", digest, "medical_knowledge_card_local") + f"# {title}\n\n{body}\n", encoding="utf-8")
        source_url = str(card.get("source_url") or f"local://{cards_path.name}#{index}")
        manifest[source] = {
            "title": title, "publisher": str(card.get("source") or "用户提供的本地知识卡（来源字段待核验）"), "source_url": source_url, "source_urls": [source_url], "retrieved_at": str(card.get("retrieved_at") or "local-import"), "published_at": str(card.get("source_updated_at") or "unknown"), "jurisdiction": "international", "language": str(card.get("source_language") or "zh-CN"), "document_type": "medical_knowledge_card_local", "review_status": "local_imported", "version": "unknown", "raw_sha256": digest, "license": card.get("license", ""), "evidence_id": card.get("evidence_id", ""),
        }
        rows.append({"source": source, "title": title, "status": "imported", "chars": len(body), "output": str(output)})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    rag_dir = args.root / "rag"
    manifest_path = rag_dir / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cards = _import_cards(rag_dir, manifest)
    drugs = _import_drug_archives(rag_dir, manifest)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {"cards": cards, "drugs": drugs, "active_statuses": sorted(ACTIVE_STATUSES)}
    (rag_dir / "local_import_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cards": sum(row.get("status") == "imported" for row in cards), "drugs": sum(row.get("status") == "imported" for row in drugs), "card_status": cards[0].get("status") if cards else "none"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
