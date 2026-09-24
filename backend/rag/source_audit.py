"""Audit active and excluded RAG sources against the provenance registry."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def audit(root: Path) -> dict[str, Any]:
    rag_dir = root / "rag" if (root / "rag").exists() else root / "backend" / "rag"
    docs_dir = rag_dir / "documents"
    cleaned_dir = rag_dir / "sources" / "cleaned"
    manifest = json.loads((rag_dir / "source_manifest.json").read_text(encoding="utf-8"))
    rows = []
    paths = {path.stem: path for path in docs_dir.glob("*.md")}
    paths.update({path.stem: path for path in cleaned_dir.glob("*.md")})
    for local_dir in (rag_dir / "sources" / "cleaned_local_drugs", rag_dir / "sources" / "cleaned_local_cards"):
        paths.update({path.stem: path for path in local_dir.glob("*.md")})
    for source, path in sorted(paths.items()):
        entry = manifest.get(source)
        status = "unregistered" if not isinstance(entry, dict) else entry.get("review_status", "unknown")
        rows.append({
            "source": source,
            "file": str(path),
            "status": status,
            "indexed": status in {"source_curated", "local_imported"},
            "content_hash": hashlib.sha256(path.read_bytes()).hexdigest(),
            "publisher": entry.get("publisher") if isinstance(entry, dict) else None,
            "source_url": entry.get("source_url") if isinstance(entry, dict) else None,
            "version": entry.get("version") if isinstance(entry, dict) else None,
        })
    active = [row for row in rows if row["indexed"]]
    snapshot_manifest_path = rag_dir / "source_snapshot_manifest.json"
    snapshot_manifest = json.loads(snapshot_manifest_path.read_text(encoding="utf-8")) if snapshot_manifest_path.exists() else {"snapshot_count": 0, "snapshots": []}
    return {
        "schema_version": "1.0",
        "status": "audited",
        "active_source_count": len(active),
        "document_file_count": len(rows),
        "active_sources": active,
        "excluded_sources": [row for row in rows if not row["indexed"]],
        "source_snapshot_count": snapshot_manifest.get("snapshot_count", 0),
        "source_snapshots": snapshot_manifest.get("snapshots", []),
        "policy": "Only source_curated and local_imported manifest entries are indexed; cleaned source files take precedence over legacy summaries.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = audit(args.root)
    rag_dir = args.root / "rag" if (args.root / "rag").exists() else args.root / "backend" / "rag"
    out = args.out or rag_dir / "source_audit_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"active_source_count": report["active_source_count"], "document_file_count": report["document_file_count"]}, ensure_ascii=False))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
