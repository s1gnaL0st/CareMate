"""Fetch registered official source pages into an auditable local snapshot.

The snapshots are provenance artifacts, not blindly indexed content. Curated
Markdown entries remain the reviewed, bounded representation used by RAG.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit
from urllib.request import Request, urlopen


def fetch(root: Path) -> dict[str, object]:
    rag_dir = root / "rag"
    manifest = json.loads((rag_dir / "source_manifest.json").read_text(encoding="utf-8"))
    snapshot_dir = rag_dir / "sources" / "raw"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    seen: set[str] = set()
    for source, entry in manifest.items():
        if entry.get("review_status") != "source_curated":
            continue
        for url in entry.get("source_urls", []):
            if url in seen:
                continue
            seen.add(url)
            parts = urlsplit(url)
            fetch_url = urlunsplit((parts.scheme, parts.netloc, quote(parts.path, safe="/%:@"), parts.query, parts.fragment))
            request = Request(fetch_url, headers={"User-Agent": "Smart-Health-Assistant/1.0 source-audit"})
            with urlopen(request, timeout=30) as response:
                body = response.read()
                final_url = response.geturl()
                content_type = response.headers.get("Content-Type", "")
            digest = hashlib.sha256(body).hexdigest()
            suffix = ".pdf" if "application/pdf" in content_type.lower() or url.lower().split("?", 1)[0].endswith(".pdf") else ".html"
            filename = f"{digest[:16]}{suffix}"
            (snapshot_dir / filename).write_bytes(body)
            rows.append({
                "source": source,
                "requested_url": url,
                "final_url": final_url,
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "content_type": content_type,
                "sha256": digest,
                "local_path": str(Path("rag/sources/raw") / filename),
                "file_type": suffix.lstrip("."),
            })
    output = {"schema_version": "1.0", "snapshot_count": len(rows), "snapshots": rows}
    out = rag_dir / "source_snapshot_manifest.json"
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    result = fetch(args.root)
    print(json.dumps({"snapshot_count": result["snapshot_count"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
