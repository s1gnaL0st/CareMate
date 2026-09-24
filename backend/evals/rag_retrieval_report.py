"""Compute Recall@K and MRR against a reviewed query-to-section set."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any


def load_cases(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not row.get("query") or not row.get("gold_sections"):
            raise ValueError(f"invalid RAG case at line {number}")
        rows.append(row)
    if not rows:
        raise ValueError("RAG case set is empty")
    return rows


def section_key(document: Any) -> str:
    metadata = getattr(document, "metadata", {}) or {}
    return f"{metadata.get('source', '')}::{metadata.get('section', '')}"


def evaluate(root: Path, cases_path: Path, ks: tuple[int, ...] = (1, 3, 5)) -> dict[str, Any]:
    sys.path.insert(0, str(root / "backend"))
    from rag.knowledge_base import HealthKnowledgeBase

    cases = load_cases(cases_path)
    kb = HealthKnowledgeBase()
    rows: list[dict[str, Any]] = []
    for case in cases:
        started = time.perf_counter()
        documents = kb.retrieve(str(case["query"]), k=max(ks))
        elapsed_ms = (time.perf_counter() - started) * 1000
        ranked = [section_key(doc) for doc in documents]
        gold = set(str(item) for item in case["gold_sections"])
        gold_sources = {item.split("::", 1)[0] for item in gold}
        ranked_sources = [key.split("::", 1)[0] for key in ranked]
        reciprocal_rank = next((1 / (index + 1) for index, key in enumerate(ranked) if key in gold), 0.0)
        rows.append({
            "id": case.get("id"),
            "query": case["query"],
            "gold_sections": sorted(gold),
            "ranked_sections": ranked,
            "reciprocal_rank": reciprocal_rank,
            "latency_ms": round(elapsed_ms, 2),
            "recall_at": {str(k): len(set(ranked[:k]) & gold) / len(gold) for k in ks},
            "source_recall_at": {str(k): len(set(ranked_sources[:k]) & gold_sources) / len(gold_sources) for k in ks},
        })
    dataset_origin = {str(case.get("dataset_origin", "unknown")) for case in cases}
    return {
        "schema_version": "1.0",
        "status": "measured_scenario_curated_source_mapping",
        "dataset": str(cases_path),
        "sample_count": len(rows),
        "section_count": len({str(item) for case in cases for item in case["gold_sections"]}),
        "cases_per_section": len(rows) // len({str(item) for case in cases for item in case["gold_sections"]}),
        "dataset_origin": sorted(dataset_origin),
        "metrics": {
            **{f"recall_at_{k}": mean(row["recall_at"][str(k)] for row in rows) for k in ks},
            **{f"source_recall_at_{k}": mean(row["source_recall_at"][str(k)] for row in rows) for k in ks},
            "mrr": mean(row["reciprocal_rank"] for row in rows),
            "mean_latency_ms": mean(row["latency_ms"] for row in rows),
        },
        "cases": rows,
        "note": "Scenario-curated queries mapped to source sections. Source-level Recall@K is the primary system metric; section-level Recall@K is diagnostic only. This does not measure clinical answer correctness and is not a naturally sampled user benchmark.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    cases = args.cases or args.root / "backend" / "evals" / "rag_cases.jsonl"
    report = evaluate(args.root, cases)
    out = args.out or args.root / "backend" / "evals" / "reports" / "rag_retrieval_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
