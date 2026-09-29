"""Compare one-shot RAG with bounded Agentic RAG on the same cases."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any


def load_cases(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def section_key(doc: Any) -> str:
    metadata = getattr(doc, "metadata", {}) or {}
    return f"{metadata.get('source', '')}::{metadata.get('section', '')}"


def scores(documents: list[Any], case: dict[str, Any]) -> dict[str, float]:
    ranked = [section_key(doc) for doc in documents]
    gold = set(str(item) for item in case["gold_sections"])
    gold_sources = {item.split("::", 1)[0] for item in gold}
    source_ranked = [item.split("::", 1)[0] for item in ranked]
    rr = next((1 / (index + 1) for index, item in enumerate(ranked) if item in gold), 0.0)
    return {
        "recall_at_1": len(set(ranked[:1]) & gold) / len(gold),
        "recall_at_3": len(set(ranked[:3]) & gold) / len(gold),
        "recall_at_5": len(set(ranked[:5]) & gold) / len(gold),
        "source_recall_at_1": len(set(source_ranked[:1]) & gold_sources) / len(gold_sources),
        "source_recall_at_3": len(set(source_ranked[:3]) & gold_sources) / len(gold_sources),
        "source_recall_at_5": len(set(source_ranked[:5]) & gold_sources) / len(gold_sources),
        "mrr": rr,
    }


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round((len(ordered) - 1) * p)))]


async def run(root: Path, cases_path: Path, output: Path, max_rounds: int) -> dict[str, Any]:
    sys.path.insert(0, str(root / "backend"))
    from dotenv import load_dotenv
    load_dotenv(root / "backend" / ".env")
    from agentic_rag import retrieve_until_sufficient
    from agents.llm import get_chat_llm, resolve_model_settings
    from rag.knowledge_base import HealthKnowledgeBase

    cases = load_cases(cases_path)
    kb = HealthKnowledgeBase()
    controller_llm = get_chat_llm("fast", streaming=False, temperature=0.0)
    model_settings = resolve_model_settings()
    baseline_rows: list[dict[str, Any]] = []
    agentic_rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases, 1):
        query = str(case["query"])
        started = time.perf_counter()
        # Use the async path for both arms so the Agentic first round can
        # reuse the exact same cache entry; baseline still makes one retrieval
        # call and never performs a follow-up query.
        baseline_docs = await kb.aretrieve(query, k=5)
        baseline_ms = (time.perf_counter() - started) * 1000
        baseline_rows.append({**scores(baseline_docs, case), "latency_ms": baseline_ms, "retrieval_calls": 1})

        started = time.perf_counter()
        agentic = await retrieve_until_sufficient(query, kb, k=5, max_rounds=max_rounds,
                                                  controller_llm=controller_llm,
                                                  rerank_llm=controller_llm)
        agentic_ms = (time.perf_counter() - started) * 1000
        agentic_rows.append({**scores(agentic.documents, case), "latency_ms": agentic_ms,
                             "retrieval_calls": len(agentic.rounds), "sufficient": agentic.sufficient,
                             "missing_facets": agentic.missing_facets, "stop_reason": agentic.stop_reason})
        if index % 50 == 0:
            print(f"processed {index}/{len(cases)}")

    metric_names = ("recall_at_1", "recall_at_3", "recall_at_5", "source_recall_at_1", "source_recall_at_3", "source_recall_at_5", "mrr", "latency_ms", "retrieval_calls")
    def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        result = {name: mean(float(row[name]) for row in rows) for name in metric_names}
        latencies = [float(row["latency_ms"]) for row in rows]
        result["latency_p50_ms"] = percentile(latencies, 0.50)
        result["latency_p95_ms"] = percentile(latencies, 0.95)
        result["sufficient_rate"] = mean(float(row.get("sufficient", True)) for row in rows)
        return result

    report = {
        "schema_version": "rag_ab_v1",
        "dataset": str(cases_path),
        "sample_count": len(cases),
        "baseline": {"name": "one_shot_hybrid_rag", "metrics": aggregate(baseline_rows)},
        "agentic": {"name": "bounded_iterative_agentic_rag", "max_rounds": max_rounds, "metrics": aggregate(agentic_rows)},
        "controller": {"provider": model_settings.provider, "model": model_settings.model,
                        "purpose": "LLM decides evidence sufficiency, follow-up query, and candidate evidence reranking"},
        "delta": {name: aggregate(agentic_rows)[name] - aggregate(baseline_rows)[name] for name in metric_names},
        "notes": [
            f"Dataset contains {len(cases)} cases; the 500-case file is a deterministic 5x synthetic expansion, while rag_cases.jsonl is the original 100-case scenario-curated source mapping set.",
            "The baseline uses exactly one HealthKnowledgeBase.retrieve call; Agentic RAG can issue up to max_rounds aretrieve calls.",
            "Agentic controller uses the configured LLM; this run records the resolved provider and model.",
            "Recall measures source-section mapping, not clinical answer correctness.",
        ],
        "rows": {"baseline": baseline_rows, "agentic": agentic_rows},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"baseline": report["baseline"]["metrics"], "agentic": report["agentic"]["metrics"], "delta": report["delta"]}, ensure_ascii=False, indent=2))
    print(f"wrote {output}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--cases", type=Path, default=Path(__file__).with_name("rag_cases_500.jsonl"))
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("reports") / "rag_agentic_comparison_500.json")
    parser.add_argument("--max-rounds", type=int, default=3)
    args = parser.parse_args()
    asyncio.run(run(args.root, args.cases, args.out, args.max_rounds))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
