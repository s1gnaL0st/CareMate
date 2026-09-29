"""Run end-to-end Agent metrics on the synthetic/reviewed benchmark contract."""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * p)))
    return ordered[index]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(row["latency_ms"]) for row in rows if row.get("latency_ms") is not None]
    route = [float(row.get("route_correct", 0)) for row in rows]
    task = [float(row.get("task_complete", 0)) for row in rows]
    citation = [float(row["citation_faithfulness"]) for row in rows if row.get("citation_faithfulness") is not None]
    overtriage = [row for row in rows if row.get("safety_expected") == "NON_URGENT"]
    overtriage_rate = (sum(bool(row.get("safety_predicted_non_urgent") is False) for row in overtriage) / len(overtriage)) if overtriage else None
    return {
        "sample_count": len(rows),
        "route_accuracy": statistics.mean(route) if route else None,
        "task_completion_rate": statistics.mean(task) if task else None,
        "citation_faithfulness_mean": statistics.mean(citation) if citation else None,
        "overtriage_rate": overtriage_rate,
        "latency_ms": {
            "mean": statistics.mean(latencies) if latencies else None,
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
        },
        "cost_usd_mean": statistics.mean([float(row["cost_usd"]) for row in rows if row.get("cost_usd") is not None]) if any(row.get("cost_usd") is not None for row in rows) else None,
    }


async def run(root: Path, cases_path: Path, output: Path, use_judge: bool, case_timeout_seconds: float) -> dict[str, Any]:
    sys.path.insert(0, str(root / "backend"))
    from dotenv import load_dotenv
    load_dotenv(root / "backend" / ".env")
    from agents.graph_new import master_app
    from evals.agent_quality import score_evidence_faithfulness, judge_execution_offline
    from langchain_core.messages import HumanMessage

    cases = load_rows(cases_path)
    rows = []
    for case in cases:
        started = time.perf_counter()
        state: dict[str, Any] = {}
        error = None
        try:
            state = await asyncio.wait_for(
                master_app.ainvoke({"messages": [HumanMessage(content=case["input"])], "user_info": {}}),
                timeout=case_timeout_seconds,
            )
        except Exception as exc:  # keep the benchmark fail-closed per case
            error = type(exc).__name__
        latency_ms = (time.perf_counter() - started) * 1000
        # The current Supervisor graph records routing in its task contract;
        # active_agent/next_agent are legacy compatibility fields and remain
        # empty by design. Treat any expected agent present in the planned or
        # completed tasks as a correct route.
        task_results = dict(state.get("task_results", {}) or {})
        planned_agents = {
            str(task.agent if hasattr(task, "agent") else task.get("agent", ""))
            for task in (state.get("task_queue", []) or [])
        }
        completed_agents = {
            str(result.get("agent", ""))
            for result in task_results.values()
            if result.get("agent")
        }
        routed_agents = planned_agents | completed_agents
        actual_agent = ",".join(sorted(routed_agents))
        expected = set(str(item) for item in case.get("expected_agents", []))
        route_correct = bool(expected & routed_agents)
        results = task_results
        task_complete = (
            all(str(item.get("status")) == "completed" for item in results.values())
            if results else bool(state.get("final_response") or state.get("run_status") in {"completed", "success"})
        )
        faithfulness = score_evidence_faithfulness(state).get("provenance_coverage")
        row: dict[str, Any] = {
            "id": case["id"], "suite": case.get("suite"), "actual_agent": actual_agent,
            "route_correct": route_correct, "task_complete": task_complete,
            "citation_faithfulness": faithfulness, "latency_ms": latency_ms,
            "cost_usd": state.get("cost_usd") or state.get("estimated_cost_usd"),
            "error": error,
        }
        # Safety overtriage is measured on emergency_cases_200.jsonl, where
        # every negative has an explicit NON_URGENT label. Do not infer a
        # safety gold label from ordinary end-to-end prompts.
        if use_judge and not error:
            row["llm_judge"] = await judge_execution_offline(case["prompt"], state)
        rows.append(row)
        if len(rows) % 25 == 0:
            print(f"processed {len(rows)}/{len(cases)}")
    report = {
        "schema_version": "e2e_eval_v1", "status": "measured_runtime_synthetic_regression",
        "dataset": str(cases_path), "dataset_origin": "synthetic_regression_only",
        "metrics": summarize(rows), "llm_judge_enabled": use_judge,
        "rows": rows,
        "notes": [
            "Synthetic regression data is not human-labelled clinical evidence.",
            "Citation faithfulness is provenance coverage unless an offline LLM judge is enabled.",
            "Overtriage rate is computed only for cases explicitly labelled NON_URGENT.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    print(f"wrote {output}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--cases", type=Path, default=Path(__file__).with_name("e2e_cases_100.jsonl"))
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("reports") / "e2e_eval_latest.json")
    parser.add_argument("--judge", action="store_true", help="invoke the configured LLM as an offline answer-quality judge")
    parser.add_argument("--case-timeout", type=float, default=45.0, help="hard timeout per graph case")
    args = parser.parse_args()
    asyncio.run(run(args.root, args.cases, args.out, args.judge, max(1.0, args.case_timeout)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
