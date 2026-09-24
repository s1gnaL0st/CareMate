"""Run the checked-in agent regression dataset against the current graph."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from openinference.instrumentation import using_attributes

from evals import (
    EvalCase,
    EvalScore,
    evaluate_run,
    evaluate_tools_with_deepeval,
    extract_eval_run,
    load_cases,
)
from observability import configure_observability


DEFAULT_DATASET = Path(__file__).with_name("cases.jsonl")
MODE_TO_AGENT = {
    "clinic": "clinic_agent",
    "insurance": "insurance_agent",
    "report": "report_agent",
    "pharmacy": "pharmacy_agent",
    "general": "advisor_agent",
}


async def _execute_case(case: EvalCase):
    from agents.graph_new import master_app

    initial_state = {
        "messages": [HumanMessage(content=case.input)],
        "user_info": {},
        "next_agent": "",
        "active_agent": MODE_TO_AGENT.get(case.chat_mode, "advisor_agent"),
        "requested_mode": case.chat_mode,
    }
    with using_attributes(
        session_id=f"eval:{case.id}",
        tags=["evaluation"],
    ):
        return await master_app.ainvoke(initial_state)


async def run_dataset(
    cases: list[EvalCase],
    provider: str,
) -> list[EvalScore]:
    scores: list[EvalScore] = []
    for case in cases:
        state = await _execute_case(case)
        run = extract_eval_run(case, state)
        score = evaluate_run(case, run)
        if provider == "deepeval" and case.expected_tools is not None:
            tool_score = evaluate_tools_with_deepeval(case, run)
            score = replace(
                score,
                tool_score=tool_score,
                passed=score.route_score == 1.0 and tool_score == 1.0,
            )
        scores.append(score)
    return scores


async def run_dataset_with_artifact(
    cases: list[EvalCase],
    provider: str,
) -> tuple[list[EvalScore], list[dict[str, object]]]:
    scores: list[EvalScore] = []
    observations: list[dict[str, object]] = []
    for case in cases:
        try:
            state = await _execute_case(case)
            run = extract_eval_run(case, state)
            score = evaluate_run(case, run)
            if provider == "deepeval" and case.expected_tools is not None:
                tool_score = evaluate_tools_with_deepeval(case, run)
                score = replace(
                    score,
                    tool_score=tool_score,
                    passed=score.route_score == 1.0 and tool_score == 1.0,
                )
            observations.append({
                "case_id": case.id,
                "expected_agent": case.expected_agent,
                "actual_agent": run.actual_agent,
                "expected_tools": list(case.expected_tools or ()),
                "tools_called": list(run.tools_called),
                "route_score": score.route_score,
                "tool_score": score.tool_score,
                "passed": score.passed,
            })
        except Exception as exc:
            score = EvalScore(case.id, 0.0, 0.0 if case.expected_tools is not None else None, False)
            observations.append({
                "case_id": case.id,
                "expected_agent": case.expected_agent,
                "actual_agent": None,
                "expected_tools": list(case.expected_tools or ()),
                "tools_called": [],
                "route_score": 0.0,
                "tool_score": score.tool_score,
                "passed": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
        scores.append(score)
    return scores, observations


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--case", dest="case_id", help="run one case by ID")
    parser.add_argument(
        "--provider",
        choices=("local", "deepeval"),
        default=os.environ.get("EVAL_PROVIDER", "local"),
    )
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = _parse_args()
    cases = load_cases(args.dataset)
    if args.case_id:
        cases = [case for case in cases if case.id == args.case_id]
        if not cases:
            raise SystemExit(f"unknown eval case: {args.case_id}")

    observability_runtime = configure_observability()
    try:
        scores, observations = asyncio.run(run_dataset_with_artifact(cases, args.provider))
    finally:
        observability_runtime.shutdown()
    for score in scores:
        tool = "skip" if score.tool_score is None else f"{score.tool_score:.2f}"
        status = "PASS" if score.passed else "FAIL"
        print(
            f"{status} {score.case_id} route={score.route_score:.2f} tools={tool}"
        )
    passed = sum(score.passed for score in scores)
    artifact = args.dataset.parent / "reports" / "routing_eval_latest.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(
        json.dumps({
            "schema_version": "1.0",
            "status": "measured_runtime_run",
            "dataset": str(args.dataset),
            "provider": args.provider,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sample_count": len(observations),
            "passed_count": passed,
            "route_accuracy": sum(item["route_score"] == 1.0 for item in observations) / len(observations),
            "observations": observations,
            "note": "Runtime agent evaluation. Provider/model availability and tool execution errors are included as observed failures.",
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {artifact}")
    print(f"Summary: {passed}/{len(scores)} passed ({args.provider})")
    return 0 if passed == len(scores) else 1


if __name__ == "__main__":
    raise SystemExit(main())
