"""Deterministic offline metrics for the checked-in evaluation evidence.

This report intentionally distinguishes measured metrics from missing datasets.
It does not call an LLM, mutate production data, or turn an absent benchmark
into a misleading zero.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} must contain JSON objects")
        rows.append(value)
    return rows


def _route_metrics(cases: list[dict[str, Any]]) -> dict[str, Any]:
    artifact_path = Path(__file__).with_name("reports") / "routing_eval_latest.json"
    if artifact_path.exists():
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        observations = artifact.get("observations", [])
        labels = [str(row.get("expected_agent", "")) for row in observations]
        predictions = [str(row.get("actual_agent", "")) for row in observations]
        matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for expected, actual in zip(labels, predictions, strict=True):
            matrix[expected][actual] += 1
        classes = sorted(set(labels) | set(predictions))
        per_class = {}
        for name in classes:
            tp = matrix[name][name]
            support = sum(matrix[name].values())
            predicted = sum(matrix[source][name] for source in classes)
            precision = tp / predicted if predicted else 0.0
            recall = tp / support if support else 0.0
            per_class[name] = {"support": support, "precision": precision, "recall": recall,
                               "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}
        macro_f1 = sum(row["f1"] for row in per_class.values()) / len(per_class) if per_class else 0.0
        return {"status": "measured_runtime_run", "sample_count": len(observations),
                "accuracy": artifact.get("route_accuracy"), "macro_f1": macro_f1,
                "provider": artifact.get("provider"), "timestamp": artifact.get("timestamp"),
                "confusion_matrix": {key: dict(value) for key, value in matrix.items()},
                "per_class": per_class,
                "note": artifact.get("note", "")}
    labels = [str(row["expected_agent"]) for row in cases]
    # The checked-in cases are the expected-route regression contract.  No
    # model prediction is present, so this is contract coverage, not a claim
    # of live model accuracy.
    predictions = [str(row["expected_agent"]) for row in cases]
    matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for expected, actual in zip(labels, predictions, strict=True):
        matrix[expected][actual] += 1
    classes = sorted(set(labels) | set(predictions))
    per_class = {}
    for name in classes:
        tp = matrix[name][name]
        support = sum(matrix[name].values())
        predicted = sum(matrix[source][name] for source in classes)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        per_class[name] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": (2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
        }
    macro_f1 = sum(row["f1"] for row in per_class.values()) / len(per_class) if per_class else 0.0
    return {
        "status": "contract_coverage_only",
        "sample_count": len(cases),
        "accuracy": None,
        "macro_f1": None,
        "expected_label_coverage": len(set(labels)) / len(labels) if labels else None,
        "confusion_matrix": {key: dict(value) for key, value in matrix.items()},
        "per_class": per_class,
        "note": "cases.jsonl contains expected routes but no recorded model predictions; do not present this as live routing accuracy.",
    }


def _trajectory_metrics(trajectories: list[dict[str, Any]]) -> dict[str, Any]:
    if not trajectories:
        return {"status": "not_available", "sample_count": 0}
    turns = [row.get("turns", []) for row in trajectories]
    answer_rate = sum(any(item.get("action") == "answer" for item in row) for row in turns) / len(turns)
    asks = [sum(item.get("action") == "ask" for item in row) for row in turns]
    duplicate_count = sum(
        int((row.get("reward") or {}).get("details", {}).get("duplicate_questions", 0) or 0)
        for row in trajectories
    )
    tool_failures = sum(
        sum(bool(item.get("tool_error")) for item in row)
        for row in turns
    )
    evidence_recalls = [
        (row.get("reward") or {}).get("details", {}).get("evidence_recall")
        for row in trajectories
    ]
    evidence_recalls = [float(value) for value in evidence_recalls if value is not None]
    return {
        "status": "pilot",
        "sample_count": len(trajectories),
        "answer_rate": answer_rate,
        "mean_turns": sum(float(row.get("turn_count", len(row))) for row, _ in zip(trajectories, turns, strict=True)) / len(trajectories),
        "mean_ask_count": sum(asks) / len(asks),
        "duplicate_question_rate": duplicate_count / len(trajectories),
        "tool_failure_rate": tool_failures / sum(max(len(row), 1) for row in turns),
        "evidence_recall_mean": sum(evidence_recalls) / len(evidence_recalls) if evidence_recalls else None,
        "terminal_reasons": dict(Counter(str(row.get("terminal_reason", "unknown")) for row in trajectories)),
        "cases": [
            {
                "case_id": row.get("case_id"),
                "family": row.get("family"),
                "turns": row.get("turn_count"),
                "actions": [item.get("action") for item in row.get("turns", [])],
            }
            for row in trajectories
        ],
        "note": "three checked-in trajectories; this is a pilot smoke set, not a generalization benchmark.",
    }


def _safety_implementation_metrics(root: Path) -> dict[str, Any]:
    skill_path = root / "backend" / "skills" / "emergency_triage" / "skill.py"
    test_path = root / "backend" / "test_clinic_action_adapter.py"
    if not skill_path.exists():
        return {"status": "not_available", "sample_count": 0}
    source = skill_path.read_text(encoding="utf-8")
    counts: dict[str, int] = {}
    for level in ("CRITICAL", "URGENT", "PROMPT"):
        match = re.search(rf"_{level}_PATTERNS\s*=\s*\[(.*?)\]\s*\n\s*\n", source, re.S)
        counts[level.lower()] = len(re.findall(r"\(r[\"']", match.group(1))) if match else 0
    test_count = 0
    if test_path.exists():
        test_source = test_path.read_text(encoding="utf-8")
        test_count = len(re.findall(r"def test_.*(?:emergency|triage|red_flag)", test_source, re.I))
    return {
        "status": "implementation_coverage",
        "rule_counts": counts,
        "total_rule_patterns": sum(counts.values()),
        "related_contract_tests": test_count,
        "note": "Rule inventory and unit-test coverage only; no labeled clinical dataset, so precision/recall/F1 are not inferred.",
    }


def _emergency_red_flag_metrics(root: Path) -> dict[str, Any]:
    report_path = root / "backend" / "evals" / "reports" / "emergency_triage_report.json"
    if not report_path.exists():
        return {"status": "not_available", "sample_count": 0,
                "reason": "No reviewed emergency rule case report has been generated."}
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"status": "not_available", "sample_count": 0, "reason": "Emergency report is invalid."}
    return {"status": report.get("status", "measured_reviewed_rule_case_set"),
            "sample_count": report.get("sample_count", 0),
            **dict(report.get("metrics", {})),
            "per_level": report.get("per_level", {}),
            "note": report.get("note", "")}


def _adapter_contract_metrics(root: Path) -> dict[str, Any]:
    """Measure the checked-in action/harness contract without a model call."""
    test_path = root / "backend" / "test_clinic_action_adapter.py"
    source_path = root / "backend" / "agents" / "clinic_action_adapter.py"
    if not test_path.exists() or not source_path.exists():
        return {"status": "not_available", "sample_count": 0}
    source = source_path.read_text(encoding="utf-8")
    actions = set(re.findall(r'"(ask|check|lookup|search|answer)"', source))
    tests = test_path.read_text(encoding="utf-8")
    contract_names = (
        "patient_answer_after_ask_is_replayed_as_tool_result",
        "simulated_two_turn_loop_reaches_answer_after_tool_replay",
        "ask_rejects_a_duplicate_question",
        "emergency_veto_is_deterministic_and_precedes_model_actions",
        "sanitize_clinic_answer_removes_training_and_untrusted_citations",
    )
    covered = sum(name in tests for name in contract_names)
    return {
        "status": "implementation_coverage",
        "exposed_actions": sorted(actions),
        "five_action_contract_complete": actions == {"ask", "check", "lookup", "search", "answer"},
        "key_contract_tests": len(contract_names),
        "key_contract_tests_present": covered,
        "contract_test_coverage": covered / len(contract_names),
        "note": "Protocol and harness coverage only; this is not a model-quality percentage.",
    }


def _rag_corpus_metrics(root: Path) -> dict[str, Any]:
    """Report corpus and metadata facts; ranking quality needs query labels."""
    manifest_path = root / "backend" / "rag" / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cleaned_dir = root / "backend" / "rag" / "sources" / "cleaned"
    docs_dir = root / "backend" / "rag" / "documents"
    source_dirs = [directory for directory in (cleaned_dir, root / "backend" / "rag" / "sources" / "cleaned_local_drugs", root / "backend" / "rag" / "sources" / "cleaned_local_cards") if directory.exists()]
    if not source_dirs:
        source_dirs = [docs_dir]
    if not any(directory.exists() for directory in source_dirs):
        return {"status": "not_available", "sample_count": 0}
    files = sorted(
        path for source_dir in source_dirs for path in source_dir.glob("*.md")
        if manifest.get(path.stem, {}).get("review_status") in {"source_curated", "local_imported"}
    )
    try:
        import sys
        sys.path.insert(0, str(root / "backend"))
        from rag.knowledge_base import HealthKnowledgeBase
        chunks = HealthKnowledgeBase()._load_documents()
        metadata_complete = sum(
            all(str(doc.metadata.get(key, "")).strip() for key in (
                "source", "section", "publisher", "source_url", "retrieved_at",
                "jurisdiction", "document_type", "review_status", "version", "content_hash",
            ))
            for doc in chunks
        )
    except Exception as exc:
        return {"status": "corpus_inventory", "document_count": len(files), "error": type(exc).__name__}
    return {
        "status": "corpus_inventory",
        "document_count": len(files),
        "active_source_count": sum(
            entry.get("review_status") in {"source_curated", "local_imported"}
            for entry in manifest.values()
        ),
        "chunk_count": len(chunks),
        "metadata_complete_rate": metadata_complete / len(chunks) if chunks else None,
        "retrieval_strategy": "BM25 0.3 + dense retrieval 0.7; MMR for dense-result de-duplication/re-ranking",
        "note": "source_curated and local_imported documents are counted; local_imported material is user-provided and not independently verified.",
    }


def _rag_retrieval_metrics(root: Path) -> dict[str, Any]:
    report_path = root / "backend" / "evals" / "reports" / "rag_retrieval_report.json"
    if not report_path.exists():
        return {"status": "not_available", "sample_count": 0, "reason": "RAG retrieval report has not been generated."}
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"status": "not_available", "sample_count": 0, "reason": "RAG retrieval report is invalid."}
    return {
        "status": report.get("status", "measured_reviewed_query_set"),
        "sample_count": report.get("sample_count", 0),
        **dict(report.get("metrics", {})),
        "note": report.get("note", ""),
    }


def _retry_contract_metrics(root: Path) -> dict[str, Any]:
    test_path = root / "backend" / "test_clinic_action_adapter.py"
    graph_test_path = root / "backend" / "test_graph_new.py"
    if not test_path.exists():
        return {"status": "not_available", "sample_count": 0}
    text = test_path.read_text(encoding="utf-8")
    graph_text = graph_test_path.read_text(encoding="utf-8") if graph_test_path.exists() else ""
    cases = {
        "transient_retry_success": "retries_transient_failure_with_bounded_attempts" in text,
        "non_transient_not_retried": "does_not_retry_non_transient_request_error" in text,
        "task_retry_contract": "executor_retries_transient_agent_failure" in graph_text,
        "fallback_contract": "fallback_plan_is_available_for_model_outage" in graph_text,
    }
    return {
        "status": "implementation_coverage",
        "contract_cases": cases,
        "contract_pass_rate": sum(cases.values()) / len(cases),
        "note": "Failure-injection success rate and availability uplift require runtime run logs; contract tests do not establish them.",
    }


def _llm_judge_contract_metrics(root: Path) -> dict[str, Any]:
    adapter = root / "backend" / "evals" / "deepeval_adapter.py"
    return {
        "status": "available_as_runner",
        "provider_adapter": str(adapter),
        "judge_metrics": ["tool_correctness"],
        "scored_cases": 0,
        "note": "DeepEval is wired as an optional judge; no network-backed judge score is claimed until a run artifact is checked in.",
    }


def _self_evolution_metrics(root: Path) -> dict[str, Any]:
    report_path = root / "backend" / "evals" / "reports" / "self_evolution_eval_latest.json"
    if not report_path.exists():
        return {
            "status": "not_available",
            "sample_count": 0,
            "reason": "No paired baseline/candidate graph-run artifact is checked in.",
        }
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"status": "not_available", "sample_count": 0, "reason": "Self-evolution report is invalid."}
    overall = dict(report.get("overall", {}))
    return {
        "status": report.get("status", "measured_local_paired_graph_run"),
        "sample_count": overall.get("case_count", 0),
        **overall,
        "by_suite": report.get("by_suite", {}),
        "timestamp": report.get("timestamp"),
        "note": report.get("note", ""),
    }


def build_report(root: Path) -> dict[str, Any]:
    cases_path = root / "backend" / "evals" / "cases.jsonl"
    trajectory_path = root / "trajectories.jsonl"
    cases = _read_jsonl(cases_path) if cases_path.exists() else []
    trajectories = _read_jsonl(trajectory_path) if trajectory_path.exists() else []
    unavailable = {
        "status": "not_available",
        "sample_count": 0,
        "reason": "No frozen labeled dataset or paired baseline/candidate observations are checked in.",
    }
    return {
        "schema_version": "1.0",
        "source_files": [
            str(cases_path),
            str(trajectory_path),
            str(root / "backend" / "evals" / "rag_cases.jsonl"),
            str(root / "backend" / "evals" / "reports" / "rag_retrieval_report.json"),
            str(root / "backend" / "evals" / "emergency_cases.jsonl"),
            str(root / "backend" / "evals" / "reports" / "emergency_triage_report.json"),
            str(root / "backend" / "evals" / "reports" / "routing_eval_latest.json"),
            str(root / "backend" / "evals" / "reports" / "self_evolution_eval_latest.json"),
        ],
        "metrics": {
            "supervisor_routing": _route_metrics(cases),
            "agent_loop": _trajectory_metrics(trajectories),
            "emergency_rule_implementation": _safety_implementation_metrics(root),
            "emergency_red_flag": _emergency_red_flag_metrics(root),
            "rag_retrieval": unavailable,
            "self_evolution_baseline_vs_candidate": _self_evolution_metrics(root),
            "model_retry_and_fallback": unavailable,
            "action_adapter_contract": _adapter_contract_metrics(root),
            "rag_corpus": _rag_corpus_metrics(root),
            "rag_retrieval": _rag_retrieval_metrics(root),
            "retry_fallback_contract": _retry_contract_metrics(root),
            "llm_as_judge": _llm_judge_contract_metrics(root),
        },
    }


def _markdown(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    routing = metrics["supervisor_routing"]
    loop = metrics["agent_loop"]
    safety = metrics["emergency_rule_implementation"]
    adapter = metrics["action_adapter_contract"]
    rag = metrics["rag_corpus"]
    retry = metrics["retry_fallback_contract"]
    rag_retrieval = metrics["rag_retrieval"]
    evolution = metrics["self_evolution_baseline_vs_candidate"]
    lines = [
        "# Offline Evaluation Report",
        "",
        "This report is generated from checked-in evidence. `contract_coverage_only` and `pilot` must not be described as production or general benchmark results.",
        "",
        "## Measured",
        "",
        f"- Supervisor routing: status={routing.get('status')}, n={routing['sample_count']}, accuracy={routing.get('accuracy')}, macro-F1={routing.get('macro_f1')}, provider={routing.get('provider', 'n/a')}.",
        f"- GRPO trajectory pilot: n={loop['sample_count']}, answer rate={loop.get('answer_rate')}, mean turns={loop.get('mean_turns')}, mean asks={loop.get('mean_ask_count')}, duplicate-question rate={loop.get('duplicate_question_rate')}, tool-failure rate={loop.get('tool_failure_rate')}, mean evidence recall={loop.get('evidence_recall_mean') }.",
        f"- Emergency rule implementation: {safety.get('total_rule_patterns', 0)} patterns (CRITICAL={safety.get('rule_counts', {}).get('critical', 0)}, URGENT={safety.get('rule_counts', {}).get('urgent', 0)}, PROMPT={safety.get('rule_counts', {}).get('prompt', 0)}), {safety.get('related_contract_tests', 0)} related contract tests; no clinical P/R/F1 inferred.",
        f"- Emergency red-flag rule cases: n={metrics['emergency_red_flag'].get('sample_count')}, binary precision={metrics['emergency_red_flag'].get('binary_red_flag', {}).get('precision')}, recall={metrics['emergency_red_flag'].get('binary_red_flag', {}).get('recall')}, F1={metrics['emergency_red_flag'].get('binary_red_flag', {}).get('f1')}; reviewed synthetic cases only.",
        f"- Action adapter contract: five_action_complete={adapter.get('five_action_contract_complete')}, test coverage={adapter.get('contract_test_coverage')}; protocol coverage only.",
        f"- RAG corpus inventory: documents={rag.get('document_count')}, chunks={rag.get('chunk_count')}, metadata-complete={rag.get('metadata_complete_rate')}; measured ranking report is included separately.",
        f"- RAG retrieval ranking: n={rag_retrieval.get('sample_count')}, source Recall@1/3/5={rag_retrieval.get('source_recall_at_1')}/{rag_retrieval.get('source_recall_at_3')}/{rag_retrieval.get('source_recall_at_5')}, section Recall@1/3/5={rag_retrieval.get('recall_at_1')}/{rag_retrieval.get('recall_at_3')}/{rag_retrieval.get('recall_at_5')}, MRR={rag_retrieval.get('mrr')}, mean latency={rag_retrieval.get('mean_latency_ms')} ms; scenario-curated set only.",
        f"- Retry/fallback contract coverage: {retry.get('contract_pass_rate')}; runtime success/availability uplift not measured.",
        f"- Self-evolution paired graph run: status={evolution.get('status')}, n={evolution.get('sample_count')}, baseline score={evolution.get('baseline_score')}, candidate score={evolution.get('candidate_score')}, baseline pass rate={evolution.get('baseline_pass_rate')}, candidate pass rate={evolution.get('candidate_pass_rate')}, gate passed={evolution.get('gate_passed')}.",
        "",
        "## Not Measured",
        "",
        "- Supervisor routing generalization: the runtime artifact contains only 5 regression cases and includes provider/tool failures; do not present it as a broad routing benchmark.",
        f"- RAG citation hit rate: not measured; the retrieval report has {rag_retrieval.get('sample_count')} scenario-curated queries mapped to official-source sections and measures source localization, not citation correctness or clinical answer quality.",
        "- Retry and DeepSeek fallback success rate: no failure-injection run log.",
        "- LLM-as-Judge answer quality: optional DeepEval adapter exists, but no network-backed judge run is checked in.",
        "",
        "## Evidence",
        "",
        *[f"- `{path}`" for path in report["source_files"]],
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--md-out", type=Path)
    args = parser.parse_args()
    report = build_report(args.root)
    json_out = args.json_out or args.root / "backend" / "evals" / "reports" / "evaluation_report.json"
    md_out = args.md_out or args.root / "backend" / "evals" / "reports" / "evaluation_report.md"
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_out.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    print(f"wrote {json_out}")
    print(f"wrote {md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
