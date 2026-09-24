"""Evaluate the deterministic emergency triage rule set on reviewed cases."""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


def _score(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}


def evaluate(root: Path, cases_path: Path) -> dict[str, Any]:
    import sys
    sys.path.insert(0, str(root / "backend"))
    from skills.emergency_triage.skill import EmergencyTriageSkill

    cases = [json.loads(line) for line in cases_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    skill = EmergencyTriageSkill()
    rows = []
    latencies = []
    for case in cases:
        start = time.perf_counter()
        result = skill.run(symptoms_text=case["text"])
        latencies.append((time.perf_counter() - start) * 1000)
        actual = result.level
        expected = case["expected_level"]
        rows.append({"id": case["id"], "expected_level": expected, "actual_level": actual,
                     "expected_positive": expected != "NON_URGENT",
                     "predicted_positive": actual != "NON_URGENT",
                     "correct": expected == actual})
    tp = sum(row["expected_positive"] and row["predicted_positive"] for row in rows)
    fp = sum(not row["expected_positive"] and row["predicted_positive"] for row in rows)
    fn = sum(row["expected_positive"] and not row["predicted_positive"] for row in rows)
    levels = ["CRITICAL", "URGENT", "PROMPT", "NON_URGENT"]
    confusion = {level: dict(Counter(row["actual_level"] for row in rows if row["expected_level"] == level)) for level in levels}
    per_level = {}
    for level in levels:
        level_tp = sum(row["expected_level"] == level and row["actual_level"] == level for row in rows)
        level_fp = sum(row["expected_level"] != level and row["actual_level"] == level for row in rows)
        level_fn = sum(row["expected_level"] == level and row["actual_level"] != level for row in rows)
        per_level[level] = {"support": sum(row["expected_level"] == level for row in rows), **_score(level_tp, level_fp, level_fn)}
    return {
        "schema_version": "1.0", "status": "measured_reviewed_rule_case_set",
        "dataset": str(cases_path), "sample_count": len(rows),
        "metrics": {"binary_red_flag": _score(tp, fp, fn),
                    "exact_level_accuracy": sum(row["correct"] for row in rows) / len(rows),
                    "macro_f1": mean(item["f1"] for item in per_level.values()),
                    "mean_latency_ms": mean(latencies)},
        "confusion_matrix": confusion, "per_level": per_level, "cases": rows,
        "note": "Reviewed synthetic rule cases, not clinical data; binary red-flag metrics group CRITICAL, URGENT, and PROMPT as positive."
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    cases = args.cases or args.root / "backend" / "evals" / "emergency_cases.jsonl"
    out = args.out or args.root / "backend" / "evals" / "reports" / "emergency_triage_report.json"
    report = evaluate(args.root, cases)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
