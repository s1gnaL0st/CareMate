"""Expand the reviewed 100-case RAG set into a reproducible 500-case stress set.

The generated variants preserve gold sections and are marked synthetic; they
are useful for regression/stress comparison, not a substitute for annotation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

TEMPLATES = (
    "{query}",
    "请直接回答：{query}",
    "如果患者咨询，{query}",
    "临床健康问答中，{query}",
    "请结合可靠医学资料说明，{query}",
)


def build(source: Path, target: Path) -> int:
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("source dataset is empty")
    output: list[dict] = []
    for row in rows:
        for variant, template in enumerate(TEMPLATES, 1):
            item = dict(row)
            item["id"] = f"{row.get('id', 'case')}-aug{variant}"
            item["query"] = template.format(query=str(row["query"]).rstrip("。？?"))
            item["variant"] = variant
            item["dataset_origin"] = "scenario_curated_source_mapping_synthetic_5x"
            item["review_status"] = "synthetic_regression_only"
            output.append(item)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in output) + "\n", encoding="utf-8")
    return len(output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path(__file__).with_name("rag_cases.jsonl"))
    parser.add_argument("--target", type=Path, default=Path(__file__).with_name("rag_cases_500.jsonl"))
    args = parser.parse_args()
    count = build(args.source, args.target)
    print(f"wrote {count} cases to {args.target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
