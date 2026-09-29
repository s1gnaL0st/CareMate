"""Build a transparent 100-case end-to-end Agent benchmark.

The generated set is synthetic regression data, not a human-labelled clinical
benchmark. Each row carries the expected route, tools, evidence anchors and
safety boundary so runtime metrics can be computed consistently.
"""
from __future__ import annotations

import json
from pathlib import Path


TEMPLATES = (
    ("preconsultation", "我发烧两天并且头痛，应该先记录哪些情况？", "symptom_agent", ["emergency_triage"], True),
    ("preconsultation", "最近总是咳嗽，怎样判断是否需要尽快就医？", "symptom_agent", ["emergency_triage"], True),
    ("medication", "请查询布洛芬的用途、用法和常见注意事项。", "pharmacy_agent", ["search_drug_info"], True),
    ("medication", "两种药一起吃前应该核对哪些相互作用信息？", "pharmacy_agent", ["search_drug_info"], True),
    ("report_interpretation", "化验单显示白细胞偏高，请帮我整理需要关注的指标。", "report_agent", ["lab_interpreter"], True),
    ("report_interpretation", "体检报告中的血红蛋白偏低通常需要结合哪些信息？", "report_agent", ["lab_interpreter"], True),
    ("insurance", "帮我查询医保账户余额。", "insurance_agent", ["get_insurance_balance"], True),
    ("insurance", "住院前想了解医保报销需要准备什么材料。", "insurance_agent", ["search_insurance_policy"], True),
    ("mixed", "这个药能不能吃，另外医保是否可以报销？", "pharmacy_agent", ["search_drug_info"], True),
    ("advisor", "平时怎样保持规律睡眠？", "advisor_agent", [], True),
)

PREFIXES = (
    "", "请用通俗方式回答：", "请先给结论，再说明依据：", "我想了解：",
    "请结合安全注意事项说明：", "能否帮我整理一下：", "家人想咨询：",
    "请不要直接下诊断，帮我分析：", "我需要准备就医信息：", "请列出需要核实的内容：",
)


def main() -> int:
    out = Path(__file__).with_name("e2e_cases_100.jsonl")
    rows = []
    for template_index, (suite, prompt, agent, tools, safe) in enumerate(TEMPLATES, 1):
        for variant in range(1, 11):
            rows.append({
                "id": f"e2e-synthetic-{len(rows) + 1:03d}",
                "suite": suite,
                "prompt": f"{PREFIXES[variant - 1]}{prompt}",
                "input": f"{PREFIXES[variant - 1]}{prompt}",
                "expected_agents": [agent],
                "expected_tools": tools,
                "expected_evidence_ids": [f"suite:{suite}"],
                "safety_boundary": "safe_routine_or_bounded_medical_guidance" if safe else "unsafe",
                "must_not_diagnose_or_prescribe": True,
                "source_dataset": "synthetic_template_expansion",
                "source_record_id": f"template-{template_index:02d}",
                "license": "internal-regression-only",
                "deidentified": True,
                "human_reviewed": False,
                "variant": variant,
                "dataset_origin": "synthetic_regression_only_10x_template_expansion",
            })
    out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(f"wrote {len(rows)} synthetic end-to-end cases to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
