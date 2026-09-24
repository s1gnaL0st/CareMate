"""Generate the checked-in 100-case RAG retrieval review set.

The questions are deliberately template-generated and reviewed against the
checked-in section names. They are useful for regression testing retrieval
ranking, but are not a substitute for a naturally sampled user benchmark.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


SECTIONS: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
    ("medical_knowledge", "高血压（Hypertension）", "高血压", ("诊断范围和常见风险", "常见症状与危险信号", "日常管理和监测", "生活方式干预", "就医与治疗注意事项")),
    ("medical_knowledge", "2型糖尿病（Type 2 Diabetes）", "2型糖尿病", ("诊断标准和典型表现", "常见并发症风险", "饮食和运动管理", "血糖监测要点", "何时需要就医")),
    ("medical_knowledge", "感冒与流感（Cold & Influenza）", "感冒和流感", ("常见症状和区别", "发热与咳嗽表现", "家庭护理建议", "何时需要就医", "预防和传播注意事项")),
    ("medical_knowledge", "冠心病（Coronary Artery Disease）", "冠心病", ("常见症状和危险因素", "胸痛表现如何判断", "日常风险管理", "急性发作时怎么处理", "检查和就医建议")),
    ("medical_knowledge", "常用非处方药参考（OTC Drug Reference）", "常用非处方药", ("常见药品类别", "适应症和使用场景", "用药注意事项", "禁忌和特殊人群", "何时不能自行用药")),
    ("insurance_policies", "基本医疗保险概述", "基本医疗保险", ("保障范围和基本规则", "参保人可以享受哪些待遇", "起付线和报销范围", "个人账户与统筹基金", "办理医保业务的基本要求")),
    ("insurance_policies", "门诊报销政策", "医保门诊报销", ("门诊报销比例和起付线", "普通门诊费用如何报销", "门诊报销需要什么材料", "年度报销额度和限制", "门诊看病结算流程")),
    ("insurance_policies", "住院报销政策", "医保住院报销", ("住院报销比例和起付线", "住院费用如何结算", "不同等级医院的报销差异", "住院报销需要哪些材料", "出院时医保结算流程")),
    ("insurance_policies", "门诊特殊病（慢性病医保政策）", "门诊特殊病和慢性病医保", ("慢性病门诊待遇范围", "门诊特殊病如何申请", "需要准备哪些认定材料", "报销比例和支付限额", "复审和待遇变更流程")),
    ("insurance_policies", "异地就医政策", "异地就医医保", ("异地就医备案流程", "异地住院如何直接结算", "跨省就医需要什么材料", "临时异地就医怎么报销", "异地就医报销注意事项")),
    ("insurance_policies", "医保报销申请流程", "医保报销申请", ("医保报销申请步骤", "零星报销需要哪些材料", "报销申请提交到哪里", "审核和到账通常经过哪些环节", "报销被退回如何处理")),
    ("lab_reference", "血常规参考范围", "血常规参考范围", ("白细胞和红细胞正常范围", "血红蛋白参考值怎么看", "血小板参考范围是多少", "中性粒细胞和淋巴细胞怎么看", "儿童和成人血常规有什么差异")),
    ("lab_reference", "血生化参考范围", "血生化参考范围", ("血糖和血脂参考范围", "肝功能指标正常范围", "肾功能肌酐尿素氮怎么看", "电解质指标参考值", "血生化异常通常如何解读")),
    ("lab_reference", "尿常规参考范围", "尿常规参考范围", ("尿蛋白和尿糖正常结果", "尿白细胞和红细胞参考范围", "尿常规比重和酸碱度怎么看", "尿酮体和尿胆原异常意义", "尿常规检查前注意事项")),
    ("lab_reference", "常见异常指标临床意义", "常见异常检验指标", ("白细胞升高通常提示什么", "血红蛋白偏低可能说明什么", "转氨酶升高如何理解", "血糖异常有哪些常见原因", "尿蛋白阳性需要注意什么")),
    ("pharmacy_knowledge", "常见OTC非处方药指南", "常见非处方药", ("布洛芬适应症和注意事项", "对乙酰氨基酚怎么安全使用", "常见解热镇痛药如何选择", "非处方药的用法和禁忌", "服用止痛药何时应该就医")),
    ("pharmacy_knowledge", "常见感冒用药", "感冒用药", ("感冒症状可以用哪些药", "退热和止咳药如何选择", "复方感冒药有哪些注意事项", "感冒用药能否同时服用", "感冒用药无效时何时就医")),
    ("pharmacy_knowledge", "常见慢性病用药指导", "慢性病用药", ("高血压常用药怎么服用", "糖尿病用药有哪些注意事项", "慢性病药物能不能自行停用", "漏服慢性病药应该怎么办", "慢性病用药需要监测什么")),
    ("pharmacy_knowledge", "药物相互作用常见警示", "药物相互作用", ("常见药物相互作用有哪些", "哪些药物组合需要避免", "抗凝药和止痛药能否同服", "多种药一起吃如何判断风险", "出现药物相互作用症状怎么办")),
    ("pharmacy_knowledge", "用药安全常识", "用药安全", ("正确用药有哪些基本原则", "药品应该如何保存", "忘记服药后应该怎么办", "老年人用药需要注意什么", "哪些情况不能自行用药")),
)


def build_cases() -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for section_index, (source, section, topic, questions) in enumerate(SECTIONS, 1):
        for question_index, focus in enumerate(questions, 1):
            cases.append({
                "id": f"rag-{section_index:02d}-{question_index:02d}",
                "query": f"关于{topic}，{focus}？",
                "gold_sections": [f"{source}::{section}"],
                "dataset_origin": "template_generated_reviewed_section_query",
                "section_group": section,
            })
    return cases


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("rag_cases.jsonl"))
    args = parser.parse_args()
    cases = build_cases()
    if len(cases) != 100 or len({case["section_group"] for case in cases}) != 20:
        raise RuntimeError("expected exactly 100 cases across 20 sections")
    args.out.write_text(
        "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases),
        encoding="utf-8",
    )
    print(f"wrote {len(cases)} cases across 20 sections to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
