"""Generate reviewed-scope retrieval cases for the Chinese official corpus."""
from __future__ import annotations

import json
from pathlib import Path


CASES = (
    ("cn_cerebrovascular_guideline_2024", "第二章 脑血管病急救", "出现口角歪斜、言语不清或一侧肢体无力时应该怎么办？"),
    ("cn_cerebrovascular_guideline_2024", "一、提高早期识别能力", "怎么快速识别疑似卒中？"),
    ("cn_cerebrovascular_guideline_2024", "第三章 脑血管病预防", "脑血管病有哪些预防策略？"),
    ("cn_cerebrovascular_guideline_2024", "1.高血压控制", "控制高血压对预防脑卒中有什么意义？"),
    ("cn_cerebrovascular_guideline_2024", "第四章缺血性卒中和TIA 的临床管理", "缺血性卒中临床管理需要关注什么？"),
    ("cn_obesity_guideline_2024", "一、概述", "肥胖症为什么需要规范诊疗？"),
    ("cn_obesity_guideline_2024", "二、肥胖症的病因学", "肥胖症可能与哪些因素有关？"),
    ("cn_obesity_guideline_2024", "四、肥胖症的定义、诊断标准、分型、分期及相关疾病", "肥胖症如何定义和分期？"),
    ("cn_obesity_guideline_2024", "1.基于体质指数的诊断标准", "BMI 如何用于肥胖症判断？"),
    ("cn_obesity_guideline_2024", "3.高血压", "肥胖与高血压有什么关系？"),
    ("cn_myco_pneumonia_guideline_2025", "六、临床表现", "儿童肺炎支原体肺炎有哪些临床表现？"),
    ("cn_myco_pneumonia_guideline_2025", "十、MP 病原学检查", "肺炎支原体感染可以做哪些病原学检查？"),
    ("cn_myco_pneumonia_guideline_2025", "十二、诊断", "儿童肺炎支原体肺炎如何诊断？"),
    ("cn_myco_pneumonia_guideline_2025", "十三、鉴别诊断", "儿童肺炎支原体肺炎需要和哪些疾病鉴别？"),
    ("cn_myco_pneumonia_guideline_2025", "十四、常见肺内外并发症的早期识别和诊断", "儿童肺炎支原体肺炎有哪些并发症需要警惕？"),
    ("cn_hypertension_standard_2025", "5 血压测量", "成人血压筛查应该怎样测量？"),
    ("cn_hypertension_standard_2025", "6 筛查与预防", "成人高血压筛查和预防包括什么？"),
    ("cn_hypertension_standard_2025", "7 诊断与评估", "成人高血压诊断和评估依据是什么？"),
    ("cn_hypertension_standard_2025", "8.1 治疗原则", "高血压治疗的基本原则是什么？"),
    ("cn_hypertension_standard_2025", "8.3 随访管理", "高血压患者需要怎样随访？"),
    ("cn_hypertension_standard_2025", "8.4 转诊", "哪些高血压情况需要转诊？"),
    ("cn_nhsa_drug_catalog_2025", "一、目录构成", "国家医保药品目录的目录构成是什么？"),
    ("cn_nhsa_drug_catalog_2025", "四、限定支付范围", "医保药品目录的限定支付范围是什么意思？"),
    ("cn_commercial_drug_catalog_2025", "商业健康保险创新药品目录（2025年）", "商业健康保险创新药品目录和基本医保目录有什么区别？"),
    ("local_card_ff99c3ba0341", "过敏症关键信息", "出现药物过敏或其他过敏症状时需要注意什么？"),
)


def main() -> int:
    out = Path(__file__).with_name("rag_cases.jsonl")
    # Four paraphrases per reviewed information need gives a 100-case source-
    # localization suite without pretending that it is a naturally sampled
    # clinical-user benchmark.
    variants = (
        "{query}",
        "请说明{topic}。",
        "如果患者咨询{topic}，知识库应检索哪些资料？",
        "关于{topic}，临床问答需要重点参考什么？",
    )
    rows = []
    for anchor_index, (source, section, query) in enumerate(CASES, 1):
        topic = query.rstrip("？。")
        for variant_index, template in enumerate(variants, 1):
            rows.append({
                "id": f"rag-cn-official-{len(rows) + 1:03d}",
                "query": template.format(query=query, topic=topic),
                "gold_sections": [f"{source}::{section}"],
                "gold_sources": [source],
                "dataset_origin": "scenario_curated_chinese_official_sources_paraphrased",
                "review_status": "retrieval_scope_reviewed",
                "anchor_id": f"anchor-{anchor_index:03d}",
                "variant": variant_index,
            })
    out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(f"wrote {len(rows)} curated retrieval cases to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
