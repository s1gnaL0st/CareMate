# Offline Evaluation Report

This report is generated from checked-in evidence. `contract_coverage_only` and `pilot` must not be described as production or general benchmark results.

## Measured

- Supervisor routing: status=measured_runtime_run, n=5, accuracy=0.2, macro-F1=0.06666666666666668, provider=local.
- GRPO trajectory pilot: n=3, answer rate=1.0, mean turns=9.0, mean asks=3.0, duplicate-question rate=0.0, tool-failure rate=0.0, mean evidence recall=1.0.
- Emergency rule implementation: 23 patterns (CRITICAL=10, URGENT=8, PROMPT=5), 3 related contract tests; no clinical P/R/F1 inferred.
- Emergency red-flag rule cases: n=48, binary precision=0.9629629629629629, recall=0.8666666666666667, F1=0.912280701754386; reviewed synthetic cases only.
- Action adapter contract: five_action_complete=True, test coverage=1.0; protocol coverage only.
- RAG corpus inventory: documents=390, chunks=3333, metadata-complete=1.0; measured ranking report is included separately.
- RAG retrieval ranking: n=100, source Recall@1/3/5=0.88/0.92/0.96, section Recall@1/3/5=0.26/0.35/0.42, MRR=0.3155, mean latency=89.6451 ms; scenario-curated set only.
- Retry/fallback contract coverage: 1.0; runtime success/availability uplift not measured.
- Self-evolution paired graph run: status=measured_local_paired_graph_run, n=9, baseline score=1.0, candidate score=1.0, baseline pass rate=1.0, candidate pass rate=1.0, gate passed=True.

## Not Measured

- Supervisor routing generalization: the runtime artifact contains only 5 regression cases and includes provider/tool failures; do not present it as a broad routing benchmark.
- RAG citation hit rate: not measured; the retrieval report has 100 scenario-curated queries mapped to official-source sections and measures source localization, not citation correctness or clinical answer quality.
- Retry and DeepSeek fallback success rate: no failure-injection run log.
- LLM-as-Judge answer quality: optional DeepEval adapter exists, but no network-backed judge run is checked in.

## Evidence

- `F:\沉淀\Smart-Health-Assistant-main\backend\evals\cases.jsonl`
- `F:\沉淀\Smart-Health-Assistant-main\trajectories.jsonl`
- `F:\沉淀\Smart-Health-Assistant-main\backend\evals\rag_cases.jsonl`
- `F:\沉淀\Smart-Health-Assistant-main\backend\evals\reports\rag_retrieval_report.json`
- `F:\沉淀\Smart-Health-Assistant-main\backend\evals\emergency_cases.jsonl`
- `F:\沉淀\Smart-Health-Assistant-main\backend\evals\reports\emergency_triage_report.json`
- `F:\沉淀\Smart-Health-Assistant-main\backend\evals\reports\routing_eval_latest.json`
- `F:\沉淀\Smart-Health-Assistant-main\backend\evals\reports\self_evolution_eval_latest.json`
