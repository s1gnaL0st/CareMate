# Agent 可观测与评估

本项目提供一个不依赖商业平台的最小闭环：LangGraph 运行链路通过 OpenInference 转为 OpenTelemetry Span，以标准 OTLP 输出到 Jaeger；回归样例由仓库管理，默认本地评分，也可使用 DeepEval。LangSmith 仍作为兼容选项保留。

## 生命周期

| 阶段 | 项目内做法 | 产物 |
| --- | --- | --- |
| 开发 | 修改 Agent、Prompt、Skill 或 RAG 文档，并补单元测试与 `evals/cases.jsonl` | 可复现代码和样例 |
| 监控 | OpenInference 自动记录节点耗时、模型调用、视觉调用和工具调用 | OTLP Trace |
| 评估 | 执行路由与工具选择基线，使用本地评分或 DeepEval | case 级通过/失败与进程退出码 |
| 迭代 | 根据失败 case 定位 Jaeger Span，修复后重复运行 | 可比较的回归结果 |

这里记录的是 **Agent 执行状态与工具调用进度**，不展示或承诺模型的私有思考链。

## 本地链路追踪

在仓库根目录启动 Jaeger：

```bash
docker compose -f compose.observability.yml up -d
```

在 `backend/.env` 中启用 OTLP：

```dotenv
OBSERVABILITY_PROVIDER=otel
OTEL_SERVICE_NAME=smart-health-assistant
OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://localhost:4318/v1/traces
TRACE_CONTENT=false
TRACE_IMAGES=false
```

启动后访问 `http://localhost:16686`，选择 `smart-health-assistant` 查看 Trace。输出采用标准 OTLP/HTTP，因此可将端点替换为 Phoenix、Grafana Tempo、SigNoz 或其他兼容后端；也支持通用的 `OTEL_EXPORTER_OTLP_ENDPOINT`，代码会追加 `/v1/traces`。`compose.observability.yml` 使用内存存储，仅适合本地调试。

设置 `OBSERVABILITY_PROVIDER=langsmith` 并配置 `LANGSMITH_API_KEY`（或旧名 `LANGCHAIN_API_KEY`）可继续使用原有 LangSmith 链路；相同内容开关会映射为 LangSmith 的输入、输出与元数据隐藏配置。设置 `none` 则关闭观测。

## 隐私默认值

`TRACE_CONTENT=false` 会隐藏输入、输出、消息文本、Prompt、调用参数和工具定义；图片还需单独设置 `TRACE_IMAGES=true` 才会进入 Span。为防止 base64 图片从 LangChain 的通用输入属性泄露，只有两个开关同时为 `true` 时才记录输入；Embedding 向量始终隐藏。后端日志也不打印用户资料、对话原文或模型原始路由结果。

医疗报告可能包含姓名、证件号、条码和检查结果。开发与 CI 应只使用脱敏或合成样例，不要把真实报告、完整回答或含个人信息的 Trace 提交到仓库。若显式开启内容追踪，应先确认后端访问控制、保留期限和删除机制。

## 回归评估

默认数据集位于 `backend/evals/cases.jsonl`，每行包含 `id`、`input`、`expected_agent`，可选 `expected_tools` 与 `chat_mode`。运行会调用当前配置的模型，因此需要对应 API Key：

```bash
cd backend

# 内置严格评分：路由必须一致，声明工具时工具集合必须一致
uv run python -m evals
uv run python -m evals --case route-report-lab

# 可选 DeepEval 工具评分
uv run --extra eval python -m evals --provider deepeval
```

评估运行器会复用相同的观测配置，并将匿名 `eval:<case_id>` 写入 session 属性，因此启用 `otel` 后可在 Jaeger 中按失败 case 排查。DeepEval judge 复用 `LLM_PROVIDER` 的多厂商配置，不强制 OpenAI；本项目默认设置 `DEEPEVAL_TELEMETRY_OPT_OUT=1`。命令只输出 case ID 和分数，任一 case 失败时返回非零退出码，便于接入 CI。新增能力时至少增加一个正常路径和一个安全边界样例。

## 端到端 Agent 质量评测

`backend/evals/agent_quality.py` 在离线成对 Graph 运行中记录 Planner DAG 合法性、期望 Agent 覆盖率、依赖准确率，以及 Agent 结果的证据来源 ID 覆盖。离线 runner 可显式开启有界 RAG 摘录捕获（每个来源最多 2,000 字符、每个任务最多 8 条）；摘录仅存在于本轮内存评测状态，持久化前剥离，不进入线上响应或通用追踪。没有来源正文时，语义忠实度标记为不可评估，而不是 0 分。

需要离线语义审计时，显式运行：

```bash
cd backend
uv run python -m evals.run_self_evolution_eval --llm-judge
```

Judge 逐条评估计划质量、回答完整度及回答主张与来源证据的支持关系，并输出主张、支持等级、证据 ID 和理由。仅当样本实际带有来源正文且 Judge 给主张提供了明确支持判定时才计算支持率；未判定主张不计分，缺失指标保持 `null`。不符合输出结构的 Judge 结果会标记 `invalid_output` 并跳过，不会中断整轮，也不会生成失败结论。Judge 仅作离线诊断，不参与候选门禁，也不会进入在线问诊路径。低分只生成 `staging_review_required` 候选；经人工核验前，不会写入冻结集或作为真实 FailureCase 宣称。

现有真实 Graph 对照数据集仍为 9 条人工构造样例。新增指标可在这 9 条上验证管线，但不能把它说成扩展后的代表性 benchmark；增加公开医疗数据前，需先核验许可、去标识状态、中文问诊适配度，并由人工标注 Planner 期望任务/依赖及可引用证据。

### 统一端到端基准（新增）

仓库新增 `evals/e2e_cases_100.jsonl` 和 `evals/e2e_eval.py`，每条样本记录期望 Agent、工具、证据锚点和安全边界。当前 100 条是 10 个模板的确定性变体扩展，标记为 `synthetic_regression_only`，不能表述为 100 条人工标注临床数据。

```bash
cd backend
uv run python evals/build_e2e_benchmark.py
uv run python evals/e2e_eval.py --cases evals/e2e_cases_100.jsonl
```

评测输出：任务完成率、路由准确率、证据来源覆盖、过度拦截率、平均延迟、P50/P95 延迟和成本字段。工具超时、模型异常和 Graph 失败按 case 记录，不能静默当作成功；完整性不足时报告应标记为 partial，而不是生成完整 benchmark 结论。

### LLM-as-a-Judge

`evals/agent_quality.py` 已支持显式离线 Judge，使用当前 `LLM_PROVIDER/LLM_MODEL` 对计划质量、回答完整度及主张-证据支持关系进行结构化评分。Judge 只用于离线诊断，不参与线上回答或自进化发布门禁。后续若用于回答质量四维评分，应增加 `correctness / completeness / safety / readability` 字段，并人工抽检至少 20% 样本后报告 Judge-人工一致率；在完成抽检前不得填写一致率。

### 安全集扩展

`evals/emergency_cases_200.jsonl` 在原有 48 条 reviewed synthetic safety cases 外，增加了提示注入/对抗性医疗 prompt 和“看似吓人但实际非急症”的负例，并按 `case_type` 分组报告。命令：

```bash
cd backend
uv run python evals/expand_emergency_cases.py
uv run python evals/emergency_triage_report.py \
  --cases evals/emergency_cases_200.jsonl \
  --out evals/reports/emergency_triage_report_200.json
```

过度拦截率只在明确标记为 `NON_URGENT` 的负例上计算；对抗性 prompt 的安全通过率单独报告，不能与原有红旗识别 Precision/Recall 混成一个数字。
