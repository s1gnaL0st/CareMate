# Smart Health Assistant：医疗 Agent 平台项目说明

## 1. 项目定位

Smart Health Assistant 是一个面向健康咨询、预问诊、检验报告辅助解读、用药信息查询和医保事务的研究型 Agent 平台。核心目标不是让单个大模型直接回答，而是将请求拆成可路由、可执行、可验证、可回放的任务，并在医疗安全边界内聚合结果。

系统不提供临床诊断、处方开具或急诊替代处置，输出用于帮助用户整理信息、理解资料和选择下一步行动。

## 2. 总体架构

```text
Web / API / 图片或 PDF
          ↓ SSE
FastAPI 会话层
          ↓
Agent Runtime
  Context · Budget · Snapshot · Trace
          ↓
LangGraph Supervisor
  Intent Gate → Planner → Executor → Verifier → Responder
                     ↑          ↓
                 Repair / Replan / Handoff
          ↓
领域 Agent 与受控工具
  symptom · report · pharmacy · insurance · advisor
          ↓
RAG · 规则引擎 · MCP · OCR · 用户记忆
          ↓
MySQL · Redis · MinIO · OpenTelemetry · Offline Evaluation
```

职责边界是项目的关键设计：LangGraph 负责状态图和流程编排；Agent Runtime 负责单次运行的预算、快照、超时和审计；Supervisor 负责意图识别、任务拆解和调度；Verifier 与安全 Harness 负责结果门禁。Runtime 是 LangGraph 的运行边界，不替换 LangGraph。

## 3. 意图识别与任务编排

请求先经过确定性急症规则和结构化 Intent Gate，主要分为 `emergency`、`pure_chat`、`mixed`、`health`。模型输出必须通过 Schema 校验；模型异常、工具失败和证据不足均采用保守兜底，不把异常视为安全通过。

Planner 将复合请求转换为任务 DAG，并检查任务 ID、依赖无环、Agent 白名单、输入切片、重试次数和安全边界。无依赖任务按波次并行执行，有依赖任务按顺序执行。

Verifier 输出 `pass / partial / fail / unsafe`：`pass` 代表结果满足约束；`partial` 保留已完成结果并补充缺失任务；`fail` 局部重跑或重新规划；`unsafe` 停止普通模型链路，输出保守建议或转人工。次数锁、内容锁和运行预算限制重规划，避免 Agent 无限循环。

## 4. 领域 Agent 与工具隔离

| Agent | 负责内容 |
|---|---|
| `symptom_agent` | 症状收集、红旗识别和风险分流 |
| `report_agent` | 检验报告 OCR、指标解释和参考范围检索 |
| `pharmacy_agent` | 药品信息、相互作用和用药安全边界 |
| `insurance_agent` | 医保政策检索、账户和事务工具 |
| `advisor_agent` | 通用健康科普和生活方式建议 |

每个 Agent 使用独立 Prompt、工具白名单、上下文切片、输出契约和停止条件。领域结果统一封装为 `status / summary / evidence / risk_level / uncertainty / next_action`，供 Verifier 做一致性检查。

## 5. Agentic RAG

知识库采用父子块结构：子块负责精确命中，父块负责补充完整章节上下文。基础检索链路为：

```text
查询规范化 → BM25 + Dense 召回 → RRF 融合
→ MMR 去重 → 子块命中 → 父块聚合 → 来源注入 → Verifier 核查
```

Agentic RAG 将知识库检索作为 Agent 可调用工具，而不是固定前置步骤。控制器记录每轮 query、来源 ID、证据覆盖、缺失证据面和停止原因：Agent 首轮检索后观察证据缺口，必要时生成面向缺口的下一轮查询，合并候选并用 DeepSeek 重排，直到证据充分或达到预算。证据不足会显式进入回答的不确定性，不由模型静默补全。

检索指标按评测协议分开记录：

- **Agentic RAG 100 条 A/B 场景集**：普通 RAG 的 Source Recall@1/3/5 为 88%/92%/96%，Agentic RAG 为 90%/93%/93%；章节级 Recall@3 从 35% 提升到 52%（+17 个百分点），MRR 从 0.3155 提升到 0.4995（+0.1840）。

Source Recall 衡量是否定位到期望来源，章节级 Recall 衡量是否命中期望章节；两者都不等同于医学答案正确率或引用忠实度。

## 6. 医疗安全 Harness

安全链路采用：

```text
模型输出 + 确定性急症规则 + 工具结果
        → Verifier 门禁 → 紧急否决
```

确定性规则优先级高于模型输出。系统内置 23 条急症识别规则，覆盖严重胸痛、呼吸困难、卒中样表现、严重过敏和大出血等红旗信号；高危场景直接拦截普通模型链路。

系统禁止确诊、开处方、越权给药和伪造引用。基于 48 条人工合成安全样本，红旗识别 Precision 为 96.3%、Recall 为 86.7%、F1 为 91.2%，风险分级准确率为 89.6%。这些是安全回归指标，不是临床验证结果。

## 7. 分层记忆与用户级 Memory Evolution

记忆分为四层：

```text
显式画像 → 表达偏好 → 当前轮工作记忆 → 近期健康事件 / 历史记录
```

`HealthEvent` 使用 `candidate / possible / active / resolved / archived` 状态机。用户连续描述近期咳嗽、发热等症状时形成近期事件；用户明确表示“好了/恢复了”后标记为 `resolved`；历史记录保留，但不默认注入无关回答。家人代问、模型生成文本和 Prompt 指令不会直接写入健康记忆。

新增用户级 `Memory Evolution`：

```text
Conversation Trace → Shadow Memory Agent
  → 安全与冲突检查 → MemoryCandidate / HealthEvent
  → Markdown Projection
```

- 用户明确表达低风险偏好时立即提取，例如“以后说简单一点”“先给结论”；
- 普通对话每 10 条用户消息触发一次记忆维护；
- 数据库中的 `MemoryCandidate` 和 `HealthEvent` 是事实源；
- 生成 Hermes 风格的 `profile.md`、`preferences.md`、`active_health_context.md` 和 `health_history.md`；
- Markdown 只是可审计投影，不能成为系统指令；
- `POST /api/v1/users/me/memory/reset` 只清除投影，不删除数据库记录和审计证据。

偏好只能改变表达方式，不能覆盖急症规则、用药边界、工具权限或人工转接策略。当前 Shadow Memory Agent 采用确定性提取器，避免 LLM 直接把健康推断写入长期档案。

## 8. 工程化自进化

系统级自进化与用户记忆演化分开：

```text
ExperienceRecord → FailureCase → SkillProposal
→ 冻结评测集 → baseline/candidate 对照
→ 安全与质量门禁 → 人工审核 → SkillDeployment
```

自进化只生成候选 Skill、Prompt 或评测案例，不在线修改模型权重、核心 Prompt、急症规则或工具权限。候选必须经过脱敏、回归评测、安全门禁和人工审核后才能发布，并支持 `deploy / rollback / reset`。

## 9. 可观测性与端到端评测

OpenTelemetry / OpenInference / Jaeger 记录节点、模型、工具、运行预算、延迟和验证状态；默认不记录医疗原文和图片。

在 100 条合成回归样本上，将 28 条在线补测结果与 72 条扩展预算补测结果按 case ID 合并：

| 指标 | 结果 |
|---|---:|
| 任务完成率 | 97.0% |
| 路由准确率 | 80.0% |
| 平均延迟 | 19.9 秒 |
| P95 延迟 | 49.5 秒 |
| LLM-as-Judge 协议遵循率 | 97/100，97.0% |
| Planner 计划质量均分 | 0.849 / 1 |
| 回答完整性均分 | 0.829 / 1 |

Judge 使用 JSON Schema、few-shot 示例、字段校验和失败重试，当前尚未经过人工校准；上述分数是离线工程诊断，不是临床质量结论。引用忠实度和过度拦截率尚未形成可比的完整数据集，因此不将缺失值解释为 0。

主要报告：

- `backend/evals/reports/e2e_eval_merged_100_judge_v2.json`
- `backend/evals/reports/e2e_eval_timeout_retry_judge_v2.json`
- `backend/evals/reports/e2e_eval_28_judge_v3.json`

## 10. 面试演示主线

用户提出“近三个月血压升高，想了解降压药，还想知道医保能否报销”。Supervisor 识别 `mixed` 请求，拆分趋势分析、药品科普和医保事务任务；无依赖任务并行执行，Agentic RAG 按证据缺口多轮检索，Verifier 检查证据和安全边界，最终按风险、用药边界和医保政策聚合回答。随后可以展示用户偏好投影、近期健康事件、Skill 发布以及 rollback/reset。

## 11. 技术边界

- 系统是医疗辅助信息工具，不替代医生诊断和急诊处置；
- 合成评测指标不代表临床效果；
- 用户记忆和系统自进化均受安全门禁约束；
- 健康事件保留来源、状态和时间，不把模型推断当作诊断事实；
- 记忆重置不等于删除数据库数据，真实删除需走用户数据删除流程。
