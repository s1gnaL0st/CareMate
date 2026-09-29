# Agent Runtime 与分层记忆改进

## 目标

本次改造把“Agent 能运行”和“Agent 如何持续变好”拆成清晰边界：LangGraph 继续负责可组合的业务工作流，Agent Runtime 负责一次运行的生命周期、预算、可观测性和恢复入口；Memory 负责受控地提供用户上下文；离线 Evolution 负责从经验和反馈中提出候选改进。

## 运行时边界

`AgentRuntime` 包装现有的 compiled LangGraph，不替换 LangGraph。每次运行拥有 `RuntimeContext`、`RuntimeBudget` 和 `RuntimeSnapshot`，统计步骤、工具调用、输入/输出 token、事件数与耗时，并在超时或预算耗尽时安全停止。SSE 保留既有 `node_*`、`tool_*`、`text`、`card`、`finish` 事件，同时增加 `runtime_start` 与 `runtime_end`。

```text
请求
  -> Runtime Context Builder
  -> Profile / Preference / Recent Health Retrieval
  -> LangGraph Planner -> Executor -> Verifier -> Responder
  -> Health Memory Write Gate
  -> Experience Record -> Offline Evaluation / Evolution
```

## 分层记忆

- **画像记忆**：年龄、职业、专业程度等显式资料，只用于表达适配。
- **偏好记忆**：用户确认过的表达偏好，经过审核后进入 `MemoryCandidate`；不能改变安全规则、诊断边界或工具权限。
- **工作记忆**：当前请求和当前健康问题，只在本轮使用。
- **近期健康事件**：`HealthEvent` 保存一段时间内的用户自述症状和状态；它不是诊断结果。
- **历史记忆**：事件进入 `resolved` 或 `archived` 后保留，用于用户主动查询，不默认注入普通回答。

## HealthEvent 状态机

`candidate -> possible -> active -> stale -> resolved -> archived`

第一版使用确定性关键词和时间规则，避免把一次提问写成确诊。第一人称且带“这几天/最近/一直/还是”等表达可提升为 `active`；弱信号为 `possible`；只有用户明确说“好了/恢复了/不咳了”等才进入 `resolved`。重新出现时应创建新事件，而不是复活旧历史。家人代问、模型生成文本和 Prompt 指令不会写入健康记忆。

## 写入与注入门控

Runtime 前只检索 `candidate/possible/active` 且与本轮问题相关的事件，注入脱敏摘要和置信度。Runtime 后只从用户最新消息经过确定性 extractor，再由 Memory Write Gate 写入事件与证据引用。`resolved` 事件仍保存证据，但不出现在默认 prompt 中；用户明确询问“上次感冒持续多久”时，再走历史查询。

## 自进化与 Agentic RAG

### 用户级 Memory Evolution

在分层记忆之上增加 `Shadow Memory Agent`。它不参与当前回答，也不修改模型权重；主 Agent 图完成后执行受控的用户级记忆维护。用户明确表达的低风险表达偏好立即提取，普通对话每 10 条用户消息触发一次记忆检索与投影刷新。数据库中的 `MemoryCandidate`、`HealthEvent` 是事实源，`data/user_memory/<user_id>/` 下的 `profile.md`、`preferences.md`、`active_health_context.md` 和 `health_history.md` 是 Hermes 风格的可审计 Markdown 投影。

记忆维护流程为：`Conversation Trace → Shadow Extractor → Conflict/Safety Gate → accepted MemoryCandidate/HealthEvent → Markdown Projection`。偏好只能改变表达方式，不能覆盖急症规则、用药边界或工具权限；健康推断不能直接写入长期档案。投影支持独立 reset（`POST /api/v1/users/me/memory/reset`），数据库记录和审计证据保留，后续可扩展按版本回滚。

运行时经验记录包含 Runtime 指标、验证状态、工具轨迹和脱敏结果。离线流程可以聚合失败案例、反馈和检索质量，生成候选 skill、提示或评测集；候选必须经过回归、安全和人工审核后才能发布。后续 Agentic RAG 可在 Runtime 中加入 query decomposition、来源可信度、时间新鲜度和证据一致性 verifier，但不能让检索结果越权覆盖安全策略。

### 发布、回滚与重置

`SkillProposal` 的 `approved` 只代表通过评测和人工审核，不等于线上启用。`SkillDeployment` 保存每个 `base_skill` 的线上版本指针：

- `deploy`：只允许 approved 候选进入 active 指针；
- `rollback`：交换当前版本和上一版本，不删除任何候选、评测或审核记录；
- `reset`：停用线上指针，恢复为没有动态技能注入的基线行为；
- Runtime 只读取 active 且对应 proposal 仍为 approved 的版本。

这样可以在发现安全回退、质量下降或延迟异常时快速恢复，同时保留完整审计轨迹。reset 是能力开关，不是数据删除；用户健康记忆也不应通过技能 reset 被清空。

## 面试 Demo

用户第一天说“这几天一直咳嗽”，系统生成 `active respiratory_symptoms`；第三天问运动建议时检索该事件；第六天说“完全好了”，事件变为 `resolved`；第二十天制定跑步计划时不再注入感冒上下文；用户主动询问病程时再展示历史时间线。

## 隐私与安全

只保存必要的短摘要和证据引用，避免原始医疗文本扩散；推断事件必须标注 `inferred`，用户自述标注 `user_reported`。任何记忆都不能改变急症识别、工具授权、处方边界或人工转接规则。
