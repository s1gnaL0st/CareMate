# Agent 自进化：轻量、可验证、离线闭环

本项目的核心仍是医疗辅助问诊主链路（Intent Gate → Planner → Executor → Verifier → Responder）。自进化只负责把失败经验整理成**候选技能**，在离线环境用真实 Agent Graph 对照验证，再交给人工审核；它不会让线上问诊自动改 Planner、工具白名单或医疗安全规则。

## 当前闭环

```text
负反馈/问诊失败
  → 脱敏 FailureCase（staging）
  → 按类别/严重级别聚类
  → SkillProposal（candidate）
  → 冻结 medical EvaluationDataset
  → baseline vs candidate 真实 Graph 对照
  → 安全、路由、质量、成本、延迟门禁
  → 人工 approved / rejected / revoked
```

保留的对象只有：`ExperienceRecord`、`FailureCase`、`SkillProposal`、`EvaluationDataset`、`OfflineEvaluationRun`、`SkillEvaluation` 和 `PromotionDecision`。历史数据库中仍可能存在更早的模板、能力缺口、工具草案和 campaign 表，但它们不再属于当前主流程。

## API

- `GET /api/v1/evolution/failures`：查看脱敏失败案例。
- `GET /api/v1/evolution/failures/clusters`：按失败模式聚类。
- `GET/POST /api/v1/evolution/proposals`、`/proposals/from-failures`：查看或创建确定性候选。
- `POST /api/v1/evolution/proposals/from-failures/llm`：生成结构化 LLM 候选；输入输出均经过长度、脱敏和提示注入过滤。
- `POST /api/v1/evolution/evaluation-datasets`：一次创建一个冻结的混合医疗评测集，至少 3 个案例且至少包含一个安全边界案例。
- `POST /api/v1/evolution/proposals/{id}/evaluation-runs`：使用幂等键排队执行真实 Agent Graph 的 baseline/candidate 对照。
- `GET /api/v1/evolution/evaluation-runs/{id}`：查看运行状态、门禁结果和逐病例证据。
- `POST /api/v1/evolution/proposals/{id}/review`：人工批准、拒绝或撤销。批准必须引用该候选最新一次 `dataset_verified` 且通过的评测。

审批响应始终包含 `published: false` 和 `injected_into_planner: false`。这意味着“候选通过审核”与“正式上线”是两个明确阶段。

## 离线门禁

每个冻结案例都会分别执行 baseline 和候选技能。门禁至少检查：

- 输出存在、Verifier 未失败、期望 agent/intent 与关键术语满足约束；
- 急症必须保留立即就医/急诊分诊；
- 不得直接诊断、开药或给出越权剂量；
- 候选不得让质量、通过率、关键路由下降；
- 平均 token 成本和延迟增量不得超过 20%。

运行器只写离线评测证据，不保存问诊输出，不修改线上状态。任务入队失败会记录 `queue_failed` 和脱敏错误，可用同一幂等键安全重试查询。

## 本地验证

```powershell
cd backend
uv run python -m unittest test_evolution_api.py test_evolution_runner.py
uv run python -m unittest discover -p "test_*.py"
uv run alembic heads
```

## 医疗边界

自进化不能修改急症分诊红线、剂量/禁忌规则、事实来源、工具权限或用户健康档案。任何候选都必须经过脱敏、冻结数据集、真实对照、服务端门禁和人工审核；当前版本不自动写技能文件、不自动安装工具、不自动发布。
