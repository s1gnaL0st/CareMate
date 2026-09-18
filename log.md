# 项目更新日志

## 2026-09-10：继续完成 P1 后端能力

### 本次更新

- 确认并同步报告分析已通过 Arq 在后台 Worker 执行，API 不再阻塞等待 OCR/报告解析。
- 报告分析失败时最多执行 3 次，使用指数退避；最终失败持久化为 `dead_letter`，同时保留错误信息和 `attempt_count`。
- 新增 OCR 页面/文本块结构归一化，统一输出 `page_number`、`text`、`blocks`、`type`、`bbox` 和 `confidence` 字段。
- 同步 `.env.example` 中的 ToolExecutor 配置示例。
- 将已经存在的文件元数据、用户数据删除、LLM 可靠性和结构化输出 fallback 能力更新为已完成状态。
- 将尚未实现的 OCR/报告任务主动取消单独保留为 TODO。
- 新增文档解析结构化输出测试。

### 验证

```text
test_document_parser.py：通过
```

## 2026-09-09：增加 ToolExecutor / middleware 工具治理层

### 本次更新

- 新增 `backend/tool_executor.py`，为 LangChain 工具提供统一执行边界。
- 接入 `clinic`、`report`、`pharmacy`、`insurance`、`advisor` 五个领域 Agent。
- 保留原有工具名称、Pydantic 参数 Schema 和成功返回格式，兼容现有 ReAct Agent、SSE 事件和前端卡片。
- 增加工具白名单，限制 Agent 只能调用自身允许的工具。
- 限制 `load_skill` 的动态 Skill 调用范围，避免跨领域越权。
- 增加工具超时、有限重试和指数退避。
- 增加统一错误 JSON：超时、权限拒绝和工具异常分别返回标准错误类型。
- 增加 `before_tool`、`after_tool`、`on_tool_error` middleware Hook。
- 默认启用安全日志，只记录工具名、Agent、Run/Task 标识和耗时，不记录医疗输入输出。
- 新增 `ToolExecutionContext`，支持传递 `user_id`、`conversation_id`、`agent_run_id`、`task_id` 和 `request_id`。
- 新增 `backend/test_tool_executor.py`，覆盖 Schema 保持、Hook、重试、超时和 Skill 权限校验。
- 更新 `todo.md` 和 `backend/docs/agent_loop_task.md`。

### 配置项

```env
# 工具单次调用超时时间，默认 30 秒
TOOL_TIMEOUT_SECONDS=30

# 工具失败后的最大重试次数，默认 1 次
TOOL_MAX_RETRIES=1
```

对应配置位于 `backend/config.py`：

- `tool_timeout_seconds`
- `tool_max_retries`

### 当前调用链

```text
ReAct Agent
  → ToolExecutor 包装器
  → 权限校验
  → before middleware
  → 超时 / 重试
  → 原始 Tool 或 Skill
  → after / error middleware
  → ToolMessage
```

### 验证结果

```text
uv run python -m unittest \
  test_llm.py test_vision.py test_main_config.py test_observability.py \
  test_evals.py test_graph_new.py test_tool_executor.py

Ran 51 tests
OK
```

### 后续可选增强

- 将工具调用审计写入数据库。
- 增加工具级幂等键，尤其是带外部副作用的工具。
- 增加按工具类型配置重试策略，区分只读工具和写操作工具。
- 将工具调用指标接入 OpenTelemetry/Prometheus。
- 对高风险医疗工具增加人工确认流程。
