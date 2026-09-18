# Supervisor AgentLoop 与 Task 使用说明

本文说明 `backend/agents/graph_new.py` 的改良架构，重点介绍 Supervisor
AgentLoop、动态 Task、任务依赖和 Executor 的执行方式。

## 1. 这张图解决什么问题

这张图是一次用户请求的 Supervisor 工作流。它负责判断请求、生成任务计划、
调度已有的领域 Agent、检查结果，并组织最终回答。

```text
START
  |
  v
intent_gate -- emergency --> safety_response --> END
  |
  | health / mixed / pure_chat
  v
planner
  |-- needs_clarification --> clarification_response --> END
  |-- handoff / exceeded limit --> handoff --> END
  v
executor --> verifier
              |-- pass     --> responder --> END
              |-- partial  --\
              |-- fail      --> planner  (bounded replan)
              |-- unsafe   --> safety_response --> END
              |-- exhausted --> handoff --> END
```

这里的 `intent_gate`、`planner`、`executor`、`verifier` 和 `responder` 是
LangGraph 的静态节点，属于 Supervisor 控制层；`symptom_agent`、
`report_agent`、`pharmacy_agent`、`insurance_agent` 和 `chat_agent` 是领域能力。

`Task` 是运行时生成的数据，不是一个 LangGraph 节点。也就是说，复合问题
需要几个任务、每个任务有什么依赖，是 Planner 在本次请求中决定的；图本身不
需要为每个可能的任务预先注册节点。

## 2. AgentLoop 是什么

AgentLoop 是一次请求在 Supervisor 图中的完整生命周期：

```text
用户消息
  -> 意图与安全判断
  -> 生成结构化计划
  -> 执行计划中的领域任务
  -> 核验结果
  -> 最终回答
```

如果核验发现部分任务失败，流程会回到 Planner 重新规划，再经过 Executor
和 Verifier。这个循环不是无限循环：代码通过 `MAX_REPLANS = 2` 限制自动重
规划次数；计划连续不变或达到上限后，转到 `handoff`，建议人工或医生继续处
理。

一次 AgentLoop 使用一个共享 State。主要字段如下：

| 字段 | 类型 | 作用 |
| --- | --- | --- |
| `messages` | `list[BaseMessage]` | 对话上下文，通常保留最近消息 |
| `user_info` | `dict[str, Any]` | 年龄、病史、地区等用户信息 |
| `requested_mode` | `str` | 前端请求的模式，作为 Planner 的参考信息 |
| `intent` | `IntentName` | `emergency`、`health`、`mixed` 或 `pure_chat` |
| `health_text` | `str` | Intent Gate 归一化后的请求 |
| `task_queue` | `list[PlannedTask]` | 本轮待执行任务 |
| `task_results` | `dict[str, TaskResult]` | 以 task id 为键保存执行结果 |
| `verify_status` | `VerifyStatus` | 核验状态 |
| `replan_count` | `int` | 已经发生的重规划次数 |
| `final_response` | `str` | 面向用户的最终文本 |

API 入口会把请求转换成初始 State，然后调用 `master_app`。应用入口已经从
旧图切换到：

```python
from agents.graph_new import master_app

initial_state = {
    "messages": [HumanMessage(content="我最近血压升高，还想知道用药是否安全")],
    "user_info": {"age": 45, "medical_history": "高血压"},
    "requested_mode": "general",
}

result = await master_app.ainvoke(initial_state)
```

生产 API 使用 `astream_events(..., version="v2")` 流式运行，但领域 Agent 的中
间回答是内部证据，不会逐条作为用户回答输出；前端主要接收最终的
`responder`、`safety_response`、`clarification_response` 或 `handoff` 结果。

## 3. Task 的数据结构

代码中的任务模型是 Pydantic `PlannedTask`：

```python
class PlannedTask(BaseModel):
    id: str
    agent: AgentName
    objective: str
    input_slice: str
    depends_on: list[str] = []
```

字段含义：

- `id`：当前计划内唯一的任务 ID。依赖关系通过它引用任务。
- `agent`：执行者的逻辑名称，只能使用受限的 `AgentName` 枚举。
- `objective`：任务目标，说明这个任务要解决什么问题。
- `input_slice`：给执行 Agent 的最小必要输入。它不应包含无关的整段对话。
- `depends_on`：前置任务 ID 列表。为空表示可以在第一波执行。

一个具体 Task 可以写成：

```python
PlannedTask(
    id="symptom_risk",
    agent="symptom_agent",
    objective="评估持续胸闷和气短的风险并给出分诊建议",
    input_slice="用户持续两天胸闷、气短，尚未提供血压和体温",
    depends_on=[],
)
```

Planner 输出的不是单个 Task，而是 `Plan`：

```python
class Plan(BaseModel):
    tasks: list[PlannedTask]
    needs_clarification: bool = False
    clarification_question: str | None = None
```

因此，State 中的计划类似于：

```python
{
    "task_queue": [
        PlannedTask(
            id="symptom_risk",
            agent="symptom_agent",
            objective="评估症状风险",
            input_slice="持续胸闷、气短",
        ),
        PlannedTask(
            id="drug_safety",
            agent="pharmacy_agent",
            objective="评估当前用药的安全信息",
            input_slice="用户想知道当前用药是否安全",
            depends_on=["symptom_risk"],
        ),
        PlannedTask(
            id="insurance_policy",
            agent="insurance_agent",
            objective="说明相关医保政策",
            input_slice="用户想了解该情况是否涉及医保报销",
        ),
    ]
}
```

`objective` 是任务的语义契约，`input_slice` 是实际发送给领域 Agent 的输入。
当前 Executor 主要通过 `input_slice` 调用领域 Agent，因此 Planner 应让
`input_slice` 本身足够明确；`objective` 会保存在结果中，并用于计划识别、
核验和后续组织结果。

## 4. Planner 如何产生和约束 Task

正常路径中，Planner 调用结构化 LLM，让模型返回符合 `Plan` schema 的结果，
而不是依赖关键词路由。随后 `_validate_plan()` 执行确定性校验：

1. 计划至少包含一个任务，且最多 `MAX_PLAN_TASKS = 6` 个任务。
2. Task ID 必须唯一。
3. `depends_on` 中的 ID 必须存在，不能依赖自己。
4. 依赖图不能出现环。
5. 缺失的 `input_slice` 会使用归一化请求补齐。
6. `agent` 只能是系统允许的领域 Agent。

LLM 不可用或返回格式不合法时，代码才使用 `_fallback_plan()`。fallback 是
服务降级路径，不是正常的业务规划机制；它包含少量关键词提示，便于模型服务
故障时仍然走一条受控路径。

如果信息不足，Planner 可以返回：

```python
Plan(
    tasks=[],
    needs_clarification=True,
    clarification_question="请补充药品名称、剂量和每天服用次数。",
)
```

这时不会进入 Executor，而是由 `clarification_response` 直接向用户追问。

## 5. Dependency 与执行波次

`depends_on` 表示数据或业务上的前置关系，而不是“调用哪个 LangGraph 节点”。
例如：药物安全分析需要先知道症状风险，可以表达为：

```text
symptom_risk ─────> drug_safety
insurance_policy    （无依赖）
```

Executor 通过 `_dependency_waves()` 将任务转换成拓扑执行波次：

```text
wave 1: symptom_risk, insurance_policy
wave 2: drug_safety
```

规则是：

- 当前没有未完成依赖的任务进入同一波。
- 同一波内的任务用 `asyncio.gather()` 并行执行。
- 下一波必须等待上一波结束。
- 如果无法找到可执行任务，说明计划存在环，执行会失败；正常情况下该问题
  已经在 Planner 校验阶段被拦截。

这相当于一个小型 DAG 调度器：

```text
任务计划 DAG
      |
      v
拓扑排序 + 分波
      |
      v
并行执行独立任务
      |
      v
把结果传给后继任务
```

## 6. Executor 如何执行一个 Task

Executor 的核心步骤如下：

```python
results = dict(state.get("task_results", {}))

for wave in _dependency_waves(state["task_queue"]):
    runnable = []

    for task in wave:
        if task 已经成功完成:
            continue
        if 有依赖任务失败:
            results[task.id] = failed(dependency_failed)
        else:
            runnable.append(task)

    await asyncio.gather(*(run_task(task) for task in runnable))
```

每个可运行 Task 会根据 `agent` 查找领域节点：

```python
_DOMAIN_NODES = {
    "symptom_agent": clinic_node,
    "report_agent": report_node,
    "pharmacy_agent": pharmacy_node,
    "insurance_agent": insurance_node,
    "chat_agent": advisor_node,
}
```

实际调用等价于：

```python
domain_node = _DOMAIN_NODES[task.agent]
domain_state = _agent_input_slice(state, task, results)
domain_result = await domain_node(domain_state)
```

Executor 不会把整个 Supervisor State 原样传给领域 Agent。`_agent_input_slice()`
只构造一个窄输入：

- 当前 Task 的 `input_slice` 作为 HumanMessage；
- `depends_on` 中已经成功完成的任务摘要作为 SystemMessage；
- `user_info` 保留给领域 Agent 使用；
- 清空 `active_agent` 和 `next_agent`，避免领域 Agent 误认为自己负责再次路由。

例如，`drug_safety` 收到的上下文大致是：

```text
上游任务摘要：
symptom_risk -> 已完成 -> 建议尽快就医并监测症状

当前任务输入：
用户想知道当前用药是否安全
```

领域 Agent 返回 State 后，Executor 从其中提取最后一条 `AIMessage`，保存为
`TaskResult`：

```python
{
    "agent": "pharmacy_agent",
    "objective": "评估当前用药的安全信息",
    "status": "completed",
    "text": "请提供药品名称和剂量后再进行相互作用判断。",
}
```

如果 Agent 抛出异常，或者没有产生可用文本，则保存失败结果：

```python
{
    "agent": "pharmacy_agent",
    "objective": "评估当前用药的安全信息",
    "status": "failed",
    "error": "TimeoutError",
}
```

依赖任务失败时，后继任务不会被盲目调用，而是记录：

```python
{
    "status": "failed",
    "error": "dependency_failed",
}
```

## 7. Verifier、重规划与最终回答

Executor 只负责执行，不负责决定最终回答是否安全完整。Verifier 会检查：

- 是否产生了任务结果；
- 是否存在失败任务；
- 领域结果是否触发医疗安全红线。

核验结果的处理方式：

| `verify_status` | 后续动作 |
| --- | --- |
| `pass` | 交给 `responder` 汇总回答 |
| `partial` | 将失败信息交给 Planner，尝试重新规划 |
| `fail` | 重新规划，补充或替换失败任务 |
| `unsafe` | 立即进入 `safety_response` |
| `exhausted` | 进入 `handoff`，停止自动尝试 |

`responder` 只读取已执行的任务结果，并调用快速模型生成面向用户的回答。它
不应自行补造医学事实、下最终诊断或开具处方；当风险较高时，应明确建议就医。

## 8. 一个完整例子

用户请求：

```text
我最近血压一直升高，想知道正在吃的药是否安全，也想问医保能不能报销。
```

Planner 可以生成：

```text
T1 symptom_risk
  agent: symptom_agent
  depends_on: []

T2 drug_safety
  agent: pharmacy_agent
  depends_on: [T1]

T3 insurance_policy
  agent: insurance_agent
  depends_on: []
```

执行时：

```text
第一波：T1、T3 并行
第二波：T1 成功后执行 T2
核验：检查 T1、T2、T3 是否成功及是否触碰安全红线
回答：Responder 汇总三个任务的已验证结果
```

如果 T1 失败，T2 会被标记为 `dependency_failed`，但 T3 仍可独立完成。Verifier
会把失败信息交给 Planner；如果重规划仍然失败或计划没有变化，系统会转人工，
而不是无限重复调用 Agent。

## 9. 开发时的边界和注意事项

### Supervisor 节点与领域 Agent

Supervisor 节点控制流程和状态；领域 Agent 负责具体领域分析。新增领域能力时，
通常需要：

1. 实现一个符合现有 Agent 接口的异步领域节点；
2. 给 `AgentName` 增加允许值；
3. 在 `_DOMAIN_NODES` 注册映射；
4. 更新 Planner 的结构化提示和确定性测试。

不应把每个动态 Task 直接注册成 `StateGraph` 节点，因为 Task 数量和依赖图是在
运行时才确定的。

### 输入隔离

`input_slice` 应尽量只包含该领域所需信息。上游结果只通过显式的
`depends_on` 传递，避免一个 Agent 看到不相关任务的全部结果，也减少隐私泄漏
和上下文膨胀风险。

### 工具调用与 ToolExecutor

领域 Agent 仍使用 LangGraph 的 `create_react_agent` 和内部 ToolNode，但传给
ReAct Agent 的工具会先经过 `ToolExecutor` 包装。它保留原工具名和 Pydantic
参数 Schema，因此不会改变模型的 function-calling 契约或前端卡片解析格式。

`ToolExecutor` 统一负责：

- Agent 工具白名单和 `load_skill` 的动态 Skill 白名单；
- 工具超时、有限重试和安全错误响应；
- 注入 `agent_run_id`、`task_id` 等非敏感上下文；
- `before_tool`、`after_tool`、`on_tool_error` middleware Hook；
- 默认只记录工具名、Agent 和耗时，不记录医疗输入输出。

成功结果保持原始 JSON 或文本；只有执行器生成的异常才转换成统一的错误 JSON。
因此 `main.py` 仍可通过 `astream_events` 监听 `on_tool_start/on_tool_end`，而不需要
再写一套工具调度器。当前 ToolExecutor 是进程内执行边界，工具调用审计持久化和
分布式幂等仍属于后续增强项。

### 结果与持久化

当前 `task_queue` 和 `task_results` 是一次 AgentLoop 的运行时 State，主要服务
于本轮调度、核验和回答；它们不是独立的数据库任务队列。若未来需要断点续跑、
异步任务或跨请求恢复，应另外设计持久化任务表和任务状态机，不能仅依赖当前
内存中的 State。

当前版本已经增加数据库级的 AgentLoop 任务账本，见 `agent_runs` 和
`agent_tasks`。它不引入 Redis：数据库保存运行和任务的事实状态，图仍负责
调度。普通无登录调用以及未绑定运行记录的测试仍然使用内存 State。

### 暂停与恢复

暂停接口只设置 `pause_requested`，不会强行中断正在进行的模型调用。Executor
完成当前执行波次后检查该标志，并将 Run 置为 `paused`。恢复接口根据原始
`agent_run_id` 重建计划和已完成结果，直接从 Executor 继续；已完成任务不会
再次调用领域 Agent。

```text
POST /api/v1/agent-runs/{run_id}/pause
GET  /api/v1/agent-runs/{run_id}
POST /api/v1/agent-runs/{run_id}/resume
```

### 崩溃恢复

每个被 Executor 领取的任务都会写入 `running`、`attempt_count`、
`lease_until`。如果进程在任务完成结果落库前崩溃，任务会遗留为过期的
`running`；下一次恢复运行时，Executor 会先把过期任务改回 `pending`，再原子
领取并重试。因此当前语义是“任务边界的至少一次执行”，领域 Agent 或外部工具
若有副作用，仍需要使用任务 ID 做幂等设计。

### 测试

新图的确定性测试位于 `backend/test_graph_new.py`，重点覆盖：

- 图是否成功编译；
- 依赖环是否被拒绝；
- 执行波次是否符合拓扑顺序；
- 依赖摘要是否按任务隔离；
- LLM 结构化计划和 fallback 计划是否可用；
- 急症路径是否绕过普通 LLM 规划。

可运行：

```powershell
cd backend
uv run python -m unittest test_graph_new.py
```

相关实现：[`backend/agents/graph_new.py`](../backend/agents/graph_new.py)
