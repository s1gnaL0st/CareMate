# 现有医疗 Agent 的五动作接入契约

本文档给 PC 上负责修改 Agent 的 AI 使用。目标是只把本地模型接到
`clinic_node`，不能直接替换整个 Agent。

## 1. 接入边界

必须保留：

- 原 Supervisor
- Intent Gate
- EmergencyTriageSkill
- Verifier / Guardrail
- 原有 RAG 和药物工具

只替换：

```text
clinic_node 使用的医疗问诊模型
```

不要全局覆盖：

```text
LLM_PROVIDER
LLM_BASE_URL
LLM_API_KEY
LLM_MODEL
```

因为 Supervisor、Planner、RAG 等节点可能仍需要原模型。正确做法是单独创建：

```python
clinic_model = ChatOpenAI(
    model="grpo-200",
    base_url="http://127.0.0.1:8000/v1",
    api_key="dummy",
    temperature=0,
)
```

其他节点保持原配置。

## 2. 诊室模型只能看到五个工具

不要把项目全部工具和模型五工具混在一起绑定。

错误：

```python
clinic_model.bind_tools(
    all_project_tools
    + ["ask", "check", "lookup", "search", "answer"]
)
```

正确：

```python
clinic_tools = [
    ask_schema,
    check_schema,
    lookup_schema,
    search_schema,
    answer_schema,
]

clinic_model_with_tools = clinic_model.bind_tools(clinic_tools)
```

模型训练时只见过：

```text
ask
check
lookup
search
answer
```

如果绑定项目现有：

```text
search_drug_info
check_drug_interaction
emergency_triage
symptom_scorer
```

模型可能输出错误名称或格式，不属于有效行为。

Supervisor 可以继续使用原有工具；只有诊室节点使用五工具。

## 3. 动作映射

### ask

模型输出：

```json
{
  "name": "ask",
  "arguments": {
    "question": "症状持续多久了？"
  }
}
```

适配：

```text
向患者提出一个明确问题
一次只能问一个问题
保存 question_id / state
防止重复追问
```

不要在同一轮同时问多个问题。

### check

模型输出：

```json
{
  "name": "check",
  "arguments": {
    "item": "药物相互作用：华法林,阿司匹林"
  }
}
```

适配规则：

| item 内容 | 路由 |
|---|---|
| 药物、相互作用 | 原 `check_drug_interaction` |
| 血压、心率、血氧、体温 | 原生命体征规则 |
| 红旗症状 | 原 red flag / emergency rules |
| 已有检验或用户提供的指标 | 项目报告解析器 |
| 无法确定 | 向用户确认，不要编造检查结果 |

不要让模型自己伪造检查结果。

### lookup

模型输出：

```json
{
  "name": "lookup",
  "arguments": {
    "query": "妊娠期腹痛处置"
  }
}
```

适配：

```text
查询项目本地医疗知识库 / RAG
返回真实 chunks 和 citation IDs
```

### search

模型输出：

```json
{
  "name": "search",
  "arguments": {
    "query": "最新指南关键词"
  }
}
```

适配：

```text
走项目的受控外部搜索
必须保留来源和引用
不允许无约束开放搜索
```

### answer

这是终止动作。模型输出：

```json
{
  "name": "answer",
  "arguments": {
    "content": "分诊：尽快就医\n建议：...\n证据：..."
  }
}
```

`answer` 后必须立即结束 tool loop，不能再追加 tool response。

## 4. answer 解析

不要把模型输出整段直接展示给用户。先解析成项目规范：

```python
def parse_answer(content: str):
    triage = None
    summary = content
    evidence_ids = []

    for line in content.splitlines():
        if line.startswith("分诊："):
            raw = line.split("：", 1)[1].strip()
            triage = normalize_triage(raw)
        elif line.startswith("建议："):
            summary = line.split("：", 1)[1].strip()
        elif line.startswith("证据："):
            evidence_ids = [
                item.strip()
                for item in line.split("：", 1)[1]
                .replace("，", ",")
                .split(",")
                if item.strip()
            ]

    if triage is None:
        triage = infer_triage_from_text(content)

    return {
        "triage": triage,
        "summary": summary,
        "model_evidence_ids": evidence_ids,
    }
```

分诊标签必须统一：

```text
急诊       -> emergency
立即急诊   -> emergency
尽快就医   -> urgent
当天就医   -> urgent
门诊       -> routine
自我照护   -> self_care
```

无法解析时：

```text
交给项目 Verifier 或安全规则处理
不要让空 triage 直接结束
```

## 5. 模型证据 ID 不能直接透传

模型训练环境里的证据 ID 是：

```text
ev_uri_care
ev_stroke_red_flags
ev_pregnancy_care
```

这些 ID 只对训练环境有意义。

在现有 Agent 中：

```text
最终 citation 必须来自项目 lookup/search 的真实工具结果
```

处理方式：

1. 保留模型输出的 `model_evidence_ids` 仅用于调试。
2. 最终回答使用项目工具返回的真实 citation IDs。
3. 如果模型引用不存在的 ID，Verifier 应丢弃。

绝对不要让模型生成的 `ev_*` 直接出现在用户界面或引用系统。

## 6. 急诊不能被模型单独决定

执行顺序必须是：

```text
用户输入
-> 原 EmergencyTriageSkill / red flag rules
-> 如果触发，直接 emergency
-> 未触发才进入 clinic_node 模型
```

模型输出：

```text
分诊：门诊
```

不能覆盖已有规则识别出的急诊。

建议在最终输出前再跑一次：

```text
emergency_veto(model_answer, accumulated_red_flags)
```

## 7. 工具循环

正确流程：

```python
messages = initial_messages

for turn in range(8):
    response = clinic_model_with_tools.invoke(messages)
    message = response
    messages.append(message)

    if not message.tool_calls:
        break

    for call in message.tool_calls:
        if call.name == "answer":
            return parse_answer(call.arguments["content"])

        result = execute_clinic_tool(call.name, call.arguments)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result, ensure_ascii=False),
            }
        )
```

必须保留：

```text
tool_call_id
```

不要只把工具结果拼成普通 assistant/user 文本，否则模型的 tool-call 协议会错位。

`ask` 需要单独处理。示例：

```python
if call.name == "ask":
    pending_question_call_id = call.id
    return call.arguments["question"]

result = execute_clinic_tool(call.name, call.arguments)
messages.append(
    {
        "role": "tool",
        "tool_call_id": call.id,
        "name": call.name,
        "content": json.dumps(result, ensure_ascii=False),
    }
)
```

患者回复下一轮追问时，不能只追加新的普通 `user` 消息。必须转换成上一轮
`ask` 的 tool 结果：

```python
messages.append(
    {
        "role": "tool",
        "tool_call_id": pending_question_call_id,
        "name": "ask",
        "content": patient_reply,
    }
)
```

否则模型每轮看到的仍然是“刚提出主诉、没有任何回答”，自然会持续 `ask`。

## 8. 状态和防重复

适配层需要维护：

```text
asked_questions
returned_evidence
tool_call_history
triage_signals
red_flags
turn_count
```

规则：

- 重复追问：拦截并要求换问题。
- 重复 tool call：返回已有结果或强制 answer。
- 超过 8 轮：进入安全 fallback，不再无限循环。
- answer 为空：交给 Verifier，不能让流程直接结束。

## 9. 推荐的 per-node 配置

```yaml
llm:
  supervisor:
    provider: original
  planner:
    provider: original
  clinic:
    provider: custom
    base_url: http://127.0.0.1:8000/v1
    api_key: dummy
    model: grpo-200
```

如果项目目前只有全局环境变量，也必须创建独立 clinic client，不要直接覆盖
全局环境变量。

## 10. 验收测试

先做两轮 smoke：

```text
用户：我最近头晕
模型：ask -> 头晕持续多久了？
用户：大约一周，有时候恶心
模型：ask/check -> 继续问红旗或检查
```

验收条件：

```text
finish_reason == tool_calls
工具名只在 ask/check/lookup/search/answer 中
tool_call_id 原样回传
answer 后循环立即结束
没有模型 ev_* 直接透传
急诊规则优先于模型
```

再跑 20 条 validation：

```text
unknown tool rate
parse failure rate
duplicate question rate
early-answer rate
emergency bypass rate
```

全部稳定后，才考虑让模型参与 Planner 或最终 Responder。

## 11. 旧版模型为什么会一直 ask

旧 Cold-start 数据中，每个病例平均有 2.72 次 `ask`，最多 3 次，然后
进入 `lookup/check/search/answer`。这个模型不是“永远不会 answer”的
模型。

受控回填测试中，`grpo-200` 在以下病例均正常终止：

```text
dental:      ask -> ask -> ask -> lookup -> check -> search -> answer
uri:         lookup -> ask -> ask -> ask -> check -> search -> answer
chest_pain:  ask -> ask -> lookup -> search -> check -> answer
```

如果接入后无限 ask，优先检查：

1. 每一轮是否保留完整 `messages`，还是重新创建会话。
2. `ask` 后的患者回复是否以 `role=tool` 回填。
3. `lookup/check/search` 的结果是否回填并继续调用模型。
4. `answer` 是否被错误地当成普通 tool，继续回填并再次询问。
5. 服务端是否正确把 OpenAI 的 arguments JSON 字符串转换成 Qwen
   chat template 需要的对象。

不要通过删除 tools 或把 `ask` 后的问题改成普通文本解决。这会造成训练和
推理协议错位。
