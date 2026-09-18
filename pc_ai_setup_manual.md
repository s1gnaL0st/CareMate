# PC AI 接入手册：本地医疗 Agent 模型 API

本手册用于指导一台 PC 上的 AI 配置助手，把现有医疗 Agent 的 LLM API
切换到本项目的本地模型服务。

## 0. 最终目标

改造前：

```text
PC Agent -> 云端 OpenAI-compatible API
```

改造后：

```text
PC Agent -> 本地模型 API -> Qwen3.5-2B + Medical Agent LoRA
```

本地模型服务提供：

```text
GET  /health
GET  /v1/models
POST /v1/chat/completions
```

必须使用 OpenAI `tools/tool_calls` 协议，不能只把模型返回当普通文本。

## 1. 先确认信息

PC AI 开始修改前，必须先向用户确认：

```text
GPU_HOST
GPU_API_PORT
SERVED_MODEL_NAME
PC 是否能访问 GPU_HOST
现有 Agent 使用 OpenAI SDK 还是 LangChain/LangGraph
Agent 当前工具名称
Agent 是否启用 streaming
```

默认值：

```text
GPU_API_PORT=8000
SERVED_MODEL_NAME=grpo-200
OPENAI_API_KEY=dummy
```

推荐使用 SSH 隧道，而不是直接把无认证服务暴露到公网：

```bash
ssh -N -L 8000:127.0.0.1:8000 USER@GPU_HOST
```

SSH 隧道建立后，PC 上的 `base_url` 使用：

```text
http://127.0.0.1:8000/v1
```

## 2. 在 GPU 机器启动模型服务

当前推荐模型：

```text
runs/grpo-encounter-v1/checkpoint-200
```

启动命令：

```bash
cd /data/sjs/chendian/MedGPT/medagent-rl

CUDA_VISIBLE_DEVICES=0 \
/data/sjs/anaconda3/envs/medgpt/bin/python \
scripts/serve_openai_compat.py \
  --model /data/sjs/chendian/MedGPT/MedicalGPT/outputs-sft-qwen35-2b-v1-merged \
  --adapter runs/grpo-encounter-v1/checkpoint-200 \
  --served-model-name grpo-200 \
  --host 0.0.0.0 \
  --port 8000
```

如果需要后台运行：

```bash
tmux new-session -d -s medical-agent-api \
  'cd /data/sjs/chendian/MedGPT/medagent-rl; \
   CUDA_VISIBLE_DEVICES=0 \
   /data/sjs/anaconda3/envs/medgpt/bin/python \
   scripts/serve_openai_compat.py \
     --model /data/sjs/chendian/MedGPT/MedicalGPT/outputs-sft-qwen35-2b-v1-merged \
     --adapter runs/grpo-encounter-v1/checkpoint-200 \
     --served-model-name grpo-200 \
     --host 0.0.0.0 \
     --port 8000 \
     > runs/pc-agent-api.log 2>&1'
```

检查：

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/models
```

预期：

```json
{"status":"ok","model":"grpo-200"}
```

```json
{
  "object": "list",
  "data": [
    {"id": "grpo-200", "object": "model", "owned_by": "local"}
  ]
}
```

## 3. 工具名称和参数必须严格统一

模型训练使用的工具动作空间只有：

```text
ask
check
lookup
search
answer
```

不要在 PC Agent 中改成别的名字。不要把 `answer` 改成 `submit`，除非同时
增加兼容层。

完整工具 schema：

```python
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "ask",
            "description": "向患者提出追问。",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "一个需要患者回答的医学问题。",
                    }
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check",
            "description": "查询已有检查结果或医学指标。",
            "parameters": {
                "type": "object",
                "properties": {
                    "item": {
                        "type": "string",
                        "description": "需要查询的检查或指标名称。",
                    }
                },
                "required": ["item"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "查询医学知识库。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "医学知识库查询词。",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "搜索外部医学资源。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "外部资源搜索词。",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "answer",
            "description": "给出最终回答或诊断，并结束问诊。",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "最终回答内容。",
                    }
                },
                "required": ["content"],
            },
        },
    },
]
```

## 4. 使用 OpenAI Python SDK

PC Agent 的模型客户端改为：

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="dummy",
)

response = client.chat.completions.create(
    model="grpo-200",
    messages=messages,
    tools=TOOLS,
    tool_choice="auto",
    temperature=0,
)
```

不要删除 `tools` 参数。如果不传 `tools`，模型就不会按工具调用模式工作。

## 5. 使用 LangChain / LangGraph

```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="grpo-200",
    base_url="http://127.0.0.1:8000/v1",
    api_key="dummy",
    temperature=0,
)

model_with_tools = model.bind_tools(TOOLS)
```

如果 Agent 原来通过环境变量配置：

```bash
OPENAI_BASE_URL=http://127.0.0.1:8000/v1
OPENAI_API_KEY=dummy
OPENAI_MODEL=grpo-200
```

PC AI 必须把项目内所有实际使用的模型配置都统一修改，不能只改 `.env.example`
或文档。

## 6. 正确响应形态

模型调用工具时返回：

```json
{
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "",
        "tool_calls": [
          {
            "id": "call_xxx",
            "type": "function",
            "function": {
              "name": "ask",
              "arguments": "{\"question\": \"症状持续多久了？\"}"
            }
          }
        ]
      },
      "finish_reason": "tool_calls"
    }
  ]
}
```

注意：

- `arguments` 是 JSON 字符串，不是 Python dict。
- 消费前必须 `json.loads(tool_call.function.arguments)`。
- `finish_reason` 是 `tool_calls`。
- 不要把 `tool_calls` 丢弃后只读取 `message.content`。

模型给出最终回答时：

```json
{
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "分诊：尽快就医\n建议：...\n证据：ev_...",
        "tool_calls": null
      },
      "finish_reason": "stop"
    }
  ]
}
```

## 7. Agent 工具循环

正确循环：

```python
response = client.chat.completions.create(
    model="grpo-200",
    messages=messages,
    tools=TOOLS,
    tool_choice="auto",
    temperature=0,
)

message = response.choices[0].message
messages.append(message.model_dump(exclude_none=True))

if message.tool_calls:
    for tool_call in message.tool_calls:
        name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)

        result = execute_tool(name, arguments)

        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result, ensure_ascii=False),
            }
        )
else:
    final_answer = message.content
```

下一轮继续把完整 `messages` 发回模型。

不要自己根据工具名伪造工具结果。必须让 PC Agent 原有工具执行器执行
`ask/check/lookup/search/answer`。

## 8. Streaming

服务支持：

```python
stream = client.chat.completions.create(
    model="grpo-200",
    messages=messages,
    tools=TOOLS,
    tool_choice="auto",
    temperature=0,
    stream=True,
)

for chunk in stream:
    delta = chunk.choices[0].delta
    print(delta)
```

如果现有 Agent 的流式 tool-call 聚合器不完整，先设置：

```python
stream=False
```

功能稳定后再开启流式。

## 9. 直接用 curl 验收

检查模型列表：

```bash
curl http://127.0.0.1:8000/v1/models
```

检查工具调用：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "grpo-200",
    "temperature": 0,
    "max_tokens": 128,
    "messages": [
      {"role": "user", "content": "我最近头晕。"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "ask",
          "description": "向患者提出追问。",
          "parameters": {
            "type": "object",
            "properties": {
              "question": {"type": "string"}
            },
            "required": ["question"]
          }
        }
      }
    ]
  }'
```

验收通过标准：

```text
HTTP 200
finish_reason == tool_calls
tool_calls[0].function.name == ask
arguments 可被 json.loads 解析
```

## 10. PC AI 修改任务清单

PC AI 应完成：

1. 扫描项目，找出所有 LLM client、base URL、API key 和 model 配置。
2. 把模型配置改为 `http://127.0.0.1:8000/v1`、`dummy`、`grpo-200`。
3. 保留原 Agent 的 LangGraph 流程。
4. 确认发送请求时包含完整 `tools=TOOLS`。
5. 确认工具 schema 名称和参数与本手册完全一致。
6. 确认 tool loop 会回传 `role="tool"` 和 `tool_call_id`。
7. 跑一次 `ask` smoke。
8. 跑一次完整两轮手工会话。
9. 报告修改文件、启动命令、验证输出和剩余问题。

PC AI 不要做：

1. 不要改 GPU 机器上的训练文件。
2. 不要重命名五个工具。
3. 不要把模型输出的 JSON schema 当工具结果。
4. 不要在模型服务里重复执行 Agent 自己的业务逻辑。
5. 不要把无认证 API 直接暴露到公网。
6. 不要删除原有云端 API fallback，先保留可回滚配置。

## 11. 常见问题

### Connection refused

检查：

```bash
curl http://GPU_HOST:8000/health
```

可能需要：

- 确认模型服务已启动。
- 确认 `--host 0.0.0.0`。
- 检查防火墙。
- 优先使用 SSH 隧道。

### 404 Not Found

`base_url` 必须包含 `/v1`：

```text
http://127.0.0.1:8000/v1
```

不是：

```text
http://127.0.0.1:8000
```

### model not found

使用的名称必须是：

```text
grpo-200
```

不是本地目录名，也不是 `Qwen3.5-2B`。

### 没有 tool_calls

检查：

- 请求是否传入 `tools`
- Agent 是否调用 `bind_tools`
- 工具名称是否完全匹配
- 是否把 `stream=False` 作为第一步
- 是否错误覆盖了 system prompt

### CUDA OOM

确保同一个 GPU 上没有其他模型服务：

```bash
nvidia-smi
tmux ls
```

停止旧服务：

```bash
tmux kill-session -t medical-agent-api
```

### 工具参数 JSON 解析失败

`function.arguments` 是字符串：

```python
arguments = json.loads(tool_call.function.arguments)
```

不要直接当 dict 使用。

## 12. 模型切换

GRPO-200：

```text
--adapter runs/grpo-encounter-v1/checkpoint-200
--served-model-name grpo-200
```

Cold-start baseline：

```text
--adapter runs/coldstart-toolcall
--served-model-name coldstart
```

PC 侧必须同步修改 `model`，不能服务端叫 `coldstart`，PC 仍请求
`grpo-200`。

## 13. 回滚

接入前先记录原始配置：

```text
LLM_BASE_URL
LLM_API_KEY
LLM_MODEL
```

回滚时恢复这三个值，并重新启动 PC Agent。

## 14. 给 PC AI 的执行提示词

可以把下面这段直接交给 PC AI：

```text
阅读 pc_ai_setup_manual.md。
先扫描并定位我的 Agent 中所有 LLM API 配置、tools 定义和 tool loop。
只修改 PC Agent 项目，不修改 GPU 机器训练代码。
将 OpenAI-compatible 配置改为：
base_url=http://127.0.0.1:8000/v1
api_key=dummy
model=grpo-200
保留 LangGraph 原有流程。
确保请求始终传 tools，工具名严格使用 ask/check/lookup/search/answer。
确保消费 finish_reason=tool_calls、tool_call_id 和 JSON arguments。
先使用 stream=False。
完成后运行 health、models、ask tool call 和完整两轮回传工具结果验证。
列出修改过的文件、验证命令、实际输出和任何未解决问题。
```

