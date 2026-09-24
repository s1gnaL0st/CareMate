# Smart Health Assistant 产品说明书

版本：v1.0  
文档状态：基于当前代码实现的产品与技术说明  
产品类型：医疗辅助问诊与健康信息服务系统  
当前在线模型：DeepSeek API  
主要运行方式：FastAPI + LangGraph + Chroma + MySQL + Redis

---

## 1. 产品概述

### 1.1 产品定位

Smart Health Assistant 是一个面向个人用户的医疗辅助问诊系统，提供以下服务：

- 健康咨询与症状信息收集；
- 预问诊与急症风险分流；
- 检验报告和常见指标的辅助解读；
- 药品信息、药物相互作用和用药安全科普；
- 医保余额、缴费、消费、报销及异地就医政策查询；
- 基于中文医学资料的检索增强问答；
- 多轮对话、任务暂停、任务恢复和人工接管；
- Agent 运行轨迹、失败案例和离线策略评测。

系统的定位是“医疗辅助信息服务”，不是诊疗系统。系统不能替代医生、药师或医保经办人员，不能依据对话直接确诊、开具处方或替代急诊处置。

### 1.2 产品目标

产品目标不是单纯生成一段看似自然的回答，而是将健康问题拆解为可审计的任务，交给具备不同工具权限和知识边界的领域 Agent 完成，再由安全控制层进行校验。

核心目标包括：

1. 对健康请求进行安全分流，优先识别急症信号；
2. 对复合问题进行任务拆分，避免单个模型承担所有知识和工具职责；
3. 让领域 Agent 只访问与自身职责相关的工具和上下文；
4. 让回答尽可能基于可追溯的医学、药品和医保资料；
5. 在 Agent 失败时进行有限重试、局部修复或人工接管；
6. 将线上运行经验沉淀为脱敏失败案例，支持离线候选策略评测；
7. 在医疗场景中优先保证安全、可解释、可回放和可审计。

### 1.3 非目标

当前版本不包含以下能力：

- 不进行在线自动修改模型参数；
- 不在线自动重写系统提示词；
- 不自动删除或放宽急症规则；
- 不自动发布候选策略到生产 Planner；
- 不自动安装新工具；
- 不提供临床诊断、处方和个性化剂量决定；
- 不把用户生成内容自动当作医学事实写入知识库。

---

## 2. 用户与使用场景

### 2.1 目标用户

- 需要进行初步健康信息整理的普通用户；
- 想在就医前整理症状、用药和既往史的用户；
- 需要理解常见检验指标含义的用户；
- 需要查询药品基本信息和相互作用风险的用户；
- 需要查询医保业务信息的用户。

### 2.2 典型场景

#### 场景 A：症状预问诊

用户描述症状后，系统首先判断是否存在急症红旗。如果没有明确急症信号，系统由症状 Agent 通过有限多轮追问收集：

- 症状开始时间；
- 严重程度和变化趋势；
- 伴随症状；
- 对呼吸、意识、活动能力的影响；
- 既往病史和当前用药；
- 过敏史和重要风险因素。

输出为风险分级、建议补充的信息、就医建议和必要的证据引用。

#### 场景 B：复合健康请求

例如：

> 最近血压偏高，正在吃一种药，想知道报告怎么看以及医保能不能报销。

Supervisor 会将请求拆分为：

```text
symptom_agent    评估症状和紧急风险
       ↓
pharmacy_agent   检查用药安全

report_agent     解读报告指标        ┐
insurance_agent  查询医保政策        ┘ 可并行
```

症状评估完成后，药品 Agent 才能使用必要的风险上下文；报告解读和医保查询如果没有依赖关系，可以并行执行。

#### 场景 C：检验报告解读

报告 Agent 从用户文本或 OCR 结果中识别指标，调用报告解读技能和医学知识库，解释指标是否超出参考范围、可能代表什么以及何时需要就医。系统不能仅凭单项指标下诊断。

#### 场景 D：用药咨询

药品 Agent 可以查询药品信息、相互作用、OTC 相关知识和药店信息。涉及处方药、儿童、孕妇、老年人、肝肾功能异常或急症表现时，系统提高安全等级并建议咨询医生或药师。

#### 场景 E：医保政策查询

医保 Agent 使用账户数据工具和医保政策检索工具。账户数据与公开政策需要区分：前者是用户业务数据，后者是来源受控的政策知识。

---

## 3. 总体架构

### 3.1 分层架构

```text
┌──────────────────────────────────────────────────────────────┐
│                      用户交互层                              │
│ Web / Mobile / SSE 流式事件 / 卡片化结果 / 暂停与恢复         │
└──────────────────────────────────────────────────────────────┘
                              │
┌──────────────────────────────────────────────────────────────┐
│                      API 与会话层                             │
│ FastAPI / 鉴权 / 限流 / 会话 / ChatRun / AgentRun             │
└──────────────────────────────────────────────────────────────┘
                              │
┌──────────────────────────────────────────────────────────────┐
│                   Supervisor AgentLoop                       │
│ Intent Gate → Planner → Executor → Verifier → Responder       │
│                         ↑                 │                    │
│                         └── 局部修复 / 重规划 ─┘                │
└──────────────────────────────────────────────────────────────┘
                              │
┌──────────────────────────────────────────────────────────────┐
│                      领域 Agent 层                            │
│ symptom / report / pharmacy / insurance / chat                │
│ 独立 Prompt、工具白名单、ReAct 循环、上下文切片、输出信封       │
└──────────────────────────────────────────────────────────────┘
                              │
┌──────────────────────────────────────────────────────────────┐
│                 工具、知识与安全控制层                         │
│ RAG / Chroma / BM25 / MMR / RRF / MCP / 规则 / ToolExecutor    │
└──────────────────────────────────────────────────────────────┘
                              │
┌──────────────────────────────────────────────────────────────┐
│                持久化、可观测性与评估层                         │
│ MySQL / Redis / OTel + Jaeger / 轨迹 / 失败案例 / 离线评测       │
└──────────────────────────────────────────────────────────────┘
```

### 3.2 主流程

```text
用户输入
  ↓
API 校验、鉴权、会话加载
  ↓
Intent Gate
  ├─ 确定性急症规则命中 → Safety Response
  └─ 普通健康请求       → Planner
                              ↓
                         任务 DAG
                              ↓
                         Executor
                              ↓
                  领域 Agent 并行/串行执行
                              ↓
                         Verifier
                  ┌───────────┼───────────┐
                  │           │           │
                pass       partial       unsafe
                  │           │           │
               Responder  局部修复/重规划  安全响应/接管
```

### 3.3 为什么采用主从式结构

医疗场景不适合多个模型自由辩论后投票。自由协商可能造成错误共识、责任不清和额外延迟。因此系统采用：

- Supervisor 统一控制任务边界；
- 领域 Agent 在有限职责内自主选择工具和执行步骤；
- Verifier 独立检查结果；
- 确定性规则拥有安全否决权；
- 达到重试上限后进入 handoff。

这种结构重点体现的是专业化、权限隔离、并行调度、局部故障恢复和可审计性，而不是 Agent 之间的开放式聊天。

---

## 4. Supervisor AgentLoop

核心实现位于 `backend/agents/graph_new.py`，使用 LangGraph 的 `StateGraph` 构建状态图。

### 4.1 共享状态

一个 AgentLoop turn 使用共享状态承载控制信息：

```python
{
    "messages": [...],
    "user_info": {...},
    "intent": "health | mixed | emergency | pure_chat",
    "health_text": "规范化后的用户请求",
    "task_queue": [...],
    "task_results": {...},
    "plan_signature": "...",
    "replan_count": 0,
    "verify_status": "pass | partial | fail | unsafe | exhausted",
    "verify_issues": [...],
    "repair_targets": [...],
    "repair_rounds": 0,
    "repair_history": [...],
    "execution_metrics": {...}
}
```

共享状态不是把所有内部消息无条件暴露给所有 Agent。Executor 会根据任务依赖构造输入切片，Verifier 还会构造独立审计视图。

### 4.2 Intent Gate

Intent Gate 有两层：

1. 先运行确定性急症规则；
2. 未直接命中高危规则时，再调用 DeepSeek 进行结构化意图识别。

意图分类：

- `pure_chat`：没有健康诉求的闲聊；
- `health`：健康相关问题；
- `mixed`：健康诉求与闲聊混合；
- `emergency`：明确出现急救危险信号。

模型输出受到 `IntentDecision` Pydantic Schema 限制：

```json
{
  "intent": "health",
  "normalized_request": "用户真正想解决的问题",
  "confidence": 0.92
}
```

如果模型服务不可用，系统使用保守的关键词兜底，不把服务故障当成安全通过。

### 4.3 Planner / Supervisor

Planner 负责将规范化请求转换成任务计划。每个任务必须包含：

```json
{
  "id": "medicine-check",
  "agent": "pharmacy_agent",
  "objective": "检查当前药品的相互作用和用药风险",
  "input_slice": "用户涉及的药品问题",
  "depends_on": ["symptom-check"]
}
```

Planner 约束：

- 最多 6 个任务；
- 任务 ID 不重复；
- 依赖必须存在且不能形成环；
- 任务输入只包含目标 Agent 需要的信息；
- 不能生成诊断或处方指令；
- 信息不足时可以请求澄清；
- 处方药或个性化用药通常要求先完成症状 Agent；
- 独立任务允许并行。

Planner 的输出先经过 Pydantic 校验，再经过依赖图校验。校验失败时使用保守 fallback plan。

### 4.4 Executor

Executor 是工作流执行器，不将自身称为领域 Agent。它负责：

- 按 DAG 计算依赖波次；
- 并行运行同一波次中无依赖的 Agent；
- 为 Agent 构造上下文切片；
- 使用 `ToolExecutor` 统一执行工具；
- 控制 Agent 级重试和指数退避；
- 持久化任务状态；
- 响应暂停请求；
- 发现目标 Agent 需要修复时进行局部重跑。

Executor 的任务状态包括 `pending`、`running`、`completed`、`failed`、`obsolete` 等。带有数据库 `AgentTask` 记录时，任务通过租约机制领取，进程崩溃后可恢复过期任务。

### 4.5 领域 Agent 结果信封

Executor 不再只保存一段文本，而是生成统一结果信封：

```json
{
  "agent": "pharmacy_agent",
  "status": "completed",
  "summary": "药品之间存在需要关注的相互作用风险",
  "evidence": [
    {
      "tool": "check_drug_interaction",
      "source_type": "drug_interaction",
      "source_ids": ["drug-interaction-001"],
      "source_urls": [],
      "version": "local-snapshot-v1",
      "retrieved_at": "runtime",
      "content_hash": "..."
    }
  ],
  "risk_level": "unknown",
  "uncertainty": [],
  "next_action": "建议咨询药师",
  "tool_calls": ["check_drug_interaction"],
  "evidence_count": 1,
  "attempt_count": 1
}
```

当前统一信封由 Supervisor 侧生成，保留既有 Agent 的自然语言输出以兼容前端。后续可以进一步让各领域 Agent 直接输出强类型领域 Schema。

### 4.6 Verifier

Verifier 是安全和质量门禁，不是普通领域 Agent。它构建隔离审计视图，只接收：

- 原始用户输入；
- 各任务的结构化结果信封；
- 工具调用名称；
- 证据来源元数据；
- Agent 状态和错误信息。

Verifier 不读取领域 Agent 的完整 ReAct 消息和隐藏推理，降低执行者影响审查者的风险。

当前检查内容包括：

- 是否存在任务结果；
- 是否有任务执行失败；
- 是否出现高风险不当表述；
- 是否需要安全响应；
- 是否需要定向修复。

Verifier 输出：

```json
{
  "verify_status": "partial",
  "verify_issues": ["pharmacy_agent 执行失败"],
  "repair_targets": [
    {
      "task_id": "medicine-check",
      "target_agent": "pharmacy_agent",
      "required_action": "重试该领域 Agent 并返回可验证结果",
      "issue_id": "a91f..."
    }
  ]
}
```

### 4.7 局部修复

当 Verifier 指定目标任务时，系统不默认重跑整张图：

1. 根据 `task_id` 找到目标任务；
2. 沿任务 DAG 找到所有依赖该任务的下游任务；
3. 将目标任务和下游任务标记为 stale；
4. 保留不相关且已通过的上游结果；
5. 对目标 Agent 注入 `required_action`；
6. 重新执行受影响子图；
7. 对同一 issue 生成 fingerprint，避免重复触发；
8. 修复轮次达到上限后进入 handoff。

例如：

```text
symptom_agent ──→ pharmacy_agent ──→ responder
report_agent   ──────────────────────┘
```

如果 `pharmacy_agent` 需要补充证据，系统会重跑 `pharmacy_agent` 及其下游，而复用已经通过的 `symptom_agent`、`report_agent` 结果。

### 4.8 Responder

Responder 只基于已通过 Verifier 的结果生成用户可见回答。单一症状 Agent 的问诊轮次可以直接返回，以避免第二次模型改写追问内容；多个领域 Agent 结果则由 Responder 统一组织。

Responder 的基本约束：

- 不补造未出现在证据中的医学事实；
- 不直接诊断或开处方；
- 保留必要的风险提示和免责声明；
- 适配移动端阅读；
- 不能覆盖 emergency veto 或安全结果。

---

## 5. 领域 Agent 设计

### 5.1 symptom_agent

实现入口：`backend/agents/clinic.py`。

职责：

- 多轮症状收集；
- 急症红旗检查；
- 风险分级；
- 就医分诊；
- 在信息不足时继续追问；
- 发现高风险时触发 emergency veto。

当前在线调用使用 DeepSeek API。历史 GRPO 五动作适配层仍保留，用于离线或实验路径，不是当前在线问诊的必需模型。

诊所 Agent 的动作协议为：

```text
ask      向用户追问
check    执行红旗、风险或相互作用检查
lookup   查询患者档案
search   检索医学证据
answer   结束当前问诊轮次
```

动作执行经过适配层，形成 `tool_call → result → reasoning` 闭环。动作层有重复追问检查、参数校验、引用过滤和问题轮次上限。

### 5.2 report_agent

实现入口：`backend/agents/report.py`。

report_agent 使用独立系统提示和报告相关工具，通过 ReAct 子 Agent 运行。主要流程：

1. 提取用户提到的指标和数值；
2. 调用 `lab_interpreter` 技能进行结构化指标判断；
3. 从医学知识库检索参考范围和解释资料；
4. 对正常和异常项进行分层说明；
5. 给出复查和就医建议；
6. 添加“不能替代面诊”的说明。

它不能根据单项报告直接做诊断或预测长期趋势。

### 5.3 pharmacy_agent

实现入口：`backend/agents/pharmacy.py`。

pharmacy_agent 使用独立工具白名单和 ReAct 循环，工具包括：

- `search_drug_info`：查询药品信息；
- `check_drug_interaction`：检查药物相互作用；
- `find_nearby_pharmacy`：查找药店；
- `get_otc_recommendation`：检索 OTC 相关建议；
- 受控加载药品相关技能。

药品工具优先使用 MCP 或本地实现，工具结果会携带知识源名称和版本摘要。处方药、特殊人群和严重症状会提高安全要求。

### 5.4 insurance_agent

实现入口：`backend/agents/insurance.py`。

insurance_agent 使用独立 ReAct 子 Agent 和数据库/政策工具，处理：

- 医保余额；
- 缴费记录；
- 消费明细；
- 异地就医信息；
- 医保政策检索。

账户工具必须经过用户身份和权限控制。政策检索结果应携带来源、版本和更新时间，不应将模型猜测当成当前政策。

### 5.5 chat_agent

chat_agent 处理非医疗闲聊和一般表达，不能因为用户使用健康词语就越权承担专业医疗任务。

### 5.6 领域 Agent 与 Skill 的关系

Skill 是能力模块或工具，Agent 是具有目标、上下文、工具选择和停止条件的任务执行者。例如：

```text
pharmacy_agent
  ├── search_drug_info tool
  ├── check_drug_interaction tool
  ├── medication_calculator skill
  └── RAG retrieval capability
```

因此 Executor 和 Verifier 是工作流控制基础设施；真正的领域 Agent 是 symptom、report、pharmacy、insurance 和 chat 节点。

---

## 6. 工具执行与权限控制

### 6.1 ToolExecutor

`ToolExecutor` 是所有领域工具的统一运行边界，负责：

- 工具白名单校验；
- 动态 Skill 白名单校验；
- 超时控制；
- 有限重试；
- 工具调用日志；
- 用户、会话、AgentRun、Task ID 上下文注入；
- 将错误转换为受控的 JSON 错误结果。

领域 Agent 不能直接越过 ToolExecutor 调用任意工具。

### 6.2 工具分类

```text
医疗安全工具：emergency_triage、risk_assessor
症状工具：红旗检查、患者档案查询
报告工具：lab_interpreter
药品工具：药品查询、相互作用、OTC 检索
医保工具：余额、缴费、消费、政策查询
检索工具：RAG、MCP 只读知识工具
```

### 6.3 MCP

MCP 用于接入只读外部知识工具，例如药品或文献查询。外部工具不可用时，系统可以回退到本地实现，但回退结果必须保持安全边界和来源标记。

---

## 7. 医疗安全设计

### 7.1 安全控制层

系统使用三类信号共同控制：

```text
模型决策
  + 确定性急症规则
  + 工具结果
  + Verifier 门禁
  + emergency veto
```

模型不能覆盖确定性急症规则。出现明确急症信号时，系统优先输出立即急救或急诊建议。

### 7.2 安全规则

当前安全 Harness 包含 23 条急症识别规则，覆盖呼吸、意识、胸痛、出血、卒中样表现等风险方向，并支持否定表达和风险分级。

安全规则需要满足：

- 规则优先于自然语言生成；
- 规则结果可解释；
- 规则不能被候选策略删除或放宽；
- 高风险结果不进入普通 Responder 改写链路；
- 高风险场景可直接 handoff。

### 7.3 医疗越权约束

系统禁止：

- 直接声称用户确诊某疾病；
- 对处方药做无监督个性化开药决定；
- 给出未经核验的剂量调整指令；
- 使用“保证治愈”等绝对化表述；
- 在缺少来源时伪造引用；
- 将用户个人偏好当作医学事实。

### 7.4 安全评测

当前有 48 条人工审核合成安全案例，评测结果：

- Precision：96.3%；
- Recall：86.7%；
- F1：91.2%；
- 风险分级准确率：89.6%。

上述数据属于初步规则集评测，不代表临床数据集结果。

---

## 8. RAG 医学知识库

### 8.1 数据范围

当前活动知识库包含约 390 个来源：

- 中文临床指南与健康资料；
- 医保目录和政策资料；
- 检验参考范围；
- 药品清洗文本；
- 用户提供并经过来源标记的医学知识卡。

未通过来源审核或被标记为不使用的资料不进入活动索引。

### 8.2 数据处理流程

```text
原始文件 / 用户知识卡
  ↓
格式解析
  ↓
文本清洗
  ↓
来源登记与 SHA-256
  ↓
人工/规则审核
  ↓
YAML provenance front matter
  ↓
父块 / 子块切分
  ↓
Chroma 向量索引 + BM25 索引
```

来源元数据包括：

- 标题；
- 发布者；
- 来源 URL 或本地来源 ID；
- 文档类型；
- 语言和适用地区；
- 发布/更新时间；
- 版本；
- 原始文件摘要；
- 审核状态。

### 8.3 父子块检索

父子块的目标是分离“召回粒度”和“上下文粒度”：

- 子块较小，负责精准匹配；
- 父块保留完整章节语境；
- 命中多个子块时通过 `parent_id` 去重；
- 最终返回父块，避免回答只看到截断句子。

### 8.4 检索链路

```text
用户查询
  ├── BM25 子块召回
  └── Dense 子块召回
          └── Dense 分支 MMR 去重重排
                  ↓
              RRF 融合
                  ↓
          parent_id 聚合去重
                  ↓
          返回父块与来源元数据
```

当前配置：

- BM25 权重：0.3；
- Dense 权重：0.7；
- Dense MMR：`k=4`、`fetch_k=20`；
- RRF 常数：`c=60`；
- 默认向量库：Chroma；
- 默认中文嵌入模型：`BAAI/bge-small-zh-v1.5`。

注意：MMR 是 Dense 分支内部的多样性重排，RRF 是 BM25 与 Dense 两路排名融合，两者不是同一个算法。

### 8.5 检索评测

在 100 条人工审核中文场景查询集上，来源级指标为：

- Recall@1：88%；
- Recall@3：92%；
- Recall@5：96%；
- 平均延迟：约 89.6 ms。

这些指标代表来源定位能力，不等同于答案正确率、引用忠实度或临床有效性。

---

## 9. 会话、任务与持久化

### 9.1 MySQL

MySQL 保存：

- 用户和会话；
- ChatRun；
- AgentRun；
- AgentTask；
- 任务计划和结果；
- 评估数据集与候选；
- 经验记录和失败案例；
- 人工审核决定。

### 9.2 Redis

Redis 用于：

- 会话和检索缓存；
- 分布式锁；
- 限流；
- 短期状态协调。

数据库是 Agent 任务持久化的主要事实来源，Redis 不替代任务审计记录。

### 9.3 SSE 与事件回放

FastAPI 通过 SSE 推送：

- 节点开始/结束；
- 工具调用开始/结束；
- 文本增量；
- 卡片化结果；
- 暂停；
- 错误；
- 完成。

事件写入 `AgentRunEvent`，每个事件有递增序号。客户端可以使用 `Last-Event-ID` 或事件游标重新获取断线期间的事件。

### 9.4 暂停与恢复

用户可以暂停 AgentLoop。系统在任务边界检查暂停请求：

- 已完成任务不重复执行；
- 运行中的任务释放租约；
- 恢复时重新领取未完成任务；
- 任务状态持久化，避免进程重启导致整轮丢失。

---

## 10. 轨迹采集与可观测性

### 10.1 轨迹采集

系统监听 LangChain/LangGraph 事件：

- `on_chain_start/end`：节点执行；
- `on_tool_start/end`：工具调用；
- `on_chat_model_end`：模型用量；
- Verifier 输出：状态和问题；
- 结束、取消和异常：运行结果。

一次运行结束后，系统生成脱敏 `ExperienceRecord`，记录：

- run ID；
- 脱敏输入；
- 意图；
- 计划签名；
- 节点名称；
- 工具名称；
- 结果摘要；
- Verifier 状态；
- 安全标记；
- 延迟和 token 指标；
- 运行结局。

原始工具参数不直接写入经验记录，避免把敏感输入扩散到离线学习数据。

### 10.2 OpenTelemetry

系统支持 OpenTelemetry 和 Jaeger 链路追踪，用于观察：

- API 请求耗时；
- Supervisor 节点耗时；
- 领域 Agent 调用；
- 工具耗时和失败；
- LLM token 使用；
- RAG 检索耗时；
- 重试和 handoff。

### 10.3 Agent 指标

当前状态和任务记录可支持以下指标：

- 各 Agent 成功率和失败率；
- 工具失败原因分布；
- 平均重试次数；
- 局部修复轮次；
- 修复目标 Agent 分布；
- evidence 数量；
- handoff 率；
- token 和延迟变化；
- 任务并行波次；
- 各 Agent 的输入输出规模。

---

## 11. 评估驱动的受控 Agent 演进

### 11.1 定位

当前自进化不是在线自动改模型，而是一个离线候选策略评估基础设施。

```text
线上轨迹
  ↓
脱敏 ExperienceRecord
  ↓
FailureCase
  ↓
失败模式聚类
  ↓
SkillProposal 候选策略
  ↓
冻结评测集
  ↓
真实 Agent Graph baseline/candidate 回放
  ↓
质量、安全、路由、成本、延迟门禁
  ↓
人工审核
```

### 11.2 策略存在形式

候选策略在数据库中以 `SkillProposal` 形式保存：

```json
{
  "base_skill": "medical_interview",
  "version": "0.1.0",
  "trigger": "发现重复追问或遗漏红旗检查",
  "content": "先完成安全红旗检查，再澄清症状时长和严重程度",
  "source_failure_ids": ["failure-001"],
  "status": "candidate"
}
```

离线运行时转换成 `offline_candidate_guidance` overlay，注入候选 Graph 的 Planner/领域 Agent 输入。它不会自动修改线上 System Prompt。

### 11.3 冻结评测集

冻结集是版本化、不可随意修改的评测快照。每条案例包括：

- 脱敏 prompt；
- case key；
- 期望意图或 Agent；
- 必须出现的安全行为；
- 禁止出现的表述；
- 数据集版本和 fingerprint。

当前本地成对评测使用 9 条案例，分为：

- regression：验证旧能力不退化；
- holdout：检查未直接针对的案例；
- safety_boundary：检查急症和越权边界。

### 11.4 门禁规则

候选策略必须满足：

- candidate score 不低于 baseline；
- candidate pass rate 不低于 baseline；
- 安全违规为 0；
- 路由回归为 0；
- 平均 token 成本增量不超过 20%；
- 平均延迟增量不超过 20%；
- 评测结果必须完整覆盖冻结集。

### 11.5 当前结果和边界

当前 9 条真实 Agent Graph 对照运行结果：

- baseline score：1.0；
- candidate score：1.0；
- baseline pass rate：100%；
- candidate pass rate：100%；
- 安全违规：0；
- 路由回归：0；
- 门禁：通过。

这证明候选策略没有破坏当前评测集上的质量和安全边界，但不能证明策略带来了效果提升。当前人工批准接口仍明确返回 `published: false` 和 `injected_into_planner: false`，没有线上灰度和自动回滚。

---

## 12. API 与前端交互

### 12.1 对话 API

对话接口负责：

- 校验输入长度和消息角色；
- 加载或创建会话；
- 创建 ChatRun/AgentRun；
- 启动 LangGraph；
- 通过 SSE 推送事件；
- 保存 assistant 消息和运行状态。

### 12.2 AgentRun API

AgentRun 相关能力包括：

- 查询运行状态；
- 查询任务 DAG 和每个 Agent 的状态；
- 获取事件游标后的 SSE 事件；
- 请求暂停；
- 恢复暂停或租约过期的运行。

### 12.3 演进 API

主要接口包括：

- 查看脱敏失败案例；
- 查看失败聚类；
- 根据失败案例创建候选策略；
- 创建冻结评测集；
- 排队执行 baseline/candidate 评测；
- 查询逐案例结果；
- 人工批准、拒绝或撤销候选。

演进 API 的批准不等于线上发布，当前发布字段始终为 false。

---

## 13. 配置与部署

### 13.1 核心配置

```text
LLM_PROVIDER=deepseek
DEEPSEEK_MODEL=deepseek-flash
DEEPSEEK_BASE_URL=https://api.deepseek.com
CLINIC_LLM_ENABLED=false
VECTOR_STORE=chroma
CHROMA_PERSIST_DIR=...
```

当前在线诊所问诊关闭本地 GRPO 专用端点，统一使用 DeepSeek。GRPO 适配代码保留用于实验和离线验证。

### 13.2 启动流程

```text
1. 准备 Python 3.12+
2. 安装 backend 依赖
3. 配置 .env，不提交密钥
4. 启动 MySQL 和 Redis
5. 执行数据库迁移
6. 导入并构建 Chroma 索引
7. 启动 FastAPI/Uvicorn
8. 使用 health/readiness 接口检查依赖
```

### 13.3 健康检查

就绪检查至少验证：

- MySQL；
- Redis；
- 向量库目录或远程向量后端；
- 应用基本运行状态。

### 13.4 数据安全要求

- API Key 仅放在本地 `.env` 或密钥管理系统；
- 不上传运行日志、原始下载文件、压缩包和个人数据；
- 轨迹写入前做邮箱、手机号、身份证号和 token 脱敏；
- 医疗知识来源需要保存来源和版本；
- 用户档案和医保账户数据按用户权限隔离。

---

## 14. 评测体系

### 14.1 安全评测

评估急症识别、否定表达、风险分级和 emergency veto。关注 Precision、Recall、F1、分级准确率以及危险漏放行。

### 14.2 RAG 评测

以来源级 Recall@K、MRR、延迟和父块去重结果评估检索。当前 100 条查询集为人工审核中文场景集，不代表生产用户分布。

### 14.3 路由评测

记录 Planner 的实际 Agent 路由、任务完成情况、失败原因和工具错误。当前小规模路由 artifact 受模型服务和工具服务状态影响，不作为广泛泛化能力证明。

### 14.4 演进评测

使用冻结集分别运行 baseline 和 candidate，保留逐案例 score、通过状态、安全违规、路由回归、token 变化和延迟变化。

### 14.5 工程测试

当前测试覆盖：

- Agent 图路由和依赖波次；
- 局部重跑与 issue 去重；
- Verifier 隔离审计视图；
- 工具权限和重试；
- RAG 来源元数据；
- 动作适配协议；
- 演进 API 和离线评测。

---

## 15. 当前版本能力清单

### 已实现

- Supervisor + 领域 ReAct Agent 编排；
- 任务 DAG、并行波次和依赖校验；
- Agent 工具白名单；
- DeepSeek 在线问诊；
- 父子块 Chroma RAG；
- BM25 + Dense MMR + RRF；
- 确定性急症安全规则；
- Verifier 隔离审计视图；
- 结构化结果信封；
- 局部修复、下游 stale 和 issue 去重；
- SSE、任务暂停恢复、MySQL/Redis 持久化；
- 轨迹脱敏和离线演进评估。

### 实验或受控离线能力

- GRPO 五动作模型适配；
- 候选 SkillProposal overlay；
- baseline/candidate 策略对照；
- 本地演进评测 artifact。

### 尚未实现

- 自动生产灰度；
- 自动发布候选策略；
- 自动回滚线上策略；
- 在线自动更新 System Prompt；
- 有临床代表性的外部大规模验证集；
- 成对 baseline/candidate 的稳定效果提升结论。

---

## 16. 后续产品路线

### P1：加强 Agent 契约

- 为症状、报告、药品、医保 Agent 分别定义 Pydantic 输出模型；
- 将 `risk_level`、`uncertainty`、`evidence` 从 Supervisor 推断字段升级为 Agent 原生字段；
- 增加证据完整性和引用一致性检查。

### P2：加强定向修复

- 让 Verifier 输出标准 issue 类型；
- 为每类 issue 配置对应 Agent 和修复工具；
- 支持仅重跑受影响子图；
- 统计修复成功率、节省 token 和节省延迟。

### P3：加强评测数据

- 扩大真实用户脱敏案例；
- 增加专家审核和双人标注；
- 将回归、holdout、安全集分开维护；
- 增加答案引用忠实度和事实正确性指标。

### P4：受控发布

在医疗安全规则不被候选覆盖的前提下，后续可以增加：

- 版本注册；
- 影子流量；
- 小比例灰度；
- 线上指标监控；
- 明确的人工回滚；
- 发布前后审计记录。

自动发布不应成为默认行为，尤其不能允许候选策略自动放宽急症 veto、处方限制或证据要求。

---

## 17. 产品价值总结

Smart Health Assistant 的核心价值不是“调用了多个大模型”，而是将医疗辅助问诊拆成可控制的专业任务：

```text
用户问题
  → 安全分流
  → Supervisor 任务规划
  → 领域 Agent 独立执行
  → 工具和知识库增强
  → 结构化证据交换
  → 独立 Verifier 审计
  → 局部修复或人工接管
  → 可追溯回答
```

这种架构让系统在医疗场景中同时具备：

- 专业职责隔离；
- 工具权限隔离；
- 并行和依赖调度；
- 局部失败恢复；
- 来源可追溯；
- 安全规则优先；
- 运行过程可观测；
- 策略演进可验证。

最终产品定位是：

> 一个以安全和可审计为第一约束、以 Supervisor 编排领域 Agent 为核心、以 RAG 和确定性工具为事实支撑、以离线评测推动受控演进的医疗辅助问诊系统。
