# Agent 自进化论文笔记与 Smart Health Assistant 接入建议

> 阅读范围：`C:\Users\jssun\Desktop\论文合集\agent自进化` 中的 6 篇 PDF。\
> 目标：提炼可复用的工程机制，并判断哪些机制适合医疗健康助手。\
> 结论先行：本项目最值得接入的是“失败驱动的技能、模板和记忆演化 + 离线验证发布”闭环，而不是直接在生产环境做在线强化学习或自博弈。

## 1. 总体结论

这 6 篇论文可以归为三条路线：

| 路线 | 论文 | 核心对象 | 对本项目的价值 |
|---|---|---|---|
| Skill evolution | SkillWeaver、AutoSkill、EvoSkill | 可复用的技能/程序/提示模块 | 最高。项目已有 `backend/skills`，接入边界清晰 |
| Memory evolution | MemSkill、AutoSkill | 记忆提取、更新、裁剪策略 | 高。适合用户偏好和医疗上下文，但安全要求最高 |
| Tool/capability evolution | STELLA、Tool-R0 | 工具库、工作流模板、能力边界 | 中高。先做注册、测试和难例生成，不做在线自动安装或 RL |

推荐的目标闭环：

```text
任务执行 -> 记录轨迹和验证问题 -> 聚类失败模式
       -> 生成候选 skill/template/memory skill
       -> 独立回归集评测 -> 通过门槛才晋级
       -> 版本化发布 -> 后续请求检索并注入
```

这个闭环的关键是“候选”和“正式库”分离。模型可以提出改进，但不能凭一次成功请求直接修改生产技能，更不能自动改写医疗安全规则。

## 2. 逐篇笔记

### 2.1 SkillWeaver: Web Agents can Self-Improve by Discovering and Honing Skills

文件：`2504.07079v1.pdf`

**问题。** Web agent 每次从头规划会重复犯相同错误；成功轨迹中包含可抽象、可组合、可复用的操作程序。

**方法。** 论文采用三阶段：

1. `systematic exploration`：探索任务并收集轨迹。
2. `skill practice`：练习技能，使用奖励模型判断效果，并合成可调用 API。
3. `skill honing`：测试、调试和验证技能，修复参数、调用顺序等问题。

技能不只是自然语言总结，还可以成为带参数的 reusable API，并支持组合成更高层技能。

**主要启发。** 成功轨迹本身不是技能；技能必须有明确触发条件、参数约束、成功判定和独立测试。论文也暴露了两个重要失败模式：agent 找不到合适 API、API 参数错误，以及“没有明显异常”但实际任务并未完成的 evaluation loophole。

**项目映射。**

- executor 产出的任务轨迹可作为技能候选的来源。
- verifier 可提供 success/failure 信号，但医疗任务不能只依赖 LLM judge。
- `backend/skills/<name>/SKILL.md + skill.py` 已经是天然的技能物化格式。
- `backend/evals/cases.jsonl` 可扩展为 candidate skill 的 holdout 验证集。

**不能直接照搬。** Web 操作成功标准通常比医疗回答简单。医疗技能需要 deterministic safety assertions、工具返回 schema、领域规则和人工审核；“模型觉得成功”不能成为发布条件。

### 2.2 STELLA: Self-Evolving LLM Agent for Biomedical Research

文件：`2507.02004v1.pdf`

**核心结构。** STELLA 将多 agent 分成 Manager、Dev、Critic 和 Tool Creation Agent，并维护两个可演化组件：

- `Template Library`：成功的推理路径、任务计划和工作流模板。
- `Tool Ocean`：可动态扩展的工具、API、数据库集合。

Critic 根据中间结果发现 capability gap，Tool Creation Agent 再搜索、创建、测试和调试工具。随着 test-time trials 增加，系统可以积累经验并提高后续任务表现。

**项目可吸收部分。**

- 保存通过验证的 planner 模板，而不是每次从零生成计划。
- 建立 capability gap registry，记录缺少哪个工具或哪个 schema 导致任务失败。
- 建立候选工具 registry，记录来源、权限、输入输出 schema、测试状态和审核状态。

**医疗边界。** Tool Creation Agent 只能生成候选描述或代码草案。涉及 emergency triage、剂量计算、处方建议和医保规则的工具，必须经过白名单、离线测试和人工审批，不能由在线请求自动启用。

### 2.3 MemSkill: Learning and Evolving Memory Skills for Self-Evolving Agents

文件：`2602.02474v2.pdf`

**问题。** 记忆系统的瓶颈不只在“存不存”，还在于什么时候提取、写什么、如何更新、何时合并或裁剪。把这些逻辑全部硬编码会导致跨任务泛化差。

**方法。** 系统包含三个角色：

- `controller`：选择当前上下文需要的 memory skills。
- `executor`：按所选技能生成结构化记忆更新。
- `designer`：从 hard cases 和失败聚类中修改现有技能或生成新技能。

记忆操作可包括 INSERT、UPDATE，以及删除/裁剪等维护操作；controller 和 skill bank 交替优化。

**项目可落地部分。**

- 把用户偏好、对话摘要、健康上下文的写入策略模块化。
- 使用 hard-case buffer 收集用户纠正、verifier 失败、矛盾信息和评测失败。
- 对每条记忆记录 provenance、confidence、timestamp、source 和 expiry。
- 支持用户查看、纠正和删除自己的记忆。

**最大风险。** 模型推断出的“可能有某疾病”不能直接写入用户长期健康档案；医疗事实、用户偏好和安全策略必须分开存储。候选记忆应先进入 pending/candidate 区，经安全过滤和策略确认后才进入可检索记忆。

### 2.4 Tool-R0: Self-Evolving LLM Agents for Tool-Learning from Zero Data

文件：`2602.21320v1.pdf`

**核心思想。** Generator 生成位于 Solver 能力边界附近的挑战任务，Solver 使用真实工具解决任务，再用可验证奖励驱动双方共同进化；系统可以从零数据开始。

**对项目的启发。** 可以生成用于压力测试的边界案例，覆盖：路由错误、planner 漏掉依赖、工具参数错误、上下文污染、长对话记忆错误和安全边界误判。挑战案例应进入 eval staging，经过人工确认后才进入正式回归集。

**不建议接入的部分。** 当前项目不宜直接做 online RL 或医疗领域 self-play。奖励设计难、成本高，生成器还可能放大错误或制造未经审查的医学回答。先把它当作“自动生成评测难例”的思想来源。

### 2.5 AUTOSKILL: Experience-Driven Lifelong Learning via Skill Self-Evolution

文件：`2603.01145v2.pdf`

**核心思想。** 从重复用户交互中抽取稳定的偏好和要求，形成标准化、可组合、可迁移的 skill；不需要重新训练基础模型。生命周期是：观察经验、抽象技能、维护/演化、检索相关技能、注入后续请求。

**适合本项目的例子。** “用少术语解释”“回答先给结论”“老人模式下分步表达”“每次涉及风险都保留就医提示”等可以成为用户偏好技能或表达模板。

**关键隔离。** 用户偏好不能覆盖医疗安全规则；“用户喜欢简短回答”可以影响表达长度，但不能删除必要的风险提示。健康事实也不能和偏好放进同一种 skill。

### 2.6 EvoSkill: Automated Skill Discovery for Multi-Agent Systems

文件：`2603.02766v1.pdf`

**核心结构。** Executor 执行任务；Proposer 分析失败轨迹和 ground truth，提出新技能或修改建议；Skill-Builder 将建议物化为带 trigger metadata 的 `SKILL.md`、可选 helper script 和 reference。系统维护 Pareto frontier/top-k programs，只有在 held-out validation set 上提升的候选才保留，并记录 feedback history 避免重复失败。

**对项目的直接价值。** 这与现有技能目录最匹配。项目不需要先重写 agent，只需增加 proposal、evaluation、promotion 三层，把“写出 skill”与“允许正式使用 skill”分开。

**医疗化改造。** ground truth、用户原始健康信息和敏感轨迹不能未经脱敏直接写入 skill 文件；发布条件应是多维门槛，而非单一平均分，例如正确性不下降、安全违规为零、关键路由不回退、成本和延迟在预算内。

## 3. 与现有项目的映射

| 论文机制 | 项目已有位置 | 建议新增位置 | 接入方式 |
|---|---|---|---|
| 轨迹/经验采集 | `backend/agents/graph_new.py`、OpenTelemetry | `experience` 服务或事件表 | 记录任务目标、计划、工具调用、结果、验证问题、耗时 |
| 技能执行 | `backend/skills/`、动态发现 | `candidate/`、`published/` 或版本元数据 | 沿用 `SKILL.md + skill.py`，增加状态和版本 |
| 失败分析 | `verifier`、`handoff`、任务错误 | `failure_cases`、失败聚类 | 将 verifier issue 和用户纠正转成可复现案例 |
| 技能提案 | `skill_proposals`、`evolution_proposer.py` | 候选谱系与反馈迭代 | 结构化离线 LLM proposer 生成，不直接生效 |
| 候选评测 | `backend/evals/`、冻结数据集、`evaluation_campaigns` | 实际候选执行器 | 逐案例比较 baseline/candidate，三套数据集成套门禁 |
| 晋级发布 | 当前没有 | promotion policy、审核记录 | 通过门槛且无安全回退才进入 published |
| 任务恢复 | `agent_persistence.py`、`AgentRun/AgentTask` | 关联 experience_id | 不改变当前恢复语义，只补充证据关联 |
| 成功模板 | planner、现有任务定义 | `template_library` | 仅检索已经验证的模板 |
| 记忆技能 | 用户 profile/对话摘要相关逻辑 | `memory_skills`、`memory_candidates` | 写入前安全过滤、可追溯、可撤销 |
| 能力缺口 | 工具调用失败、handoff | `capability_gaps` | 统计缺工具、schema 不匹配和权限问题 |

## 4. 分阶段实施建议

### P0：观测和离线技能改进闭环

这是最值得马上做的一阶段，风险和收益都可控。

1. 在每次 AgentLoop 结束时保存 `ExperienceRecord`：输入摘要、intent、plan signature、task results、tool calls、verify status/issues、latency、token/cost、最终结果和安全标记。
2. 将失败、部分成功、用户纠正和人工标记收集为 `FailureCase`，保留可脱敏的最小复现输入。
3. 离线 proposer 按失败模式生成 `SkillProposal`，内容包括触发条件、技能正文、预期修复点、影响范围和测试案例。项目现已实现结构化 LLM 候选生成、提示注入隔离、父提案谱系和内容指纹去重。
4. 在独立冻结数据集上运行 baseline 与 candidate，产生带逐案例证据的 `SkillEvaluation`；项目现已强制精确覆盖数据集快照。
5. regression、holdout、safety-boundary 三套评测组成 `EvaluationCampaign` 且全部通过后才允许人工批准；当前批准仍不发布到线上。

建议的最小对象：

```text
ExperienceRecord
  id, run_id, task_id, intent, input_redacted, plan_signature
  tool_calls, result, verify_status, verify_issues, safety_flags, metrics

SkillProposal
  id, base_skill, version, trigger, content, source_failure_ids
  status(candidate|evaluating|approved|rejected), created_at

SkillEvaluation
  proposal_id, dataset_version, baseline_score, candidate_score
  safety_violations, route_regressions, cost_delta, latency_delta

PromotionDecision
  proposal_id, decision, reasons, reviewer, rollback_to, decided_at
```

### P1：Template Library 与安全记忆

- 保存高频且已验证的工作流模板，planner 只注入与 intent/task shape 匹配的模板。
- 先做低风险的用户表达偏好和服务流程偏好。
- 医疗上下文采用结构化字段，并附来源、置信度、时间和有效期。
- 记忆写入前运行安全过滤；矛盾信息进入 review/pending，而不是覆盖旧值。

### P2：Capability gap 与 Skill Composer

- 统计“没有工具”“工具 schema 不匹配”“权限不足”“工具返回不可验证”等缺口。
- 允许组合多个已验证技能形成候选工作流。
- 可以生成新工具的接口草案和测试草案，但不自动安装、不自动放入生产白名单。

### P3：自动挑战集生成

借鉴 Tool-R0 的 Generator 思路，从失败案例和当前低置信度案例生成边界测试，覆盖路由、参数、长上下文和安全拒答。生成案例先进入 staging，人工抽样确认后再进入正式回归集。

## 5. 发布门槛与评测指标

技能或模板的发布不能只看总体成功率，至少应同时满足：

- 任务正确性：route correctness、tool correctness、answer groundedness。
- 安全性：医疗安全违规为零；emergency triage 不得回退；不得生成越权诊断、处方或剂量建议。
- 稳定性：holdout 集提升或不下降，关键 case 无回退。
- 记忆质量：recall、precision、contradiction rate、staleness/expiry、用户撤销生效率。
- 工程指标：延迟、token、成本、工具失败率和重试次数。
- 可维护性：触发条件明确、版本可追溯、候选可回滚、失败历史可查询。

可以在现有 `backend/evals/cases.jsonl` 之外增加三类数据集：

1. `regression`：现有稳定能力，任何关键回退都阻止发布。
2. `holdout`：不参与技能生成，用来判断真实泛化。
3. `safety_boundary`：急症、剂量、禁忌、越权和不确定性场景，安全失败必须是硬门槛。

## 6. 医疗安全边界

以下内容不应由在线自进化自动改变：

- emergency triage 的红线、升级就医策略和安全响应模板。
- 剂量计算、药物禁忌和处方相关规则。
- 医保/保险规则的事实来源和权限边界。
- 用户长期健康档案中的事实字段。
- 工具白名单、外部数据访问权限和审计策略。

允许自动化的范围是：候选技能生成、候选模板生成、失败聚类、脱敏难例生成、离线评测和改进建议。正式发布应至少保留自动门禁、审核记录和回滚能力。

## 7. 具体改造顺序

建议按以下顺序落地：

1. 先在 `graph_new.py` 的 executor/verifier 边界统一生成 experience 记录，避免一开始改动 planner 行为。
2. 在 `agent_persistence.py` 和 `models.py` 增加 experience/failure/proposal/evaluation 的持久化；必要时补 Alembic migration。
3. 在 `backend/evals/` 增加 baseline-vs-candidate runner 和 safety boundary checks。
4. 为 `backend/skills/` 增加候选/发布状态与版本索引，复用现有技能格式。
5. 评测稳定后再接入已验证模板；最后再做 memory skill 和 capability gap。

不建议第一步修改在线 planner，让它立刻学习新技能。先让系统“能观察、能解释、能评测、能回滚”，再开放经过验证的检索注入。

## 8. 一句话决策

当前项目最合适的自进化方案是：**把 verifier 发现的失败变成脱敏经验，把经验变成候选 skill/template/memory skill，在独立回归集和医疗安全集上验证后版本化发布；在线系统只消费已发布能力。**
