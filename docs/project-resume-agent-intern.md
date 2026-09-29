# Smart Health Assistant｜医疗 Agent 平台与健康咨询系统

**项目类型：**个人项目　　**求职方向：**Agent / 大模型应用 / Python 后端实习

## 项目概述

面向健康咨询、预问诊、报告解读、用药咨询和医保事务设计的研究型 Agent 平台。采用 Supervisor + 领域 Agent + Skill/Tool + Verifier 架构，将用户请求拆解为可并行执行、可验证、可重规划和可回放的任务。

## 技术栈

Python、FastAPI、LangGraph、LangChain、DeepSeek API、Chroma、BM25、SQLAlchemy、MySQL、Redis、MinIO、Alembic、SSE、OpenTelemetry、Jaeger、MCP、Docker。

## 主要工作与技术亮点

- **Supervisor 意图识别：**设计 `emergency / pure_chat / mixed / health` 四类 Intent Gate，结合确定性急症规则和结构化 Schema 校验，避免普通问答覆盖高风险信号；支持复合请求提取和领域 Agent 路由。
- **任务 DAG 编排：**将复合请求拆解为带依赖关系的任务图，校验 ID 唯一、依赖无环、Agent 白名单、重试上限和输入切片；无依赖任务按波次并行，降低串行调用开销。
- **Plan-Solve-Verify-Replan：**实现 Planner、Executor、Verifier、Responder 闭环；对 `partial / fail / unsafe` 分别执行局部补丁、局部重跑/重规划或安全熔断，使用次数锁、内容锁和时间预算防止死循环。
- **领域 Agent 隔离：**拆分症状评估、报告解读、药品咨询、医保事务和通用健康顾问 Agent；通过独立 Prompt、工具白名单、上下文切片和结果信封限制越权调用。
- **Agentic RAG 检索链路：**实现 BM25 + Dense 召回、RRF 融合、MMR 去重重排、父子块聚合和来源注入；新增有界多轮检索控制器，第一轮观察证据缺口后生成面向适应症、风险、流程等缺口的下一轮查询，直到证据充分或达到预算。
- **检索评测：**在 100 条 Agentic RAG A/B 场景集上，普通 RAG 的 Source Recall@1/3/5 为 88%/92%/96%，Agentic RAG 为 90%/93%/93%；章节 Recall@3 从 35% 提升至 52%，MRR 从 0.3155 提升至 0.4995；明确区分来源定位、章节命中与医学答案正确率。
- **Agent Runtime：**在不替换 LangGraph 的前提下实现 Runtime Context、步骤/工具/Token/超时预算、运行快照、SSE 生命周期和经验记录，为限额、回放和恢复提供统一边界。
- **多模态与异步任务：**支持报告图片/PDF 上传、Vision/MinerU/PaddleOCR 可替换 OCR、MinIO 对象存储和 Worker 异步分析；Redis 负责缓存、限流、幂等和任务协调。
- **分层记忆：**将用户画像、表达偏好、当前轮上下文、近期健康事件和历史记录分层；实现 HealthEvent 状态机，支持症状合并、康复标记、过期、历史保留和相关性注入。
- **用户级 Memory Evolution：**实现 Shadow Memory Agent，显式偏好即时提取、普通对话每 10 轮触发维护，基于 `MemoryCandidate/HealthEvent` 生成可审计的 Hermes 风格 Markdown 记忆投影，并提供独立 reset，不影响医疗安全规则和历史审计数据。
- **工程化自进化：**构建 ExperienceRecord → FailureCase → SkillProposal → 冻结评测集 → baseline/candidate 对照 → 安全门禁 → 人工审核 → SkillDeployment 闭环。
- **发布恢复：**实现 Skill 的 `deploy / rollback / reset`；回滚只切换版本指针，重置只停用动态技能注入，不删除评测、失败案例和审计证据。
- **可观测与安全：**接入 OpenInference/OpenTelemetry/Jaeger，默认隐藏医疗原文和图片；补充路由、Planner DAG、工具依赖、证据覆盖和红旗安全评测，48 条合成红旗案例达到 Precision 96.3%、Recall 86.7%、F1 91.2%。
- **端到端评测与质量诊断：**合并 28 条在线补测样本与 72 条扩展预算补测样本，形成完整 100 条合成回归口径：任务完成率 97.0%、路由准确率 80.0%，平均延迟 19.9 秒、P95 49.5 秒；对全部 100 条引入 DeepSeek LLM-as-Judge，结构化输出遵循率 97.0%，有效样本计划质量均分 0.849、回答完整性均分 0.829。结果未经人工校准，不能解读为临床结论。

## 典型 Demo

用户提出“近三个月血压升高，想吃降压药，还想知道医保能否报销”。系统识别 mixed health intent，拆分趋势分析、科普检索和医保事务任务；并行执行无依赖任务；Verifier 检查证据、安全和逻辑一致性；缺信息时只补充局部任务；最终按健康风险、用药边界和医保政策聚合回答。

## 设计边界

系统不自动诊断、开处方或生成越权剂量；自进化不在线修改模型权重、急症规则或工具权限；所有候选技能必须脱敏、评测、审核后才能发布，并支持回滚和基线重置。
