# Smart Health Assistant 完整化路线图

目标：将当前 AI 应用原型升级为可部署、可恢复、可测试的健康服务平台，逐步移除 Demo 属性。

## 总体架构目标

```text
React → Nginx/API Gateway → FastAPI
                           ├─ MySQL：业务数据唯一事实来源
                           ├─ Redis：缓存、限流、锁、任务队列
                           ├─ MinIO/S3：报告图片和附件
                           ├─ Chroma/Qdrant：向量检索
                           └─ Worker：OCR、视觉识别、报告解析
```

## P0：必须完成

### 基础工程

- [x] 增加 `docker-compose.yml`，一键启动 MySQL、Redis、MinIO、后端
- [x] 增加 `dev/test/prod` 配置，使用 `pydantic-settings` 管理环境变量
- [x] 统一 API 前缀为 `/api/v1`（保留旧路径兼容）
- [x] 增加统一错误响应、请求 `request_id` 和结构化日志
- [x] 移除前端硬编码 API 地址，模型配置通过环境变量管理
- [x] 补充后端和前端 `.env.example`

验收：新机器执行 `docker compose up` 后可以启动完整服务。

### MySQL + SQLAlchemy

- [x] 安装 SQLAlchemy 2.x、`asyncmy`、Alembic
- [x] 创建异步 Engine、连接池和 Session 管理
- [x] 建立 Alembic 迁移流程
- [x] 创建 `users`、`user_profiles`、`conversations`、`messages` 表
- [x] 创建 `medical_reports` 表（分析结果 JSON 可扩展）
- [x] 创建 `medications`、`medication_tasks` 表
- [x] 创建 `insurance_accounts`、`insurance_transactions` 表
- [x] 创建 `audit_logs` 表
- [x] 为用户、会话、消息序号、创建时间等字段建立索引和唯一约束
- [x] 增加 Repository 数据访问边界（完整 Service 编排待继续抽取）
- [x] 使用 Pydantic Schema 隔离 ORM Model
- [x] 将 seed 数据和生产代码分离

验收：删除前端 `MOCK_*` 后，保险、报告、药品任务仍可从数据库正常读取。

### 用户与会话

- [x] 增加注册、登录、刷新、退出和当前用户接口
- [x] 使用 JWT Bearer Token
- [x] 增加 `user_id`、`conversation_id`
- [x] 增加会话创建、列表、消息详情、删除接口
- [x] 用户消息入库后再调用 Agent
- [x] AI 最终回答入库
- [x] 前端在存在 token 时页面刷新恢复最近聊天
- [x] 服务端校验会话归属并决定 `active_agent`
- [x] 从消息表恢复 LangGraph 上下文

### Supervisor AgentLoop

- [x] 实现 `Intent Gate → Planner → Executor → Verifier → Responder` Supervisor 图
- [x] Planner 使用结构化 `Plan` / `PlannedTask`，按请求动态生成任务
- [x] 校验 Task ID、Agent 白名单、依赖存在性及依赖环
- [x] 按 dependency DAG 分波执行，独立 Task 并行执行
- [x] Verifier 失败后的有限 Replan（最多 2 次，并检测重复计划）
- [x] 使用 MySQL 持久化 `AgentRun` / `AgentTask` 及任务结果
- [x] 支持任务边界暂停与恢复，恢复时跳过已完成 Task
- [x] 使用 Task lease 处理进程崩溃后的过期任务恢复
- [x] 增加统一 `ToolExecutor` / middleware，集中处理工具权限、超时、重试和 Hook
- [ ] 使用 LangGraph checkpointer 持久化完整 Graph State
- [ ] 使用 `interrupt()` / `Command(resume=...)` 实现完整 Human-in-the-loop
- [ ] 支持任意 Graph 节点或模型调用位置的精确暂停恢复
- [ ] 将 AgentLoop 拆为独立 Worker，并支持跨进程调度
- [x] 增加 AgentLoop Task 最大重试次数、退避和死信记录
- [ ] 持久化任务事件、调度信息和完整 Trace
- [x] 接入 MCP 工具调用（药品信息、药物相互作用、医保政策；MCP 不可用时自动回退本地实现）
- [x] 将 Replan 升级为保留已完成结果的增量补丁策略

### Agent 自进化（离线安全闭环）

- [x] 每次持久化 AgentRun 记录脱敏 `ExperienceRecord`（输入、计划、节点、工具名、验证结果、指标）
- [x] 将失败、部分成功、取消和安全路径归档为 staging `FailureCase`
- [x] 按失败类别和严重级别生成有界离线聚类，供提案器选择样本
- [x] 增加仅保存用户表达/服务偏好的 `pending` 记忆候选，支持用户接受、拒绝和撤销
- [x] 将用户已接受、未过期且通过提示注入过滤的表达偏好安全注入最终 Responder
- [x] 将可溯源的负向评分、安全反馈和用户纠正脱敏、幂等归档为 staging 失败案例
- [x] 增加仅生成 candidate 状态的离线 `SkillProposal`，不会自动修改线上 Planner
- [x] 接入结构化离线 LLM Proposer，隔离失败摘要提示注入并记录候选内容指纹、模型来源和父提案谱系
- [x] 增加能力缺口登记、指纹去重、出现次数累计和人工解决/拒绝/重开流程，不自动安装工具
- [x] 从脱敏失败记录生成 regression/safety-boundary 挑战案例 staging，并通过人工审核控制后续数据集整理
- [x] 将人工接受的 challenge cases 冻结为版本化 regression/holdout/safety 数据集快照
- [x] 将技能/模板评测绑定冻结数据集，服务端计算门禁并持久化逐案例证据，禁止手工 passed 绕过晋级
- [x] 用不可变 EvaluationCampaign 强制 regression/holdout/safety 三套评测成套通过后才允许人工批准
- [x] 从 verifier 通过的成功经验提取工作流模板候选，完成独立评测、最新通过门禁和人工审核，不自动注入 Planner
- [x] 支持将多个已批准且最新评测通过的技能提案组合为工作流候选，并复用模板评测审核门禁
- [x] 从能力缺口生成脱敏工具接口 Schema 与测试草案，支持审计审核但不生成代码、不安装工具
- [x] 增加 regression/holdout/safety 对比评测门禁，安全违规和关键路由回退硬失败
- [x] 对自进化观测做邮箱、手机号、身份证、JWT 脱敏及长度/深度限制
- [x] SkillEvaluation/PromotionDecision 持久化及人工审核工作流（只允许离线候选审核，不自动发布）
- [ ] 自动生成并发布技能、模板或记忆（需正式线审批和回滚机制）

### SSE 完整性

- [x] 增加 SSE 心跳
- [x] 支持客户端断开检测并标记主动取消
- [x] 设置 LLM 事件超时、最大执行时间和输出长度限制
- [x] 保存 `pending/completed/failed/cancelled` 状态
- [x] 使用 Redis 幂等键，避免重复提交
- [x] 统一 `start/token/tool/card/finish/error` 事件协议（现有事件兼容）
- [x] 处理断线重连和已消费事件（事件 ID、持久化账本及 after 游标接口）

验收：刷新、断网、重复点击时，不产生重复消息和悬挂任务。

## P1：体现后端能力

### Redis

- [x] 使用 `redis.asyncio`
- [x] 缓存药品查询、医保余额、政策查询和 RAG 结果
- [x] 为 `/chat`、`/vision-chat` 增加按用户/IP 限流
- [x] 使用分布式锁防止同一报告重复解析
- [x] 设置 TTL、缓存版本和主动失效机制
- [x] 明确 Redis 不是永久聊天数据源

### 异步任务

- [x] 使用 Arq、Celery 或 RQ
- [x] 图片识别、OCR、报告解析改为后台任务
- [x] 增加任务状态查询接口
- [x] 支持失败重试、退避和死信记录
- [x] 前端通过轮询或 SSE 获取任务进度

推荐流程：上传图片 → MinIO → 创建分析记录 → Redis 队列 → Worker 识别 → MySQL 保存 → SSE 通知。

### 文件与隐私

- [x] 图片不长期保存在本地磁盘，改用 MinIO/S3
- [x] 数据库只保存对象 key、大小、类型、哈希
- [x] 限制 MIME 类型、文件大小和图片尺寸
- [x] 使用图片哈希避免重复上传
- [x] 日志禁止输出医疗图片、完整病历和敏感信息
- [x] 敏感字段加密或脱敏
- [x] 增加用户数据删除接口

### LLM 可靠性

- [x] 增加模型调用超时、有限重试和备用模型
- [x] 主模型失败时支持备用模型
- [x] 限制输入字符长度（Token 精确计数待补）
- [x] 记录模型耗时、Token 和估算成本
- [x] 处理结构化输出解析失败并走 fallback
- [x] 增加 Prompt Injection 防护
- [x] 统一医疗免责声明和紧急情况提示

### OCR 与文档解析 Worker（MinerU）

- [x] 将 API 服务和 OCR Worker 拆成独立容器，避免阻塞 FastAPI
- [x] PDF 检查报告优先使用 MinerU 解析版面、表格和文本
- [x] 普通中文报告图片接入 PaddleOCR/PP-Structure，MinerU 作为复杂文档补充
- [ ] 药品盒图片使用 OCR + 视觉模型 + 药品数据库进行交叉校验
- [ ] 为 MinerU 固定版本、模型文件和运行参数
- [x] OCR Worker 单独使用 Python 3.11/3.12 镜像，验证与当前 Python 3.13 后端的兼容性
- [x] 统一输出页码、文本块、类型、坐标和置信度等 JSON 结构
- [ ] 保存原图、解析 Markdown/JSON 和裁剪图片的对象存储地址
- [ ] 低置信度、表格异常和手写内容进入人工复核状态
- [x] 增加 OCR/报告任务超时、重试、退避和死信记录
- [x] 增加 OCR/报告任务主动取消机制
- [x] 使用 `report_id` 加 Redis 分布式锁，保证报告解析幂等
- [ ] OCR 完成后将结构化文本切片并写入 RAG 向量库

验收：上传 PDF 或报告图片后，任务可异步完成，结果、状态和来源均可查询；重复提交不会重复解析。

## P2：去掉假数据感

- [x] 保险余额、消费记录、缴费记录全部来自 MySQL（按认证用户查询；无账户返回明确空状态）
- [x] 药品信息和相互作用从可版本化知识源读取（`pharmacy_knowledge.md` SHA-256 版本随工具结果返回）
- [x] 报告列表接入后端 API
- [x] 医院列表、健康习惯后端数据模型和接口完成（无数据返回空状态）
- [x] 移除前端 `frontend/src/data/mockData.ts` 的生产依赖
- [x] 报告、医院列表增加分页、筛选、排序、空状态和错误状态；健康目标支持空状态
- [x] 增加回答反馈接口（评分、分类、评论，并校验消息/会话归属）
- [x] 工具输出统一为版本化 JSON Schema（核心保险/药品工具已增加 `schema_version`）
- [x] RAG 返回来源、章节、更新时间和引用片段
- [x] 支持知识库增量更新，不每次全量重建

## P3：生产和简历加分项

### 测试

- [ ] Repository 单元测试
- [ ] MySQL 集成测试
- [ ] Redis 缓存、限流、锁测试
- [ ] SSE 流程和断线测试
- [ ] 文件上传安全测试
- [x] Agent 路由和工具调用测试
- [ ] 并发、超时、重试测试
- [x] GitHub Actions 自动执行 lint、test、build

### 监控

- [ ] OpenTelemetry 覆盖 API、Agent、工具和数据库
- [x] 增加 Prometheus 指标和 Grafana Dashboard
- [ ] 使用 Jaeger 查看链路
- [x] 记录 P50/P95 延迟、错误率、缓存命中率、任务失败率（Prometheus HTTP Histogram、缓存与任务计数器支持 PromQL）
- [x] 增加 `/health/live` 和 `/health/ready`
- [x] readiness 检查 MySQL、Redis、向量库

### 部署

- [x] 多阶段 Dockerfile
- [x] Nginx 反向代理
- [x] Gunicorn/Uvicorn 多 worker
- [x] 数据库备份和恢复脚本
- [x] 日志轮转
- [x] CI/CD 部署文档
- [x] 配置生产 CORS、HTTPS 和限流

## 推荐实施顺序

1. MySQL、SQLAlchemy、Alembic、用户和会话
2. 消息持久化和可靠 SSE
3. 移除保险、药品、报告相关 mock
4. Redis 缓存、限流、分布式锁
5. MinIO + MinerU/PaddleOCR + 异步报告解析 Worker
6. 鉴权、隐私、安全和错误处理
7. 集成测试、监控和 Docker 部署
8. RAG 引用、评估看板和成本统计

## 简历验收标准

完成前 5 项后，项目可以描述为：

> 基于 FastAPI、LangGraph 构建多 Agent 健康服务平台，使用 SQLAlchemy Async + MySQL 持久化用户、会话、消息及报告数据，使用 Redis 实现缓存、限流、分布式锁和异步任务调度，通过 MinIO 管理医疗图片，并使用 SSE 实现 Agent 执行过程和报告分析结果的实时推送。

面试需要能解释：MySQL 与向量库的职责划分、SSE 断线处理、Redis 缓存失效、报告幂等解析、多实例一致性和医疗隐私保护。
