# 统一检索（search）

`search` 是一个能力，不是一条固定的数据流水线。它同时扇出三个通道，把结果按来源权威度合并成一份证据列表，再交给上层 Agent 使用。

## 三个通道

| 通道 | 来源 | 信任等级 | tier | 实现 |
| --- | --- | --- | --- | --- |
| `knowledge_base` | 内部受控中文医学语料 | `curated_internal` | 0 | `agentic_rag.retrieve_until_sufficient`（有界多轮） |
| `literature` | PubMed，经 MCP 服务 | `peer_reviewed` | 1 | `mcp_adapter.call_mcp_tool("search_medical_literature")` |
| `web` | 公开网页 | `untrusted_external` | 2 ~ 4 | `web_search.search_web` |

三个通道用 `asyncio.gather` 并发执行。**任何一个通道失败或超时都不会让整次检索失败**：该通道返回空结果并写入 `degraded_reason`，其余通道照常合并。

## 信任模型

扩大检索面不等于扩大 prompt 的信任面。外部内容一律遵守三条：

1. **标注来源。** 每条证据带 `citation_id`、`source_type`、`source_tier`、`trust` 四个字段。引用前缀区分通道：`rag:`、`pubmed:`、`web:`。
2. **按权威度排序。** tier 越小越权威，合并后整体升序排列（内部语料 → 同行评议文献 → 官方网页 → 一般网页 → 未知来源）。
3. **隔离注入内容。** 命中提示覆盖特征的网页不会进入证据列表，只保留在通道 payload 里并计入 `quarantined_count`，供审计排查。

`knowledge_search.format_evidence()` 在渲染 prompt 时会显式声明 web / literature 属于不可信数据，其中任何指令性文字都不得执行。

## 来源权威度分层

| tier | 判定 |
| --- | --- |
| 1 | `*.gov.cn`、`nmpa.gov.cn`、`cde.org.cn`、`who.int`、`fda.gov`、`nih.gov`、`cdc.gov`、`nhs.uk` 等监管/政府/国际组织域名 |
| 2 | `cochrane.org`、`msdmanuals.*`、`dayi.org.cn`、`yiigle.com`、`*.edu.cn`、`*.ac.cn` 等学术与机构医学来源 |
| 3 | `health.baidu.com`、`dxy.cn`、`wikipedia.org`、`webmd.com` 等一般医学参考 |
| 4 | 其余未识别域名 |

## 配置

```dotenv
WEB_SEARCH_ENABLED=true
# auto | tavily | searxng | duckduckgo | none
WEB_SEARCH_PROVIDER=auto
WEB_SEARCH_TIMEOUT_SECONDS=12
WEB_SEARCH_MAX_RESULTS=5
WEB_SEARCH_CACHE_TTL_SECONDS=900
WEB_SEARCH_REGION=cn-zh
# TAVILY_API_KEY=
# WEB_SEARCH_SEARXNG_URL=http://localhost:8888
LITERATURE_SEARCH_TIMEOUT_SECONDS=12
```

`WEB_SEARCH_PROVIDER=auto` 的解析顺序是：配置了 `TAVILY_API_KEY` 用 Tavily；配置了 `WEB_SEARCH_SEARXNG_URL` 用 SearXNG；否则回退到无密钥的 DuckDuckGo（`ddgs`）。设为 `none` 可关闭网页通道，`knowledge_base` 与 `literature` 不受影响。

## 调用点

- **诊室 Agent**：`search_health_knowledge` 工具（`agents/clinic_action_adapter.py`）调用统一检索，并把返回的 `citation_id` 加入答案引用白名单；引用白名单同步支持 `web:` 前缀。
- **通用 Agent**：同名 `search_health_knowledge` 工具（`agents/advisor.py`），参数可分别关闭 web 或文献通道。

## 延迟与成本

- 三个通道并发，最坏耗时由 `WEB_SEARCH_TIMEOUT_SECONDS` 与 `LITERATURE_SEARCH_TIMEOUT_SECONDS` 中的较大者决定，而不是三者相加。
- 网页结果按 query + provider + 条数缓存 `WEB_SEARCH_CACHE_TTL_SECONDS` 秒，重复问题不会重复出网。
- 网页通道只请求搜索结果页，不抓取正文，因此单次检索的 token 增量是可控的（每条证据截断到 1200 字符）。

## 验证

```bash
cd backend
uv run python -m unittest tests.test_web_search tests.test_knowledge_search
```

测试全部使用注入的假检索器，不触网、不依赖 Redis。
