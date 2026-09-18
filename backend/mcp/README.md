# MCP 工具接入

当前首批只读知识工具已通过 `backend/mcp` 适配层接入：

- `search_drug_info`
- `check_drug_interaction`
- `search_insurance_policy`

适配层提供 MCP 风格的工具注册、参数转发和超时边界。开发环境默认使用本地进程内 Server 实现，便于项目开箱即用；调用失败会自动回退原有本地数据库/RAG 实现，不会中断 Agent 主链路。

可通过环境变量控制：

```env
MCP_ENABLED=true
MCP_TIMEOUT_SECONDS=30
```

后续接入远程 MCP Server 时，只需替换 `mcp/client.py` 的传输实现，领域 Agent 和前端工具协议无需改动。
