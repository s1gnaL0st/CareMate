from __future__ import annotations

import asyncio
import inspect
import os
import sys
from typing import Any

from config import get_settings
from .registry import get_tool

class MCPUnavailableError(RuntimeError):
    pass

async def call_mcp_tool(name: str, **arguments: Any) -> Any:
    settings = get_settings()
    if not settings.mcp_enabled:
        raise MCPUnavailableError("MCP disabled")
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except Exception as exc:
        raise MCPUnavailableError("MCP SDK unavailable") from exc
    remote_url = ""
    remote_tool = name
    if name == "search_drug_info":
        remote_url, remote_tool = os.getenv("MCP_OPENFDA_URL", settings.mcp_openfda_url), "openfda_drug_profile"
    elif name == "search_medical_literature":
        remote_url, remote_tool = os.getenv("MCP_PUBMED_URL", settings.mcp_pubmed_url), "pubmed_search_articles"
    if remote_url:
        try:
            from mcp.client.streamable_http import streamable_http_client
            async with streamable_http_client(remote_url) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await asyncio.wait_for(session.initialize(), settings.mcp_timeout_seconds)
                    result = await asyncio.wait_for(
                        session.call_tool(remote_tool, arguments=arguments),
                        settings.mcp_timeout_seconds,
                    )
                    text = result.content[0].text if result.content else ""
                    import json
                    return json.dumps({"schema_version": "1.0", "tool": name, "source": "remote_mcp", "found": bool(text), "data": text}, ensure_ascii=False)
        except Exception as exc:
            raise MCPUnavailableError(f"Remote MCP tool '{name}' failed") from exc

    params = StdioServerParameters(
        command=sys.executable,
        args=["-u", os.path.join(os.path.dirname(__file__), "..", "mcp_server.py"), name],
        env=os.environ.copy(),
    )
    try:
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await asyncio.wait_for(session.initialize(), settings.mcp_timeout_seconds)
                result = await asyncio.wait_for(
                    session.call_tool(name, arguments=arguments), settings.mcp_timeout_seconds
                )
                return result.content[0].text if result.content else ""
    except Exception as exc:
        raise MCPUnavailableError(f"MCP tool '{name}' failed") from exc
