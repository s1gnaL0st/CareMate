"""MCP client facade with bounded timeout and graceful fallback support."""
from __future__ import annotations

import asyncio
import inspect
from typing import Any

from config import get_settings
from .registry import get_tool


class MCPUnavailableError(RuntimeError):
    """Raised when an MCP tool is not configured or cannot be called."""


async def call_mcp_tool(name: str, **arguments: Any) -> Any:
    settings = get_settings()
    if not settings.mcp_enabled:
        raise MCPUnavailableError("MCP is disabled")
    handler = get_tool(name)
    if handler is None:
        raise MCPUnavailableError(f"MCP tool '{name}' is not registered")
    try:
        result = handler(**arguments)
        if inspect.isawaitable(result):
            result = await asyncio.wait_for(result, timeout=settings.mcp_timeout_seconds)
        return result
    except asyncio.TimeoutError as exc:
        raise MCPUnavailableError(f"MCP tool '{name}' timed out") from exc
    except MCPUnavailableError:
        raise
    except Exception as exc:
        raise MCPUnavailableError(f"MCP tool '{name}' failed") from exc
