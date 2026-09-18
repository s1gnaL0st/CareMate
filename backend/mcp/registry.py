"""Small in-process MCP tool registry.

Handlers use the same contract as an MCP server: named tools with JSON-like
arguments and JSON-serializable results. Keeping registration separate makes
the transport replaceable without changing domain agents.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

MCPHandler = Callable[..., Any] | Callable[..., Awaitable[Any]]
_TOOLS: dict[str, MCPHandler] = {}


def register_tool(name: str, handler: MCPHandler | None = None):
    """Register a handler directly or return a decorator."""
    def registrar(fn: MCPHandler) -> MCPHandler:
        _TOOLS[name] = fn
        return fn

    return registrar(handler) if handler is not None else registrar


def get_tool(name: str) -> MCPHandler | None:
    return _TOOLS.get(name)
