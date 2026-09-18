"""Stable MCP payload envelope shared by read-only health tools."""
from __future__ import annotations

from typing import Any


def envelope(tool: str, **payload: Any) -> dict[str, Any]:
    return {"schema_version": "1.0", "tool": tool, **payload}
