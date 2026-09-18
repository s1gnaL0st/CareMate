from collections.abc import Callable
from typing import Any

_TOOLS: dict[str, Callable[..., Any]] = {}

def register_tool(name: str, handler=None):
    def registrar(fn):
        _TOOLS[name] = fn
        return fn
    return registrar(handler) if handler is not None else registrar

def get_tool(name: str):
    return _TOOLS.get(name)
