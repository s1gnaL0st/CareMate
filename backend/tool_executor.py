"""Common execution boundary for LangChain tools.

The domain agents still use LangGraph's prebuilt ReAct agent and ToolNode.  A
tool is wrapped before it is passed to the ReAct agent so every invocation
gets the same policy, timeout, retry and lifecycle-hook behavior.

The executor deliberately preserves successful tool return values.  Existing
frontend card renderers and agents expect the original JSON/text payload.  Only
executor-generated failures use the small JSON error envelope defined here.
"""
from __future__ import annotations

import asyncio
from contextvars import ContextVar
import inspect
import json
import logging
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool

from config import get_settings


logger = logging.getLogger("smart_health.tools")


@dataclass(frozen=True)
class ToolExecutionContext:
    """Non-sensitive identifiers attached to one tool invocation."""

    user_id: str | None = None
    conversation_id: str | None = None
    agent_run_id: str | None = None
    task_id: str | None = None
    agent_name: str | None = None
    request_id: str | None = None


_active_tool_context: ContextVar[ToolExecutionContext] = ContextVar(
    "active_tool_context", default=ToolExecutionContext()
)


def get_active_tool_context() -> ToolExecutionContext:
    """Return non-sensitive context for the tool currently being executed."""
    return _active_tool_context.get()


class ToolMiddleware(Protocol):
    """Lifecycle extension point for auditing, metrics and policy adapters."""

    async def before_tool(
        self, tool: BaseTool, args: Mapping[str, Any], context: ToolExecutionContext
    ) -> None: ...

    async def after_tool(
        self,
        tool: BaseTool,
        result: Any,
        elapsed_ms: int,
        context: ToolExecutionContext,
    ) -> None: ...

    async def on_tool_error(
        self,
        tool: BaseTool,
        error: Exception,
        elapsed_ms: int,
        context: ToolExecutionContext,
    ) -> None: ...


class ToolExecutionError(RuntimeError):
    """Base error that is safe to convert into a tool response."""


class ToolAuthorizationError(ToolExecutionError):
    """Raised when an Agent attempts to call a tool outside its allow-list."""


class ToolTimeoutError(ToolExecutionError):
    """Raised when a tool exceeds the configured execution deadline."""


@dataclass
class ToolExecutor:
    """Wrap and execute tools with a consistent production boundary."""

    context: ToolExecutionContext = field(default_factory=ToolExecutionContext)
    allowed_tools: frozenset[str] = field(default_factory=frozenset)
    allowed_dynamic_skills: frozenset[str] = field(default_factory=frozenset)
    timeout_seconds: float | None = None
    max_retries: int | None = None
    middleware: tuple[ToolMiddleware, ...] = ()

    def __post_init__(self) -> None:
        if self.timeout_seconds is None:
            self.timeout_seconds = float(get_settings().tool_timeout_seconds)
        if self.max_retries is None:
            self.max_retries = max(0, int(get_settings().tool_max_retries))
        if not self.middleware:
            self.middleware = (LoggingToolMiddleware(),)
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries cannot be negative")

    @classmethod
    def from_state(
        cls,
        state: Mapping[str, Any],
        *,
        agent_name: str,
        allowed_tools: Iterable[str],
        allowed_dynamic_skills: Iterable[str] = (),
        middleware: Iterable[ToolMiddleware] = (),
    ) -> "ToolExecutor":
        """Build an executor from the current AgentLoop state."""
        return cls(
            context=ToolExecutionContext(
                user_id=state.get("user_id"),
                conversation_id=state.get("conversation_id"),
                agent_run_id=state.get("agent_run_id"),
                task_id=state.get("task_id"),
                agent_name=agent_name,
                request_id=state.get("request_id"),
            ),
            allowed_tools=frozenset(allowed_tools),
            allowed_dynamic_skills=frozenset(allowed_dynamic_skills),
            middleware=tuple(middleware),
        )

    def wrap_tools(self, tools: Iterable[BaseTool]) -> list[BaseTool]:
        """Return ToolNode-compatible wrappers while preserving schemas/names."""
        return [self.wrap(tool) for tool in tools]

    def wrap(self, base_tool: BaseTool) -> BaseTool:
        if base_tool.name not in self.allowed_tools:
            raise ValueError(
                f"Tool '{base_tool.name}' is not allowed for "
                f"agent '{self.context.agent_name or 'unknown'}'"
            )

        async def wrapped(**kwargs: Any) -> Any:
            return await self.execute(base_tool, kwargs)

        tool_kwargs: dict[str, Any] = {
            "coroutine": wrapped,
            "name": base_tool.name,
            "description": base_tool.description,
        }
        # @tool functions are StructuredTools. Reusing the schema keeps the
        # provider's function-calling contract unchanged.
        if base_tool.args_schema is not None:
            tool_kwargs["args_schema"] = base_tool.args_schema
            tool_kwargs["infer_schema"] = False
        return StructuredTool.from_function(**tool_kwargs)

    async def execute(self, base_tool: BaseTool, args: Mapping[str, Any]) -> Any:
        """Run one tool with authorization, hooks, timeout and bounded retry."""
        normalized_args = dict(args)
        started = asyncio.get_running_loop().time()
        token = _active_tool_context.set(self.context)
        try:
            self._authorize(base_tool.name, normalized_args)
            await self._before(base_tool, normalized_args)

            attempts = int(self.max_retries or 0) + 1
            last_error: Exception | None = None
            for attempt in range(attempts):
                try:
                    result = await asyncio.wait_for(
                        self._call_underlying(base_tool, normalized_args),
                        timeout=self.timeout_seconds,
                    )
                except Exception as exc:  # tools can be sync or async
                    last_error = exc
                    if isinstance(exc, ToolAuthorizationError):
                        break
                    if attempt + 1 < attempts:
                        logger.warning(
                            "tool failed; retrying tool=%s attempt=%d/%d error=%s",
                            base_tool.name,
                            attempt + 1,
                            attempts,
                            type(exc).__name__,
                        )
                        await asyncio.sleep(min(0.25 * (2**attempt), 2.0))
                else:
                    # Middleware is observational; a broken metrics/audit hook
                    # must not turn a successful business call into a retry.
                    await self._after(base_tool, result, started)
                    return result
            assert last_error is not None
            raise last_error
        except asyncio.TimeoutError as exc:
            error = ToolTimeoutError(f"tool '{base_tool.name}' timed out")
            await self._on_error(base_tool, error, started)
            return self._error_payload(base_tool, error)
        except Exception as exc:
            await self._on_error(base_tool, exc, started)
            return self._error_payload(base_tool, exc)
        finally:
            _active_tool_context.reset(token)

    def _authorize(self, tool_name: str, args: Mapping[str, Any]) -> None:
        if tool_name != "load_skill":
            return
        skill_name = str(args.get("skill_name", "")).strip()
        if skill_name not in self.allowed_dynamic_skills:
            raise ToolAuthorizationError(
                f"skill '{skill_name or '<empty>'}' is not allowed for "
                f"agent '{self.context.agent_name or 'unknown'}'"
            )

    async def _call_underlying(
        self, base_tool: BaseTool, args: Mapping[str, Any]
    ) -> Any:
        """Call the underlying function without creating duplicate tool events.

        The outer StructuredTool is the event boundary visible to LangGraph.
        Calling the original StructuredTool.ainvoke here would emit a second
        nested on_tool_start/on_tool_end pair with the same name.
        """
        coroutine = getattr(base_tool, "coroutine", None)
        if coroutine is not None:
            return await coroutine(**dict(args))
        func = getattr(base_tool, "func", None)
        if func is not None:
            result = func(**dict(args))
            if inspect.isawaitable(result):
                return await result
            return result
        return await base_tool.ainvoke(dict(args))

    async def _before(self, tool: BaseTool, args: Mapping[str, Any]) -> None:
        for middleware in self.middleware:
            try:
                await middleware.before_tool(tool, args, self.context)
            except Exception as exc:
                logger.warning(
                    "tool middleware before hook failed tool=%s middleware=%s error=%s",
                    tool.name,
                    type(middleware).__name__,
                    type(exc).__name__,
                )

    async def _after(self, tool: BaseTool, result: Any, started: float) -> None:
        elapsed_ms = self._elapsed_ms(started)
        for middleware in self.middleware:
            try:
                await middleware.after_tool(tool, result, elapsed_ms, self.context)
            except Exception as exc:
                logger.warning(
                    "tool middleware after hook failed tool=%s middleware=%s error=%s",
                    tool.name,
                    type(middleware).__name__,
                    type(exc).__name__,
                )

    async def _on_error(self, tool: BaseTool, error: Exception, started: float) -> None:
        logger.warning(
            "tool execution failed tool=%s agent=%s error=%s",
            tool.name,
            self.context.agent_name,
            type(error).__name__,
        )
        elapsed_ms = self._elapsed_ms(started)
        for middleware in self.middleware:
            try:
                await middleware.on_tool_error(tool, error, elapsed_ms, self.context)
            except Exception as exc:
                logger.warning(
                    "tool middleware error hook failed tool=%s middleware=%s error=%s",
                    tool.name,
                    type(middleware).__name__,
                    type(exc).__name__,
                )

    @staticmethod
    def _elapsed_ms(started: float) -> int:
        return int((asyncio.get_running_loop().time() - started) * 1000)

    @staticmethod
    def _error_payload(tool: BaseTool, error: Exception) -> str:
        if isinstance(error, ToolTimeoutError):
            code = "timeout"
            message = "工具执行超时，请稍后重试。"
        elif isinstance(error, ToolAuthorizationError):
            code = "forbidden"
            message = "当前 Agent 无权调用该工具。"
        else:
            code = "tool_error"
            message = "工具暂时不可用，请稍后重试。"
        return json.dumps(
            {"success": False, "tool": tool.name, "error": code, "message": message},
            ensure_ascii=False,
        )


class LoggingToolMiddleware:
    """Safe default middleware: identifiers and timings only, never payloads."""

    async def before_tool(
        self, tool: BaseTool, args: Mapping[str, Any], context: ToolExecutionContext
    ) -> None:
        logger.info(
            "tool.start tool=%s agent=%s run_id=%s task_id=%s",
            tool.name,
            context.agent_name,
            context.agent_run_id,
            context.task_id,
        )

    async def after_tool(
        self,
        tool: BaseTool,
        result: Any,
        elapsed_ms: int,
        context: ToolExecutionContext,
    ) -> None:
        logger.info(
            "tool.end tool=%s agent=%s elapsed_ms=%d",
            tool.name,
            context.agent_name,
            elapsed_ms,
        )

    async def on_tool_error(
        self,
        tool: BaseTool,
        error: Exception,
        elapsed_ms: int,
        context: ToolExecutionContext,
    ) -> None:
        logger.warning(
            "tool.error tool=%s agent=%s elapsed_ms=%d error=%s",
            tool.name,
            context.agent_name,
            elapsed_ms,
            type(error).__name__,
        )
