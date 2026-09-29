"""Runtime facade for executing the LangGraph AgentLoop.

LangGraph remains the workflow engine.  This module owns the concerns that
belong to one agent run: identity, budgets, lifecycle metrics and a small
serializable checkpoint envelope.  Keeping this boundary separate makes it
possible to add memory, approvals and replay without coupling those concerns
to every graph node.
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field
from time import monotonic
from typing import Any, AsyncIterator, Mapping


class RuntimeBudgetExceeded(RuntimeError):
    """Raised when a run exceeds a platform-level execution budget."""


@dataclass(frozen=True)
class RuntimeBudget:
    """Hard limits for one Agent Runtime execution."""

    max_steps: int = 80
    max_tool_calls: int = 32
    max_total_tokens: int = 0
    timeout_seconds: float = 120.0

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if self.max_tool_calls < 0:
            raise ValueError("max_tool_calls cannot be negative")
        if self.max_total_tokens < 0:
            raise ValueError("max_total_tokens cannot be negative")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")


@dataclass(frozen=True)
class RuntimeContext:
    """Stable, non-sensitive identity for one run."""

    run_id: str = ""
    request_id: str = ""
    user_id: str = ""
    conversation_id: str = ""
    requested_mode: str = "general"


@dataclass
class RuntimeSnapshot:
    """Mutable run counters exposed as a serializable snapshot."""

    status: str = "created"
    steps: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    event_count: int = 0
    last_node: str = ""
    stop_reason: str = ""
    elapsed_ms: int = 0
    checkpoint_id: str = ""

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["total_tokens"] = self.total_tokens
        return value


@dataclass
class AgentRuntime:
    """Execute a compiled LangGraph app under one runtime policy boundary."""

    graph: Any
    context: RuntimeContext = field(default_factory=RuntimeContext)
    budget: RuntimeBudget = field(default_factory=RuntimeBudget)
    snapshot_state: RuntimeSnapshot = field(default_factory=RuntimeSnapshot)
    _started_at: float = field(default_factory=monotonic, init=False, repr=False)

    @classmethod
    def from_state(
        cls,
        state: Mapping[str, Any],
        *,
        graph: Any,
        request_id: str = "",
        budget: RuntimeBudget | None = None,
    ) -> "AgentRuntime":
        """Create a per-run runtime without changing the graph state contract."""
        runtime_data = state.get("runtime_context", {})
        if not isinstance(runtime_data, Mapping):
            runtime_data = {}
        context = RuntimeContext(
            run_id=str(state.get("agent_run_id") or runtime_data.get("run_id") or ""),
            request_id=str(request_id or runtime_data.get("request_id") or ""),
            user_id=str(state.get("user_id") or runtime_data.get("user_id") or ""),
            conversation_id=str(state.get("conversation_id") or runtime_data.get("conversation_id") or ""),
            requested_mode=str(state.get("requested_mode") or runtime_data.get("requested_mode") or "general"),
        )
        return cls(graph=graph, context=context, budget=budget or RuntimeBudget())

    def runtime_state(self) -> dict[str, Any]:
        """Return the state fragment safe to persist in a checkpoint."""
        return {
            "context": asdict(self.context),
            "budget": asdict(self.budget),
            "snapshot": self.snapshot_state.as_dict(),
        }

    def describe(self) -> dict[str, Any]:
        """Return runtime metadata suitable for an SSE event or trace."""
        return {
            "runtime": "agent_runtime_v1",
            "context": asdict(self.context),
            "budget": asdict(self.budget),
            "snapshot": self.snapshot_state.as_dict(),
        }

    def _update_elapsed(self) -> None:
        self.snapshot_state.elapsed_ms = int((monotonic() - self._started_at) * 1000)

    def _check_budget(self) -> None:
        self._update_elapsed()
        if self.snapshot_state.steps > self.budget.max_steps:
            self.snapshot_state.status = "budget_exceeded"
            self.snapshot_state.stop_reason = "max_steps"
            raise RuntimeBudgetExceeded("agent runtime step budget exceeded")
        if self.snapshot_state.tool_calls > self.budget.max_tool_calls:
            self.snapshot_state.status = "budget_exceeded"
            self.snapshot_state.stop_reason = "max_tool_calls"
            raise RuntimeBudgetExceeded("agent runtime tool-call budget exceeded")
        if self.budget.max_total_tokens and self.snapshot_state.total_tokens > self.budget.max_total_tokens:
            self.snapshot_state.status = "budget_exceeded"
            self.snapshot_state.stop_reason = "max_total_tokens"
            raise RuntimeBudgetExceeded("agent runtime token budget exceeded")
        if self.snapshot_state.elapsed_ms > int(self.budget.timeout_seconds * 1000):
            self.snapshot_state.status = "budget_exceeded"
            self.snapshot_state.stop_reason = "timeout"
            raise RuntimeBudgetExceeded("agent runtime timeout exceeded")

    @staticmethod
    def _usage(output: Any) -> tuple[int, int]:
        if isinstance(output, Mapping):
            usage = output.get("usage_metadata") or output.get("token_usage") or {}
        else:
            metadata = getattr(output, "response_metadata", None) or {}
            usage = getattr(output, "usage_metadata", None) or metadata.get("token_usage", {})
        if not isinstance(usage, Mapping):
            return 0, 0
        return (
            int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0),
            int(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0),
        )

    async def astream_events(
        self,
        state: Mapping[str, Any],
        *,
        version: str = "v2",
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield LangGraph events while applying runtime limits and metrics."""
        self.snapshot_state.status = "running"
        self._started_at = monotonic()
        graph_state = dict(state)
        graph_state["runtime_context"] = {
            **dict(graph_state.get("runtime_context", {}) or {}),
            **asdict(self.context),
        }
        stream = self.graph.astream_events(graph_state, version=version).__aiter__()
        try:
            while True:
                self._check_budget()
                remaining = self.budget.timeout_seconds - (monotonic() - self._started_at)
                if remaining <= 0:
                    self.snapshot_state.status = "budget_exceeded"
                    self.snapshot_state.stop_reason = "timeout"
                    raise RuntimeBudgetExceeded("agent runtime timeout exceeded")
                try:
                    event = await asyncio.wait_for(stream.__anext__(), timeout=remaining)
                except StopAsyncIteration:
                    break
                self.snapshot_state.event_count += 1
                kind = str(event.get("event", ""))
                name = str(event.get("name", ""))
                if kind == "on_chain_start":
                    self.snapshot_state.steps += 1
                    self.snapshot_state.last_node = name
                elif kind == "on_tool_start":
                    self.snapshot_state.tool_calls += 1
                    self.snapshot_state.last_node = name
                elif kind == "on_chat_model_end":
                    input_tokens, output_tokens = self._usage(event.get("data", {}).get("output"))
                    self.snapshot_state.input_tokens += input_tokens
                    self.snapshot_state.output_tokens += output_tokens
                self._check_budget()
                yield event
        except asyncio.TimeoutError as exc:
            self.snapshot_state.status = "budget_exceeded"
            self.snapshot_state.stop_reason = "timeout"
            raise RuntimeBudgetExceeded("agent runtime timeout exceeded") from exc
        except BaseException:
            if self.snapshot_state.status == "running":
                self.snapshot_state.status = "failed"
            self._update_elapsed()
            raise
        else:
            self.snapshot_state.status = "completed"
            self._update_elapsed()
        finally:
            close = getattr(stream, "aclose", None)
            if close is not None:
                await close()
