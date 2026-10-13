"""Input contract shared by the domain agents.

The Supervisor keeps its own state (``AgentLoopState`` in ``agents/graph_new``)
and never hands that whole object to a domain agent. The executor calls each
agent with the narrow slice declared here, built by
``graph_new._agent_input_slice``. Keeping the two schemas separate is
deliberate: a domain agent must not be able to read or write supervisor fields
such as ``task_queue`` or ``verify_status``.
"""
from __future__ import annotations

from typing import Any, TypedDict

from langchain_core.messages import BaseMessage


class DomainAgentInput(TypedDict):
    """Exactly the fields ``_agent_input_slice`` passes to one domain agent."""

    # System preambles plus this task's own input slice.
    messages: list[BaseMessage]

    # Canonical conversation history with real Human/AI roles. The clinic agent
    # needs it to replay a patient's answer as the result of its own ask call.
    conversation_messages: list[BaseMessage]

    # Profile injected by the API layer (name, age, history, memory, ...).
    user_info: dict[str, Any]

    # Offline evaluation opt-in. Only the evolution runner sets it to True;
    # the online API path never captures RAG excerpts.
    offline_evidence_capture: bool
