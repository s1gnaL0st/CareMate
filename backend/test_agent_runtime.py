import unittest

from agent_runtime import (
    AgentRuntime,
    RuntimeBudget,
    RuntimeBudgetExceeded,
)


class FakeGraph:
    def __init__(self, events):
        self.events = events

    async def astream_events(self, _state, version="v2"):
        self.version = version
        for event in self.events:
            yield event


class AgentRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_runtime_preserves_graph_events_and_records_counters(self):
        graph = FakeGraph([
            {"event": "on_chain_start", "name": "planner", "data": {}},
            {"event": "on_tool_start", "name": "search", "data": {}},
            {
                "event": "on_chat_model_end",
                "name": "model",
                "data": {"output": {"usage_metadata": {"input_tokens": 4, "output_tokens": 6}}},
            },
        ])
        runtime = AgentRuntime.from_state(
            {"agent_run_id": "run-1", "user_id": "user-1", "requested_mode": "general"},
            graph=graph,
            request_id="request-1",
            budget=RuntimeBudget(max_steps=4, max_tool_calls=2, timeout_seconds=2),
        )

        events = [event async for event in runtime.astream_events({"messages": []})]

        self.assertEqual(len(events), 3)
        self.assertEqual(graph.version, "v2")
        self.assertEqual(runtime.context.run_id, "run-1")
        self.assertEqual(runtime.context.request_id, "request-1")
        self.assertEqual(runtime.snapshot_state.steps, 1)
        self.assertEqual(runtime.snapshot_state.tool_calls, 1)
        self.assertEqual(runtime.snapshot_state.total_tokens, 10)
        self.assertEqual(runtime.snapshot_state.status, "completed")

    async def test_runtime_stops_when_tool_budget_is_exceeded(self):
        graph = FakeGraph([
            {"event": "on_tool_start", "name": "one", "data": {}},
            {"event": "on_tool_start", "name": "two", "data": {}},
        ])
        runtime = AgentRuntime(
            graph=graph,
            budget=RuntimeBudget(max_steps=4, max_tool_calls=1, timeout_seconds=2),
        )

        with self.assertRaises(RuntimeBudgetExceeded):
            _ = [event async for event in runtime.astream_events({"messages": []})]

        self.assertEqual(runtime.snapshot_state.status, "budget_exceeded")
        self.assertEqual(runtime.snapshot_state.stop_reason, "max_tool_calls")

    def test_runtime_state_is_serializable(self):
        runtime = AgentRuntime.from_state(
            {"agent_run_id": "run-1", "user_id": "user-1"},
            graph=FakeGraph([]),
        )

        state = runtime.runtime_state()

        self.assertEqual(state["context"]["run_id"], "run-1")
        self.assertIn("max_steps", state["budget"])
        self.assertIn("total_tokens", state["snapshot"])


if __name__ == "__main__":
    unittest.main()
