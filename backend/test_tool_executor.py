import asyncio
import json
import unittest

from langchain_core.tools import tool

from tool_executor import (
    ToolAuthorizationError,
    ToolExecutionContext,
    ToolExecutor,
)


class _Recorder:
    def __init__(self):
        self.events = []

    async def before_tool(self, tool, args, context):
        self.events.append(("before", tool.name, context.task_id))

    async def after_tool(self, tool, result, elapsed_ms, context):
        self.events.append(("after", tool.name, context.task_id))

    async def on_tool_error(self, tool, error, elapsed_ms, context):
        self.events.append(("error", tool.name, type(error).__name__))


@tool
async def add_one(value: int) -> str:
    """Add one to an integer."""
    return str(value + 1)


@tool
async def fail_tool() -> str:
    """Always fail."""
    raise RuntimeError("expected failure")


@tool
async def slow_tool() -> str:
    """Sleep longer than the executor deadline."""
    await asyncio.sleep(0.05)
    return "done"


@tool
async def load_skill(skill_name: str, params_json: str = "{}") -> str:
    """Test dynamic skill authorization."""
    return json.dumps({"skill": skill_name})


class ToolExecutorTests(unittest.TestCase):
    def test_wrap_preserves_schema_and_runs_hooks(self):
        recorder = _Recorder()
        executor = ToolExecutor(
            context=ToolExecutionContext(agent_name="clinic_agent", task_id="t1"),
            allowed_tools=frozenset({"add_one"}),
            timeout_seconds=1,
            max_retries=0,
            middleware=(recorder,),
        )
        wrapped = executor.wrap(add_one)

        result = asyncio.run(wrapped.ainvoke({"value": 2}))

        self.assertEqual(result, "3")
        self.assertEqual(wrapped.name, add_one.name)
        self.assertEqual(wrapped.args_schema.model_json_schema(), add_one.args_schema.model_json_schema())
        self.assertEqual([event[0] for event in recorder.events], ["before", "after"])

    def test_failure_is_safe_and_retried(self):
        recorder = _Recorder()
        executor = ToolExecutor(
            context=ToolExecutionContext(agent_name="report_agent"),
            allowed_tools=frozenset({"fail_tool"}),
            timeout_seconds=1,
            max_retries=1,
            middleware=(recorder,),
        )

        result = asyncio.run(executor.wrap(fail_tool).ainvoke({}))
        payload = json.loads(result)

        self.assertEqual(payload["error"], "tool_error")
        self.assertEqual([event[0] for event in recorder.events], ["before", "error"])

    def test_timeout_returns_safe_error(self):
        executor = ToolExecutor(
            context=ToolExecutionContext(agent_name="clinic_agent"),
            allowed_tools=frozenset({"slow_tool"}),
            timeout_seconds=0.01,
            max_retries=0,
        )

        result = asyncio.run(executor.wrap(slow_tool).ainvoke({}))
        self.assertEqual(json.loads(result)["error"], "timeout")

    def test_dynamic_skill_must_be_allowlisted(self):
        executor = ToolExecutor(
            context=ToolExecutionContext(agent_name="report_agent"),
            allowed_tools=frozenset({"load_skill"}),
            allowed_dynamic_skills=frozenset({"lab_interpreter"}),
            timeout_seconds=1,
            max_retries=0,
        )

        result = asyncio.run(
            executor.wrap(load_skill).ainvoke(
                {"skill_name": "medication_calculator", "params_json": "{}"}
            )
        )
        self.assertEqual(json.loads(result)["error"], "forbidden")

        with self.assertRaises(ToolAuthorizationError):
            executor._authorize("load_skill", {"skill_name": "other"})


if __name__ == "__main__":
    unittest.main()

