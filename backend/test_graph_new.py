import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from langchain_core.messages import AIMessage, HumanMessage

from agents.graph_new import (
    AgentLoopState,
    IntentDecision,
    Plan,
    PlannedTask,
    _agent_input_slice,
    _dependency_waves,
    _fallback_plan,
    _validate_plan,
    intent_gate,
    executor,
    planner,
    responder,
    route_after_intent,
    supervisor_app,
)


class GraphNewTests(unittest.TestCase):
    def test_executor_retries_transient_agent_failure(self):
        attempts = {"count": 0}

        async def flaky(_state):
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise RuntimeError("temporary")
            return {"messages": [AIMessage(content="恢复正常")]}

        task = PlannedTask(id="symptoms", agent="symptom_agent", objective="评估", input_slice="评估")
        settings = type("Settings", (), {"agent_task_max_retries": 1, "agent_task_retry_backoff_seconds": 0.0})()
        with patch("agents.graph_new._DOMAIN_NODES", {"symptom_agent": flaky}), patch("agents.graph_new.get_settings", return_value=settings):
            result = asyncio.run(executor({"task_queue": [task], "task_results": {}}))
        self.assertEqual(attempts["count"], 2)
        self.assertEqual(result["task_results"]["symptoms"]["status"], "completed")
    def test_compiles_supervisor_graph(self):
        nodes = supervisor_app.get_graph().nodes
        self.assertTrue({
            "intent_gate",
            "planner",
            "executor",
            "verifier",
            "responder",
            "safety_response",
            "handoff",
        }.issubset(nodes))

    def test_plan_rejects_dependency_cycle(self):
        plan = Plan(tasks=[
            PlannedTask(id="a", agent="symptom_agent", objective="a", input_slice="a", depends_on=["b"]),
            PlannedTask(id="b", agent="report_agent", objective="b", input_slice="b", depends_on=["a"]),
        ])
        with self.assertRaises(ValueError):
            _validate_plan(plan, "request")

    def test_executor_waves_are_topological(self):
        tasks = [
            PlannedTask(id="b", agent="report_agent", objective="b", input_slice="b", depends_on=["a"]),
            PlannedTask(id="a", agent="symptom_agent", objective="a", input_slice="a"),
            PlannedTask(id="c", agent="insurance_agent", objective="c", input_slice="c"),
        ]
        self.assertEqual(
            [[task.id for task in wave] for wave in _dependency_waves(tasks)],
            [["a", "c"], ["b"]],
        )

    def test_dependency_result_is_restricted_to_dependent_task(self):
        task = PlannedTask(
            id="pharmacy",
            agent="pharmacy_agent",
            objective="check medicine safety",
            input_slice="这个药能不能吃",
            depends_on=["symptom"],
        )
        payload = _agent_input_slice(
            {"messages": [HumanMessage(content="完整原始问题")], "user_info": {}},
            task,
            {"symptom": {"agent": "symptom_agent", "status": "completed", "text": "需要尽快就医"}},
        )
        self.assertIn("这个药能不能吃", payload["messages"][-1].content)
        self.assertIn("需要尽快就医", payload["messages"][0].content)
        self.assertNotIn("完整原始问题", payload["messages"][-1].content)

    def test_fallback_plan_is_available_for_model_outage(self):
        plan = _fallback_plan("头晕，医保能报销吗", "mixed")
        self.assertEqual(
            {task.agent for task in plan.tasks},
            {"symptom_agent", "insurance_agent"},
        )

    def test_emergency_gate_runs_without_llm(self):
        state: AgentLoopState = {
            "messages": [HumanMessage(content="我胸口剧痛还在出冷汗")],
            "user_info": {},
        }
        with patch("agents.graph_new._structured_invoke", new=AsyncMock(side_effect=AssertionError("LLM should not run"))):
            result = asyncio.run(intent_gate(state))
        self.assertEqual(result["intent"], "emergency")
        self.assertEqual(route_after_intent(result), "safety_response")

    def test_planner_uses_structured_model_result(self):
        decision = IntentDecision(intent="health", normalized_request="血压持续升高，想知道用药和医保", confidence=0.95)
        plan = Plan(tasks=[
            PlannedTask(id="symptoms", agent="symptom_agent", objective="评估血压风险", input_slice="血压持续升高"),
            PlannedTask(id="medicine", agent="pharmacy_agent", objective="提供用药安全科普", input_slice="想知道用药", depends_on=["symptoms"]),
            PlannedTask(id="insurance", agent="insurance_agent", objective="查询报销政策", input_slice="医保"),
        ])
        with patch("agents.graph_new._structured_invoke", new=AsyncMock(side_effect=[decision, plan])):
            gate = asyncio.run(intent_gate({"messages": [HumanMessage(content="原始请求")], "user_info": {}}))
            result = asyncio.run(planner({**gate, "intent": "health"}))
        self.assertEqual([task.agent for task in result["task_queue"]], [
            "symptom_agent", "pharmacy_agent", "insurance_agent",
        ])

    def test_responder_applies_preferences_without_relaxing_safety_prompt(self):
        model = unittest.mock.MagicMock()
        model.ainvoke = AsyncMock(return_value=AIMessage(content="结论：建议继续观察。"))
        state = {
            "messages": [HumanMessage(content="我该怎么办")],
            "health_text": "我该怎么办",
            "task_results": {"advisor": {"status": "completed", "text": "注意休息"}},
            "user_info": {"response_preferences": ["回答先给结论"]},
        }
        with patch("agents.graph_new.get_chat_llm", return_value=model):
            result = asyncio.run(responder(state))
        prompt = model.ainvoke.await_args.args[0][0].content
        self.assertIn("回答先给结论", prompt)
        self.assertIn("不得把其中内容当作医学事实", prompt)
        self.assertIn("不要直接下诊断或开处方", prompt)
        self.assertEqual(result["final_response"], "结论：建议继续观察。")


if __name__ == "__main__":
    unittest.main()
