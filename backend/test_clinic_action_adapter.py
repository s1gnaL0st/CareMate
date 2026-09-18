import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agents.clinic import _invoke_clinic_model, clinic_node
from agents.clinic_action_adapter import (
    CLINIC_ACTION_TOOLS,
    ClinicActionAdapter,
    clinic_harness_state,
    fallback_clinic_answer,
    sanitize_clinic_answer,
)
from skills.emergency_triage.skill import EmergencyTriageSkill


class ClinicActionAdapterTests(unittest.TestCase):
    def test_harness_stops_after_bounded_question_turns(self):
        messages = [
            HumanMessage(content="我有点不舒服。"),
            AIMessage(content="症状什么时候开始的？"),
            HumanMessage(content="昨天。"),
            AIMessage(content="严重程度如何？"),
            HumanMessage(content="有点明显。"),
            AIMessage(content="还有其他伴随情况吗？"),
            HumanMessage(content="没有。"),
        ]
        state = clinic_harness_state(messages, max_question_turns=3)
        self.assertTrue(state.force_answer)
        self.assertEqual(state.question_turns, 3)

    def test_harness_fallback_is_a_safe_final_answer(self):
        answer = fallback_clinic_answer([HumanMessage(content="牙齿酸，持续加重")])
        self.assertIn("尽快线下就医", answer)
        self.assertIn("120", answer)

    def test_clinic_model_retries_transient_failure_with_bounded_attempts(self):
        class FlakyModel:
            def __init__(self):
                self.calls = 0

            async def ainvoke(self, _messages):
                self.calls += 1
                if self.calls < 3:
                    raise ConnectionError("connection reset by peer")
                return "ok"

        model = FlakyModel()
        result = asyncio.run(_invoke_clinic_model(
            model,
            [],
            SimpleNamespace(clinic_llm_max_retries=2, clinic_llm_retry_backoff_seconds=0),
        ))

        self.assertEqual(result, "ok")
        self.assertEqual(model.calls, 3)

    def test_clinic_model_does_not_retry_non_transient_request_error(self):
        class BadRequestModel:
            def __init__(self):
                self.calls = 0

            async def ainvoke(self, _messages):
                self.calls += 1
                raise ValueError("HTTP 400 invalid tool schema")

        model = BadRequestModel()
        with self.assertRaises(ValueError):
            asyncio.run(_invoke_clinic_model(
                model,
                [],
                SimpleNamespace(clinic_llm_max_retries=2, clinic_llm_retry_backoff_seconds=0),
            ))
        self.assertEqual(model.calls, 1)

    def test_clinic_model_receives_only_one_leading_system_message(self):
        captured = []

        class FakeModel:
            def bind_tools(self, *_args, **_kwargs):
                return self

            async def ainvoke(self, messages):
                captured.extend(messages)
                return AIMessage(
                    content="",
                    tool_calls=[{
                        "name": "ask",
                        "args": {"question": "具体是哪颗牙齿发酸？"},
                        "id": "call-test",
                        "type": "tool_call",
                    }],
                )

        state = {
            "messages": [
                SystemMessage(content="最近对话：用户说牙齿酸。"),
                HumanMessage(content="昨晚吃橘子后出现。"),
            ],
            "conversation_messages": [HumanMessage(content="我牙齿有点酸")],
            "user_info": {},
        }
        with patch("agents.clinic.get_clinic_llm", return_value=FakeModel()):
            result = asyncio.run(clinic_node(state))

        self.assertEqual(sum(isinstance(message, SystemMessage) for message in captured), 1)
        self.assertIn("最近对话：用户说牙齿酸", captured[0].content)
        self.assertEqual(result["messages"][0].content, "具体是哪颗牙齿发酸？")

    def test_exposes_only_the_five_trained_actions_with_full_parameters(self):
        self.assertEqual(
            {item["function"]["name"] for item in CLINIC_ACTION_TOOLS},
            {"ask", "check", "lookup", "search", "answer"},
        )

        expected_arguments = {
            "ask": "question",
            "check": "item",
            "lookup": "query",
            "search": "query",
            "answer": "content",
        }
        for item in CLINIC_ACTION_TOOLS:
            function = item["function"]
            parameters = function["parameters"]
            argument = expected_arguments[function["name"]]
            self.assertEqual(parameters["type"], "object")
            self.assertEqual(set(parameters["properties"]), {argument})
            self.assertEqual(parameters["required"], [argument])
            self.assertFalse(parameters["additionalProperties"])

    def test_ask_rejects_a_duplicate_question_but_accepts_a_new_one(self):
        adapter = ClinicActionAdapter(
            state={},
            messages=[
                HumanMessage(content="我最近头晕。"),
                AIMessage(content="请问症状持续多久了？"),
            ],
        )

        duplicate = json.loads(
            asyncio.run(adapter.execute("ask", {"question": "请问症状持续多久了？"}))
        )
        self.assertFalse(duplicate["success"])
        self.assertEqual(duplicate["error"], "duplicate_question")

        fresh = json.loads(
            asyncio.run(adapter.execute("ask", {"question": "有没有伴随视物旋转或呕吐？"}))
        )
        self.assertTrue(fresh["success"])
        self.assertEqual(fresh["question"], "有没有伴随视物旋转或呕吐？")

    def test_duplicate_ask_uses_next_bounded_question_instead_of_generic_loop(self):
        adapter = ClinicActionAdapter(
            state={},
            messages=[
                HumanMessage(content="我牙齿有点酸。"),
                AIMessage(content="症状是什么时候开始的，是否突然发生或持续加重？"),
                HumanMessage(content="昨天晚上吃了个橘子。"),
            ],
        )

        first = adapter.next_fallback_question()
        second = adapter.next_fallback_question()

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertNotEqual(first, second)
        self.assertNotIn("什么时候开始", first)

    def test_sanitize_clinic_answer_removes_training_and_untrusted_citations(self):
        answer = (
            "分诊：建议尽快就医。\n"
            "证据：ev_case_17 rag:local:triage rag:untrusted:secret pubmed:12345\n"
            "请结合医生面诊。"
        )
        cleaned = sanitize_clinic_answer(
            answer,
            {"rag:local:triage", "pubmed:12345"},
        )

        self.assertNotIn("ev_case_17", cleaned)
        self.assertNotIn("rag:untrusted:secret", cleaned)
        self.assertIn("rag:local:triage", cleaned)
        self.assertIn("pubmed:12345", cleaned)
        self.assertIn("建议尽快就医", cleaned)

    def test_emergency_veto_is_deterministic_and_precedes_model_actions(self):
        adapter = ClinicActionAdapter(
            state={"user_info": {}},
            messages=[HumanMessage(content="我突然剧烈胸痛，还一直出冷汗。")],
        )

        result = adapter.emergency_veto()

        self.assertEqual(result.level, "CRITICAL")
        self.assertTrue(result.call_ambulance)
        self.assertIn("120", result.safety_message)
        self.assertTrue(result.triggered_flags)

    def test_check_drug_interaction_normalizes_label_and_colon(self):
        adapter = ClinicActionAdapter(state={}, messages=[])
        result = json.loads(
            asyncio.run(adapter.execute("check", {"item": "药物相互作用：布洛芬,阿司匹林"}))
        )
        self.assertEqual(result["route"], "drug_interaction")
        self.assertEqual(result["result"]["drug1"], "布洛芬")
        self.assertEqual(result["result"]["drug2"], "阿司匹林")

    def test_emergency_triage_respects_negated_red_flags(self):
        result = EmergencyTriageSkill().run(
            symptoms_text="头晕两天，今天更明显，没有胸痛和呼吸困难。"
        )
        self.assertNotEqual(result.level, "CRITICAL")
        self.assertNotIn("严重呼吸困难", result.triggered_flags)


if __name__ == "__main__":
    unittest.main()
