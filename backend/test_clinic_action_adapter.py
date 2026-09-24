import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agents.clinic import (
    CLINIC_SYSTEM_PROMPT,
    _invoke_clinic_model,
    _clinic_model_history,
    _normalize_triage_level,
    _visible_ask_text,
    clinic_node,
)
from agents.clinic_action_adapter import (
    CLINIC_ACTION_TOOLS,
    ClinicActionAdapter,
    answered_facets,
    clinic_harness_state,
    fallback_clinic_answer,
    sanitize_clinic_answer,
)
from skills.emergency_triage.skill import EmergencyTriageSkill


class ClinicActionAdapterTests(unittest.TestCase):
    def test_visible_ask_text_keeps_one_question(self):
        text = _visible_ask_text(
            "听起来确实不舒服。可能和牙齿敏感有关。\n是否肿胀？\n是否发热？",
            "是否肿胀？",
            set(),
        )
        self.assertEqual(text.count("？"), 1)
        self.assertTrue(text.endswith("是否肿胀？"))
        self.assertIn("听起来确实不舒服", text)

    def test_numeric_v2_triage_level_is_normalized(self):
        self.assertEqual(_normalize_triage_level(0), "emergency")
        self.assertEqual(_normalize_triage_level("1"), "urgent")
        self.assertEqual(_normalize_triage_level(2), "routine")
        self.assertEqual(_normalize_triage_level("3"), "self_care")
        self.assertEqual(_normalize_triage_level("not-a-level"), "not-a-level")

    def test_clinic_prompt_does_not_make_onset_question_a_default(self):
        self.assertIn("不要使用固定的首问或问题清单", CLINIC_SYSTEM_PROMPT)
        self.assertIn("不要默认询问", CLINIC_SYSTEM_PROMPT)

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
                captured.append(list(messages))
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

        self.assertEqual(sum(isinstance(message, SystemMessage) for message in captured[0]), 1)
        self.assertIn("最近对话：用户说牙齿酸", captured[0][0].content)
        self.assertIn("具体是哪颗牙齿发酸？", result["messages"][0].content)
        self.assertEqual(result["messages"][0].tool_calls[0]["id"], "call-test")

    def test_exposes_only_the_five_trained_actions_with_full_parameters(self):
        self.assertEqual(
            {item["function"]["name"] for item in CLINIC_ACTION_TOOLS},
            {"ask", "check", "lookup", "search", "answer"},
        )

        expected_arguments = {
            "ask": "question",
            "check": "check_type",
            "lookup": "field",
            "search": "query",
            "answer": "triage_level",
        }
        for item in CLINIC_ACTION_TOOLS:
            function = item["function"]
            parameters = function["parameters"]
            argument = expected_arguments[function["name"]]
            self.assertEqual(parameters["type"], "object")
            if function["name"] == "check":
                self.assertEqual(set(parameters["properties"]), {"check_type", "items"})
                self.assertEqual(parameters["required"], ["check_type", "items"])
            elif function["name"] == "answer":
                self.assertEqual(set(parameters["properties"]), {"triage_level", "content"})
                self.assertEqual(parameters["required"], ["triage_level", "content"])
            else:
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

    def test_ask_rejects_question_when_all_information_dimensions_were_answered(self):
        adapter = ClinicActionAdapter(
            state={},
            messages=[
                HumanMessage(content="我牙齿有点酸。"),
                AIMessage(content="症状什么时候开始的？"),
                HumanMessage(content="昨天晚上突然开始。"),
            ],
        )
        duplicate = json.loads(
            asyncio.run(adapter.execute("ask", {"question": "什么时候开始的？"}))
        )
        self.assertFalse(duplicate["success"])
        self.assertEqual(duplicate["error"], "duplicate_question")

    def test_negative_answer_covers_the_previous_safety_question(self):
        adapter = ClinicActionAdapter(
            state={},
            messages=[
                HumanMessage(content="我的牙齿有点疼。"),
                AIMessage(content="是否出现面部肿胀、张口困难、吞咽或呼吸困难？"),
                HumanMessage(content="没有。"),
            ],
        )
        result = json.loads(asyncio.run(adapter.execute(
            "ask", {"question": "是否出现持续剧痛、无法吞咽、流口水或意识改变？"}
        )))
        self.assertFalse(result["success"])
        self.assertEqual(result["error"], "duplicate_question")

    def test_duplicate_ask_is_returned_to_model_for_recovery(self):
        class RecoveryModel:
            def __init__(self):
                self.calls = []
                self.turn = 0

            def bind_tools(self, *_args, **_kwargs):
                return self

            async def ainvoke(self, messages):
                self.calls.append(list(messages))
                self.turn += 1
                if self.turn == 1:
                    return AIMessage(content="", tool_calls=[{
                        "name": "ask", "args": {"question": "是否出现面部肿胀？"},
                        "id": "call-first", "type": "tool_call",
                    }])
                if self.turn == 2:
                    return AIMessage(content="", tool_calls=[{
                        "name": "ask", "args": {"question": "是否出现吞咽困难？"},
                        "id": "call-duplicate", "type": "tool_call",
                    }])
                duplicate_error = next(
                    message for message in messages
                    if getattr(message, "type", "") == "tool"
                    and "duplicate_question" in message.content
                )
                assert duplicate_error
                return AIMessage(content="", tool_calls=[{
                    "name": "answer", "args": {
                        "triage_level": "self_care",
                        "content": "目前没有发现已知危险信号，可先预约牙科评估。",
                    }, "id": "call-answer", "type": "tool_call",
                }])

        model = RecoveryModel()
        state = {
            "messages": [
                HumanMessage(content="我的牙齿有点疼"),
                AIMessage(content="是否出现面部肿胀？", tool_calls=[{
                    "name": "ask", "args": {"question": "是否出现面部肿胀？"}, "id": "old", "type": "tool_call",
                }]),
                HumanMessage(content="没有"),
            ],
            "user_info": {},
        }
        with patch("agents.clinic.get_clinic_llm", return_value=model):
            result = asyncio.run(clinic_node(state))
        self.assertIn("没有发现已知危险信号", result["messages"][0].content)

    def test_compound_question_can_still_ask_unanswered_dimension(self):
        adapter = ClinicActionAdapter(
            state={},
            messages=[HumanMessage(content="昨天晚上突然开始。")],
        )
        result = json.loads(
            asyncio.run(adapter.execute("ask", {"question": "症状是否持续加重？"}))
        )
        self.assertTrue(result["success"])

    def test_compound_question_is_reduced_to_unanswered_clause(self):
        adapter = ClinicActionAdapter(
            state={},
            messages=[HumanMessage(content="昨天晚上突然开始。")],
        )
        result = json.loads(
            asyncio.run(adapter.execute("ask", {"question": "是否突然发生或持续加重？"}))
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["question"], "症状是否持续加重？")

    def test_compound_question_never_exposes_a_fragment(self):
        adapter = ClinicActionAdapter(
            state={},
            messages=[HumanMessage(content="昨天开始，日常活动正常。")],
        )
        result = json.loads(asyncio.run(adapter.execute(
            "ask", {"question": "是否突然发生或是否影响日常活动？"}
        )))
        self.assertTrue(result["success"])
        self.assertNotEqual(result["question"], "日常活动？")
        self.assertTrue(result["question"].endswith("？"))

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

    def test_sanitize_clinic_answer_removes_thinking_blocks(self):
        cleaned = sanitize_clinic_answer(
            "<think>内部推理，不应展示</think>牙齿酸可能与龋齿有关。",
            set(),
        )
        self.assertEqual(cleaned, "牙齿酸可能与龋齿有关。")

    def test_sanitize_clinic_answer_removes_section_labels_and_escaped_think(self):
        cleaned = sanitize_clinic_answer(
            "第一段：牙齿酸可能与龋齿有关。\\</think>第二段：建议观察。",
            set(),
        )
        self.assertNotIn("第一段", cleaned)
        self.assertNotIn("第二段", cleaned)
        self.assertNotIn("think", cleaned.lower())

    def test_sanitize_clinic_answer_preserves_model_wording(self):
        cleaned = sanitize_clinic_answer(
            "初步可能范围：牙齿敏感、龋齿；风险等级：一般；下一步：预约牙科治疗。",
            set(),
        )
        self.assertEqual(
            cleaned,
            "初步可能范围：牙齿敏感、龋齿；风险等级：一般；下一步：预约牙科治疗。",
        )
        self.assertIn("风险等级：", cleaned)

    def test_simulated_patient_turns_cover_dimensions_without_repeating_questions(self):
        messages = [HumanMessage(content="我有点不舒服。")]
        adapter = ClinicActionAdapter(state={}, messages=messages)

        first = json.loads(asyncio.run(adapter.execute("ask", {"question": "哪里不舒服？"})))
        self.assertTrue(first["success"])
        messages.extend([AIMessage(content=first["question"]), HumanMessage(content="胸口中央。")])

        second_adapter = ClinicActionAdapter(state={}, messages=messages)
        second = json.loads(asyncio.run(second_adapter.execute("ask", {"question": "是否突然发生或持续加重？"})))
        self.assertTrue(second["success"])
        messages.extend([AIMessage(content=second["question"]), HumanMessage(content="没有加重。")])

        third_adapter = ClinicActionAdapter(state={}, messages=messages)
        repeated = json.loads(asyncio.run(third_adapter.execute("ask", {"question": "胸口哪里不舒服？"})))
        self.assertFalse(repeated["success"])
        self.assertEqual(repeated["error"], "duplicate_question")

    def test_patient_answer_after_ask_is_replayed_as_tool_result(self):
        history = _clinic_model_history([
            HumanMessage(content="我的牙齿有点酸"),
            AIMessage(
                content="症状什么时候开始？",
                tool_calls=[{"name": "ask", "id": "call-ask", "args": {"question": "症状什么时候开始？"}}],
            ),
            HumanMessage(content="昨天晚上开始"),
        ])
        self.assertEqual(history[-1].type, "tool")
        self.assertEqual(history[-1].tool_call_id, "call-ask")
        self.assertEqual(history[-1].content, "昨天晚上开始")

    def test_simulated_two_turn_loop_reaches_answer_after_tool_replay(self):
        """The second model call must see the patient's reply as role=tool."""
        class TwoTurnModel:
            def __init__(self):
                self.calls = []
                self.turn = 0

            def bind_tools(self, *_args, **_kwargs):
                return self

            async def ainvoke(self, messages):
                self.calls.append(list(messages))
                self.turn += 1
                if self.turn == 1:
                    return AIMessage(content="", tool_calls=[{
                        "name": "ask", "args": {"question": "疼了多久？"},
                        "id": "call-1", "type": "tool_call",
                    }])
                self.assert_tool_reply(messages)
                return AIMessage(content="", tool_calls=[{
                    "name": "answer", "args": {"triage_level": "self_care", "content": "初步考虑牙齿敏感，建议预约牙科评估。"},
                    "id": "call-2", "type": "tool_call",
                }])

            def assert_tool_reply(self, messages):
                self_reply = next((m for m in messages if getattr(m, "type", "") == "tool"), None)
                assert self_reply is not None
                assert self_reply.tool_call_id == "call-1"
                assert self_reply.content == "昨天晚上开始"

        model = TwoTurnModel()
        state = {
            "messages": [HumanMessage(content="我的牙齿有点酸")],
            "user_info": {},
        }
        with patch("agents.clinic.get_clinic_llm", return_value=model):
            first = asyncio.run(clinic_node(state))
            state["messages"] = state["messages"] + [first["messages"][0], HumanMessage(content="昨天晚上开始")]
            second = asyncio.run(clinic_node(state))

        self.assertEqual(first["clinic_pending_tool_call"]["id"], "call-1")
        self.assertIn("初步考虑牙齿敏感", second["messages"][0].content)
        # The first ask turn may use one additional tool-free expression call;
        # the second clinic turn must still see the original tool reply.
        self.assertGreaterEqual(model.turn, 2)

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
            asyncio.run(adapter.execute("check", {"check_type": "drug_interaction", "items": ["布洛芬", "阿司匹林"]}))
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
