import asyncio
import json
import unittest

from langchain_core.messages import AIMessage, HumanMessage

from agents.clinic_action_adapter import (
    CLINIC_ACTION_TOOLS,
    ClinicActionAdapter,
    sanitize_clinic_answer,
)
from skills.emergency_triage.skill import EmergencyTriageSkill


class ClinicActionAdapterTests(unittest.TestCase):
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
