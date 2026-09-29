import unittest
from unittest.mock import patch

from evals.agent_quality import (
    score_evidence_faithfulness,
    score_plan_quality,
    validate_plan,
    validate_reviewed_benchmark,
    judge_execution_offline,
)


class AgentQualityTests(unittest.TestCase):
    def test_plan_detects_missing_dependency_and_cycle(self):
        self.assertFalse(validate_plan([{"id": "a", "agent": "report_agent", "depends_on": ["missing"]}])["valid"])
        self.assertTrue(validate_plan([
            {"id": "a", "agent": "report_agent", "depends_on": ["b"]},
            {"id": "b", "agent": "pharmacy_agent", "depends_on": ["a"]},
        ])["cyclic"])

    def test_plan_quality_scores_expected_agents_and_edge(self):
        result = score_plan_quality({"task_queue": [
            {"id": "symptom", "agent": "symptom_agent", "depends_on": []},
            {"id": "drug", "agent": "pharmacy_agent", "depends_on": ["symptom"]},
        ]}, {"expected_agents": ["symptom_agent", "pharmacy_agent"],
             "expected_dependencies": [["symptom", "drug"]]})
        self.assertTrue(result["passed"])
        self.assertEqual(result["quality_score"], 1.0)

    def test_missing_source_text_is_not_claimed_as_faithful(self):
        result = score_evidence_faithfulness({"final_response": "请立即就医。", "task_results": {
            "t": {"status": "completed", "summary": "请立即就医。", "evidence": []}
        }})
        self.assertEqual(result["status"], "measured_provenance_only")
        self.assertIsNone(result["lexical_support_rate"])
        self.assertEqual(result["provenance_coverage"], 0.0)

    def test_benchmark_cannot_freeze_without_source_privacy_and_review_metadata(self):
        result = validate_reviewed_benchmark([{"id": "only-one", "suite": "safety_boundary"}])
        self.assertEqual(result["status"], "not_freeze_eligible")
        reasons = {item["reason"] for item in result["errors"]}
        self.assertIn("required_provenance_missing", reasons)
        self.assertIn("must_be_true", reasons)
        self.assertIn("human_label_required", reasons)

    def test_judge_marks_faithfulness_unassessable_without_source_text(self):
        class FakeJudgment:
            content = '{"plan_score":0.8,"plan_issues":[],"claim_judgments":[{"claim":"未标注的主张"}],"answer_completeness":{"score":0.7,"notes":"覆盖基本问题"},"overall_issues":[]}'

        class FakeLLM:
            async def ainvoke(self, *_args, **_kwargs):
                return FakeJudgment()

        async def run():
            with patch("agents.llm.get_chat_llm", return_value=FakeLLM()):
                return await judge_execution_offline("问题", {"task_results": {
                    "t": {"status": "completed", "evidence": [{"source_ids": ["s1"], "text": "来源原文"}]}
                }})

        import asyncio
        result = asyncio.run(run())
        self.assertTrue(result["evidence_text_available"])
        self.assertFalse(result["faithfulness_assessable"])
        self.assertIsNone(result["supported_claim_rate"])
        self.assertEqual(result["unadjudicated_claim_count"], 1)
        self.assertEqual(result["answer_completeness"], 0.7)
        self.assertEqual(result["judge_status"], "partial_output")

    def test_judge_normalizes_ten_point_scores_and_rejects_unknown_citations(self):
        class FakeJudgment:
            content = (
                '{"plan_score":8,"plan_issues":[],"claim_judgments":['
                '{"claim":"白细胞偏高","supported":true,"support_level":"full",'
                '"evidence_ids":["invented"],"reason":"证据支持"}],'
                '"answer_completeness":{"score":9,"notes":"完整"},"overall_issues":[]}'
            )

        class FakeLLM:
            async def ainvoke(self, *_args, **_kwargs):
                return FakeJudgment()

        async def run():
            with patch("agents.llm.get_chat_llm", return_value=FakeLLM()):
                return await judge_execution_offline("问题", {"task_results": {
                    "t": {"status": "completed", "evidence": [{"source_ids": ["s1"], "text": "血常规参考"}]}
                }})

        import asyncio
        result = asyncio.run(run())
        self.assertEqual(result["judge_status"], "invalid_output")
        self.assertEqual(result["judge_error_type"], "ValueError")
        self.assertFalse(result["faithfulness_assessable"])

    def test_judge_accepts_ten_point_scores_with_traceable_citation(self):
        class FakeJudgment:
            content = (
                '{"plan_score":8,"plan_issues":[],"claim_judgments":['
                '{"claim":"白细胞参考范围见来源","supported":true,"support_level":"full",'
                '"evidence_ids":["s1"],"reason":"来源包含对应参考范围"}],'
                '"answer_completeness":{"score":9},"overall_issues":[]}'
            )

        class FakeLLM:
            async def ainvoke(self, *_args, **_kwargs):
                return FakeJudgment()

        async def run():
            with patch("agents.llm.get_chat_llm", return_value=FakeLLM()):
                return await judge_execution_offline("问题", {"task_results": {
                    "t": {"status": "completed", "evidence": [{"source_ids": ["s1"], "text": "白细胞参考范围"}]}
                }})

        import asyncio
        result = asyncio.run(run())
        self.assertEqual(result["plan_score"], 0.8)
        self.assertEqual(result["answer_completeness"], 0.9)
        self.assertEqual(result["supported_claim_rate"], 1.0)
        self.assertTrue(result["judge_output_complete"])


if __name__ == "__main__":
    unittest.main()
