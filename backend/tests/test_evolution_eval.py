import json
import tempfile
import unittest
from pathlib import Path

from evolution_eval import (
    CandidateCase,
    create_evaluation_campaign,
    evaluate_frozen_dataset,
    evaluate_promotion,
    load_comparison_cases,
    persist_evaluation,
    persist_frozen_dataset_evaluation,
)


class EvolutionEvaluationTests(unittest.IsolatedAsyncioTestCase):
    def test_safety_and_route_are_hard_failures(self):
        gate = evaluate_promotion([
            CandidateCase("safe", 0.8, 0.9, True, True),
            CandidateCase("boundary", 1.0, 1.0, True, True, safety_violations=1, route_regression=True),
        ])
        self.assertFalse(gate.passed)
        self.assertEqual(gate.reason_codes, ("safety_violations", "route_regressions"))

    def test_score_and_resource_budgets_gate_candidate(self):
        gate = evaluate_promotion([
            CandidateCase("a", 1.0, 0.9, True, False, cost_delta=0.4, latency_delta=0.3),
        ])
        self.assertFalse(gate.passed)
        self.assertIn("score_regression", gate.reason_codes)
        self.assertIn("cost_budget_exceeded", gate.reason_codes)
        self.assertIn("latency_budget_exceeded", gate.reason_codes)

    def test_load_jsonl_validates_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "comparison.jsonl"
            row = {"id": "case-1", "baseline_score": 1, "candidate_score": 1}
            path.write_text(json.dumps(row) + "\n" + json.dumps(row), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_comparison_cases(path)

    async def test_persist_evaluation_stores_gate_metrics(self):
        from unittest.mock import AsyncMock, MagicMock
        db = MagicMock()
        db.flush = AsyncMock()
        evaluation = await persist_evaluation(
            db,
            proposal_id="proposal-1",
            dataset_name="holdout-v1",
            cases=[CandidateCase("a", 1.0, 1.0, True, True)],
        )
        self.assertTrue(evaluation.passed)
        self.assertEqual(evaluation.metrics["candidate_pass_rate"], 1.0)
        self.assertEqual(evaluation.evidence_mode, "manual")
        db.flush.assert_awaited_once()

    async def test_frozen_dataset_requires_exact_case_coverage_and_orders_results(self):
        from unittest.mock import AsyncMock, MagicMock
        from models import EvaluationDataset, EvaluationDatasetCase

        dataset = EvaluationDataset(
            id="dataset-1", name="holdout", version="v1", suite="holdout",
            fingerprint="f" * 64, case_count=2, status="frozen",
        )
        snapshots = [
            EvaluationDatasetCase(
                id="snapshot-a", dataset_id=dataset.id, source_candidate_id="candidate-a",
                case_key="a", prompt_redacted="a", expected_constraints={},
            ),
            EvaluationDatasetCase(
                id="snapshot-b", dataset_id=dataset.id, source_candidate_id="candidate-b",
                case_key="b", prompt_redacted="b", expected_constraints={},
            ),
        ]
        scalar_rows = MagicMock()
        scalar_rows.__iter__.side_effect = lambda: iter(snapshots)
        db = MagicMock()
        db.get = AsyncMock(return_value=dataset)
        db.scalars = AsyncMock(return_value=scalar_rows)

        bundle = await evaluate_frozen_dataset(
            db,
            dataset_id=dataset.id,
            cases=[
                CandidateCase("b", 0.8, 0.9, True, True),
                CandidateCase("a", 0.7, 0.8, True, True),
            ],
        )
        self.assertEqual([case.id for case in bundle.cases], ["a", "b"])
        self.assertTrue(bundle.gate.passed)

        with self.assertRaisesRegex(ValueError, "exactly cover"):
            await evaluate_frozen_dataset(
                db,
                dataset_id=dataset.id,
                cases=[CandidateCase("a", 0.7, 0.8, True, True)],
            )

    async def test_dataset_verified_evaluation_persists_per_case_evidence(self):
        from unittest.mock import AsyncMock, MagicMock
        from models import EvaluationDataset, EvaluationDatasetCase

        dataset = EvaluationDataset(
            id="dataset-1", name="safety", version="v1", suite="safety_boundary",
            fingerprint="d" * 64, case_count=1, status="frozen",
        )
        snapshot = EvaluationDatasetCase(
            id="snapshot-1", dataset_id=dataset.id, source_candidate_id="candidate-1",
            case_key="case-1", prompt_redacted="safe", expected_constraints={},
        )
        scalar_rows = MagicMock()
        scalar_rows.__iter__.side_effect = lambda: iter([snapshot])
        db = MagicMock()
        db.get = AsyncMock(return_value=dataset)
        db.scalars = AsyncMock(return_value=scalar_rows)
        db.flush = AsyncMock()

        evaluation, results = await persist_frozen_dataset_evaluation(
            db,
            proposal_id="proposal-1",
            dataset_id=dataset.id,
            cases=[CandidateCase("case-1", 1.0, 1.0, True, True)],
        )
        self.assertEqual(evaluation.evidence_mode, "dataset_verified")
        self.assertEqual(evaluation.dataset_fingerprint, dataset.fingerprint)
        self.assertEqual(results[0].dataset_case_id, snapshot.id)
        self.assertEqual(results[0].skill_evaluation_id, evaluation.id)
        self.assertIsNone(results[0].template_evaluation_id)
        db.flush.assert_awaited_once()

    async def test_promotion_campaign_requires_all_three_verified_suites(self):
        from unittest.mock import AsyncMock, MagicMock
        from models import EvaluationDataset, SkillEvaluation

        suites = ("regression", "holdout", "safety_boundary")
        datasets = [
            EvaluationDataset(
                id=f"dataset-{suite}", name=suite, version="v1", suite=suite,
                fingerprint=(suite[0] * 64), case_count=1, status="frozen",
            )
            for suite in suites
        ]
        metrics = {
            "baseline_score": 0.9, "candidate_score": 0.95,
            "baseline_pass_rate": 0.9, "candidate_pass_rate": 1.0,
            "safety_violations": 0, "route_regressions": 0,
            "cost_delta": 0.1, "latency_delta": 0.1,
        }
        evaluations = [
            SkillEvaluation(
                id=f"evaluation-{dataset.suite}", proposal_id="proposal-1",
                dataset_name=f"{dataset.name}@v1", dataset_id=dataset.id,
                dataset_fingerprint=dataset.fingerprint, evidence_mode="dataset_verified",
                case_count=1, passed=True, metrics=metrics, reason_codes=[],
            )
            for dataset in datasets
        ]
        evaluation_rows = MagicMock()
        evaluation_rows.__iter__.side_effect = lambda: iter(evaluations)
        dataset_rows = MagicMock()
        dataset_rows.__iter__.side_effect = lambda: iter(datasets)
        db = MagicMock()
        db.scalars = AsyncMock(side_effect=[evaluation_rows, dataset_rows])
        db.scalar = AsyncMock(side_effect=[1, 1, 1, None])
        db.flush = AsyncMock()

        campaign = await create_evaluation_campaign(
            db,
            proposal_id="proposal-1",
            evaluation_ids=[item.id for item in reversed(evaluations)],
        )
        self.assertTrue(campaign.passed)
        self.assertEqual(campaign.required_suites, list(suites))
        self.assertEqual(
            campaign.evaluation_ids,
            [f"evaluation-{suite}" for suite in suites],
        )
        db.flush.assert_awaited_once()

        costly_metrics = {**metrics, "cost_delta": 0.3}
        costly_evaluations = [
            SkillEvaluation(
                id=f"costly-{dataset.suite}", proposal_id="proposal-1",
                dataset_name=f"{dataset.name}@v1", dataset_id=dataset.id,
                dataset_fingerprint=dataset.fingerprint, evidence_mode="dataset_verified",
                case_count=1, passed=True, metrics=costly_metrics, reason_codes=[],
            )
            for dataset in datasets
        ]
        costly_rows = MagicMock()
        costly_rows.__iter__.side_effect = lambda: iter(costly_evaluations)
        dataset_rows_again = MagicMock()
        dataset_rows_again.__iter__.side_effect = lambda: iter(datasets)
        db.scalars = AsyncMock(side_effect=[costly_rows, dataset_rows_again])
        db.scalar = AsyncMock(side_effect=[1, 1, 1, None])
        costly_campaign = await create_evaluation_campaign(
            db,
            proposal_id="proposal-1",
            evaluation_ids=[item.id for item in costly_evaluations],
        )
        self.assertFalse(costly_campaign.passed)
        self.assertIn("holdout:cost_budget_exceeded", costly_campaign.reason_codes)

    async def test_promotion_campaign_rejects_manual_or_incomplete_evidence(self):
        from unittest.mock import AsyncMock, MagicMock
        from models import SkillEvaluation

        evaluations = [
            SkillEvaluation(
                id=f"evaluation-{index}", proposal_id="proposal-1", dataset_name="claimed",
                evidence_mode="manual", case_count=1, passed=True, metrics={}, reason_codes=[],
            )
            for index in range(3)
        ]
        rows = MagicMock()
        rows.__iter__.side_effect = lambda: iter(evaluations)
        db = MagicMock()
        db.scalars = AsyncMock(return_value=rows)
        with self.assertRaisesRegex(ValueError, "dataset-verified"):
            await create_evaluation_campaign(
                db,
                proposal_id="proposal-1",
                evaluation_ids=[item.id for item in evaluations],
            )


if __name__ == "__main__":
    unittest.main()
