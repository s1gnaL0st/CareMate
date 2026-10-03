import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException

from evolution_api import (
    EvaluationDatasetRequest,
    EvaluationRunRequest,
    ReviewRequest,
    _require_reviewer,
    create_evaluation_run,
    review_proposal,
    router,
)
from models import EvaluationDataset, SkillEvaluation, SkillProposal


class EvolutionApiContractTests(unittest.TestCase):
    def test_router_only_exposes_compact_evolution_workflow(self):
        routes = {(route.path, method) for route in router.routes for method in route.methods}
        expected = {
            ("/api/v1/evolution/failures", "GET"),
            ("/api/v1/evolution/failures/clusters", "GET"),
            ("/api/v1/evolution/proposals", "GET"),
            ("/api/v1/evolution/proposals/from-failures", "POST"),
            ("/api/v1/evolution/proposals/from-failures/llm", "POST"),
            ("/api/v1/evolution/proposals/{proposal_id}", "GET"),
            ("/api/v1/evolution/evaluation-datasets", "POST"),
            ("/api/v1/evolution/evaluation-datasets", "GET"),
            ("/api/v1/evolution/evaluation-datasets/{dataset_id}", "GET"),
            ("/api/v1/evolution/proposals/{proposal_id}/evaluation-runs", "POST"),
            ("/api/v1/evolution/evaluation-runs/{run_id}", "GET"),
            ("/api/v1/evolution/proposals/{proposal_id}/review", "POST"),
        }
        self.assertEqual(routes, expected)
        self.assertFalse(any("campaign" in path or "template" in path for path, _ in routes))

    def test_review_payload_only_allows_audited_decisions(self):
        payload = ReviewRequest(
            decision="approved", reason="通过真实离线评测", evaluation_id="evaluation-1"
        )
        self.assertEqual(payload.decision, "approved")
        with self.assertRaises(ValueError):
            ReviewRequest(decision="published", reason="不应允许")

    def test_curated_dataset_requires_three_cases(self):
        case = {"prompt": "头痛怎么办", "expected_constraints": {}}
        with self.assertRaises(ValueError):
            EvaluationDatasetRequest(name="medical-v1", version="1", cases=[case, case])
        payload = EvaluationDatasetRequest(
            name="medical-v1",
            version="1",
            cases=[
                case,
                {"prompt": "胸痛并大汗", "expected_constraints": {"must_preserve_emergency_triage": True}},
                {"prompt": "解释血常规", "expected_constraints": {"expected_agent": "report_agent"}},
            ],
        )
        self.assertEqual(len(payload.cases), 3)

    def test_evaluation_run_requires_bounded_idempotency_key(self):
        payload = EvaluationRunRequest(dataset_id="dataset-1", idempotency_key="proposal-1:medical-v1")
        self.assertEqual(payload.dataset_id, "dataset-1")
        for key in ("short", "contains spaces"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                EvaluationRunRequest(dataset_id="dataset-1", idempotency_key=key)

    def test_production_reviewer_is_allowlist_based(self):
        user = MagicMock(email="reviewer@example.com")
        with patch("evolution_api.get_settings") as settings:
            settings.return_value.environment = "production"
            settings.return_value.evolution_reviewer_email_list = []
            with self.assertRaises(HTTPException) as raised:
                _require_reviewer(user)
        self.assertEqual(raised.exception.status_code, 403)


class EvaluationRunApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_queue_failure_is_persisted_and_redacted(self):
        proposal = SkillProposal(
            id="proposal-1",
            base_skill="triage",
            trigger="symptoms",
            content="candidate",
            status="candidate",
        )
        dataset = EvaluationDataset(
            id="dataset-1",
            name="medical",
            version="1",
            suite="medical",
            fingerprint="d" * 64,
            case_count=3,
            status="frozen",
        )
        db = MagicMock()
        db.scalar = AsyncMock(return_value=None)
        db.get = AsyncMock(side_effect=[proposal, dataset])
        db.commit = AsyncMock()

        async def assign_id(run):
            run.id = "run-1"

        db.refresh = AsyncMock(side_effect=assign_id)
        with patch("evolution_api._require_reviewer"), patch(
            "evolution_api.enqueue_evolution_evaluation",
            AsyncMock(side_effect=RuntimeError("redis unavailable for 13800138000")),
        ):
            with self.assertRaises(HTTPException) as raised:
                await create_evaluation_run(
                    proposal.id,
                    EvaluationRunRequest(
                        dataset_id=dataset.id,
                        idempotency_key="proposal-1:medical-v1",
                    ),
                    user=MagicMock(id="reviewer-1"),
                    db=db,
                )
        self.assertEqual(raised.exception.status_code, 503)
        run = db.add.call_args.args[0]
        self.assertEqual(run.status, "queue_failed")
        self.assertNotIn("13800138000", run.error_message)
        self.assertEqual(db.commit.await_count, 2)

    async def test_idempotency_key_cannot_be_reused_for_another_target(self):
        existing = MagicMock(proposal_id="proposal-other", dataset_id="dataset-1")
        db = MagicMock()
        db.scalar = AsyncMock(return_value=existing)
        with patch("evolution_api._require_reviewer"):
            with self.assertRaises(HTTPException) as raised:
                await create_evaluation_run(
                    "proposal-1",
                    EvaluationRunRequest(
                        dataset_id="dataset-1", idempotency_key="proposal-1:medical-v1"
                    ),
                    user=MagicMock(id="reviewer-1"),
                    db=db,
                )
        self.assertEqual(raised.exception.status_code, 409)


class ProposalReviewTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _proposal() -> SkillProposal:
        return SkillProposal(
            id="proposal-1",
            base_skill="triage",
            version="0.1.0",
            trigger="symptoms",
            content="candidate",
            source_failure_ids=["failure-1"],
            status="evaluating",
        )

    async def test_manual_pass_flag_cannot_approve_skill(self):
        proposal = self._proposal()
        evaluation = SkillEvaluation(
            id="evaluation-1",
            proposal_id=proposal.id,
            dataset_name="claimed-holdout",
            evidence_mode="manual",
            case_count=10,
            passed=True,
            metrics={},
            reason_codes=[],
        )
        db = MagicMock()
        db.get = AsyncMock(side_effect=[proposal, evaluation])
        db.scalar = AsyncMock(return_value=evaluation)
        db.commit = AsyncMock()
        with patch("evolution_api._require_reviewer"):
            with self.assertRaises(HTTPException) as raised:
                await review_proposal(
                    proposal.id,
                    ReviewRequest(
                        decision="approved",
                        reason="manual claim",
                        evaluation_id=evaluation.id,
                    ),
                    user=MagicMock(id="reviewer-1"),
                    db=db,
                )
        self.assertEqual(raised.exception.status_code, 409)
        db.commit.assert_not_awaited()

    async def test_approval_requires_latest_evaluation(self):
        proposal = self._proposal()
        selected = MagicMock(
            id="evaluation-1",
            proposal_id=proposal.id,
            passed=True,
            evidence_mode="dataset_verified",
            dataset_id="dataset-1",
        )
        latest = MagicMock(id="evaluation-2")
        db = MagicMock()
        db.get = AsyncMock(side_effect=[proposal, selected])
        db.scalar = AsyncMock(return_value=latest)
        with patch("evolution_api._require_reviewer"):
            with self.assertRaises(HTTPException) as raised:
                await review_proposal(
                    proposal.id,
                    ReviewRequest(
                        decision="approved", reason="stale run", evaluation_id=selected.id
                    ),
                    user=MagicMock(id="reviewer-1"),
                    db=db,
                )
        self.assertEqual(raised.exception.status_code, 409)

    async def test_verified_latest_evaluation_can_be_approved_but_not_published(self):
        proposal = self._proposal()
        evaluation = SkillEvaluation(
            id="evaluation-1",
            proposal_id=proposal.id,
            dataset_name="medical@1",
            dataset_id="dataset-1",
            dataset_fingerprint="d" * 64,
            evidence_mode="dataset_verified",
            case_count=3,
            passed=True,
            metrics={"safety_violations": 0, "route_regressions": 0},
            reason_codes=[],
        )
        dataset = EvaluationDataset(
            id="dataset-1",
            name="medical",
            version="1",
            suite="medical",
            fingerprint="d" * 64,
            case_count=3,
            status="frozen",
        )
        db = MagicMock()
        db.get = AsyncMock(side_effect=[proposal, evaluation, dataset])
        db.scalar = AsyncMock(side_effect=[evaluation, 3])
        db.commit = AsyncMock()
        db.refresh = AsyncMock()
        with patch("evolution_api._require_reviewer"):
            response = await review_proposal(
                proposal.id,
                ReviewRequest(
                    decision="approved",
                    reason="真实对照评测通过",
                    evaluation_id=evaluation.id,
                ),
                user=MagicMock(id="reviewer-1"),
                db=db,
            )
        self.assertEqual(proposal.status, "approved")
        self.assertFalse(response["published"])
        self.assertFalse(response["injected_into_planner"])
        db.commit.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
