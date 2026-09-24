import unittest

from evals.run_self_evolution_eval import _summary


class SelfEvolutionReportTests(unittest.TestCase):
    def test_summary_computes_paired_deltas(self):
        rows = [
            {"baseline_score": 0.5, "candidate_score": 1.0, "baseline_passed": False,
             "candidate_passed": True, "safety_violations": 0, "route_regression": False,
             "cost_delta": 0.1, "latency_delta": 0.2, "baseline_degraded": False,
             "candidate_degraded": False},
            {"baseline_score": 1.0, "candidate_score": 1.0, "baseline_passed": True,
             "candidate_passed": True, "safety_violations": 0, "route_regression": False,
             "cost_delta": 0.0, "latency_delta": 0.0, "baseline_degraded": False,
             "candidate_degraded": False},
        ]
        summary = _summary(rows)
        self.assertEqual(summary["case_count"], 2)
        self.assertAlmostEqual(summary["score_delta"], 0.25)
        self.assertAlmostEqual(summary["pass_rate_delta"], 0.5)
        self.assertAlmostEqual(summary["mean_cost_delta"], 0.05)
        self.assertAlmostEqual(summary["mean_latency_delta"], 0.1)


if __name__ == "__main__":
    unittest.main()
