import unittest

from tasks import MAX_ANALYSIS_ATTEMPTS, analysis_failure_status


class TaskPolicyTests(unittest.TestCase):
    def test_final_analysis_attempt_is_dead_letter(self):
        self.assertEqual(analysis_failure_status(MAX_ANALYSIS_ATTEMPTS), "dead_letter")

    def test_intermediate_analysis_attempt_is_retryable(self):
        self.assertEqual(analysis_failure_status(1), "retrying")


if __name__ == "__main__":
    unittest.main()
