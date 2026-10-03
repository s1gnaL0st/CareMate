import unittest

from memory_evolution import extract_explicit_preferences


class MemoryEvolutionTests(unittest.TestCase):
    def test_extracts_only_explicit_communication_preferences(self):
        values = extract_explicit_preferences("以后请说简单一点，按步骤讲，先给结论。")
        self.assertEqual({item["key"] for item in values}, {
            "prefer_plain_language", "prefer_step_by_step", "prefer_conclusion_first",
        })
        self.assertTrue(all(item["source"] == "user_explicit" for item in values))

    def test_does_not_infer_health_fact_as_preference(self):
        self.assertEqual(extract_explicit_preferences("这几天一直咳嗽，可能是肺炎"), [])


if __name__ == "__main__":
    unittest.main()
