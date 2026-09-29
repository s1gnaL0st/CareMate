import unittest
from datetime import datetime, timedelta, timezone

from health_memory import (
    event_is_relevant,
    health_context,
    infer_health_event_observation,
    should_mark_stale,
)


class HealthMemoryTests(unittest.TestCase):
    def test_repeated_first_person_symptoms_become_active(self):
        item = infer_health_event_observation("我这几天一直咳嗽，还有流鼻涕")
        self.assertEqual(item.category, "respiratory_symptoms")
        self.assertEqual(item.status, "active")
        self.assertEqual(item.confirmation, "user_reported")

    def test_single_weak_signal_is_possible(self):
        item = infer_health_event_observation("咳嗽怎么办")
        self.assertEqual(item.status, "possible")
        self.assertEqual(item.confirmation, "inferred")

    def test_recovery_is_explicit(self):
        item = infer_health_event_observation("我现在已经完全好了，不咳了")
        self.assertEqual(item.status, "resolved")

    def test_family_report_is_not_written(self):
        self.assertIsNone(infer_health_event_observation("我妈妈这几天一直咳嗽"))

    def test_resolved_events_are_not_prompt_context(self):
        result = health_context([{"id": "1", "status": "resolved", "summary": "old"}])
        self.assertEqual(result["events"], [])

    def test_relevance_and_staleness(self):
        self.assertTrue(event_is_relevant({"status": "active", "category": "respiratory_symptoms"}, "咳嗽"))
        self.assertTrue(should_mark_stale(datetime.now(timezone.utc) - timedelta(days=15)))


if __name__ == "__main__":
    unittest.main()
