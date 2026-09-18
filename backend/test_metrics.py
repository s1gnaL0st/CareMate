import unittest

from metrics import record_cache_event, record_request, record_task_event, render_prometheus


class MetricsTests(unittest.TestCase):
    def test_prometheus_output_contains_request_counter_and_latency(self):
        record_request("GET", "/health", 200, 0.012)
        payload = render_prometheus()
        self.assertIn("smart_health_http_requests_total", payload)
        self.assertIn('path="/health"', payload)
        self.assertIn("smart_health_http_request_duration_seconds_sum", payload)
        self.assertIn("smart_health_http_request_duration_seconds_bucket", payload)
        self.assertIn('le="+Inf"', payload)

    def test_cache_and_task_counters(self):
        record_cache_event("hit")
        record_task_event("dead_letter")
        payload = render_prometheus()
        self.assertIn("smart_health_cache_events_total", payload)
        self.assertIn("smart_health_task_events_total", payload)


if __name__ == "__main__":
    unittest.main()
