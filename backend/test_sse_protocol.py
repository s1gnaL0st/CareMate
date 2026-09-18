import json
import unittest

from main import _sse_payload


class SseProtocolTests(unittest.TestCase):
    def test_payload_is_json_data_frame(self):
        frame = _sse_payload({"type": "token", "content": "你好\n世界"})
        self.assertTrue(frame.endswith("\n\n"))
        self.assertTrue(frame.startswith("data: "))
        body = frame[len("data: ") : -2]
        self.assertEqual(json.loads(body), {"type": "token", "content": "你好\n世界"})

    def test_payload_preserves_unicode_without_sse_injection(self):
        frame = _sse_payload({"type": "error", "content": "异常\r\ndata: forged"})
        body = frame[len("data: ") : -2]
        payload = json.loads(body)
        self.assertEqual(payload["content"], "异常\r\ndata: forged")
        self.assertNotIn("\ndata: forged\n\n", frame)


if __name__ == "__main__":
    unittest.main()
