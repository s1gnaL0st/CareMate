import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from cache import cache_key, enforce_rate_limit
from config import get_settings


class CacheTests(unittest.TestCase):
    def test_cache_key_is_versioned_and_does_not_expose_query(self):
        key = cache_key("rag", "患者的血压是多少")
        self.assertIn(":rag:v", key)
        self.assertNotIn("患者", key)

    def test_rate_limit_rejects_after_threshold(self):
        class Client:
            def __init__(self):
                self.count = 0

            async def incr(self, _key):
                self.count += 1
                return self.count

            async def expire(self, *_args):
                return True

            async def ttl(self, _key):
                return 12

        request = type("Request", (), {"client": type("ClientInfo", (), {"host": "127.0.0.1"})()})()
        client = Client()
        with patch("cache.redis_client", client), patch.object(get_settings(), "rate_limit_requests", 1, create=True):
            asyncio.run(enforce_rate_limit(request, bucket="test"))
            with self.assertRaises(Exception) as ctx:
                asyncio.run(enforce_rate_limit(request, bucket="test"))
            self.assertEqual(ctx.exception.status_code, 429)

    def test_rate_limit_fails_open_when_redis_is_down(self):
        client = AsyncMock()
        client.incr.side_effect = RuntimeError("redis down")
        request = type("Request", (), {"client": None})()
        with patch("cache.redis_client", client):
            asyncio.run(enforce_rate_limit(request, bucket="test"))


if __name__ == "__main__":
    unittest.main()
