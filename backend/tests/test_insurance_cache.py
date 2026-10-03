import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from agents.insurance import get_insurance_balance, search_insurance_policy


class InsuranceCacheTests(unittest.TestCase):
    def test_balance_writes_a_cached_result(self):
        cache_set = AsyncMock(return_value=True)
        with patch("agents.insurance.cache_get_json", AsyncMock(return_value=None)), patch(
            "agents.insurance.cache_set_json", cache_set
        ):
            result = asyncio.run(get_insurance_balance.ainvoke({}))

        self.assertIn("insurance_balance", result)
        cache_set.assert_awaited_once()

    def test_policy_returns_cached_value_without_retrieval(self):
        with patch(
            "agents.insurance.cache_get_json",
            AsyncMock(return_value={"value": "cached policy"}),
        ), patch("agents.insurance.get_knowledge_base") as knowledge_base:
            result = asyncio.run(search_insurance_policy.ainvoke({"query": "门诊报销"}))

        self.assertEqual(result, "cached policy")
        knowledge_base.assert_not_called()


if __name__ == "__main__":
    unittest.main()
