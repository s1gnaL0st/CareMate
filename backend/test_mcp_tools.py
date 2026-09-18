import asyncio
import json
import unittest
from unittest.mock import patch

from agents.pharmacy import check_drug_interaction, search_drug_info
from agents.insurance import search_insurance_policy
from mcp_adapter import MCPUnavailableError, call_mcp_tool


class MCPToolTests(unittest.TestCase):
    def test_registered_drug_tool_returns_stable_payload(self):
        output = asyncio.run(call_mcp_tool("search_drug_info", drug_name="布洛芬"))
        payload = json.loads(output)
        self.assertEqual(payload["schema_version"], "1.0")
        self.assertTrue(payload["found"])

    def test_public_tool_falls_back_when_mcp_unavailable(self):
        with patch("agents.pharmacy.call_mcp_tool", side_effect=MCPUnavailableError()):
            output = asyncio.run(search_drug_info.ainvoke({"drug_name": "布洛芬"}))
        self.assertTrue(json.loads(output)["found"])

    def test_interaction_uses_mcp_registry(self):
        output = asyncio.run(
            check_drug_interaction.ainvoke({"drug1": "布洛芬", "drug2": "阿司匹林"})
        )
        self.assertTrue(json.loads(output)["has_interaction"])

    def test_policy_falls_back_when_mcp_disabled(self):
        with patch("agents.insurance.call_mcp_tool", side_effect=MCPUnavailableError()):
            with patch("agents.insurance._search_insurance_policy_local", return_value="fallback"):
                output = asyncio.run(search_insurance_policy.ainvoke({"query": "报销"}))
        self.assertEqual(output, "fallback")


if __name__ == "__main__":
    unittest.main()
