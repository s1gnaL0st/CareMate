"""Standalone MCP server exposing read-only health knowledge tools."""
import os
import sys

# Run as its own process (stdio transport), so make the backend package
# importable regardless of the caller's working directory.
sys.path.insert(0, os.path.dirname(__file__))

from mcp.server.fastmcp import FastMCP

from agents.pharmacy import _search_drug_info_local, _check_drug_interaction_local
from agents.insurance import _search_insurance_policy_local

server = FastMCP("smart-health-readonly")

@server.tool()
async def search_drug_info(drug_name: str) -> str:
    return await _search_drug_info_local(drug_name)

@server.tool()
def check_drug_interaction(drug1: str, drug2: str) -> str:
    return _check_drug_interaction_local(drug1, drug2)

@server.tool()
async def search_insurance_policy(query: str) -> str:
    return await _search_insurance_policy_local(query)

if __name__ == "__main__":
    server.run(transport="stdio")
