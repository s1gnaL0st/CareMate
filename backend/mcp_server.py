"""Standalone MCP server exposing read-only health knowledge tools."""
import asyncio
import os
import sys

# This project has a legacy local ``mcp`` package; load the official SDK first.
site_packages = os.path.join(sys.prefix, "Lib", "site-packages")
if os.path.isdir(site_packages):
    backend_dir = os.path.dirname(__file__)
    sys.path[:] = [p for p in sys.path if p and os.path.abspath(p) != os.path.abspath(backend_dir)]
    sys.path.insert(0, site_packages)
    for module_name in list(sys.modules):
        if module_name == "mcp" or module_name.startswith("mcp."):
            sys.modules.pop(module_name, None)

from mcp.server.fastmcp import FastMCP

sys.path.insert(0, os.path.dirname(__file__))
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
