"""
Insurance Agent Node.

Uses a ReAct sub-agent with four authenticated, database-backed tools:
  1. get_insurance_balance      - 查个人账户余额
  2. get_consumption_records    - 查医保消费明细
  3. get_payment_records        - 查缴费记录
  4. get_cross_region_info      - 异地就医信息
"""
import json
import os
from datetime import datetime, timedelta, timezone
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent
from agents.state import MainAgentState
from agents.llm import get_chat_llm
from rag.knowledge_base import get_knowledge_base
from agentic_rag import retrieve_until_sufficient
from cache import cache_get_json, cache_key, cache_set_json
from config import get_settings
from tool_executor import ToolExecutor
from mcp_adapter import MCPUnavailableError, call_mcp_tool, register_tool
from db import SessionLocal
from models import InsuranceAccount, InsuranceTransaction
from sqlalchemy import select
from tool_executor import get_active_tool_context


async def _account_for_current_user() -> InsuranceAccount | None:
    user_id = get_active_tool_context().user_id
    if not user_id:
        return None
    async with SessionLocal() as db:
        return await db.scalar(select(InsuranceAccount).where(InsuranceAccount.user_id == user_id).order_by(InsuranceAccount.updated_at.desc()))


def _missing(tool_name: str) -> str:
    return json.dumps({"schema_version": "1.0", "tool": tool_name, "available": False, "records": [], "message": "当前用户尚未绑定医保账户或暂无数据。"}, ensure_ascii=False)


async def _missing_cached(tool_name: str, key: str) -> str:
    cached = await cache_get_json(key)
    if isinstance(cached, dict) and cached.get("value"):
        return str(cached["value"])
    output = _missing(tool_name)
    await cache_set_json(key, {"value": output}, get_settings().cache_default_ttl_seconds)
    return output


# ─── TOOLS ────────────────────────────────────────────────────────────────────

@tool
async def get_insurance_balance() -> str:
    """查询用户的医保个人账户余额及统筹账户情况。"""
    account = await _account_for_current_user()
    if account is None:
        return await _missing_cached("insurance_balance", cache_key("insurance-balance", "anonymous"))
    key = cache_key("insurance-balance", account.user_id)
    cached = await cache_get_json(key)
    if isinstance(cached, dict) and cached.get("value"):
        return str(cached["value"])
    result = {
        "schema_version": "1.0",
        "tool": "insurance_balance",
        "user_id": account.user_id,
        "insurance_type": account.insurance_type,
        "region": account.region,
        "personal_account": account.personal_balance / 100,
        "updated_at": account.updated_at.isoformat() if account.updated_at else None,
    }
    output = json.dumps(result, ensure_ascii=False)
    await cache_set_json(key, {"value": output}, get_settings().cache_default_ttl_seconds)
    return output


@tool
async def get_consumption_records(months: int = 3) -> str:
    """查询用户近期医保消费明细（就诊报销记录）。months 为最近几个月，默认3个月。"""
    account = await _account_for_current_user()
    if account is None:
        return await _missing_cached("insurance_expenses", cache_key("insurance-consumption", "anonymous"))
    key = cache_key("insurance-consumption", f"{account.user_id}:{months}")
    cached = await cache_get_json(key)
    if isinstance(cached, dict) and cached.get("value"):
        return str(cached["value"])
    async with SessionLocal() as db:
        since = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=max(1, months) * 31)
        rows = list(await db.scalars(select(InsuranceTransaction).where(InsuranceTransaction.account_id == account.id, InsuranceTransaction.occurred_at >= since).order_by(InsuranceTransaction.occurred_at.desc())))
    records = [{"date": r.occurred_at.isoformat(), "hospital": r.provider_name, "amount": r.amount_cents / 100, "self_pay": r.self_pay_cents / 100, "reimbursed": r.reimbursed_cents / 100, "category": r.transaction_type, **(r.metadata_json or {})} for r in rows]
    result = {
        "schema_version": "1.0",
        "tool": "insurance_expenses",
        "user_id": account.user_id,
        "period_months": months,
        "records": records,
        "total_amount": sum(r["amount"] for r in records),
        "total_self_pay": sum(r["self_pay"] for r in records),
        "total_reimbursed": sum(r["reimbursed"] for r in records),
    }
    output = json.dumps(result, ensure_ascii=False)
    await cache_set_json(key, {"value": output}, get_settings().cache_default_ttl_seconds)
    return output


@tool
async def get_payment_records(months: int = 6) -> str:
    """查询用户的医保缴费记录，包含单位缴费和个人缴费情况。months 默认6个月。"""
    account = await _account_for_current_user()
    if account is None:
        return await _missing_cached("insurance_payments", cache_key("insurance-payment", "anonymous"))
    key = cache_key("insurance-payment", f"{account.user_id}:{months}")
    cached = await cache_get_json(key)
    if isinstance(cached, dict) and cached.get("value"):
        return str(cached["value"])
    async with SessionLocal() as db:
        since = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=max(1, months) * 31)
        rows = list(await db.scalars(select(InsuranceTransaction).where(InsuranceTransaction.account_id == account.id, InsuranceTransaction.occurred_at >= since, InsuranceTransaction.transaction_type == "payment").order_by(InsuranceTransaction.occurred_at.desc())))
    records = [{"year_month": r.occurred_at.strftime("%Y-%m"), "individual": (r.metadata_json or {}).get("individual", r.amount_cents / 100), "employer": (r.metadata_json or {}).get("employer", 0), "total": r.amount_cents / 100, "status": (r.metadata_json or {}).get("status", "已入账")} for r in rows[:months]]
    result = {
        "schema_version": "1.0",
        "tool": "insurance_payments",
        "user_id": account.user_id,
        "period_months": months,
        "records": records,
        "annual_total_individual": sum(r["individual"] for r in records),
        "annual_total_employer": sum(r["employer"] for r in records),
    }
    output = json.dumps(result, ensure_ascii=False)
    await cache_set_json(key, {"value": output}, get_settings().cache_default_ttl_seconds)
    return output


@tool
async def get_cross_region_info() -> str:
    """查询用户的异地就医备案状态及定点医院信息。"""
    account = await _account_for_current_user()
    if account is None:
        return await _missing_cached("insurance_cross_region", cache_key("insurance-cross-region", "anonymous"))
    key = cache_key("insurance-cross-region", account.user_id)
    cached = await cache_get_json(key)
    if isinstance(cached, dict) and cached.get("value"):
        return str(cached["value"])
    async with SessionLocal() as db:
        row = await db.scalar(select(InsuranceTransaction).where(InsuranceTransaction.account_id == account.id, InsuranceTransaction.transaction_type == "cross_region").order_by(InsuranceTransaction.occurred_at.desc()))
    if row is None:
        return _missing("insurance_cross_region")
    result = {
        "schema_version": "1.0",
        "tool": "insurance_cross_region",
        "user_id": account.user_id,
        **(row.metadata_json or {}),
    }
    output = json.dumps(result, ensure_ascii=False)
    await cache_set_json(key, {"value": output}, get_settings().cache_default_ttl_seconds)
    return output


async def _search_insurance_policy_local(query: str) -> str:
    """搜索医疗保险政策文件，获取报销规定、覆盖范围、申请流程等政策信息。
    适用于用户询问政策性问题，如：报销比例、门慢/门特政策、家庭共济、大病保险等。"""
    key = cache_key("insurance-policy", query)
    cached = await cache_get_json(key)
    if isinstance(cached, dict) and cached.get("value"):
        return str(cached["value"])
    kb = get_knowledge_base()
    # The policy tool is already called from an Agent runtime. Keep the tool
    # itself single-round and bounded; nested three-round Agentic retrieval
    # can exceed the outer tool budget and cause avoidable timeouts.
    retrieval = await retrieve_until_sufficient(query, kb, k=3, max_rounds=1)
    context = kb.format_context(retrieval.documents)
    if not retrieval.sufficient:
        context += "\n\n[检索提示] 当前政策证据覆盖不足：" + ", ".join(retrieval.missing_facets)
    output = context if context else "未找到相关医保政策信息，建议拨打当地医保服务热线12393咨询。"
    await cache_set_json(key, {"value": output}, get_settings().cache_rag_ttl_seconds)
    return output


@register_tool("search_insurance_policy")
async def _mcp_search_insurance_policy(query: str) -> str:
    return await _search_insurance_policy_local(query)


@tool
async def search_insurance_policy(query: str) -> str:
    """搜索医保政策；本地 RAG 默认优先，远端 MCP 通过显式开关启用。"""
    key = cache_key("insurance-policy", query)
    cached = await cache_get_json(key)
    if isinstance(cached, dict) and cached.get("value"):
        return str(cached["value"])
    # The local policy corpus is deterministic and available offline. MCP is
    # optional; keeping it opt-in prevents a slow remote initialization from
    # consuming the entire outer tool timeout before fallback can run.
    if os.getenv("INSURANCE_POLICY_MCP_FIRST", "false").strip().lower() not in {"1", "true", "yes", "on"}:
        return await _search_insurance_policy_local(query)
    try:
        return await call_mcp_tool("search_insurance_policy", query=query)
    except MCPUnavailableError:
        return await _search_insurance_policy_local(query)


# ─── AGENT ────────────────────────────────────────────────────────────────────

INSURANCE_SYSTEM_PROMPT = """你是一个专业的医保政策咨询助手，熟悉中国基本医疗保险制度。

【职责范围】
- 解答医保报销政策（门诊、住院、门慢、大病等）
- 调用工具查询用户的账户余额、消费明细、缴费记录、异地就医信息
- 说明常见的医保办理流程（异地就医备案、转诊、家庭共济等）

【工具使用规则】
- 当用户询问"余额"/"账户"时，调用 get_insurance_balance
- 当用户询问"消费"/"报销"/"明细"/"看病记录"时，调用 get_consumption_records
- 当用户询问"缴费"/"扣款"/"单位缴费"时，调用 get_payment_records
- 当用户询问"异地"/"外地就医"/"备案"时，调用 get_cross_region_info
- 当用户询问政策问题（报销比例、门慢政策、大病保险、家庭共济、申请流程等）时，调用 search_insurance_policy

【回答格式】
工具调用完成后，简洁总结关键数据（1-3句）。数字已由卡片展示，不用大量复述数字。"""

_INSURANCE_TOOLS = [
    get_insurance_balance,
    get_consumption_records,
    get_payment_records,
    get_cross_region_info,
    search_insurance_policy,
]

# Tool names that trigger frontend card rendering
INSURANCE_CARD_TOOLS = {t.name for t in _INSURANCE_TOOLS}


def _build_insurance_agent(llm, tools=None):
    """Build a ReAct sub-agent scoped to insurance tools."""
    return create_react_agent(
        llm,
        tools=tools or _INSURANCE_TOOLS,
        prompt=SystemMessage(content=INSURANCE_SYSTEM_PROMPT),
    )


async def insurance_node(state: MainAgentState) -> dict:
    """
    Insurance agent: uses a ReAct sub-agent to call insurance tools and answer queries.
    Extracts messages from the sub-graph result to be compatible with MainAgentState.
    """
    llm = get_chat_llm("balanced")
    user_info = state.get("user_info", {})

    extra_context = ""
    region = user_info.get("region", "")
    elder_mode = user_info.get("elder_mode", False)
    if region:
        extra_context += f"\n\n用户所在地区：{region}，尽量结合当地政策作答。"
    if elder_mode:
        extra_context += "\n请使用通俗易懂的语言，避免复杂的政策术语。"

    tool_executor = ToolExecutor.from_state(
        state,
        agent_name="insurance_agent",
        allowed_tools=(tool.name for tool in _INSURANCE_TOOLS),
    )
    wrapped_tools = tool_executor.wrap_tools(_INSURANCE_TOOLS)

    # Dynamically patch the system prompt if we have user context
    if extra_context:
        agent = create_react_agent(
            llm,
            tools=wrapped_tools,
            prompt=SystemMessage(content=INSURANCE_SYSTEM_PROMPT + extra_context),
        )
    else:
        agent = _build_insurance_agent(llm, wrapped_tools)

    sub_result = await agent.ainvoke({"messages": state["messages"]})
    # Extract only the new messages (all except the original input messages)
    original_count = len(state["messages"])
    new_messages = sub_result["messages"][original_count:]
    return {"messages": new_messages}
