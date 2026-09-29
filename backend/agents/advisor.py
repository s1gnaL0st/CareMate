"""
Advisor Agent Node.

Handles general health Q&A, drug information, diet/lifestyle advice.
Uses a simple RAG-style flow:
  1. Query rewriting (if conversational context is needed)
  2. LLM generation with health advisor persona
"""
import json
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from agents.state import MainAgentState
from agents.llm import get_chat_llm
from rag.knowledge_base import get_knowledge_base
from agentic_rag import retrieve_until_sufficient
from skills import get_agent_tools, load_skill
from tool_executor import ToolExecutor
from mcp_adapter import MCPUnavailableError, call_mcp_tool


class LiteratureSearchInput(BaseModel):
    query: str = Field(description="医学问题或研究主题，例如：2型糖尿病运动干预")
    max_results: int = Field(default=5, ge=1, le=20, description="返回文献数量")


@tool(args_schema=LiteratureSearchInput)
async def search_medical_literature(query: str, max_results: int = 5) -> str:
    """通过公开 PubMed MCP 检索医学文献，并返回带来源的结果。"""
    try:
        return await call_mcp_tool(
            "search_medical_literature", query=query, maxResults=max_results
        )
    except MCPUnavailableError:
        return json.dumps({
            "schema_version": "1.0", "tool": "search_medical_literature",
            "source": "fallback", "query": query,
            "message": "PubMed MCP 暂时不可用，请稍后重试。",
        }, ensure_ascii=False)


def _build_system_prompt(user_info: dict) -> str:
    name = user_info.get("name", "用户")
    age = user_info.get("age", "未知")
    history = user_info.get("medical_history", "无")
    elder_mode = user_info.get("elder_mode", False)

    lang_style = "请使用极度通俗易懂、口语化的语言，避免医学专业词汇。" if elder_mode else "请使用清晰简洁的语言。"

    return f"""你是一个名为"大健康AI助手"的专业、温暖的虚拟健康顾问。

【当前用户信息】
姓名：{name}  年龄：{age}  既往史：{history}

【对话原则】
1. **安全性**：你不能替代真正的医生下达明确的医疗诊断，你的回答只是健康参考。如遇急性严重症状（如剧烈胸痛、呼吸困难），必须在回答开头强烈建议用户拨打 120 或立即就医。
2. **语言风格**：{lang_style}
3. **结构化输出**：回答应条理清晰，利用列表进行排版以适应移动端阅读。
4. **精炼**：单次回答控制在 300 字以内，并主动通过一句温和的提问引导用户继续交流。

【可用技能工具】
- search_medical_literature：当用户需要医学研究证据、论文或最新临床发现时调用
- health_calculator：当用户询问BMI、理想体重、每日热量需求时调用
- risk_assessor：当用户想评估心血管或糖尿病风险时调用
- load_skill：通用技能加载器，可按名称调用任意已注册技能"""


async def advisor_node(state: MainAgentState) -> dict:
    """
    Advisor agent: general health Q&A with RAG context + advisor-tagged skills
    (health_calculator, risk_assessor) loaded as ReAct tools.
    """
    llm = get_chat_llm("fast")
    user_info = state.get("user_info", {})
    system_prompt = _build_system_prompt(user_info)
    offline_evidence = []

    last_user_msg = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        "",
    )

    if last_user_msg:
        kb = get_knowledge_base()
        retrieval = await retrieve_until_sufficient(last_user_msg, kb, k=3, max_rounds=3)
        docs = retrieval.documents
        if state.get("offline_evidence_capture") is True:
            offline_evidence = [
                {
                    "source_id": str(doc.metadata.get("source_id") or doc.metadata.get("parent_id") or f"{doc.metadata.get('source', 'unknown')}::{doc.metadata.get('section', '')}"),
                    "version": str(doc.metadata.get("version") or doc.metadata.get("knowledge_version") or "unknown"),
                    "text": str(doc.page_content or "")[:2000],
                }
                for doc in docs[:3]
            ]
        rag_context = kb.format_context(docs)
        if rag_context:
            system_prompt += f"\n\n## 参考知识库\n以下为相关知识内容，可作为回答参考（请勿照抄，结合用户情况灵活运用）：\n\n{rag_context}"
        system_prompt += (
            "\n\n## 检索充分性\n"
            f"本轮 Agentic RAG 已执行 {len(retrieval.rounds)} 轮检索；"
            f"证据是否充分：{retrieval.sufficient}；缺失证据面：{retrieval.missing_facets or '无'}。"
            "不得把缺失证据补写成事实，证据不足时明确说明不确定性。"
        )

    skill_tools = get_agent_tools(tags=["advisor"])
    all_tools = [search_medical_literature, load_skill, *skill_tools]
    tool_executor = ToolExecutor.from_state(
        state,
        agent_name="advisor_agent",
        allowed_tools=(tool.name for tool in all_tools),
        allowed_dynamic_skills=(tool.name for tool in skill_tools),
    )

    agent = create_react_agent(
        llm,
        tools=tool_executor.wrap_tools(all_tools),
        prompt=SystemMessage(content=system_prompt),
    )
    sub_result = await agent.ainvoke({"messages": state["messages"]})
    original_count = len(state["messages"])
    new_messages = sub_result["messages"][original_count:]
    result = {"messages": new_messages}
    if last_user_msg:
        result["rag_trace"] = retrieval.trace()
    if offline_evidence:
        result["offline_evidence"] = offline_evidence
    return result
