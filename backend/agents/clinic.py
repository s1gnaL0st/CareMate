"""Clinic agent with an isolated five-action local-model adapter."""
from __future__ import annotations

import json
import logging
import asyncio
import random
import re
from typing import Any, Mapping

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from agents.clinic_action_adapter import (
    CLINIC_ACTION_TOOLS,
    ClinicActionAdapter,
    clinic_harness_state,
    fallback_clinic_answer,
    sanitize_clinic_answer,
)
from agents.llm import get_clinic_llm
from agents.state import MainAgentState
from config import get_settings

logger = logging.getLogger("smart_health.clinic")

CLINIC_SYSTEM_PROMPT = """你是大健康 App 的 AI 预问诊助手。

你的任务是通过渐进式多轮对话收集症状信息，并给出安全的阶段性分诊建议。

规则：
1. 先关注急症信号；不要自行诊断、开处方或给个体化剂量。
2. 每次面向患者的回复都遵循“三段式”：先正面回应当前主诉，再用简短语言解释目前可能范围/风险，最后决定是否需要追问。
3. 不要使用固定的首问或问题清单。不要默认询问“什么时候开始”“是否突然发生或持续加重”；只有当这个信息确实会改变当前分诊、且患者尚未提供时，才询问它。
4. 信息已经足够支持阶段性分诊时，直接给出“初步可能范围、风险等级和下一步建议”，不要为了继续对话而追问；可以说明患者愿意时还能补充哪些信息。
5. 信息不足且补充信息会改变分诊时，使用 ask，每次只问一个基于当前上下文、最有价值的问题。不要重复已回答的信息，不要把两个独立问题合并成一个问题。
6. 需要医学证据时使用 lookup；需要受控外部医学资源时使用 search；不要编造引用。
7. 需要检查、生命体征、红旗或药物相互作用时使用 check。
8. 信息足够后，使用 answer 输出最终分诊、建议、风险提示和下一步；answer 是终止动作。
9. 不要输出训练环境的 ev_* 证据编号。只能使用 lookup/search 工具返回的真实引用。

表达方式：
- 面向患者说自然、完整、短句的中文，先承接患者刚说的内容，再解释判断依据和下一步；不要像填写表格或生成报告。
- 不要输出“初步可能范围：”“风险等级：”“下一步：”“第一段：”这类字段标签，也不要把句子写成关键词列表。
- 可以说“从你现在描述的情况看，常见可能包括……”，但不要把可能性说成确定诊断；建议应具体、简洁、可执行。
- 先让患者感到被听见：可以自然地表达理解和安抚，例如“听起来确实挺不舒服的”“先别太担心”，但不要夸张、卖萌或保证一定没事。
- 把医学判断翻译成患者听得懂的话，说明“为什么这样判断”和“接下来怎么做”；避免冷冰冰的结论、命令式短语和机械模板。
- ask 的 question 必须是完整、礼貌、自然的问题，不能只输出“持续加重？”、“日常活动？”这类残句。
- 如果患者质疑“真的假的/真的吗”，直接说明目前判断的不确定性、依据和何时需要就医，不要机械重复上一句。
- 最终 answer 通常用 2 到 4 句完成，不要为了凑三段而重复内容。

当使用 ask 时，工具参数只放最后那一个追问。模型可以在 answer 前保持工具调用链；不要把 ask 当成最终诊断。

最终回答不得声称确定诊断，不得替代医生面诊。"""


@tool
def exit_clinic_mode() -> str:
    """结束当前预问诊模式，保留给旧调用方的兼容工具。"""
    return json.dumps(
        {
            "action": "exit_mode",
            "mode": "clinic",
            "message": "已结束本次预问诊，已返回普通健康助手。",
        },
        ensure_ascii=False,
    )


def _message_content(value: Any) -> str:
    content = getattr(value, "content", value)
    return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, default=str)


def _tool_call_parts(call: Mapping[str, Any]) -> tuple[str, str, dict[str, Any]]:
    function = call.get("function") if isinstance(call.get("function"), Mapping) else call
    name = str(call.get("name") or function.get("name") or "").strip()
    call_id = str(call.get("id") or call.get("tool_call_id") or "clinic-call").strip()
    raw_args = function.get("arguments", call.get("args", {}))
    if isinstance(raw_args, str):
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError:
            args = {}
    else:
        args = dict(raw_args or {}) if isinstance(raw_args, Mapping) else {}
    return name, call_id, args


def _clinic_model_history(messages: list[Any]) -> list[Any]:
    """Replay patient answers as tool results for the preceding ``ask``.

    The GRPO checkpoint was trained on assistant tool calls followed by a
    tool-role result.  The UI/API stores the patient's reply as a HumanMessage,
    so normalize that boundary only when it immediately follows an ask call.
    """
    history: list[Any] = []
    for index, message in enumerate(messages):
        if isinstance(message, HumanMessage) and history:
            previous = history[-1]
            calls = getattr(previous, "tool_calls", None) or []
            ask_call = next((call for call in calls if str(call.get("name", "")) == "ask"), None)
            if ask_call is not None:
                call_id = str(ask_call.get("id") or "clinic-call")
                history.append(ToolMessage(content=_message_content(message), tool_call_id=call_id))
                continue
        history.append(message)
    return history


def _safety_text(veto: Any) -> str:
    return str(getattr(veto, "safety_message", "如出现明显不适或症状加重，请立即就医。"))


def _normalize_triage_level(value: Any) -> str:
    """Accept the V2 string enum and numeric enum emitted by some Qwen templates."""
    if isinstance(value, bool):
        return ""
    if isinstance(value, int) or (isinstance(value, str) and value.strip().isdigit()):
        return {
            0: "emergency",
            1: "urgent",
            2: "routine",
            3: "self_care",
        }.get(int(value), "")
    return str(value or "").strip().lower()


def _is_retryable_clinic_error(exc: BaseException) -> bool:
    """Classify transient model transport/service failures.

    Invalid requests (usually 400) must not be retried because they will
    deterministically fail again.  Timeouts, connection failures and common
    gateway/rate-limit statuses are safe to retry because this call is a
    read-only model generation.
    """
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError, OSError)):
        return True
    detail = str(exc).lower()
    transient_markers = (
        "408", "409", "425", "429", "500", "502", "503", "504",
        "timeout", "timed out", "rate limit", "temporarily unavailable",
        "connection reset", "connection refused", "server error", "bad gateway",
    )
    return any(marker in detail for marker in transient_markers)


def _visible_ask_text(content: str, question: str, citation_ids: set[str]) -> str:
    """Keep one model preamble and exactly the action-selected question.

    Tool-capable checkpoints sometimes put their own questions in assistant
    content even though the same turn also contains ``ask``. Showing both
    violates the one-question harness contract and can duplicate the action
    question. The action remains authoritative; prose before the first
    question mark is retained as the patient's acknowledgement/explanation.
    """
    text = sanitize_clinic_answer(content, citation_ids).strip()
    # Some checkpoints echo the exact action argument in ``content``.  The
    # structured ask argument is authoritative; remove that echo before
    # applying the one-question boundary below.
    if text and question:
        text = text.replace(question, "").strip()
    if text:
        match = re.search(r"[？?]", text)
        if match:
            text = text[: match.start()].rstrip(" \n，。；;：:")
    return f"{text}\n\n{question}" if text else question


async def _invoke_clinic_model(model: Any, model_messages: list[Any], settings: Any) -> Any:
    """Invoke the clinic model with bounded exponential backoff and jitter."""
    max_retries = int(getattr(settings, "clinic_llm_max_retries", 2))
    base_delay = max(0.0, float(getattr(settings, "clinic_llm_retry_backoff_seconds", 0.5)))
    for attempt in range(max_retries + 1):
        try:
            return await model.ainvoke(model_messages)
        except Exception as exc:
            retryable = _is_retryable_clinic_error(exc)
            exhausted = attempt >= max_retries
            logger.warning(
                "clinic model attempt failed attempt=%d/%d retryable=%s error=%s detail=%s",
                attempt + 1,
                max_retries + 1,
                retryable,
                type(exc).__name__,
                str(exc)[:240],
            )
            if exhausted or not retryable:
                raise
            delay = min(base_delay * (2 ** attempt), 8.0)
            # Small jitter prevents many browser sessions sharing one tunnel
            # from retrying in lockstep after a gateway hiccup.
            delay += random.uniform(0.0, min(0.25, delay * 0.25)) if delay else 0.0
            await asyncio.sleep(delay)
    raise RuntimeError("clinic model retry loop exited unexpectedly")


async def _render_ask_turn(
    model_messages: list[Any],
    question: str,
    settings: Any,
) -> str:
    """Render the visible part of an ``ask`` turn without changing its protocol.

    The GRPO policy emits ``ask`` as an action and usually leaves assistant
    content empty.  A second, tool-free call lets the same model turn that
    action into patient-facing prose.  The original tool call remains in the
    checkpoint; this call is presentation-only.
    """
    renderer = get_clinic_llm(temperature=0.2)
    instruction = SystemMessage(
        content=(
            "你现在是医疗问诊对话的表达层。根据下面的会话上下文，生成一条给患者看的自然中文回复。"
            "先承接患者刚才的主诉并给出简短、谨慎的可能解释，再自然地引出最后一个追问。"
            "不要编造新的症状或诊断，不要使用‘第一段/第二段/风险等级/下一步’等字段标签，"
            "不要输出工具调用、思考过程或证据编号。最后必须保留这句追问，原样放在回复末尾：\n"
            f"{question}"
        )
    )
    # Qwen's chat template accepts one leading system message. Merge the
    # clinic policy prompt and renderer instruction instead of sending two
    # system roles (which some OpenAI-compatible servers reject).
    context = []
    for message in model_messages:
        if isinstance(message, SystemMessage):
            context.append(message)
        elif isinstance(message, ToolMessage):
            # The expression call is not a tool-loop request. Some Qwen
            # servers reject a role=tool message when tools are omitted, so
            # expose the observation as ordinary conversation text here.
            context.append(HumanMessage(content=f"患者上一轮补充：{_message_content(message)}"))
        elif isinstance(message, AIMessage):
            content = _message_content(message).strip()
            if content:
                context.append(AIMessage(content=content))
        else:
            context.append(message)
    if context and isinstance(context[0], SystemMessage):
        context[0] = SystemMessage(content=f"{context[0].content}\n\n{instruction.content}")
        render_messages = context
    else:
        render_messages = [instruction, *context]
    rendered = await _invoke_clinic_model(renderer, render_messages, settings)
    raw_calls = getattr(rendered, "tool_calls", None) or getattr(rendered, "additional_kwargs", {}).get("tool_calls", [])
    if raw_calls:
        raise RuntimeError("ask_preamble_generation_returned_tool_call")
    text = sanitize_clinic_answer(_message_content(rendered), set()).strip()
    if not text:
        raise RuntimeError("ask_preamble_generation_empty")
    return _visible_ask_text(text, question, set())


async def clinic_node(state: MainAgentState) -> dict:
    """Run clinic-only Qwen tool calls behind a strict project adapter."""
    messages = list(state.get("messages", []))
    # Keep the original role-labelled history for duplicate-question detection
    # and emergency veto.  The local Qwen chat template, however, expects one
    # leading system message; merge any task/context preambles into the clinic
    # system prompt before sending the request.
    adapter_messages = list(state.get("conversation_messages", messages))
    adapter = ClinicActionAdapter(state=state, messages=adapter_messages)
    settings = get_settings()
    harness = clinic_harness_state(
        adapter_messages,
        max_question_turns=settings.clinic_max_question_turns,
    )
    initial_veto = adapter.emergency_veto()
    if getattr(initial_veto, "level", "") == "CRITICAL":
        return {"messages": [AIMessage(content=_safety_text(initial_veto))]}

    user_info = state.get("user_info", {}) or {}
    system = (
        f"{CLINIC_SYSTEM_PROMPT}\n\n当前用户：姓名={user_info.get('name', '您')}，"
        f"年龄={user_info.get('age', '')}，既往病史={user_info.get('medical_history', '无')}"
    )
    if adapter.answered_facets:
        system += (
            "\n\n已从患者回答中确认的信息维度："
            + "、".join(sorted(adapter.answered_facets))
            + "。不要再次询问已确认的维度；复合问题只追问其中尚未确认的部分。"
        )
    if user_info.get("elder_mode", False):
        system += "\n请使用极度通俗易懂的语言。"

    available_tools = list(CLINIC_ACTION_TOOLS)
    if harness.force_answer:
        # The harness has reached its question budget. Do not let the model
        # open another questionnaire branch; the domain skill must answer.
        available_tools = [
            item for item in available_tools if item["function"]["name"] != "ask"
        ]
        system += "\n本次问诊已达到追问上限，请直接使用 answer 输出分诊和下一步建议，不要继续追问。"
    model = get_clinic_llm(temperature=0.0).bind_tools(available_tools, tool_choice="auto")
    system_parts = [system]
    # Preserve task-level context/system preambles supplied by the planner.
    for message in messages:
        if isinstance(message, SystemMessage):
            content = _message_content(message).strip()
            if content:
                system_parts.append(content)

    # The executor passes a compact task prompt in ``messages``.  The GRPO
    # policy, however, must see the complete clinic conversation so the
    # previous ask tool call can be paired with the patient's latest reply.
    # ``conversation_messages`` is the canonical history reconstructed by the
    # API layer; using the task slice here silently restarted the interview on
    # every turn.
    model_history: list[Any] = [
        message
        for message in _clinic_model_history(adapter_messages)
        if not isinstance(message, SystemMessage)
    ]
    model_messages: list[Any] = [SystemMessage(content="\n\n".join(system_parts)), *model_history]
    max_rounds = settings.clinic_llm_max_rounds
    for _round in range(max_rounds):
        try:
            response = await _invoke_clinic_model(model, model_messages, settings)
        except Exception as exc:
            logger.warning("clinic model unavailable after retries error=%s", type(exc).__name__)
            # A local-model outage must not create another interview turn.
            # Return a complete, safe response so the patient is never left
            # with a bare question or an apparent empty result.
            return {"messages": [AIMessage(content=fallback_clinic_answer(adapter_messages))]}
        model_messages.append(response)
        raw_calls = getattr(response, "tool_calls", None) or getattr(response, "additional_kwargs", {}).get("tool_calls", [])
        if not raw_calls:
            final = sanitize_clinic_answer(_message_content(response), adapter.citation_ids)
            if not final:
                final = "【诊室模型错误】模型返回了空回答（empty_model_response）。"
            if getattr(adapter.emergency_veto(), "level", "") == "CRITICAL":
                final = _safety_text(adapter.emergency_veto())
            return {"messages": [AIMessage(content=final)]}
        for raw_call in raw_calls:
            name, call_id, arguments = _tool_call_parts(raw_call)
            if name == "ask":
                if harness.force_answer:
                    return {"messages": [AIMessage(content=fallback_clinic_answer(adapter_messages))]}
                # ``ask`` is a user-facing turn boundary. The next model call
                # belongs to the next user message.
                try:
                    result = await adapter.execute(name, arguments)
                    payload = json.loads(result)
                except Exception as exc:
                    logger.warning("clinic ask action failed error=%s", type(exc).__name__)
                    result = json.dumps(
                        {"success": False, "error": "ask_action_failed"},
                        ensure_ascii=False,
                    )
                    payload = {"success": False}
                question = str(payload.get("question", "")).strip()
                if payload.get("success") and question:
                    native_content = sanitize_clinic_answer(
                        _message_content(response), adapter.citation_ids
                    ).strip()
                    if native_content:
                        # Some V2 checkpoints already return a natural preamble
                        # alongside the action. Keep it and append the policy's
                        # exact question; no second generation is necessary.
                        visible = _visible_ask_text(
                            native_content, question, adapter.citation_ids
                        )
                    else:
                        try:
                            visible = await _render_ask_turn(
                                # ``response`` is the action being rendered;
                                # do not send an unmatched assistant tool call
                                # into the tool-free expression request.
                                model_messages[:-1],
                                question,
                                settings,
                            )
                        except Exception as exc:
                            logger.warning(
                                "clinic ask expression layer failed error=%s",
                                type(exc).__name__,
                            )
                            visible = (
                                "【诊室模型错误】追问已生成，但表达层没有生成可展示的正文 "
                                "（ask_preamble_generation_failed）。请重试本轮。\n\n"
                                f"{question}"
                            )
                    # Preserve the native assistant tool call in checkpointed
                    # state.  The visible content is presentation-only; on the
                    # next turn the tool call is replayed as a tool result.
                    return {
                        "messages": [
                            AIMessage(
                                content=visible,
                                tool_calls=[{
                                    "name": "ask",
                                    "args": {"question": question},
                                    "id": call_id,
                                    "type": "tool_call",
                                }],
                            )
                        ],
                        "clinic_pending_tool_call": {
                            "id": call_id,
                            "name": "ask",
                            "question": question,
                        },
                    }
                # Feed a rejected action back to the model so it can recover
                # with an answer or a genuinely missing dimension. Do not
                # expose a harness-generated medical sentence to the patient.
                model_messages.append(
                    ToolMessage(
                        content=result,
                        tool_call_id=call_id,
                    )
                )
                continue
            if name == "answer":
                triage_level = _normalize_triage_level(arguments.get("triage_level"))
                content = str(arguments.get("content", "")).strip()
                if triage_level not in {"emergency", "urgent", "routine", "self_care"} or not content:
                    return {"messages": [AIMessage(content="【诊室模型错误】answer 参数无效：需要 triage_level 和 content。" )]}
                final = sanitize_clinic_answer(content, adapter.citation_ids)
                veto = adapter.emergency_veto()
                if getattr(veto, "level", "") == "CRITICAL":
                    final = _safety_text(veto)
                elif triage_level == "emergency":
                    final = "建议立即前往急诊或拨打120。\n\n" + final
                return {"messages": [AIMessage(content=final)]}
            if name not in {"ask", "check", "lookup", "search"}:
                result = json.dumps({"success": False, "error": "unknown_action"}, ensure_ascii=False)
            else:
                try:
                    result = await adapter.execute(name, arguments)
                except Exception as exc:
                    logger.warning("clinic action failed action=%s error=%s", name, type(exc).__name__)
                    result = json.dumps({"success": False, "error": "action_unavailable"}, ensure_ascii=False)
            model_messages.append(ToolMessage(content=result, tool_call_id=call_id))
    return {"messages": [AIMessage(content=fallback_clinic_answer(adapter_messages))]}
