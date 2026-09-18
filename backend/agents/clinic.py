"""Clinic agent with an isolated five-action local-model adapter."""
from __future__ import annotations

import json
import logging
import asyncio
import random
from typing import Any, Mapping

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from agents.clinic_action_adapter import CLINIC_ACTION_TOOLS, ClinicActionAdapter, sanitize_clinic_answer
from agents.llm import get_clinic_llm
from agents.state import MainAgentState
from config import get_settings

logger = logging.getLogger("smart_health.clinic")

CLINIC_SYSTEM_PROMPT = """你是大健康 App 的 AI 预问诊助手。

你的任务是通过多轮对话收集症状信息，再给出安全的就医分诊建议。

规则：
1. 先关注急症信号；不要自行诊断、开处方或给个体化剂量。
2. 信息不够时使用 ask，每次只问一个尚未问过的问题。优先补齐主诉、部位、持续时间、严重程度和伴随症状。
3. 需要医学证据时使用 lookup；需要受控外部医学资源时使用 search；不要编造引用。
4. 需要检查、生命体征、红旗或药物相互作用时使用 check。
5. 信息足够后，使用 answer 输出最终分诊、建议、风险提示和下一步；answer 是终止动作。
6. 不要输出训练环境的 ev_* 证据编号。只能使用 lookup/search 工具返回的真实引用。

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


def _safety_text(veto: Any) -> str:
    return str(getattr(veto, "safety_message", "如出现明显不适或症状加重，请立即就医。"))


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


async def clinic_node(state: MainAgentState) -> dict:
    """Run clinic-only Qwen tool calls behind a strict project adapter."""
    messages = list(state.get("messages", []))
    # Keep the original role-labelled history for duplicate-question detection
    # and emergency veto.  The local Qwen chat template, however, expects one
    # leading system message; merge any task/context preambles into the clinic
    # system prompt before sending the request.
    adapter_messages = list(state.get("conversation_messages", messages))
    adapter = ClinicActionAdapter(state=state, messages=adapter_messages)
    initial_veto = adapter.emergency_veto()
    if getattr(initial_veto, "level", "") == "CRITICAL":
        return {"messages": [AIMessage(content=_safety_text(initial_veto))]}

    user_info = state.get("user_info", {}) or {}
    system = (
        f"{CLINIC_SYSTEM_PROMPT}\n\n当前用户：姓名={user_info.get('name', '您')}，"
        f"年龄={user_info.get('age', '')}，既往病史={user_info.get('medical_history', '无')}"
    )
    if user_info.get("elder_mode", False):
        system += "\n请使用极度通俗易懂的语言。"

    model = get_clinic_llm(temperature=0.0).bind_tools(
        list(CLINIC_ACTION_TOOLS), tool_choice="auto"
    )
    system_parts = [system]
    model_history: list[Any] = []
    for message in messages:
        if isinstance(message, SystemMessage):
            content = _message_content(message).strip()
            if content:
                system_parts.append(content)
        else:
            model_history.append(message)
    model_messages: list[Any] = [SystemMessage(content="\n\n".join(system_parts)), *model_history]
    max_rounds = get_settings().clinic_llm_max_rounds
    for _round in range(max_rounds):
        try:
            response = await _invoke_clinic_model(model, model_messages, get_settings())
        except Exception as exc:
            logger.warning("clinic model unavailable after retries error=%s", type(exc).__name__)
            # A transient local-model 5xx/timeout must not terminate an
            # otherwise healthy interview.  Continue with the next bounded
            # question; this is only a failure fallback and does not replace
            # model-led intent routing or normal GRPO actions.
            fallback_question = adapter.next_fallback_question()
            if fallback_question:
                return {"messages": [AIMessage(content=fallback_question)]}
            return {"messages": [AIMessage(content="当前预问诊模型暂时不可用，请稍后重试或直接联系医生。") ]}
        model_messages.append(response)
        raw_calls = getattr(response, "tool_calls", None) or getattr(response, "additional_kwargs", {}).get("tool_calls", [])
        if not raw_calls:
            final = sanitize_clinic_answer(_message_content(response), adapter.citation_ids)
            if getattr(adapter.emergency_veto(), "level", "") == "CRITICAL":
                final = _safety_text(adapter.emergency_veto())
            return {"messages": [AIMessage(content=final)]}
        for raw_call in raw_calls:
            name, call_id, arguments = _tool_call_parts(raw_call)
            if name == "ask":
                # ``ask`` is a user-facing turn boundary. The next model call
                # belongs to the next user message.
                try:
                    result = await adapter.execute(name, arguments)
                    payload = json.loads(result)
                except Exception as exc:
                    logger.warning("clinic ask action failed error=%s", type(exc).__name__)
                    payload = {"success": False}
                question = str(payload.get("question", "")).strip()
                if payload.get("success") and question:
                    return {"messages": [AIMessage(content=question)]}
                # The local model sometimes repeats an earlier ask after the
                # patient has supplied an answer.  Do not return the same
                # generic sentence forever; move to a bounded missing slot.
                fallback_question = adapter.next_fallback_question()
                if fallback_question:
                    logger.info("clinic duplicate/invalid ask; using bounded fallback question")
                    return {"messages": [AIMessage(content=fallback_question)]}
                return {
                    "messages": [
                        AIMessage(content="请补充症状持续时间、严重程度或伴随症状中的一项。")
                    ]
                }
            if name == "answer":
                final = sanitize_clinic_answer(str(arguments.get("content", "")), adapter.citation_ids)
                veto = adapter.emergency_veto()
                if getattr(veto, "level", "") == "CRITICAL":
                    final = _safety_text(veto)
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
    return {"messages": [AIMessage(content="为了安全完成预问诊，请补充症状持续时间、严重程度和伴随症状。") ]}
