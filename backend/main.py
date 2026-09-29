from contextlib import asynccontextmanager
import asyncio
from datetime import datetime, timezone
from uuid import uuid4
import logging
import os
import re
from typing import Any
from pathlib import Path
from time import perf_counter
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json
from dotenv import load_dotenv
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.middleware.base import BaseHTTPMiddleware
from langchain_core.messages import HumanMessage, AIMessage
from api_v1 import router as api_v1_router
from auth import get_current_user, get_optional_current_user, hash_password
from config import get_settings
from db import close_clients, engine, get_db, redis_client
from models import AgentRun, AgentRunEvent, ChatRun, Conversation, Message, User, UserProfile
from agent_persistence import (
    create_agent_run,
    get_run,
    get_run_snapshot,
    load_resume_state,
    mark_run_status,
    requeue_running_tasks,
    request_pause,
    resume_run,
)
from agents.insurance import INSURANCE_CARD_TOOLS
from agents.llm import LLMConfigurationError, resolve_model_settings
from agents.pharmacy import PHARMACY_TOOL_TO_CARD_TYPE
from agents.safety import UnsafePromptError, validate_untrusted_text
from agents.vision import (
    SCAN_TYPE_TO_AGENT,
    VisionInputError,
    compose_agent_message,
    normalize_scan_type,
    recognize_image,
    validate_image_content,
)
from observability import configure_observability
from cache import cache_get_json, cache_key, cache_set_json, distributed_lock, enforce_rate_limit
from metrics import record_request, render_prometheus
from evolution import build_experience_payload, load_active_memory_preferences, persist_experience
from agent_runtime import AgentRuntime, RuntimeBudget, RuntimeBudgetExceeded
from memory_manager import build_response_memory
from memory_evolution import maybe_run_shadow_memory
from health_memory import health_context, load_relevant_health_events, observe_health_message
from skill_registry import active_skills
from reports_api import router as reports_router
from users_api import router as users_router
from wellness_api import router as wellness_router
from evolution_api import router as evolution_router

load_dotenv()
settings = get_settings()
logger = logging.getLogger("smart_health")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

_observability_runtime = configure_observability()
if _observability_runtime.provider != "none":
    print(
        f"--- [Observability] {_observability_runtime.provider} tracing enabled ---",
        flush=True,
    )


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Development convenience: keep the demo login available after every
    # migration/restart without overwriting an existing password.
    if settings.environment != "production" and os.getenv("SEED_ADMIN", "true").lower() == "true":
        try:
            from db import SessionLocal
            async with SessionLocal() as db:
                admin = await db.scalar(select(User).where(User.email == "admin@caremate.local"))
                if admin is None:
                    admin = User(email="admin@caremate.local", password_hash=hash_password("123"), name="CareMate Admin")
                    db.add(admin)
                    await db.flush()
                    db.add(UserProfile(user_id=admin.id))
                    await db.commit()
                    logger.info("created development admin account: admin / 123")
        except Exception:
            logger.exception("failed to initialize development admin account")
    yield
    _observability_runtime.shutdown()
    await close_clients()


app = FastAPI(title="大健康 AI 后端", version="2.0.0", lifespan=lifespan)

class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        started = perf_counter()
        response = await call_next(request)
        record_request(request.method, request.url.path, response.status_code, perf_counter() - started)
        return response


app.add_middleware(RequestIDMiddleware)
app.add_middleware(MetricsMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(api_v1_router)
app.include_router(reports_router)
app.include_router(users_router)
app.include_router(wellness_router)
app.include_router(evolution_router)


@app.exception_handler(Exception)
async def unhandled_exception(request: Request, exc: Exception):
    logger.exception("unhandled request error request_id=%s", getattr(request.state, "request_id", ""))
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=500, content={"error": {"code": "internal_error", "message": "服务器内部错误", "request_id": getattr(request.state, "request_id", "")}})


@app.exception_handler(HTTPException)
async def http_exception(request: Request, exc: HTTPException):
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=exc.status_code, headers=exc.headers, content={"error": {"code": f"http_{exc.status_code}", "message": str(exc.detail), "request_id": getattr(request.state, "request_id", "")}})

# Lazy-load the graph to speed up startup
_master_app = None

def get_master_app():
    global _master_app
    if _master_app is None:
        from agents.graph_new import master_app
        _master_app = master_app
    return _master_app


# --- 工具名称映射 (Tool / Node display names) ---
_NODE_LABELS = {
    "intent_gate": "检查需求与安全风险",
    "planner": "规划健康任务",
    # Executor can run chat/profile, report, pharmacy, insurance or clinic
    # tasks. Keep the label neutral; the concrete agent is recorded in trace.
    "executor": "执行已规划任务",
    "verifier": "核验分析结果",
    "responder": "整理最终答复",
    "clarification_response": "等待补充信息",
    "safety_response": "生成安全提示",
    "handoff": "转入人工协助",
    "paused": "AgentLoop 已暂停",
    # Legacy graph names remain supported for existing event consumers.
    "router": "分析您的需求",
    "clinic_node": "进入预问诊模块",
    "insurance_node": "进入医保咨询模块",
    "report_node": "进入报告解读模块",
    "advisor_node": "进入健康顾问模块",
    "pharmacy_node": "进入用药咨询模块",
}

# --- 技能工具 → 前端展示标签 ---
_SKILL_LABELS: dict[str, str] = {
    # insurance tools
    "get_insurance_balance":    "正在查询医保余额…",
    "get_consumption_records":  "正在查询消费明细…",
    "get_payment_records":      "正在查询缴费记录…",
    "get_cross_region_info":    "正在查询异地就医信息…",
    "search_insurance_policy":  "正在检索医保政策知识库…",
    # pharmacy tools
    "search_drug_info":         "正在查询药品信息…",
    "check_drug_interaction":   "正在检查药物相互作用…",
    "find_nearby_pharmacy":     "正在查找附近药店…",
    "get_otc_recommendation":   "正在检索用药建议…",
    # skills
    "load_skill":               "正在加载技能模块…",
    "emergency_triage":         "正在进行急症安全评估…",
    "symptom_scorer":           "正在评估症状严重程度…",
    "health_calculator":        "正在计算健康指标…",
    "lab_interpreter":          "正在解读化验指标…",
    "risk_assessor":            "正在评估慢性病风险…",
    "medication_calculator":    "正在计算用药剂量…",
}

# --- 医保工具 → 前端卡片 payload type 映射 ---
_INSURANCE_TOOL_TO_CARD_TYPE = {
    "get_insurance_balance": "insurance_balance",
    "get_consumption_records": "insurance_expenses",
    "get_payment_records": "insurance_payments",
    "get_cross_region_info": "insurance_cross_region",
}

_REPORT_TOOL_TO_CARD_TYPE = {
    "lab_interpreter": "report_analysis",
}

_MODE_TO_AGENT = {
    "clinic": "clinic_agent",
    "insurance": "insurance_agent",
    "report": "report_agent",
    "pharmacy": "pharmacy_agent",
    "general": "advisor_agent",
    "dashboard": "advisor_agent",
}


class ChatMessage(BaseModel):
    role: str
    content: str


class UserInfo(BaseModel):
    name: str = "用户"
    age: int | None = None
    medical_history: str = "无"
    elder_mode: bool = False
    region: str = ""
    # Optional explicit communication profile.  These fields affect wording,
    # never triage, diagnosis or medical facts.
    profession: str = ""
    occupation: str = ""
    expertise_level: str = ""
    communication_style: str = ""


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    user_info: UserInfo | None = None
    chat_mode: str = "general"
    conversation_id: str | None = None
    idempotency_key: str | None = None


def _sse_payload(payload: dict, event_id: int | str | None = None) -> str:
    prefix = f"id: {event_id}\n" if event_id is not None else ""
    return f"{prefix}data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _build_lc_messages(
    messages: list[ChatMessage], *, active_agent: str = ""
) -> list[HumanMessage | AIMessage]:
    """Convert API history to LangChain messages.

    The browser currently sends only ``role`` and ``content``.  For clinic
    conversations an assistant question is nevertheless an ``ask`` tool call
    in the GRPO protocol.  Reconstruct that call before the following patient
    message so ``clinic_node`` can replay the answer as a role=tool result.
    We intentionally use a narrow question-only heuristic and never rewrite
    ordinary assistant prose.
    """
    result: list[HumanMessage | AIMessage] = []
    for index, msg in enumerate(messages):
        content = msg.content.strip()
        if not content or msg.role not in ("user", "assistant"):
            continue
        if msg.role == "user":
            result.append(HumanMessage(content=msg.content))
            continue
        question = _extract_ask_question(content)
        is_question = (
            active_agent == "clinic_agent"
            and question is not None
            and index + 1 < len(messages)
            and messages[index + 1].role == "user"
        )
        if is_question:
            result.append(AIMessage(
                content=content,
                tool_calls=[{
                    "name": "ask",
                    "args": {"question": question},
                    "id": f"clinic-ask-{index}",
                    "type": "tool_call",
                }],
            ))
        else:
            result.append(AIMessage(content=msg.content))
    return result


def _extract_ask_question(content: str) -> str | None:
    """Extract only the final patient-facing question from an ask turn.

    The visible ask response may contain a short preamble before the question.
    That preamble must remain display text, but it must never become the
    tool-call argument when browser history is reconstructed.
    """
    text = content.strip()
    if not text or not text.endswith(("？", "?")):
        return None
    match = re.search(r"([^\n。！？?]*[？?])$", text)
    return match.group(1).strip() if match else text


def _build_initial_state(
    messages: list[ChatMessage],
    user_info: UserInfo | None,
    active_agent: str,
    user_id: str | None = None,
) -> dict:
    user_info_dict = user_info.model_dump() if user_info else {}
    memory_context = build_response_memory(user_info_dict, _build_lc_messages(messages, active_agent=active_agent))
    user_info_dict["memory_context"] = memory_context.as_dict()
    return {
        "messages": _build_lc_messages(messages, active_agent=active_agent)[-10:],
        "user_info": user_info_dict,
        "memory_context": memory_context.as_dict(),
        "next_agent": "",
        "active_agent": active_agent,
        "requested_mode": active_agent,
        "user_id": user_id,
    }


def _parse_user_info_json(raw: str) -> UserInfo:
    try:
        data = json.loads(raw) if raw else {}
        if not isinstance(data, dict):
            raise ValueError("user_info must be an object")
        return UserInfo(**data)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"user_info 格式错误: {exc}") from exc


def _parse_messages_json(raw: str) -> list[ChatMessage]:
    try:
        data = json.loads(raw) if raw else []
        if not isinstance(data, list):
            raise ValueError("messages must be a list")
        return [ChatMessage(**item) for item in data if isinstance(item, dict)]
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"messages 格式错误: {exc}") from exc


def _validate_chat_input(messages: list[ChatMessage]) -> None:
    if sum(len(message.content) for message in messages) > settings.max_input_chars:
        raise HTTPException(status_code=413, detail="输入内容超过允许长度")
    try:
        for message in messages:
            validate_untrusted_text(message.content)
    except UnsafePromptError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


async def _get_owned_conversation(db: AsyncSession, conversation_id: str, user: User) -> Conversation:
    conversation = await db.scalar(select(Conversation).where(Conversation.id == conversation_id, Conversation.user_id == user.id, Conversation.status == "active"))
    if conversation is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return conversation


async def _append_message(db: AsyncSession, conversation_id: str, role: str, content: str, metadata: dict | None = None) -> Message:
    last_sequence = await db.scalar(select(func.max(Message.sequence)).where(Message.conversation_id == conversation_id))
    message = Message(conversation_id=conversation_id, role=role, content=content, sequence=(last_sequence or 0) + 1, event_metadata=metadata)
    db.add(message)
    await db.flush()
    return message


async def _prepare_persistent_state(request: ChatRequest, user: User | None, db: AsyncSession | None, active_agent: str) -> tuple[dict, Conversation | None]:
    state = _build_initial_state(request.messages, request.user_info, active_agent, user.id if user else None)
    accepted_preferences: list[str] = []
    if user is not None and db is not None:
        accepted_preferences = await load_active_memory_preferences(db, user_id=user.id)
        if accepted_preferences:
            state["user_info"]["response_preferences"] = accepted_preferences
        try:
            recent = await load_relevant_health_events(db, user_id=user.id,
                query=request.messages[-1].content if request.messages else "")
            state["recent_health_context"] = health_context(recent)
            state["published_skill_context"] = await active_skills(db)
        except Exception:
            # Memory retrieval must never make the chat path unavailable.
            state["recent_health_context"] = health_context([])
            state["published_skill_context"] = []
    if user is None or db is None or not request.conversation_id:
        memory_context = build_response_memory(
            state.get("user_info", {}),
            state.get("messages", []),
            accepted_preferences=accepted_preferences,
        )
        state["memory_context"] = memory_context.as_dict()
        state["user_info"]["memory_context"] = memory_context.as_dict()
        state.setdefault("recent_health_context", health_context([]))
        state.setdefault("published_skill_context", [])
        return state, None
    conversation = await _get_owned_conversation(db, request.conversation_id, user)
    rows = await db.scalars(select(Message).where(Message.conversation_id == conversation.id).order_by(Message.sequence).limit(settings.max_chat_messages))
    stored_messages: list[Any] = []
    for row in rows:
        if row.role == "user":
            stored_messages.append(HumanMessage(content=row.content))
        elif row.role == "assistant":
            pending = (row.event_metadata or {}).get("clinic_pending_tool_call") if row.event_metadata else None
            if pending and pending.get("name") == "ask":
                question = str(pending.get("question") or _extract_ask_question(row.content) or row.content).strip()
                stored_messages.append(AIMessage(
                    content=row.content,
                    tool_calls=[{
                        "name": "ask",
                        "args": {"question": question},
                        "id": str(pending.get("id") or "clinic-call"),
                        "type": "tool_call",
                    }],
                ))
            else:
                stored_messages.append(AIMessage(content=row.content))
    incoming = request.messages[-1:] if request.messages else []
    if incoming and (not stored_messages or stored_messages[-1].content != incoming[-1].content):
        stored_messages.append(HumanMessage(content=incoming[-1].content))
        if incoming[-1].role == "user":
            await _append_message(db, conversation.id, "user", incoming[-1].content)
    conversation.active_agent = active_agent
    await db.commit()
    state["messages"] = stored_messages[-settings.max_chat_messages:][-10:]
    memory_context = build_response_memory(
        state.get("user_info", {}),
        state["messages"],
        accepted_preferences=accepted_preferences,
    )
    state["memory_context"] = memory_context.as_dict()
    state["user_info"]["memory_context"] = memory_context.as_dict()
    return state, conversation


async def _stream_agent_events(initial_state: dict, db: AsyncSession | None = None, conversation: Conversation | None = None, request_id: str | None = None, run: ChatRun | None = None, request: Request | None = None):
    assistant_text: list[str] = []
    input_tokens = 0
    output_tokens = 0
    started_at = perf_counter()
    terminal_node = ""
    node_names: list[str] = []
    tool_names: list[str] = []
    verify_status: str | None = None
    verify_issues: list[str] = []
    safety_flags: list[str] = []
    clinic_pending_tool_call: dict | None = None
    experience_recorded = False
    runtime: AgentRuntime | None = None
    try:
        event_sequence = max(0, int(request.headers.get("Last-Event-ID", "0"))) if request is not None else 0
    except (TypeError, ValueError):
        event_sequence = 0
    if db is not None and run is not None:
        stored_max = await db.scalar(
            select(func.max(AgentRunEvent.sequence)).where(AgentRunEvent.run_id == run.id)
        )
        event_sequence = max(event_sequence, int(stored_max or 0))

    async def _emit(payload: dict) -> str:
        """Assign an SSE id and persist it for replay when a durable run exists."""
        nonlocal event_sequence
        event_sequence += 1
        if db is not None and run is not None:
            db.add(AgentRunEvent(run_id=run.id, sequence=event_sequence, payload=payload))
            await db.commit()
        return _sse_payload(payload, event_sequence)

    async def _record_experience(outcome: str) -> None:
        nonlocal experience_recorded
        if experience_recorded or db is None or not initial_state.get("agent_run_id"):
            return
        experience_recorded = True
        latest_input = ""
        for message in reversed(initial_state.get("messages", [])):
            if isinstance(message, HumanMessage):
                latest_input = str(message.content or "")
                break
        # Health memory is written only from the user's latest message. The
        # extractor is deterministic and explicitly rejects third-person text.
        if db is not None and initial_state.get("user_id") and latest_input:
            try:
                await observe_health_message(
                    db,
                    user_id=str(initial_state["user_id"]),
                    text=latest_input,
                    conversation_id=conversation.id if conversation is not None else None,
                )
                await db.flush()
            except Exception:
                logger.exception("health memory write gate failed")
            try:
                await maybe_run_shadow_memory(
                    db,
                    user_id=str(initial_state["user_id"]),
                    latest_user_text=latest_input,
                    conversation_id=conversation.id if conversation is not None else None,
                    intent=str(initial_state.get("intent") or ""),
                )
                await db.flush()
            except Exception:
                logger.exception("shadow memory maintenance failed")
        payload = build_experience_payload(
            run_id=initial_state.get("agent_run_id"),
            user_id=initial_state.get("user_id") or (run.user_id if run is not None else None),
            intent=initial_state.get("intent", "unknown"),
            input_text=latest_input,
            plan_signature=initial_state.get("plan_signature", ""),
            node_names=node_names,
            tool_names=tool_names,
            result_summary="".join(assistant_text),
            verify_status=verify_status,
            verify_issues=verify_issues,
            safety_flags=safety_flags,
            metrics={
                "latency_ms": int((perf_counter() - started_at) * 1000),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "runtime": runtime.snapshot_state.as_dict() if runtime is not None else {},
            },
            outcome=outcome,
        )
        await persist_experience(db, payload)

    try:
        print("--- [API] Starting event stream ---", flush=True)
        runtime = AgentRuntime.from_state(
            initial_state,
            graph=get_master_app(),
            request_id=request_id or "",
            budget=RuntimeBudget(
                max_steps=settings.runtime_max_steps,
                max_tool_calls=settings.runtime_max_tool_calls,
                max_total_tokens=settings.runtime_max_total_tokens,
                timeout_seconds=float(settings.llm_timeout_seconds),
            ),
        )
        initial_state["runtime_context"] = runtime.runtime_state()["context"]
        yield await _emit({"type": "runtime_start", "runtime": runtime.describe()})
        events = runtime.astream_events(initial_state, version="v2").__aiter__()
        deadline = asyncio.get_running_loop().time() + settings.llm_timeout_seconds
        while True:
            if request is not None and await request.is_disconnected():
                try:
                    await events.aclose()
                except Exception:
                    pass
                if run is not None:
                    run.status = "cancelled"
                    run.completed_at = datetime.now(timezone.utc)
                    run.latency_ms = int((perf_counter() - started_at) * 1000)
                    await _record_experience("cancelled")
                    await db.commit()
                if initial_state.get("agent_run_id"):
                    await requeue_running_tasks(initial_state["agent_run_id"])
                    await mark_run_status(initial_state["agent_run_id"], "paused")
                return
            try:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise TimeoutError("LLM stream timeout")
                event = await asyncio.wait_for(events.__anext__(), timeout=min(settings.stream_heartbeat_seconds, remaining))
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError:
                yield ": ping\n\n"
                continue
            kind = event["event"]
            node_name = event.get("name", "")

            if kind not in ("on_chat_model_stream", "on_chat_model_start", "on_chat_model_end"):
                print(f"--- [Event] {kind} | Node: {node_name} ---", flush=True)

            # 1. Count model usage.  The new Supervisor graph intentionally
            # emits only the final responder text; intermediate domain-agent
            # generations are internal evidence, not user-facing answers.
            if kind == "on_chat_model_stream":
                continue

            elif kind == "on_chat_model_end":
                response = event.get("data", {}).get("output")
                response_metadata = getattr(response, "response_metadata", None) or {}
                usage = getattr(response, "usage_metadata", None) or response_metadata.get("token_usage", {})
                if isinstance(usage, dict):
                    input_tokens += int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0)
                    output_tokens += int(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0)

            # 2. A graph node is starting (shows agent status in UI)
            elif kind == "on_chain_start":
                if node_name and node_name not in node_names:
                    node_names.append(node_name)
                label = _NODE_LABELS.get(node_name)
                if label:
                    yield await _emit({"type": "node_start", "node": node_name, "content": label})

            # 3. A graph node finished
            elif kind == "on_chain_end":
                output = event.get("data", {}).get("output")
                if isinstance(output, dict) and output.get("clinic_pending_tool_call"):
                    clinic_pending_tool_call = output["clinic_pending_tool_call"]
                if node_name == "verifier":
                    output = event.get("data", {}).get("output")
                    if isinstance(output, dict):
                        verify_status = str(output.get("verify_status") or "") or None
                        verify_issues = [str(issue) for issue in (output.get("verify_issues") or [])][:32]
                if node_name in {"responder", "safety_response", "clarification_response", "handoff", "paused"}:
                    terminal_node = node_name
                if node_name == "safety_response":
                    safety_flags.append("safety_route")
                elif node_name == "handoff":
                    safety_flags.append("human_handoff")
                if node_name in _NODE_LABELS:
                    yield await _emit({"type": "node_end", "node": node_name})

                if node_name in {"responder", "safety_response", "clarification_response", "handoff", "paused"}:
                    final_text = ""
                    if isinstance(output, dict):
                        final_text = str(output.get("final_response") or "")
                        if not final_text:
                            output_messages = output.get("messages", [])
                            if isinstance(output_messages, list):
                                for message in reversed(output_messages):
                                    if isinstance(message, AIMessage):
                                        final_text = str(message.content or "")
                                        break
                    if final_text:
                        remaining_chars = settings.max_output_chars - sum(len(part) for part in assistant_text)
                        final_text = final_text[:max(0, remaining_chars)]
                        if final_text:
                            assistant_text.append(final_text)
                            yield await _emit({"type": "text", "content": final_text})

            # 4. Tool / skill calls
            elif kind == "on_tool_start":
                tool_name = event.get("name", "tool")
                if tool_name not in tool_names:
                    tool_names.append(tool_name)
                label = _SKILL_LABELS.get(tool_name, f"正在调用：{tool_name}")
                yield await _emit({"type": "tool_start", "tool": tool_name, "content": label})

            elif kind == "on_tool_end":
                tool_name = event.get("name", "tool")
                yield await _emit({"type": "tool_end", "tool": tool_name})

                # Emit a structured card event for tools that have card mappings
                card_type = (
                    _INSURANCE_TOOL_TO_CARD_TYPE.get(tool_name)
                    or PHARMACY_TOOL_TO_CARD_TYPE.get(tool_name)
                    or _REPORT_TOOL_TO_CARD_TYPE.get(tool_name)
                )
                if card_type:
                    try:
                        raw_output = event.get("data", {}).get("output", "{}")
                        # output may be a ToolMessage or raw string
                        if hasattr(raw_output, "content"):
                            raw_output = raw_output.content
                        tool_data = json.loads(raw_output)
                        yield await _emit({"type": "card", "payload": {"type": card_type, "data": tool_data}})
                    except Exception as parse_err:
                        print(f"--- [Card] Failed to parse tool output for {tool_name}: {parse_err} ---", flush=True)

        print("--- [API] Event stream finished successfully ---", flush=True)
        # Never complete an interactive request with only a finish event. This
        # can happen when a domain task exhausts retries (for example while
        # the clinic SSH tunnel is down) and the supervisor has no verified
        # text to pass to the responder. Surface a useful diagnostic instead
        # of leaving the browser on “处理完成” with an empty answer.
        if not assistant_text:
            diagnostic = (
                "本轮没有生成可展示的回答：专业健康模块未返回结果。"
                "请检查诊室模型连接（CLINIC_LLM_BASE_URL/SSH 隧道）后重试。"
            )
            assistant_text.append(diagnostic)
            yield await _emit({"type": "text", "content": diagnostic})
        outcome = "paused" if terminal_node == "paused" else "completed"
        # Persist the user-visible conversation before auxiliary telemetry.
        # Agent-run bookkeeping uses a separate transaction and may hit a
        # MySQL lock timeout; it must never roll back a completed answer.
        if db is not None and conversation is not None:
            if assistant_text:
                metadata = {"request_id": request_id, "status": "completed"}
                if clinic_pending_tool_call:
                    metadata["clinic_pending_tool_call"] = clinic_pending_tool_call
                await _append_message(db, conversation.id, "assistant", "".join(assistant_text), metadata)
            if run is not None:
                run.status = "paused" if terminal_node == "paused" else "completed"
                run.completed_at = datetime.now(timezone.utc)
                run.input_tokens = input_tokens
                run.output_tokens = output_tokens
                run.estimated_cost_micros = int(
                    input_tokens * settings.llm_input_cost_per_million_usd
                    + output_tokens * settings.llm_output_cost_per_million_usd
                )
                run.latency_ms = int((perf_counter() - started_at) * 1000)
            await db.commit()
        try:
            await _record_experience(outcome)
        except Exception as exc:
            logger.warning("experience persistence skipped after completed chat: %s", type(exc).__name__)
        if initial_state.get("agent_run_id"):
            try:
                await mark_run_status(initial_state["agent_run_id"], outcome)
            except Exception as exc:
                logger.warning("agent run status update skipped after completed chat: %s", type(exc).__name__)
        if runtime is not None:
            yield await _emit({"type": "runtime_end", "runtime": runtime.describe()})
        yield await _emit({"type": "finish", "request_id": request_id})

    except Exception as e:
        print(f"--- [API] Event stream error: {e} ---", flush=True)
        # Roll back any pending conversation/message writes before recording
        # the failure.  Experience is intentionally persisted in a fresh
        # transaction so a failed stream still produces offline evidence.
        if db is not None:
            await db.rollback()
        safety_flags.append("stream_error")
        if db is not None and run is not None:
            run.status = "failed"
            run.error_message = str(e)[:1000]
            run.completed_at = datetime.now(timezone.utc)
            run.input_tokens = input_tokens
            run.output_tokens = output_tokens
            run.latency_ms = int((perf_counter() - started_at) * 1000)
        if db is not None:
            await _record_experience("failed")
        if db is not None:
            # Commit both the ChatRun status and the newly recorded evidence.
            # This also keeps background/offline callers correct when no
            # ChatRun object was created for the request.
            await db.commit()
        if initial_state.get("agent_run_id"):
            await mark_run_status(initial_state["agent_run_id"], "failed", error_message=str(e)[:1000])
        if isinstance(e, RuntimeBudgetExceeded):
            error_content = "本轮分析已达到运行预算，系统已安全停止。请缩短问题或稍后重试。"
        else:
            error_content = str(e)
        if runtime is not None:
            yield await _emit({"type": "runtime_end", "runtime": runtime.describe()})
        yield await _emit({"type": "error", "content": error_content, "request_id": request_id})


@app.get("/")
async def root():
    return {"message": "大健康 AI 后端 v2.0 (LangGraph)"}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/metrics", include_in_schema=False)
async def metrics():
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(render_prometheus(), media_type="text/plain; version=0.0.4")


@app.get("/api/v1/health/live")
async def liveness():
    return {"status": "ok"}


@app.get("/api/v1/health/ready")
async def readiness():
    checks: dict[str, str] = {}
    try:
        async with AsyncSession(bind=engine) as session:
            await session.execute(text("SELECT 1"))
        checks["mysql"] = "ok"
    except Exception as exc:
        logger.warning("mysql readiness failed: %s", type(exc).__name__)
        checks["mysql"] = "error"
    try:
        await redis_client.ping()
        checks["redis"] = "ok"
    except Exception as exc:
        logger.warning("redis readiness failed: %s", type(exc).__name__)
        checks["redis"] = "error"
    # Verify the configured local vector index exists without eagerly loading
    # the embedding model on every readiness probe.
    vector_backend = os.getenv("VECTOR_STORE", "chroma").lower()
    if vector_backend == "chroma":
        persist_dir = Path(os.getenv("CHROMA_PERSIST_DIR", "rag/chroma_db"))
        checks["vector_store"] = "ok" if (persist_dir / "chroma.sqlite3").exists() else "error"
    elif vector_backend == "faiss":
        index_path = Path(os.getenv("FAISS_INDEX_PATH", "rag/faiss_index"))
        checks["vector_store"] = "ok" if index_path.exists() else "error"
    else:
        # Remote stores are validated by their first retrieval; here we only
        # ensure the backend is explicitly configured.
        checks["vector_store"] = "ok" if vector_backend in {"qdrant", "pgvector"} else "error"
    ready = all(value == "ok" for value in checks.values())
    from fastapi.responses import JSONResponse
    return JSONResponse({"status": "ok" if ready else "not_ready", "checks": checks}, status_code=200 if ready else 503)


async def _recognize_image_cached(image_bytes: bytes, content_type: str, scan_type: str) -> str:
    """Deduplicate identical vision requests across workers using Redis."""
    digest = __import__("hashlib").sha256(image_bytes).hexdigest()
    key = cache_key("vision", f"{digest}:{scan_type}")
    cached = await cache_get_json(key)
    if isinstance(cached, dict) and cached.get("text"):
        return str(cached["text"])
    async with distributed_lock(f"vision:{digest}:{scan_type}") as acquired:
        if not acquired:
            raise VisionInputError("该图片正在分析，请稍后重试")
        cached = await cache_get_json(key)
        if isinstance(cached, dict) and cached.get("text"):
            return str(cached["text"])
        result = await recognize_image(image_bytes, content_type, scan_type)  # type: ignore[arg-type]
        await cache_set_json(key, {"text": result}, settings.cache_default_ttl_seconds)
        return result


async def _chat_impl(http_request: Request | None, request: ChatRequest, request_id: str = "", user: User | None = None, db: AsyncSession | None = None):
    _validate_chat_input(request.messages)
    try:
        model_settings = resolve_model_settings()
    except LLMConfigurationError as exc:
        raise HTTPException(status_code=500, detail=f"大模型配置错误：{exc}") from exc

    active_agent = _MODE_TO_AGENT.get(request.chat_mode, "advisor_agent")
    initial_state, conversation = await _prepare_persistent_state(request, user, db, active_agent)
    run = None
    if conversation is not None and user is not None and db is not None:
        run = ChatRun(id=request_id or str(uuid4()), conversation_id=conversation.id, user_id=user.id, status="pending", model_name=model_settings.model)
        db.add(run)
        await db.commit()
        initial_state["agent_run_id"] = run.id
        await create_agent_run(
            run_id=run.id,
            conversation_id=conversation.id,
            user_id=user.id,
            requested_mode=active_agent,
            initial_state=initial_state,
        )
    return StreamingResponse(
        _stream_agent_events(initial_state, db=db, conversation=conversation, request_id=request_id, run=run, request=http_request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def chat(request: ChatRequest):
    return await _chat_impl(None, request)


@app.post("/api/chat")
async def legacy_chat(request: ChatRequest, request_obj: Request):
    await enforce_rate_limit(request_obj, bucket="chat")
    return await chat(request)


@app.post("/api/v1/chat")
async def chat_v1(request: ChatRequest, request_obj: Request, user: User | None = Depends(get_optional_current_user), db: AsyncSession = Depends(get_db)):
    await enforce_rate_limit(request_obj, subject=user.id if user else None, bucket="chat")
    if user is not None and request.idempotency_key:
        key = f"chat:idem:{user.id}:{request.idempotency_key}"
        try:
            accepted = await redis_client.set(key, request.conversation_id or "", ex=86400, nx=True)
            if not accepted:
                raise HTTPException(status_code=409, detail="请求已处理，请勿重复提交")
        except HTTPException:
            raise
        except Exception as exc:
            logger.warning("idempotency store unavailable: %s", type(exc).__name__)
    return await _chat_impl(request_obj, request, request_obj.state.request_id, user, db)


@app.post("/api/v1/agent-runs/{run_id}/pause")
async def pause_agent_run(run_id: str, user: User = Depends(get_current_user)):
    if not await request_pause(run_id, user.id):
        raise HTTPException(status_code=404, detail="AgentLoop 运行不存在或已经结束")
    return {"run_id": run_id, "status": "pause_requested"}


@app.get("/api/v1/agent-runs/{run_id}")
async def agent_run_status(run_id: str, user: User = Depends(get_current_user)):
    snapshot = await get_run_snapshot(run_id, user.id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="AgentLoop 运行不存在")
    return snapshot


@app.get("/api/v1/agent-runs/{run_id}/events")
async def agent_run_events(
    run_id: str,
    after: int = 0,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return durable SSE events after a cursor for reconnecting clients."""
    if after < 0:
        raise HTTPException(status_code=400, detail="after 必须为非负整数")
    run = await db.scalar(select(AgentRun).where(AgentRun.id == run_id, AgentRun.user_id == user.id))
    if run is None:
        raise HTTPException(status_code=404, detail="AgentLoop 运行不存在")
    rows = await db.scalars(
        select(AgentRunEvent)
        .where(AgentRunEvent.run_id == run_id, AgentRunEvent.sequence > after)
        .order_by(AgentRunEvent.sequence)
        .limit(500)
    )
    events = list(rows)
    return {
        "run_id": run_id,
        "after": after,
        "next_cursor": events[-1].sequence if events else after,
        "events": [{"id": event.sequence, "data": event.payload} for event in events],
    }


@app.post("/api/v1/agent-runs/{run_id}/resume")
async def resume_agent_run(run_id: str, request_obj: Request, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    state = await load_resume_state(run_id, user.id)
    if state is None:
        raise HTTPException(status_code=404, detail="AgentLoop 运行不存在")
    run = await get_run(run_id, user.id)
    if run is None or run.status in {"completed", "failed", "cancelled"}:
        raise HTTPException(status_code=409, detail="该 AgentLoop 不能恢复")
    if not await resume_run(run_id, user.id):
        raise HTTPException(status_code=409, detail="该 AgentLoop 不能恢复")
    conversation = await db.scalar(select(Conversation).where(
        Conversation.id == run.conversation_id,
        Conversation.user_id == user.id,
    ))
    chat_run = await db.get(ChatRun, run_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="关联会话不存在")
    if chat_run is not None:
        chat_run.status = "pending"
        chat_run.error_message = None
        chat_run.completed_at = None
        await db.commit()
    return StreamingResponse(
        _stream_agent_events(
            state,
            db=db,
            conversation=conversation,
            request_id=request_obj.state.request_id,
            run=chat_run,
            request=request_obj,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/vision-chat")
@app.post("/api/v1/vision-chat")
async def vision_chat(
    request: Request,
    file: UploadFile = File(...),
    scan_type: str = Form(...),
    user_info: str = Form("{}"),
    messages: str = Form("[]"),
):
    await enforce_rate_limit(request, bucket="vision")
    normalized_scan_type = normalize_scan_type(scan_type)
    parsed_user_info = _parse_user_info_json(user_info)
    parsed_messages = _parse_messages_json(messages)
    _validate_chat_input(parsed_messages)

    image_bytes = await file.read()
    try:
        validate_image_content(image_bytes, file.content_type, settings.max_image_pixels)
    except VisionInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    async def event_generator():
        try:
            yield _sse_payload({
                "type": "tool_start",
                "tool": "vision_model",
                "content": "正在识别图片内容…",
            })
            vision_text = await _recognize_image_cached(
                image_bytes,
                file.content_type or "",
                normalized_scan_type,
            )
            yield _sse_payload({"type": "tool_end", "tool": "vision_model"})

            injected_message = compose_agent_message(normalized_scan_type, vision_text)
            vision_messages = [
                *parsed_messages[-9:],
                ChatMessage(role="user", content=injected_message),
            ]
            active_agent = SCAN_TYPE_TO_AGENT[normalized_scan_type]
            initial_state = _build_initial_state(vision_messages, parsed_user_info, active_agent, user.id if user else None)
            async for payload in _stream_agent_events(initial_state):
                yield payload

        except VisionInputError as exc:
            yield _sse_payload({"type": "error", "content": str(exc)})
        except Exception as exc:
            print(f"--- [Vision] Event stream error: {exc} ---", flush=True)
            yield _sse_payload({"type": "error", "content": "图片识别失败，请稍后再试"})

    return StreamingResponse(event_generator(), media_type="text/event-stream")
