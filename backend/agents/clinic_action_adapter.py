"""Adapter between the clinic Qwen action space and project capabilities.

The local model only sees the five actions used during GRPO training. Project
tools stay behind this boundary, so model tool names and schemas cannot widen
production permissions by accident.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Mapping, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from agents.advisor import search_medical_literature
from agents.pharmacy import check_drug_interaction
from rag.knowledge_base import get_knowledge_base
from skills.emergency_triage.skill import EmergencyTriageSkill


CLINIC_ACTION_TOOLS: tuple[dict[str, Any], ...] = (
    {
        "type": "function",
        "function": {
            "name": "ask",
            "description": "向患者提出一个尚未问过的追问。",
            "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string", "description": "一个需要患者回答的医学问题。"}},
                "required": ["question"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check",
            "description": "查询已有检查结果、生命体征、红旗规则或药物相互作用。",
            "parameters": {
                "type": "object",
                "properties": {"item": {"type": "string", "description": "需要查询的检查、指标或药品信息。"}},
                "required": ["item"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "查询项目本地医疗知识库。",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "医学知识库查询词。"}},
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "通过项目配置的受控医学资源搜索，不进行任意网页搜索。",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "外部医学资源查询词。"}},
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "answer",
            "description": "给出最终分诊回答并结束问诊。",
            "parameters": {
                "type": "object",
                "properties": {"content": {"type": "string", "description": "最终回答内容。"}},
                "required": ["content"],
                "additionalProperties": False,
            },
        },
    },
)

_EV_ID = re.compile(r"\bev_[A-Za-z0-9_-]+\b")
_DRUG_SPLIT = re.compile(r"(?:相互作用|能否同服|一起吃|联用|和|与|、|,|，|/|\+|[:：])")
_CITATION_ID = re.compile(r"(?:rag|pubmed|doi):[A-Za-z0-9_.:/-]+")
_DRUG_PREFIX = re.compile(r"^(?:药物)?相互作用\s*[:：\-]?\s*", flags=re.I)
_DRUG_INFO_HINT = re.compile(r"(?:药品|药物|说明书|适应症|用法用量|剂量|副作用|禁忌)")
_DRUG_PUNCTUATION = " \t\r\n:：,，。.;；、/\\+&和与及以及()（）[]【】{}<>《》\"'`"


def _text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)


def _clean_drug_name(value: Any) -> str:
    """Normalize a model-provided drug name before calling a pharmacy tool."""
    return _text(value).strip().strip(_DRUG_PUNCTUATION)


def _normalize_question(value: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]", "", value.casefold())


def _last_human_text(messages: Sequence[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return _text(message.content)
    return ""


def _all_human_text(messages: Sequence[BaseMessage]) -> str:
    return "\n".join(_text(message.content) for message in messages if isinstance(message, HumanMessage))[-12000:]


def _citation_label(source: str, section: str = "") -> str:
    return f"rag:{source}:{section}" if section else f"rag:{source}"


def sanitize_clinic_answer(content: str, citation_ids: set[str]) -> str:
    """Remove training-only evidence IDs and untrusted citation claims."""
    cleaned = _EV_ID.sub("", str(content or ""))
    for token in _CITATION_ID.findall(cleaned):
        if token not in citation_ids:
            cleaned = cleaned.replace(token, "")
    lines: list[str] = []
    for line in cleaned.splitlines():
        if re.search(r"(?:证据|引用|citation)\s*[:：]", line, flags=re.I):
            tokens = set(_CITATION_ID.findall(line))
            if tokens and not tokens.intersection(citation_ids):
                continue
        lines.append(line.rstrip())
    return "\n".join(lines).strip()


@dataclass
class ClinicActionAdapter:
    """Execute the five trained actions against bounded project services."""

    state: Mapping[str, Any]
    messages: Sequence[BaseMessage]
    citation_ids: set[str] = field(default_factory=set)
    asked_questions: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        for message in self.messages:
            if isinstance(message, AIMessage):
                text = _text(message.content)
                for question in re.findall(r"[^。！？!?\n]{2,120}[？?]", text):
                    self.asked_questions.add(_normalize_question(question))

    def emergency_veto(self) -> Any:
        info = self.state.get("user_info", {}) or {}
        age = info.get("age")
        try:
            age = int(age) if age not in (None, "") else None
        except (TypeError, ValueError):
            age = None
        return EmergencyTriageSkill().run(
            symptoms_text=_all_human_text(self.messages),
            age=age,
        )

    def _is_duplicate_question(self, question: str) -> bool:
        normalized = _normalize_question(question)
        if not normalized:
            return True
        return any(
            normalized == old or SequenceMatcher(None, normalized, old).ratio() >= 0.9
            for old in self.asked_questions
        )

    async def execute(self, name: str, arguments: Mapping[str, Any]) -> str:
        args = dict(arguments)
        if name == "ask":
            question = _text(args.get("question", "")).strip()
            if self._is_duplicate_question(question):
                return json.dumps({"success": False, "action": "ask", "error": "duplicate_question", "message": "这个问题已经问过，请换一个缺失信息继续追问。"}, ensure_ascii=False)
            self.asked_questions.add(_normalize_question(question))
            return json.dumps({"success": True, "action": "ask", "question": question}, ensure_ascii=False)
        if name == "check":
            return await self._check(_text(args.get("item", "")).strip())
        if name == "lookup":
            return await self._lookup(_text(args.get("query", "")).strip())
        if name == "search":
            return await self._search(_text(args.get("query", "")).strip())
        raise ValueError(f"unsupported clinic action: {name}")

    async def _check(self, item: str) -> str:
        if not item:
            return json.dumps({"success": False, "action": "check", "error": "missing_item"}, ensure_ascii=False)
        if any(word in item for word in ("相互作用", "能否同服", "一起吃", "联用")):
            interaction_text = _DRUG_PREFIX.sub("", item.strip())
            candidates = [
                _clean_drug_name(part)
                for part in _DRUG_SPLIT.split(interaction_text)
                if _clean_drug_name(part) and len(_clean_drug_name(part)) <= 80
            ]
            if len(candidates) >= 2:
                result = await check_drug_interaction.ainvoke({"drug1": candidates[-2], "drug2": candidates[-1]})
                return json.dumps({"success": True, "action": "check", "route": "drug_interaction", "result": json.loads(_text(result))}, ensure_ascii=False)
            return json.dumps({"success": False, "action": "check", "error": "need_two_drugs", "message": "请提供需要比较的两种药品名称。"}, ensure_ascii=False)
        if _DRUG_INFO_HINT.search(item):
            drug_name = _clean_drug_name(re.sub(r"^(?:查询|查|了解|请问)\s*", "", item))
            if drug_name:
                result = await search_drug_info.ainvoke({"drug_name": drug_name})
                return json.dumps({"success": True, "action": "check", "route": "drug_info", "drug_name": drug_name, "result": json.loads(_text(result))}, ensure_ascii=False)
        if any(word in item for word in ("症状", "红旗", "急症", "生命体征")):
            triage = self.emergency_veto()
            return json.dumps({"success": True, "action": "check", "route": "emergency_triage", "result": triage.model_dump()}, ensure_ascii=False)
        return await self._lookup(item)

    async def _lookup(self, query: str) -> str:
        if not query:
            return json.dumps({"success": False, "action": "lookup", "error": "missing_query"}, ensure_ascii=False)
        docs = await get_knowledge_base().aretrieve(query, k=4)
        evidence: list[dict[str, str]] = []
        for doc in docs:
            source = str(doc.metadata.get("source", "unknown"))
            section = str(doc.metadata.get("section", ""))
            citation_id = _citation_label(source, section)
            self.citation_ids.add(citation_id)
            evidence.append({"citation_id": citation_id, "source": source, "section": section, "content": doc.page_content[:1800]})
        return json.dumps({"success": True, "action": "lookup", "query": query, "evidence": evidence}, ensure_ascii=False)

    async def _search(self, query: str) -> str:
        if not query:
            return json.dumps({"success": False, "action": "search", "error": "missing_query"}, ensure_ascii=False)
        raw = await search_medical_literature.ainvoke({"query": query, "max_results": 5})
        parsed: Any
        try:
            parsed = json.loads(_text(raw))
        except (TypeError, json.JSONDecodeError):
            parsed = {"message": _text(raw)[:4000]}
        if isinstance(parsed, dict):
            for row in parsed.get("results", parsed.get("articles", [])) or []:
                if isinstance(row, dict):
                    value = row.get("pmid") or row.get("doi") or row.get("url")
                    if value:
                        self.citation_ids.add(f"pubmed:{value}")
        return json.dumps({"success": True, "action": "search", "query": query, "result": parsed}, ensure_ascii=False)
