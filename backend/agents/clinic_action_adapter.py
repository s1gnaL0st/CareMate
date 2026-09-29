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
from agents.pharmacy import check_drug_interaction, search_drug_info
from rag.knowledge_base import get_knowledge_base
from skills.emergency_triage.skill import EmergencyTriageSkill


CLINIC_ACTION_TOOLS: tuple[dict[str, Any], ...] = (
    {
        "type": "function",
        "function": {
            "name": "ask",
            "description": "向患者追问一个当前最关键的临床问题。",
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
            "description": "执行确定性的红旗、药物相互作用或生命体征规则。",
            "parameters": {
                "type": "object",
                "properties": {
                    "check_type": {"type": "string", "enum": ["red_flags", "drug_interaction", "vital_thresholds"]},
                    "items": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["check_type", "items"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "读取当前患者的结构化病例字段。",
            "parameters": {
                "type": "object",
                "properties": {"field": {"type": "string", "enum": ["age", "sex", "history", "medications", "allergies", "uploaded_reports", "profile", "all"]}},
                "required": ["field"],
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
                "properties": {
                    "triage_level": {"type": "string", "enum": ["emergency", "urgent", "routine", "self_care"]},
                    "content": {"type": "string", "description": "面向患者的最终回复。"},
                },
                "required": ["triage_level", "content"],
                "additionalProperties": False,
            },
        },
    },
)

_EV_ID = re.compile(r"\bev_[A-Za-z0-9_-]+\b")
_THINK_BLOCK = re.compile(r"\\?<(?:think|analysis)>.*?\\?</(?:think|analysis)>", flags=re.I | re.S)
_THINK_TAG = re.compile(r"\\?</?(?:think|analysis)>", flags=re.I)
_SECTION_LABEL = re.compile(r"\s*(?:第一段|第二段|第三段|回答|解释|建议|补充信息)\s*[:：]\s*", flags=re.I)
_DRUG_SPLIT = re.compile(r"(?:相互作用|能否同服|一起吃|联用|和|与|、|,|，|/|\+|[:：])")
_CITATION_ID = re.compile(r"(?:rag|pubmed|doi):[A-Za-z0-9_.:/-]+")
_DRUG_PREFIX = re.compile(r"^(?:药物)?相互作用\s*[:：\-]?\s*", flags=re.I)
_DRUG_INFO_HINT = re.compile(r"(?:药品|药物|说明书|适应症|用法用量|剂量|副作用|禁忌)")
_DRUG_PUNCTUATION = " \t\r\n:：,，。.;；、/\\+&和与及以及()（）[]【】{}<>《》\"'`"

# These are conversation dimensions, not disease-specific rules.  They let
# the harness recognize that a patient's answer covered a question even when
# the next model question uses different wording.
_QUESTION_FACET_PATTERNS: dict[str, tuple[str, ...]] = {
    "onset": ("什么时候", "何时", "多久", "开始", "起病", "出现", "以来", "昨天", "今天", "前天", "刚才", "突然"),
    "location": ("哪里", "哪颗", "哪一颗", "部位", "位置", "哪侧", "左侧", "右侧", "中央", "胸口", "单颗", "多颗", "一颗", "一片", "上颌", "下颌"),
    "severity": ("多严重", "严重程度", "多明显", "几分", "影响吃饭", "影响睡觉", "疼痛程度"),
    "trend": ("加重", "减轻", "好转", "恶化", "持续", "反复", "越来越", "变化"),
    "trigger": ("诱因", "什么会", "吃冷", "吃热", "吃甜", "吃酸", "活动时", "什么情况下"),
    "associated": ("伴随", "同时", "有没有", "有无", "没有", "还会", "另外"),
    # Generic safety dimension. The exact wording remains model-selected; these
    # aliases only let the harness recognize that a previously asked safety
    # dimension was answered, even when the patient replies "没有".
    "red_flag": (
        "红旗", "肿胀", "张口", "吞咽", "呼吸", "流口水", "麻木",
        "意识", "昏厥", "剧烈胸痛", "胸痛", "持续剧痛",
    ),
}

@dataclass(frozen=True)
class ClinicHarnessState:
    """Generic turn policy; domain skills still decide question content.

    The harness must not infer dental (or any other domain) slots from
    keywords.  It only counts assistant turns that contained a question and
    takes away ``ask`` after a bounded number of turns, so a tool-calling model
    cannot create an endless interview.
    """

    question_turns: int
    max_question_turns: int
    force_answer: bool


def _facets_in_text(text: str) -> set[str]:
    """Return generic symptom-information dimensions present in text."""
    value = str(text or "").casefold()
    return {
        facet
        for facet, patterns in _QUESTION_FACET_PATTERNS.items()
        if any(pattern.casefold() in value for pattern in patterns)
    }


def answered_facets(messages: Sequence[BaseMessage]) -> set[str]:
    """Infer coarse dimensions answered by the patient.

    A short reply such as ``没有`` carries no facet keywords by itself. When
    it follows an assistant question, the question's facets are nevertheless
    answered (negatively), so later paraphrases cannot reopen the same line
    of questioning.
    """
    facets: set[str] = set()
    previous_assistant: BaseMessage | None = None
    for message in messages:
        if isinstance(message, HumanMessage):
            facets.update(_facets_in_text(_text(message.content)))
            if previous_assistant is not None:
                facets.update(_facets_in_text(_text(previous_assistant.content)))
            previous_assistant = None
        elif isinstance(message, AIMessage):
            previous_assistant = message
        else:
            previous_assistant = None
    return facets


def question_facets(question: str) -> set[str]:
    return _facets_in_text(question)


def _trim_answered_question(question: str, answered: set[str]) -> str:
    """Remove already-covered clauses from a compound model question."""
    parts = [part.strip() for part in re.split(r"[，,；;]|或|以及|并且", question) if part.strip()]
    if len(parts) <= 1:
        return question.strip()
    remaining = [part for part in parts if not (question_facets(part) and question_facets(part).issubset(answered))]
    if len(remaining) == len(parts):
        return question.strip()
    if not remaining:
        return ""
    text = "，".join(remaining).rstrip("？?").strip()
    # A clause-level reduction must remain a complete patient-facing question.
    # The model may naturally emit a compound question such as
    # ``是否突然发生或持续加重？``; removing the answered clause must not leave
    # fragments like ``持续加重？`` or ``日常活动？`` in the UI.
    if len(remaining) == 1 and text and not re.search(r"(?:是否|有没有|有无|哪|什么|多久|何时|多少|是否有)", text):
        if re.match(r"^(?:持续|突然|逐渐|反复|越来越|明显|严重|影响)", text):
            text = f"症状是否{text}"
        else:
            text = f"是否影响{text}"
    if text and not text.endswith(("？", "?")):
        text += "？"
    return text


_GENERIC_FALLBACK_QUESTIONS = (
    "请补充一个与当前症状相关的重要信息？",
    "还可以补充症状变化或伴随情况中的一项吗？",
)


def clinic_harness_state(messages: Sequence[BaseMessage], *, max_question_turns: int = 4) -> ClinicHarnessState:
    """Compute generic stop-policy state from assistant output only."""
    question_turns = 0
    # Count only the current interview segment. A completed answer without a
    # question starts a fresh segment if the patient later opens another topic.
    for message in reversed(messages):
        if not isinstance(message, AIMessage):
            continue
        if re.search(r"[^。！？!?\n]{2,160}[？?]", _text(message.content)):
            question_turns += 1
            continue
        break
    max_question_turns = max(1, int(max_question_turns))
    return ClinicHarnessState(
        question_turns=question_turns,
        max_question_turns=max_question_turns,
        force_answer=question_turns >= max_question_turns,
    )


def fallback_clinic_answer(messages: Sequence[BaseMessage]) -> str:
    """Safe final response when the model remains unavailable after retries."""
    return (
        "根据目前提供的信息，暂时不能确认具体原因。建议尽快线下就医评估，"
        "先避免可能加重症状的刺激，并记录症状变化。"
        "如果出现明显加重、剧烈疼痛、呼吸或吞咽困难、意识异常等情况，"
        "请立即就医或拨打120。"
    )


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
    cleaned = _THINK_BLOCK.sub("", str(content or ""))
    cleaned = _THINK_TAG.sub("", cleaned)
    # Some Qwen serving templates wrap a plain response in an answer tag even
    # when no tool call is present. The tag is transport syntax, not patient
    # content, so remove only the wrapper.
    cleaned = re.sub(r"</?(?:answer|assistant)>", "", cleaned, flags=re.I)
    cleaned = _SECTION_LABEL.sub("\n", cleaned)
    cleaned = _EV_ID.sub("", cleaned)
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
    answered_facets: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.answered_facets = answered_facets(self.messages)
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
        if any(
            normalized == old or SequenceMatcher(None, normalized, old).ratio() >= 0.9
            for old in self.asked_questions
        ):
            return True
        facets = question_facets(question)
        # A compound question is considered covered only when every dimension
        # it asks about is already present in the patient's answers.
        # Safety questions often contain generic words such as "是否" or
        # "持续" that also match ordinary facets. Treat that whole safety
        # cluster as one dimension once the prior safety question was answered.
        if (
            "red_flag" in facets
            and "red_flag" in self.answered_facets
            and facets.issubset({"red_flag", "associated", "onset", "trend"})
        ):
            return True
        return bool(facets) and facets.issubset(self.answered_facets)

    def next_fallback_question(self) -> str | None:
        """Return the next bounded question after a duplicate model action.

        This is only reached after the model has emitted a repeated ``ask``.
        It is intentionally independent of intent routing and does not replace
        the GRPO model; it prevents one malformed/repeated tool call from
        producing the same generic sentence on every user turn.
        """
        for question in _GENERIC_FALLBACK_QUESTIONS:
            if not self._is_duplicate_question(question):
                self.asked_questions.add(_normalize_question(question))
                return question
        return None

    async def execute(self, name: str, arguments: Mapping[str, Any]) -> str:
        args = dict(arguments)
        if name == "ask":
            question = _trim_answered_question(
                _text(args.get("question", "")).strip(), self.answered_facets
            )
            if self._is_duplicate_question(question):
                return json.dumps({"success": False, "action": "ask", "error": "duplicate_question", "message": "这个问题已经问过，请换一个缺失信息继续追问。"}, ensure_ascii=False)
            self.asked_questions.add(_normalize_question(question))
            return json.dumps({"success": True, "action": "ask", "question": question}, ensure_ascii=False)
        if name == "check":
            check_type = _text(args.get("check_type", "")).strip()
            items = args.get("items")
            if check_type not in {"red_flags", "drug_interaction", "vital_thresholds"} or not isinstance(items, list) or not items:
                return json.dumps({"success": False, "action": "check", "error": "invalid_check_arguments", "message": "check 需要合法的 check_type 和非空 items。"}, ensure_ascii=False)
            return await self._check(check_type, [str(item) for item in items])
        if name == "lookup":
            return await self._lookup(_text(args.get("field", "")).strip())
        if name == "search":
            return await self._search(_text(args.get("query", "")).strip())
        raise ValueError(f"unsupported clinic action: {name}")

    async def _check(self, check_type: str, items: list[str]) -> str:
        item = "，".join(items)
        if check_type == "drug_interaction":
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
        if check_type == "red_flags":
            triage = self.emergency_veto()
            return json.dumps({"success": True, "action": "check", "route": "emergency_triage", "result": triage.model_dump()}, ensure_ascii=False)
        if check_type == "vital_thresholds":
            results = []
            for raw in items:
                name, _, value = raw.partition(":")
                try:
                    numeric = float(value)
                except ValueError:
                    results.append({"name": name or raw, "value": value, "triggered": False, "error": "invalid_value"})
                    continue
                threshold = 140 if name in {"systolic_bp", "收缩压"} else None
                results.append({"name": name, "value": numeric, "triggered": bool(threshold and numeric >= threshold), "severity": "high" if threshold and numeric >= threshold else "normal"})
            return json.dumps(results, ensure_ascii=False)
        return json.dumps({"success": False, "action": "check", "error": "unsupported_check_type"}, ensure_ascii=False)

    async def _lookup(self, field: str) -> str:
        allowed = {"age", "sex", "history", "medications", "allergies", "uploaded_reports", "profile", "all"}
        if field not in allowed:
            return json.dumps({"success": False, "action": "lookup", "error": "invalid_field"}, ensure_ascii=False)
        info = dict(self.state.get("user_info", {}) or {})
        mapping = {"age": "age", "sex": "sex", "history": "medical_history", "medications": "medications", "allergies": "allergies", "uploaded_reports": "uploaded_reports"}
        remembered = info.get("response_preferences") or (info.get("memory_context") or {}).get("active_preferences") or []
        if not isinstance(remembered, list):
            remembered = []
        profile = [str(item) for item in remembered if str(item).strip()]
        if field == "all":
            result = {key: info.get(key) for key in mapping.values() if info.get(key) not in (None, "", [], {})}
            if profile:
                result["remembered_profile"] = profile[:10]
        elif field == "profile":
            result = {"profile": profile[:10]}
        else:
            key = mapping[field]
            result = {field: info.get(key)} if info.get(key) not in (None, "", [], {}) else {field: None}
        return json.dumps(result, ensure_ascii=False)

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
