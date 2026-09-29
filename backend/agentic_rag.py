"""Agent-led iterative retrieval with explicit evidence sufficiency checks."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any, Protocol


class Retriever(Protocol):
    async def aretrieve(self, query: str, k: int = 3) -> list[Any]: ...


@dataclass
class RetrievalRound:
    round: int
    query: str
    source_ids: list[str]
    new_source_ids: list[str]
    evidence_count: int
    missing_facets: list[str]

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class AgenticRetrievalResult:
    documents: list[Any] = field(default_factory=list)
    rounds: list[RetrievalRound] = field(default_factory=list)
    sufficient: bool = False
    missing_facets: list[str] = field(default_factory=list)
    stop_reason: str = ""

    def trace(self) -> dict[str, Any]:
        return {
            "schema_version": "agentic_rag_v1",
            "sufficient": self.sufficient,
            "missing_facets": self.missing_facets,
            "stop_reason": self.stop_reason,
            "rounds": [item.as_dict() for item in self.rounds],
            "source_ids": [_source_id(doc) for doc in self.documents],
        }


async def _rerank_documents(documents: list[Any], llm: Any | None, query: str) -> list[Any]:
    """Use the configured chat model as a bounded evidence reranker."""
    if llm is None or len(documents) < 2:
        return documents
    evidence = "\n".join(
        f"[{index}] {_source_id(doc)}: {_text(doc)[:1400]}"
        for index, doc in enumerate(documents)
    )
    try:
        response = await llm.ainvoke(
            "你是医疗知识库证据重排器。根据用户问题，按证据与问题的直接相关性、"
            "权威性和可支持性从高到低排序。不要回答问题，只输出 JSON："
            "{\"order\":[整数索引...] }。必须包含所有索引且不能重复。\n"
            f"用户问题：{query}\n候选证据：\n{evidence}"
        )
        raw = getattr(response, "content", response)
        raw = raw if isinstance(raw, str) else str(raw)
        match = re.search(r"\{.*\}", raw, re.S)
        order = json.loads(match.group(0)).get("order", []) if match else []
        order = [int(index) for index in order]
        if sorted(order) != list(range(len(documents))):
            return documents
        return [documents[index] for index in order]
    except Exception:
        return documents


def _source_id(doc: Any) -> str:
    metadata = getattr(doc, "metadata", {}) or {}
    return str(metadata.get("parent_id") or metadata.get("source_id") or
               f"{metadata.get('source', 'unknown')}::{metadata.get('section', '')}")


def _text(doc: Any) -> str:
    return str(getattr(doc, "page_content", "") or "").casefold()


def _facets(query: str) -> list[tuple[str, str]]:
    text = str(query or "").casefold()
    if any(term in text for term in ("药", "吃", "服用", "相互作用", "otc")):
        return [("indication", "适应症 用法 用量"), ("safety", "禁忌 不良反应 相互作用 合并用药")]
    if any(term in text for term in ("报告", "化验", "指标", "血常规", "检查")):
        return [("reference", "参考范围 指标 临床意义"), ("follow_up", "复查 就医 风险")]
    if any(term in text for term in ("医保", "报销", "备案", "缴费")):
        return [("policy", "报销 政策 目录"), ("process", "流程 条件 材料 地区")]
    if any(term in text for term in ("咳嗽", "发热", "疼", "头晕", "症状", "不舒服")):
        return [("risk", "危险信号 红旗 就医"), ("care", "处理 建议 注意事项")]
    return [("definition", "定义 原因 机制"), ("action", "处理 建议 注意事项")]


def _missing(documents: list[Any], facets: list[tuple[str, str]]) -> list[str]:
    corpus = " ".join(_text(doc) for doc in documents)
    return [name for name, terms in facets if not any(term.casefold() in corpus for term in terms.split())]


async def retrieve_until_sufficient(
    query: str,
    retriever: Retriever,
    *,
    k: int = 3,
    max_rounds: int = 3,
    min_sources: int = 2,
    controller_llm: Any | None = None,
    rerank_llm: Any | None = None,
) -> AgenticRetrievalResult:
    """Let a retrieval controller search, observe gaps, and search again.

    A round is not considered complete merely because it returned documents:
    the controller checks source diversity and coverage of query-specific
    evidence facets before stopping.
    """
    facets = _facets(query)
    documents: list[Any] = []
    seen: set[str] = set()
    rounds: list[RetrievalRound] = []
    base_query = str(query).strip()
    facet_queries = {name: f"{base_query} {terms}" for name, terms in facets}
    used_queries: set[str] = set()
    missing = [name for name, _ in facets]
    llm_next_query: str | None = None

    async def finish(sufficient: bool, missing_facets: list[str], reason: str) -> AgenticRetrievalResult:
        ranked = await _rerank_documents(documents, rerank_llm, base_query)
        return AgenticRetrievalResult(ranked, rounds, sufficient, missing_facets, reason)

    for index in range(max(1, max_rounds)):
        if llm_next_query:
            current_query = llm_next_query
            llm_next_query = None
        elif index == 0:
            current_query = base_query
        else:
            target = next((name for name in missing if facet_queries[name] not in used_queries), None)
            if target is None:
                break
            current_query = facet_queries[target]
        used_queries.add(current_query)
        found = await retriever.aretrieve(current_query, k=k)
        new_ids: list[str] = []
        for doc in found:
            source_id = _source_id(doc)
            if source_id not in seen:
                seen.add(source_id)
                documents.append(doc)
                new_ids.append(source_id)
        missing = _missing(documents, facets)
        # The production path can let the configured LLM inspect the current
        # evidence and choose the next query. Deterministic facet rules remain
        # the bounded fallback for offline tests and transient model failures.
        if controller_llm is not None and index + 1 < max_rounds:
            try:
                evidence = "\n".join(
                    f"- {_source_id(doc)}: {_text(doc)[:1200]}" for doc in documents
                )
                response = await controller_llm.ainvoke(
                    "你是医疗知识库检索控制器。判断当前证据是否足够回答用户问题。"
                    "如果不足，提出一个下一轮检索查询；如果足够则停止。"
                    "只输出 JSON：{\"sufficient\":true/false,\"missing\":[\"...\"],\"next_query\":\"...\"}。\n"
                    f"用户问题：{base_query}\n当前证据：\n{evidence}"
                )
                raw = getattr(response, "content", response)
                raw = raw if isinstance(raw, str) else str(raw)
                match = re.search(r"\{.*\}", raw, re.S)
                decision = json.loads(match.group(0)) if match else {}
                if bool(decision.get("sufficient")) and len(seen) >= min_sources:
                    return await finish(True, [], "llm_evidence_sufficient")
                next_query = str(decision.get("next_query") or "").strip()
                if next_query and next_query not in used_queries:
                    llm_next_query = next_query
                    missing = list(decision.get("missing") or missing or ["llm_follow_up"])
            except Exception:
                # Retrieval evaluation must remain reproducible if the remote
                # controller is unavailable; the rule-based path continues.
                pass
        rounds.append(RetrievalRound(index + 1, current_query, [_source_id(doc) for doc in found], new_ids, len(documents), missing))
        if len(seen) >= min_sources and not missing:
            return await finish(True, [], "evidence_sufficient")
        if not new_ids and index + 1 < max_rounds:
            # Do not waste a round repeating an exhausted query; the next
            # query targets the next missing evidence facet.
            continue
    missing = _missing(documents, facets)
    reason = "max_rounds" if len(rounds) >= max_rounds else "no_more_queries"
    return await finish(len(seen) >= min_sources and not missing, missing, reason)
