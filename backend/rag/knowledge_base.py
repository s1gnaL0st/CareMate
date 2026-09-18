"""
RAG Knowledge Base for the Smart Health Assistant.

Retrieval strategy: Hybrid BM25 (30%) + Dense MMR (70%) via EnsembleRetriever.
  - BM25 catches exact medical term matches (高血压, 血红蛋白, 门诊报销 …)
  - Dense MMR retrieves semantically similar chunks with diversity enforcement
  - EnsembleRetriever merges both lists via Reciprocal Rank Fusion (RRF)

Thread safety: asyncio.Lock + run_in_executor keeps the blocking HuggingFace model
load off the event loop. Double-checked locking prevents duplicate init on concurrent
first requests.
"""
from __future__ import annotations

import asyncio
import re
import hashlib
import json
from functools import lru_cache
from pathlib import Path
from datetime import datetime, timezone
from typing import List

from langchain_classic.retrievers import EnsembleRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from rag.embeddings import get_embeddings
from rag.vectorstores import get_vectorstore
from cache import cache_get_json, cache_key, cache_set_json
from config import get_settings

DOCS_DIR = Path(__file__).parent / "documents"

# Chinese-aware separators: sentence endings → paragraph breaks → finer boundaries
_CHINESE_SEPARATORS = ["。", "！", "？", "；", "\n\n", "\n", "，", " ", ""]


class HealthKnowledgeBase:
    """
    Hybrid RAG retriever with pluggable embedding and vector-store backends.
    Lazily initialized on first retrieval call.
    """

    def __init__(self) -> None:
        self._retriever: EnsembleRetriever | None = None
        self._initialized = False
        self._lock = asyncio.Lock()

    # ── Initialization ────────────────────────────────────────────────────────

    async def _ensure_initialized(self) -> None:
        """Async-safe lazy init: runs the blocking model load in a thread pool."""
        if self._initialized:
            return
        async with self._lock:
            if self._initialized:
                return  # another coroutine finished init while we waited
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._sync_init)

    def _sync_init(self, *, rebuild: bool = False) -> None:
        """Blocking initialization — called via run_in_executor or directly from CLI."""
        embeddings = get_embeddings()
        documents = self._load_documents()
        vectorstore = get_vectorstore(embeddings, documents, rebuild=rebuild)

        dense_retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 4, "fetch_k": 20},
        )
        sparse_retriever = BM25Retriever.from_documents(documents, k=3)

        self._retriever = EnsembleRetriever(
            retrievers=[sparse_retriever, dense_retriever],
            weights=[0.3, 0.7],
        )
        self._initialized = True

    def rebuild(self) -> None:
        """Drop and rebuild the vector store from source documents (synchronous)."""
        self._initialized = False
        self._retriever = None
        self._sync_init(rebuild=True)

    def incremental_update(self) -> dict[str, int]:
        """Update only knowledge-base files whose content changed.

        A small manifest (stored next to the configured vector store) tracks
        source-file hashes.  Changed chunks are upserted into stores that
        support metadata deletion (Chroma/Qdrant); for append-only backends we
        still add only the changed chunks, avoiding an expensive full rebuild.
        """
        embeddings = get_embeddings()
        docs = self._load_documents()
        by_source: dict[str, list[Document]] = {}
        for doc in docs:
            by_source.setdefault(str(doc.metadata.get("source", "")), []).append(doc)

        # Keep the manifest in the Chroma directory (or alongside FAISS).
        manifest_path = Path(__file__).parent / ".rag_manifest.json"
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            previous = {}
        current: dict[str, str] = {}
        for path in sorted(DOCS_DIR.glob("*.md")):
            current[path.stem] = hashlib.sha256(path.read_bytes()).hexdigest()
        changed = [src for src, digest in current.items() if previous.get(src) != digest]
        removed = [src for src in previous if src not in current]
        if not changed and not removed:
            return {"changed": 0, "removed": 0, "chunks": 0}

        try:
            vectorstore = get_vectorstore(embeddings, None, rebuild=False)
        except ValueError:
            # No persisted index yet: bootstrap once, then record the
            # manifest so subsequent runs remain incremental.
            self._sync_init()
            manifest_path.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
            return {"changed": len(changed), "removed": 0, "chunks": len(docs)}
        backend = __import__("os").getenv("VECTOR_STORE", "chroma").lower()
        # Chroma exposes a stable collection API, allowing stale chunks to be
        # removed before upserting replacements.
        if backend == "chroma" and hasattr(vectorstore, "_collection"):
            for source in [*changed, *removed]:
                vectorstore._collection.delete(where={"source": source})
        update_docs = [doc for src in changed for doc in by_source.get(src, [])]
        if update_docs:
            vectorstore.add_documents(update_docs)

        # Rebuild the in-process hybrid retriever from the updated store.
        sparse = BM25Retriever.from_documents(docs, k=3)
        dense = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 20})
        self._retriever = EnsembleRetriever(retrievers=[sparse, dense], weights=[0.3, 0.7])
        self._initialized = True
        manifest_path.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"changed": len(changed), "removed": len(removed), "chunks": len(update_docs)}

    # ── Document Loading & Chunking ───────────────────────────────────────────

    def _load_documents(self) -> List[Document]:
        """
        Load Markdown files from documents/, split by ## section headers, then
        apply RecursiveCharacterTextSplitter with Chinese-aware separators.
        Each chunk carries {source, section} metadata.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=_CHINESE_SEPARATORS,
        )
        docs: List[Document] = []
        for md_file in sorted(DOCS_DIR.glob("*.md")):
            source = md_file.stem
            updated_at = datetime.fromtimestamp(md_file.stat().st_mtime, tz=timezone.utc).isoformat()
            text = md_file.read_text(encoding="utf-8")
            # Split on lines starting with "## " to extract named sections
            sections = re.split(r"\n(?=## )", text)
            for section in sections:
                lines = section.strip().splitlines()
                if not lines:
                    continue
                heading = lines[0].lstrip("# ").strip()
                body = "\n".join(lines[1:]).strip()
                if not body:
                    continue
                chunks = splitter.create_documents(
                    [body],
                    metadatas=[{"source": source, "section": heading, "updated_at": updated_at}],
                )
                docs.extend(chunks)
        return docs

    # ── Retrieval ─────────────────────────────────────────────────────────────

    async def aretrieve(self, query: str, k: int = 3) -> List[Document]:
        """
        Async hybrid retrieval — safe to call from FastAPI handlers.
        Returns at most k de-duplicated chunks ranked by RRF score.
        """
        key = cache_key("rag", f"{query}:{k}")
        cached = await cache_get_json(key)
        if isinstance(cached, list):
            return [Document(page_content=str(item.get("content", "")), metadata=item.get("metadata", {})) for item in cached if isinstance(item, dict)]
        await self._ensure_initialized()
        assert self._retriever is not None
        results = await self._retriever.ainvoke(query)
        selected = results[:k]
        await cache_set_json(
            key,
            [{"content": d.page_content, "metadata": d.metadata} for d in selected],
            get_settings().cache_rag_ttl_seconds,
        )
        return selected

    async def amulti_query_retrieve(self, queries: List[str], k: int = 4) -> List[Document]:
        """
        Multi-query retrieval (RAG Implementation best practice).

        Runs the hybrid retriever for each query in parallel, then merges and
        de-duplicates results. Improves recall for complex medical questions
        where a single query phrasing misses relevant chunks.

        Args:
            queries: 2-4 varied phrasings of the same information need.
            k: Maximum unique documents to return after deduplication.
        """
        await self._ensure_initialized()
        assert self._retriever is not None

        # Run all queries concurrently
        tasks = [self._retriever.ainvoke(q) for q in queries]
        per_query_results = await asyncio.gather(*tasks)

        # Deduplicate by page_content while preserving first-seen order
        seen: set[str] = set()
        merged: List[Document] = []
        for docs in per_query_results:
            for doc in docs:
                key = doc.page_content
                if key not in seen:
                    seen.add(key)
                    merged.append(doc)
                if len(merged) >= k * 2:
                    break

        return merged[:k]

    def retrieve(self, query: str, k: int = 3) -> List[Document]:
        """
        Synchronous retrieval — used inside LangChain @tool functions where
        an event loop may already be running.
        """
        if not self._initialized:
            self._sync_init()
        assert self._retriever is not None
        results = self._retriever.invoke(query)
        return results[:k]

    # ── Formatting ────────────────────────────────────────────────────────────

    @staticmethod
    def format_context(docs: List[Document]) -> str:
        """
        Render retrieved chunks as a labeled context block for injection into
        the agent system prompt.

        Example output:
            [medical_knowledge · 高血压（Hypertension）]
            高血压是指动脉血压持续升高…

            ---

            [lab_reference · 血常规参考范围]
            红细胞计数（RBC）…
        """
        if not docs:
            return ""
        parts: List[str] = []
        for doc in docs:
            src = doc.metadata.get("source", "")
            sec = doc.metadata.get("section", "")
            label = f"[{src} · {sec}]" if sec else f"[{src}]"
            parts.append(f"{label}\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)


@lru_cache(maxsize=1)
def get_knowledge_base() -> HealthKnowledgeBase:
    """Singleton accessor — returns the same instance for the process lifetime."""
    return HealthKnowledgeBase()
