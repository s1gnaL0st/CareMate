"""
RAG Knowledge Base for the Smart Health Assistant.

Retrieval strategy: BM25 (30%) + dense retrieval (70%) for dual-path recall;
MMR is used inside dense retrieval for diversity-aware de-duplication and
re-ranking. EnsembleRetriever merges both ranked lists.

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
CLEANED_DOCS_DIR = Path(__file__).parent / "sources" / "cleaned"
SOURCE_MANIFEST = Path(__file__).parent / "source_manifest.json"

# Chinese-aware separators: sentence endings → paragraph breaks → finer boundaries
_CHINESE_SEPARATORS = ["。", "！", "？", "；", "\n\n", "\n", "，", " ", ""]
_PARENT_CONTEXT_LIMIT = 6000


class HealthKnowledgeBase:
    """
    Hybrid RAG retriever with pluggable embedding and vector-store backends.
    Lazily initialized on first retrieval call.
    """

    def __init__(self) -> None:
        self._retriever: EnsembleRetriever | None = None
        self._parents: dict[str, Document] = {}
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
            c=60,
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

        # Keep the manifest alongside the configured local vector index.
        manifest_path = Path(__file__).parent / ".rag_manifest.json"
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            previous = {}
        current: dict[str, str] = {}
        source_dirs = [directory for directory in (CLEANED_DOCS_DIR, Path(__file__).parent / "sources" / "cleaned_local_drugs", Path(__file__).parent / "sources" / "cleaned_local_cards") if directory.exists()]
        if not source_dirs:
            source_dirs = [DOCS_DIR]
        for source_dir in source_dirs:
            for path in sorted(source_dir.glob("*.md")):
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
        self._retriever = EnsembleRetriever(retrievers=[sparse, dense], weights=[0.3, 0.7], c=60)
        self._initialized = True
        manifest_path.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"changed": len(changed), "removed": len(removed), "chunks": len(update_docs)}

    # ── Document Loading & Chunking ───────────────────────────────────────────

    def _load_documents(self) -> List[Document]:
        """
        Load Markdown files from documents/, split by ## section headers, then
        apply RecursiveCharacterTextSplitter with Chinese-aware separators.
        Each chunk carries provenance metadata from source_manifest.json.
        Unregistered documents fail closed instead of entering the index without
        a traceable publisher and URL.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=_CHINESE_SEPARATORS,
        )
        manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
        docs: List[Document] = []
        parents: dict[str, Document] = {}
        source_dirs = [directory for directory in (CLEANED_DOCS_DIR, Path(__file__).parent / "sources" / "cleaned_local_drugs", Path(__file__).parent / "sources" / "cleaned_local_cards") if directory.exists()]
        if not source_dirs:
            source_dirs = [DOCS_DIR]
        md_files = [path for source_dir in source_dirs for path in sorted(source_dir.glob("*.md"))]
        for md_file in md_files:
            source = md_file.stem
            provenance = manifest.get(source)
            if not isinstance(provenance, dict):
                continue
            if provenance.get("review_status") not in {"source_curated", "local_imported"}:
                continue
            content_hash = hashlib.sha256(md_file.read_bytes()).hexdigest()
            updated_at = datetime.fromtimestamp(md_file.stat().st_mtime, tz=timezone.utc).isoformat()
            text = md_file.read_text(encoding="utf-8")
            # Cleaned source files carry provenance in YAML front matter; it
            # is metadata, not retrieval content.
            if text.startswith("---\n"):
                _, _, text = text.partition("\n---\n")
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
                parent_id = f"{source}::{heading}"
                parent_content = body[:_PARENT_CONTEXT_LIMIT]
                parent_metadata = {
                    "source": source,
                    "section": heading,
                    "parent_id": parent_id,
                    "chunk_type": "parent",
                    "content_hash": content_hash,
                    "updated_at": updated_at,
                    **provenance,
                    "source_urls": json.dumps(provenance.get("source_urls", []), ensure_ascii=False),
                }
                parents[parent_id] = Document(page_content=parent_content, metadata=parent_metadata)
                chunks = splitter.create_documents(
                    [body],
                    metadatas=[{
                        "source": source,
                        "section": heading,
                        "parent_id": parent_id,
                        "chunk_type": "child",
                        "content_hash": content_hash,
                        "updated_at": updated_at,
                        **provenance,
                        "source_urls": json.dumps(provenance.get("source_urls", []), ensure_ascii=False),
                    }],
                )
                for child_index, child in enumerate(chunks):
                    child.metadata["child_index"] = child_index
                docs.extend(chunks)
        self._parents = parents
        return docs

    def _expand_to_parents(self, documents: List[Document], limit: int) -> List[Document]:
        """Map ranked child hits back to unique parent sections."""
        selected: list[Document] = []
        seen: set[str] = set()
        for child in documents:
            parent_id = str(child.metadata.get("parent_id", ""))
            parent = self._parents.get(parent_id)
            if parent is None:
                parent = child
                parent_id = f"{child.metadata.get('source', '')}::{child.metadata.get('section', '')}"
            if parent_id in seen:
                continue
            seen.add(parent_id)
            metadata = dict(parent.metadata)
            metadata["matched_child"] = child.page_content
            metadata["matched_child_index"] = child.metadata.get("child_index", 0)
            selected.append(Document(page_content=parent.page_content, metadata=metadata))
            if len(selected) >= limit:
                break
        return selected

    # ── Retrieval ─────────────────────────────────────────────────────────────

    async def aretrieve(self, query: str, k: int = 3) -> List[Document]:
        """
        Async hybrid retrieval — safe to call from FastAPI handlers.
        Returns at most k de-duplicated chunks ranked by RRF score.
        """
        key = cache_key("rag_parent_v2", f"{query}:{k}")
        cached = await cache_get_json(key)
        if isinstance(cached, list):
            return [Document(page_content=str(item.get("content", "")), metadata=item.get("metadata", {})) for item in cached if isinstance(item, dict)]
        await self._ensure_initialized()
        assert self._retriever is not None
        results = await self._retriever.ainvoke(query)
        selected = self._expand_to_parents(results, k)
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

        merged_children: List[Document] = []
        for docs in per_query_results:
            for doc in docs:
                merged_children.append(doc)
                if len(merged_children) >= k * 4:
                    break
        return self._expand_to_parents(merged_children, k)

    def retrieve(self, query: str, k: int = 3) -> List[Document]:
        """
        Synchronous retrieval — used inside LangChain @tool functions where
        an event loop may already be running.
        """
        if not self._initialized:
            self._sync_init()
        assert self._retriever is not None
        results = self._retriever.invoke(query)
        return self._expand_to_parents(results, k)

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
