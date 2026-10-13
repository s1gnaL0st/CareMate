"""Web search as an explicit retrieval channel.

Web content is untrusted external data. Every result is normalised, labelled
``trust="untrusted_external"``, tiered by source authority and screened for
prompt-override attempts before it can reach a prompt. The channel degrades to
an explanatory empty result instead of blocking the caller's request.

Providers are resolved in this order when ``WEB_SEARCH_PROVIDER=auto``:
Tavily (API key configured) -> SearXNG (instance URL configured) -> DuckDuckGo
(keyless, requires the ``ddgs`` package).
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import logging
import re
from typing import Any, Awaitable, Callable, Iterable, Mapping
from urllib.parse import urlparse

import httpx

from agents.safety import contains_injection_pattern
from cache import cache_get_json, cache_key, cache_set_json
from config import Settings, get_settings


logger = logging.getLogger("smart_health.web_search")

SCHEMA_VERSION = "web_search_v1"
UNTRUSTED_TRUST_LABEL = "untrusted_external"
MAX_TITLE_CHARS = 300
MAX_SNIPPET_CHARS = 600

# Source authority tiers. Tier 1 is regulator / government / WHO level, tier 2
# is peer-reviewed or academic, tier 3 is general medical reference, and tier 4
# is anything unrecognised. Lower is more authoritative.
_TIER_1_SUFFIXES = (
    "nmpa.gov.cn", "cde.org.cn", "nhc.gov.cn", "chinacdc.cn", "gov.cn",
    "who.int", "fda.gov", "nih.gov", "cdc.gov", "ema.europa.eu", "nhs.uk",
)
_TIER_2_SUFFIXES = (
    "pubmed.ncbi.nlm.nih.gov", "ncbi.nlm.nih.gov", "cochrane.org",
    "msdmanuals.cn", "msdmanuals.com", "dayi.org.cn", "yiigle.com",
    "mayoclinic.org", "edu.cn", "ac.cn",
)
_TIER_3_SUFFIXES = (
    "health.baidu.com", "baidu.com", "dxy.cn", "39.net", "wikipedia.org",
    "webmd.com", "healthline.com", "medsci.cn", "sina.com.cn", "qq.com",
)


def source_tier(domain: str) -> int:
    """Map a hostname to an authority tier (1 best, 4 unknown)."""
    host = str(domain or "").strip().lower().strip(".")
    if not host:
        return 4
    for suffix in _TIER_1_SUFFIXES:
        if host == suffix or host.endswith("." + suffix):
            return 1
    for suffix in _TIER_2_SUFFIXES:
        if host == suffix or host.endswith("." + suffix):
            return 2
    for suffix in _TIER_3_SUFFIXES:
        if host == suffix or host.endswith("." + suffix):
            return 3
    return 4


def _clean_text(value: Any, limit: int) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]


@dataclass(frozen=True)
class WebResult:
    title: str
    url: str
    domain: str
    snippet: str
    source_tier: int
    published_at: str = ""
    safety_flagged: bool = False

    @property
    def citation_id(self) -> str:
        digest = hashlib.sha1(self.url.encode("utf-8")).hexdigest()[:10]
        return f"web:{self.domain}:{digest}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "citation_id": self.citation_id,
            "source_type": "web",
            "source_tier": self.source_tier,
            "trust": UNTRUSTED_TRUST_LABEL,
            "title": self.title,
            "url": self.url,
            "domain": self.domain,
            "snippet": self.snippet,
            "published_at": self.published_at,
            "safety_flagged": self.safety_flagged,
        }


def normalize_result(raw: Mapping[str, Any]) -> WebResult | None:
    """Normalise one provider row; return None when it is unusable."""
    if not isinstance(raw, Mapping):
        return None
    url = _clean_text(
        raw.get("url") or raw.get("href") or raw.get("link"), 2000
    )
    if not url.startswith(("http://", "https://")):
        return None
    domain = (urlparse(url).hostname or "").lower()
    if not domain:
        return None
    title = _clean_text(raw.get("title"), MAX_TITLE_CHARS) or domain
    snippet = _clean_text(
        raw.get("body") or raw.get("snippet") or raw.get("content"),
        MAX_SNIPPET_CHARS,
    )
    return WebResult(
        title=title,
        url=url,
        domain=domain,
        snippet=snippet,
        source_tier=source_tier(domain),
        published_at=_clean_text(
            raw.get("published") or raw.get("published_at") or raw.get("date"), 60
        ),
        safety_flagged=contains_injection_pattern(f"{title} {snippet}"),
    )


async def _search_duckduckgo(
    query: str, max_results: int, settings: Settings
) -> list[Mapping[str, Any]]:
    def run() -> list[Mapping[str, Any]]:
        from ddgs import DDGS

        with DDGS(timeout=settings.web_search_timeout_seconds) as ddgs:
            return list(
                ddgs.text(
                    query,
                    region=settings.web_search_region,
                    max_results=max_results,
                )
            )

    return await asyncio.to_thread(run)


async def _search_tavily(
    query: str, max_results: int, settings: Settings
) -> list[Mapping[str, Any]]:
    payload = {
        "api_key": settings.tavily_api_key,
        "query": query,
        "max_results": max_results,
        "search_depth": "basic",
    }
    async with httpx.AsyncClient(timeout=settings.web_search_timeout_seconds) as client:
        response = await client.post("https://api.tavily.com/search", json=payload)
        response.raise_for_status()
        data = response.json()
    rows = data.get("results") if isinstance(data, Mapping) else None
    return list(rows or [])


async def _search_searxng(
    query: str, max_results: int, settings: Settings
) -> list[Mapping[str, Any]]:
    endpoint = settings.web_search_searxng_url.rstrip("/") + "/search"
    async with httpx.AsyncClient(timeout=settings.web_search_timeout_seconds) as client:
        response = await client.get(
            endpoint,
            params={"q": query, "format": "json", "language": "zh-CN"},
        )
        response.raise_for_status()
        data = response.json()
    rows = data.get("results") if isinstance(data, Mapping) else None
    return list(rows or [])


def resolve_provider(settings: Settings, requested: str | None = None) -> str:
    """Resolve the effective provider name without touching the network."""
    name = (requested or settings.web_search_provider or "auto").strip().lower()
    if name != "auto":
        return name
    if settings.tavily_api_key.strip():
        return "tavily"
    if settings.web_search_searxng_url.strip():
        return "searxng"
    return "duckduckgo"


def _empty_payload(
    query: str, reason: str, *, provider: str = "", degraded: str = ""
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "channel": "web",
        "provider": provider,
        "query": query,
        "result_count": 0,
        "degraded_reason": reason,
        "detail": degraded,
        "results": [],
    }


async def search_web(
    query: str,
    *,
    max_results: int | None = None,
    provider: str | None = None,
    use_cache: bool = True,
    searcher: Callable[..., Awaitable[Iterable[Mapping[str, Any]]]] | None = None,
) -> dict[str, Any]:
    """Search the public web and return labelled, tiered, bounded results.

    ``searcher`` is injectable so tests and offline runs can supply a
    deterministic provider without touching the network.
    """
    settings = get_settings()
    query = _clean_text(query, 400)
    if not query:
        return _empty_payload(query, "empty_query")
    if not settings.web_search_enabled:
        return _empty_payload(query, "web_search_disabled")

    limit = int(max_results or settings.web_search_max_results)
    provider_name = resolve_provider(settings, provider)
    if provider_name in {"", "none", "off", "disabled"}:
        return _empty_payload(query, "provider_disabled", provider=provider_name)

    cache_key_value = cache_key(
        "websearch", f"{provider_name}:{limit}:{query}"
    )
    if use_cache:
        try:
            cached = await cache_get_json(cache_key_value)
            if (
                isinstance(cached, dict)
                and cached.get("schema_version") == SCHEMA_VERSION
            ):
                return cached
        except Exception:  # cache is an optimisation, never a dependency
            logger.debug("web search cache read skipped", exc_info=True)

    degraded = ""
    try:
        if searcher is not None:
            rows = await asyncio.wait_for(
                searcher(query, limit, settings),
                timeout=settings.web_search_timeout_seconds + 2,
            )
        elif provider_name == "tavily":
            rows = await _search_tavily(query, limit, settings)
        elif provider_name == "searxng":
            rows = await _search_searxng(query, limit, settings)
        elif provider_name == "duckduckgo":
            rows = await _search_duckduckgo(query, limit, settings)
        else:
            return _empty_payload(
                query, "unknown_provider", provider=provider_name
            )
    except asyncio.TimeoutError:
        logger.warning("web search timed out provider=%s", provider_name)
        rows, degraded = [], "timeout"
    except Exception as exc:
        logger.warning(
            "web search failed provider=%s error=%s",
            provider_name,
            type(exc).__name__,
        )
        rows, degraded = [], f"provider_error:{type(exc).__name__}"

    results: list[WebResult] = []
    seen: set[str] = set()
    for raw in rows or []:
        normalized = normalize_result(raw if isinstance(raw, Mapping) else {})
        if normalized is None or normalized.url in seen:
            continue
        seen.add(normalized.url)
        results.append(normalized)

    # Authoritative domains first; dated items before undated ones at equal tier.
    results.sort(key=lambda item: (item.source_tier, 0 if item.published_at else 1))
    results = results[:limit]

    retrieved_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "channel": "web",
        "provider": provider_name,
        "query": query,
        "result_count": len(results),
        "degraded_reason": degraded,
        "quarantined_count": sum(1 for item in results if item.safety_flagged),
        "results": [
            {**item.as_dict(), "retrieved_at": retrieved_at} for item in results
        ],
    }
    if use_cache and not degraded:
        try:
            await cache_set_json(
                cache_key_value, payload, settings.web_search_cache_ttl_seconds
            )
        except Exception:
            logger.debug("web search cache write skipped", exc_info=True)
    return payload
