"""Redis-backed cache, rate limiting, and distributed locks."""
from __future__ import annotations

import hashlib
import json
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import HTTPException, Request, status

from config import get_settings
from db import redis_client
from metrics import record_cache_event


def cache_key(namespace: str, value: str, *, version: str | None = None) -> str:
    settings = get_settings()
    prefix = f"{settings.cache_key_prefix}:{namespace}:v{version or settings.cache_version}"
    digest = hashlib.sha256(value.strip().encode("utf-8")).hexdigest()[:32]
    return f"{prefix}:{digest}"


async def cache_get_json(key: str) -> Any | None:
    try:
        raw = await redis_client.get(key)
        record_cache_event("hit" if raw is not None else "miss")
        return json.loads(raw) if raw is not None else None
    except Exception:
        record_cache_event("error")
        return None


async def cache_set_json(key: str, value: Any, ttl: int) -> bool:
    try:
        await redis_client.set(key, json.dumps(value, ensure_ascii=False), ex=max(1, ttl))
        return True
    except Exception:
        return False


async def cache_delete(key: str) -> bool:
    try:
        return bool(await redis_client.delete(key))
    except Exception:
        return False


async def invalidate_cache(namespace: str, value: str, *, version: str | None = None) -> bool:
    """Actively evict a known cache entry after a source-data update."""
    return await cache_delete(cache_key(namespace, value, version=version))


async def invalidate_namespace(namespace: str) -> int:
    """Evict every version of a namespace after a knowledge-source rebuild."""
    settings = get_settings()
    pattern = f"{settings.cache_key_prefix}:{namespace}:v*"
    removed = 0
    try:
        async for key in redis_client.scan_iter(match=pattern, count=100):
            removed += int(await redis_client.delete(key))
    except Exception:
        return removed
    return removed


async def enforce_rate_limit(request: Request, *, subject: str | None = None, bucket: str = "api") -> None:
    settings = get_settings()
    client_ip = request.client.host if request.client else "unknown"
    identity = subject or client_ip
    key = f"{settings.cache_key_prefix}:rate:{bucket}:{identity}"
    try:
        count = await redis_client.incr(key)
        if count == 1:
            await redis_client.expire(key, settings.rate_limit_window_seconds)
        if count > settings.rate_limit_requests:
            retry_after = await redis_client.ttl(key)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="请求过于频繁，请稍后再试",
                headers={"Retry-After": str(max(1, retry_after))},
            )
    except HTTPException:
        raise
    except Exception:
        return


@asynccontextmanager
async def distributed_lock(name: str, *, timeout: int | None = None, blocking_timeout: int | None = None) -> AsyncIterator[bool]:
    settings = get_settings()
    lock = redis_client.lock(
        f"{settings.cache_key_prefix}:lock:{name}",
        timeout=timeout or settings.lock_timeout_seconds,
        blocking_timeout=blocking_timeout if blocking_timeout is not None else settings.lock_blocking_timeout_seconds,
    )
    acquired = False
    try:
        acquired = bool(await lock.acquire())
        yield acquired
    finally:
        if acquired:
            try:
                await lock.release()
            except Exception:
                pass
