"""Async SQLAlchemy and Redis clients."""
from collections.abc import AsyncIterator

from redis.asyncio import Redis, from_url
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from config import get_settings

settings = get_settings()
engine = create_async_engine(settings.database_url, pool_pre_ping=True, pool_recycle=1800)
SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
redis_client: Redis = from_url(settings.redis_url, decode_responses=True)


async def get_db() -> AsyncIterator[AsyncSession]:
    async with SessionLocal() as session:
        yield session


async def close_clients() -> None:
    await redis_client.aclose()
    await engine.dispose()
