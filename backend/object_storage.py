"""Small async adapter around the synchronous MinIO/S3-compatible client."""
from __future__ import annotations

import asyncio
import io

from config import get_settings


def _client():
    from minio import Minio

    settings = get_settings()
    return Minio(
        settings.object_storage_endpoint,
        access_key=settings.object_storage_access_key,
        secret_key=settings.object_storage_secret_key,
        secure=settings.object_storage_secure,
    )


def _ensure_bucket(client, bucket: str) -> None:
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)


async def put_object(object_key: str, content: bytes, content_type: str) -> None:
    settings = get_settings()

    def write() -> None:
        client = _client()
        _ensure_bucket(client, settings.object_storage_bucket)
        client.put_object(
            settings.object_storage_bucket,
            object_key,
            io.BytesIO(content),
            length=len(content),
            content_type=content_type,
        )

    await asyncio.to_thread(write)


async def get_object(object_key: str) -> bytes:
    settings = get_settings()

    def read() -> bytes:
        response = _client().get_object(settings.object_storage_bucket, object_key)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    return await asyncio.to_thread(read)


async def delete_object(object_key: str) -> None:
    settings = get_settings()
    await asyncio.to_thread(_client().remove_object, settings.object_storage_bucket, object_key)
