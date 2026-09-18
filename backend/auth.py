"""Password hashing and JWT authentication helpers."""
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pwdlib import PasswordHash
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config import get_settings
from db import get_db, redis_client
from models import User

password_hash = PasswordHash.recommended()
bearer_scheme = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    return password_hash.hash(password)


def verify_password(password: str, hashed: str) -> bool:
    return password_hash.verify(password, hashed)


def _create_token(user_id: str, token_type: str, expires_in: int) -> str:
    settings = get_settings()
    now = datetime.now(timezone.utc)
    payload = {"sub": user_id, "type": token_type, "jti": str(uuid4()), "iat": now, "exp": now + timedelta(seconds=expires_in)}
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def create_access_token(user_id: str) -> tuple[str, int]:
    expires_in = get_settings().access_token_minutes * 60
    return _create_token(user_id, "access", expires_in), expires_in


def create_refresh_token(user_id: str) -> str:
    return _create_token(user_id, "refresh", 60 * 60 * 24 * 30)


def decode_token(token: str, expected_type: str = "access") -> dict:
    settings = get_settings()
    payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    if payload.get("type", "access") != expected_type:
        raise jwt.InvalidTokenError("unexpected token type")
    return payload


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    if not credentials:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="需要登录")
    settings = get_settings()
    try:
        payload = decode_token(credentials.credentials)
        user_id = payload.get("sub")
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="登录凭证无效") from exc
    try:
        if payload.get("jti") and await redis_client.exists(f"auth:revoked:{payload['jti']}"):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="登录凭证已失效")
    except HTTPException:
        raise
    except Exception:
        pass
    user = await db.scalar(select(User).where(User.id == user_id, User.is_active.is_(True)))
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="用户不存在或已停用")
    return user


async def get_optional_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> User | None:
    """Resolve a bearer token when present; legacy anonymous chat remains supported."""
    if credentials is None:
        return None
    return await get_current_user(credentials, db)
