"""Versioned REST endpoints for authentication and conversation history."""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy import desc, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from auth import bearer_scheme, create_access_token, create_refresh_token, decode_token, get_current_user, hash_password, verify_password
from db import get_db, redis_client
from models import Conversation, Message, User, UserProfile
from schemas import ConversationCreate, ConversationResponse, LoginRequest, MessageResponse, TokenResponse, UserCreate, UserResponse

router = APIRouter(prefix="/api/v1")
auth_router = APIRouter(prefix="/auth", tags=["auth"])
conversation_router = APIRouter(prefix="/conversations", tags=["conversations"])


@auth_router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(payload: UserCreate, db: AsyncSession = Depends(get_db)):
    user = User(email=str(payload.email).lower(), password_hash=hash_password(payload.password), name=payload.name)
    db.add(user)
    try:
        await db.flush()
        db.add(UserProfile(user_id=user.id))
        await db.commit()
        await db.refresh(user)
    except IntegrityError as exc:
        await db.rollback()
        raise HTTPException(status_code=409, detail="邮箱已注册") from exc
    token, expires_in = create_access_token(user.id)
    return TokenResponse(access_token=token, refresh_token=create_refresh_token(user.id), expires_in=expires_in, user=user)


@auth_router.post("/login", response_model=TokenResponse)
async def login(payload: LoginRequest, db: AsyncSession = Depends(get_db)):
    user = await db.scalar(select(User).where(User.email == str(payload.email).lower()))
    if user is None or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="邮箱或密码错误")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="用户已停用")
    token, expires_in = create_access_token(user.id)
    return TokenResponse(access_token=token, refresh_token=create_refresh_token(user.id), expires_in=expires_in, user=user)


@auth_router.post("/refresh", response_model=TokenResponse)
async def refresh(credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme), db: AsyncSession = Depends(get_db)):
    if credentials is None:
        raise HTTPException(status_code=401, detail="需要 refresh token")
    try:
        payload = decode_token(credentials.credentials, expected_type="refresh")
    except Exception as exc:
        raise HTTPException(status_code=401, detail="refresh token 无效") from exc
    user = await db.scalar(select(User).where(User.id == payload.get("sub"), User.is_active.is_(True)))
    if user is None:
        raise HTTPException(status_code=401, detail="用户不存在或已停用")
    token, expires_in = create_access_token(user.id)
    return TokenResponse(access_token=token, refresh_token=create_refresh_token(user.id), expires_in=expires_in, user=user)


@auth_router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme)):
    if credentials is None:
        return
    try:
        payload = decode_token(credentials.credentials)
        ttl = max(1, int(payload["exp"] - __import__("time").time()))
        await redis_client.set(f"auth:revoked:{payload['jti']}", "1", ex=ttl)
    except Exception:
        return


@auth_router.get("/me", response_model=UserResponse)
async def me(user: User = Depends(get_current_user)):
    return user


@conversation_router.post("", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED)
async def create_conversation(payload: ConversationCreate, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    conversation = Conversation(user_id=user.id, title=payload.title)
    db.add(conversation)
    await db.commit()
    await db.refresh(conversation)
    return conversation


@conversation_router.get("", response_model=list[ConversationResponse])
async def list_conversations(user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    result = await db.scalars(select(Conversation).where(Conversation.user_id == user.id).order_by(desc(Conversation.updated_at)))
    return list(result)


async def _owned_conversation(conversation_id: str, user: User, db: AsyncSession) -> Conversation:
    conversation = await db.scalar(select(Conversation).where(Conversation.id == conversation_id, Conversation.user_id == user.id))
    if conversation is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return conversation


@conversation_router.get("/{conversation_id}/messages", response_model=list[MessageResponse])
async def list_messages(conversation_id: str, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    await _owned_conversation(conversation_id, user, db)
    result = await db.scalars(select(Message).where(Message.conversation_id == conversation_id).order_by(Message.sequence))
    return list(result)


@conversation_router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(conversation_id: str, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    conversation = await _owned_conversation(conversation_id, user, db)
    await db.delete(conversation)
    await db.commit()


router.include_router(auth_router)
router.include_router(conversation_router)
