"""Small persistence boundary used by API services.

Keeping database queries here makes it possible to replace MySQL or add a
read-model later without coupling route handlers to ORM details.
"""
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from models import Conversation, Message


class ConversationRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def for_user(self, conversation_id: str, user_id: str) -> Conversation | None:
        return await self.session.scalar(select(Conversation).where(Conversation.id == conversation_id, Conversation.user_id == user_id))

    async def list_for_user(self, user_id: str) -> list[Conversation]:
        result = await self.session.scalars(select(Conversation).where(Conversation.user_id == user_id).order_by(desc(Conversation.updated_at)))
        return list(result)

    async def messages(self, conversation_id: str, limit: int = 50) -> list[Message]:
        result = await self.session.scalars(select(Message).where(Message.conversation_id == conversation_id).order_by(Message.sequence).limit(limit))
        return list(result)
