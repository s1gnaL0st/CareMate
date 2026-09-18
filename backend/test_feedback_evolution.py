import unittest
from unittest.mock import AsyncMock, MagicMock

from models import ExperienceRecord, Message
from schemas import FeedbackCreate
from wellness_api import create_feedback


class FeedbackEvolutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_negative_answer_feedback_enters_failure_staging(self):
        message = Message(
            id="message-1",
            conversation_id="conversation-1",
            role="assistant",
            content="旧回答",
            sequence=2,
            event_metadata={"request_id": "run-1"},
        )
        experience = ExperienceRecord(
            id="experience-1",
            run_id="run-1",
            intent="health",
            input_redacted="用户问题",
            plan_signature="plan-a",
            verify_status="pass",
            outcome="completed",
        )
        db = MagicMock()
        db.scalar = AsyncMock(side_effect=[message, experience, None])
        db.flush = AsyncMock()
        db.commit = AsyncMock()
        db.refresh = AsyncMock()
        response = await create_feedback(
            FeedbackCreate(
                message_id=message.id,
                rating=-1,
                category="incorrect",
                comment="回答中的事实不正确",
            ),
            user=MagicMock(id="user-1"),
            db=db,
        )
        self.assertEqual(response.rating, -1)
        self.assertEqual(db.add.call_count, 2)
        staged = db.add.call_args_list[1].args[0]
        self.assertEqual(staged.category, "user_correction")
        self.assertEqual(staged.experience_id, experience.id)
        self.assertEqual(staged.source_feedback_id, response.id)
        db.commit.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
