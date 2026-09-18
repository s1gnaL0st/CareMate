import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from main import ChatMessage, ChatRequest, UserInfo, _prepare_persistent_state


class MemoryRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_authenticated_request_receives_only_server_loaded_preferences(self):
        request = ChatRequest(
            messages=[ChatMessage(role="user", content="你好")],
            user_info=UserInfo(name="测试用户"),
        )
        db = MagicMock()
        user = MagicMock(id="user-1")
        with patch("main.load_active_memory_preferences", new=AsyncMock(return_value=["回答先给结论"])):
            state, conversation = await _prepare_persistent_state(request, user, db, "advisor_agent")
        self.assertIsNone(conversation)
        self.assertEqual(state["user_info"]["response_preferences"], ["回答先给结论"])

    async def test_anonymous_request_never_loads_persistent_preferences(self):
        request = ChatRequest(messages=[ChatMessage(role="user", content="你好")])
        loader = AsyncMock(return_value=["should not load"])
        with patch("main.load_active_memory_preferences", new=loader):
            state, _ = await _prepare_persistent_state(request, None, None, "advisor_agent")
        loader.assert_not_awaited()
        self.assertNotIn("response_preferences", state["user_info"])


if __name__ == "__main__":
    unittest.main()
