import unittest

from pydantic import ValidationError

from schemas import FeedbackCreate, HabitGoalCreate, HabitGoalUpdate


class WellnessContractTests(unittest.TestCase):
    def test_habit_goal_defaults_and_bounds(self):
        goal = HabitGoalCreate(title="每日饮水")
        self.assertEqual(goal.target_value, 1)
        self.assertEqual(goal.unit, "次")
        with self.assertRaises(ValidationError):
            HabitGoalCreate(title="", target_value=0)

    def test_habit_goal_update_restricts_status_values(self):
        self.assertEqual(HabitGoalUpdate(current_value=3, status="completed").status, "completed")
        with self.assertRaises(ValidationError):
            HabitGoalUpdate(status="deleted")

    def test_feedback_rating_and_comment_limits(self):
        feedback = FeedbackCreate(message_id="m1", rating=1, category="helpful")
        self.assertEqual(feedback.rating, 1)
        with self.assertRaises(ValidationError):
            FeedbackCreate(rating=2)
        with self.assertRaises(ValidationError):
            FeedbackCreate(rating=0, comment="x" * 1001)


if __name__ == "__main__":
    unittest.main()
