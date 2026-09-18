"""add hospital directory, habit goals, and answer feedback"""
from alembic import op
import sqlalchemy as sa

revision = "007_wellness_directory"
down_revision = "006_agent_loop_persistence"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "hospitals",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("city", sa.String(100), nullable=False),
        sa.Column("address", sa.String(300), nullable=False),
        sa.Column("tags", sa.JSON(), nullable=False),
        sa.Column("distance_km", sa.Float(), nullable=True),
        sa.Column("phone", sa.String(50), nullable=False),
        sa.Column("is_ad", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_index("ix_hospitals_city_name", "hospitals", ["city", "name"])
    op.create_table(
        "habit_goals",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("title", sa.String(200), nullable=False),
        sa.Column("target_value", sa.Integer(), nullable=False),
        sa.Column("current_value", sa.Integer(), nullable=False),
        sa.Column("unit", sa.String(30), nullable=False),
        sa.Column("color", sa.String(30), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_index("ix_habit_goals_user_id", "habit_goals", ["user_id"])
    op.create_index("ix_habit_goals_user_status", "habit_goals", ["user_id", "status"])
    op.create_table(
        "answer_feedback",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("message_id", sa.String(36), sa.ForeignKey("messages.id", ondelete="SET NULL"), nullable=True),
        sa.Column("conversation_id", sa.String(36), sa.ForeignKey("conversations.id", ondelete="SET NULL"), nullable=True),
        sa.Column("rating", sa.Integer(), nullable=False),
        sa.Column("category", sa.String(50), nullable=True),
        sa.Column("comment", sa.String(1000), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.UniqueConstraint("user_id", "message_id", name="uq_feedback_user_message"),
    )
    op.create_index("ix_answer_feedback_user_id", "answer_feedback", ["user_id"])
    op.create_index("ix_answer_feedback_message_id", "answer_feedback", ["message_id"])
    op.create_index("ix_answer_feedback_conversation_id", "answer_feedback", ["conversation_id"])


def downgrade() -> None:
    op.drop_table("answer_feedback")
    op.drop_table("habit_goals")
    op.drop_table("hospitals")
