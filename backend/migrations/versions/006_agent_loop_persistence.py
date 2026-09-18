"""add durable AgentLoop runs and task ledger"""

from alembic import op
import sqlalchemy as sa


revision = "006_agent_loop_persistence"
down_revision = "005_analysis_result_object"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "agent_runs",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("conversation_id", sa.String(length=36), sa.ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("user_id", sa.String(length=36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("status", sa.String(length=30), nullable=False, server_default="pending"),
        sa.Column("pause_requested", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("plan_signature", sa.String(length=40), nullable=False, server_default=""),
        sa.Column("replan_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("requested_mode", sa.String(length=50), nullable=False, server_default="general"),
        sa.Column("initial_state", sa.JSON(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("lease_until", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    op.create_index("ix_agent_runs_user_status", "agent_runs", ["user_id", "status"])
    op.create_index("ix_agent_runs_conversation_id", "agent_runs", ["conversation_id"])
    op.create_index("ix_agent_runs_status", "agent_runs", ["status"])

    op.create_table(
        "agent_tasks",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("run_id", sa.String(length=36), sa.ForeignKey("agent_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("task_key", sa.String(length=80), nullable=False),
        sa.Column("agent", sa.String(length=50), nullable=False),
        sa.Column("objective", sa.String(length=500), nullable=False),
        sa.Column("input_slice", sa.Text(), nullable=False),
        sa.Column("depends_on", sa.JSON(), nullable=True),
        sa.Column("status", sa.String(length=30), nullable=False, server_default="pending"),
        sa.Column("attempt_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("result", sa.JSON(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("lease_until", sa.DateTime(), nullable=True),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), onupdate=sa.func.now()),
        sa.UniqueConstraint("run_id", "task_key", name="uq_agent_tasks_run_key"),
    )
    op.create_index("ix_agent_tasks_run_status", "agent_tasks", ["run_id", "status"])
    op.create_index("ix_agent_tasks_run_id", "agent_tasks", ["run_id"])
    op.create_index("ix_agent_tasks_status", "agent_tasks", ["status"])


def downgrade() -> None:
    op.drop_index("ix_agent_tasks_status", table_name="agent_tasks")
    op.drop_index("ix_agent_tasks_run_id", table_name="agent_tasks")
    op.drop_index("ix_agent_tasks_run_status", table_name="agent_tasks")
    op.drop_table("agent_tasks")
    op.drop_index("ix_agent_runs_status", table_name="agent_runs")
    op.drop_index("ix_agent_runs_conversation_id", table_name="agent_runs")
    op.drop_index("ix_agent_runs_user_status", table_name="agent_runs")
    op.drop_table("agent_runs")
