"""add redacted experience, failure cases, and offline skill proposals"""

from alembic import op
import sqlalchemy as sa


revision = "008_self_evolution"
down_revision = "007_wellness_directory"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "experience_records",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("run_id", sa.String(36), sa.ForeignKey("agent_runs.id", ondelete="SET NULL"), nullable=True),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("intent", sa.String(30), nullable=False, server_default="unknown"),
        sa.Column("input_redacted", sa.Text(), nullable=False),
        sa.Column("plan_signature", sa.String(40), nullable=False, server_default=""),
        sa.Column("node_names", sa.JSON(), nullable=True),
        sa.Column("tool_calls", sa.JSON(), nullable=True),
        sa.Column("result_summary", sa.Text(), nullable=False),
        sa.Column("verify_status", sa.String(20), nullable=True),
        sa.Column("verify_issues", sa.JSON(), nullable=True),
        sa.Column("safety_flags", sa.JSON(), nullable=True),
        sa.Column("metrics", sa.JSON(), nullable=True),
        sa.Column("outcome", sa.String(30), nullable=False, server_default="completed"),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_index("ix_experience_records_run_id", "experience_records", ["run_id"])
    op.create_index("ix_experience_records_user_id", "experience_records", ["user_id"])
    op.create_index("ix_experience_records_user_created", "experience_records", ["user_id", "created_at"])
    op.create_index("ix_experience_records_verify_status", "experience_records", ["verify_status"])

    op.create_table(
        "failure_cases",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("experience_id", sa.String(36), sa.ForeignKey("experience_records.id", ondelete="CASCADE"), nullable=False),
        sa.Column("category", sa.String(50), nullable=False),
        sa.Column("severity", sa.String(20), nullable=False, server_default="medium"),
        sa.Column("repro_input_redacted", sa.Text(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="staging"),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_index("ix_failure_cases_experience_id", "failure_cases", ["experience_id"])
    op.create_index("ix_failure_cases_status", "failure_cases", ["status"])
    op.create_index("ix_failure_cases_status_severity", "failure_cases", ["status", "severity"])
    op.create_index("ix_failure_cases_category_created", "failure_cases", ["category", "created_at"])

    op.create_table(
        "skill_proposals",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("base_skill", sa.String(100), nullable=False),
        sa.Column("version", sa.String(40), nullable=False, server_default="0.1.0"),
        sa.Column("trigger", sa.String(500), nullable=False, server_default=""),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("source_failure_ids", sa.JSON(), nullable=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="candidate"),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_index("ix_skill_proposals_status", "skill_proposals", ["status"])
    op.create_index("ix_skill_proposals_status_created", "skill_proposals", ["status", "created_at"])


def downgrade() -> None:
    op.drop_table("skill_proposals")
    op.drop_table("failure_cases")
    op.drop_table("experience_records")
