"""add verified skill composition and tool interface drafts"""

from alembic import op
import sqlalchemy as sa


revision = "013_skill_composer_tool_drafts"
down_revision = "012_feedback_memory_loop"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "workflow_template_candidates",
        sa.Column("source_skill_proposal_ids", sa.JSON(), nullable=True),
    )
    op.create_table(
        "tool_interface_drafts",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("fingerprint", sa.String(64), nullable=False),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("purpose", sa.Text(), nullable=False),
        sa.Column("input_schema", sa.JSON(), nullable=False),
        sa.Column("output_schema", sa.JSON(), nullable=False),
        sa.Column("test_cases", sa.JSON(), nullable=False),
        sa.Column("source_gap_ids", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="candidate"),
        sa.Column("reviewer_id", sa.String(36), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("review_reason", sa.Text(), nullable=True),
        sa.Column("reviewed_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.UniqueConstraint("fingerprint", name="uq_tool_interface_drafts_fingerprint"),
    )
    op.create_index("ix_tool_interface_drafts_fingerprint", "tool_interface_drafts", ["fingerprint"], unique=True)
    op.create_index("ix_tool_interface_drafts_status", "tool_interface_drafts", ["status"])
    op.create_index("ix_tool_interface_drafts_status_created", "tool_interface_drafts", ["status", "created_at"])
    op.create_table(
        "tool_draft_decisions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("draft_id", sa.String(36), sa.ForeignKey("tool_interface_drafts.id", ondelete="CASCADE"), nullable=False),
        sa.Column("reviewer_id", sa.String(36), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("decision", sa.String(20), nullable=False),
        sa.Column("previous_status", sa.String(20), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_index("ix_tool_draft_decisions_draft_id", "tool_draft_decisions", ["draft_id"])
    op.create_index("ix_tool_draft_decisions_draft_created", "tool_draft_decisions", ["draft_id", "created_at"])


def downgrade() -> None:
    op.drop_table("tool_draft_decisions")
    op.drop_table("tool_interface_drafts")
    op.drop_column("workflow_template_candidates", "source_skill_proposal_ids")
