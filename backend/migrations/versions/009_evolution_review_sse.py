"""add evolution review audit trail and durable SSE event ledger"""

from alembic import op
import sqlalchemy as sa


revision = "009_evolution_review_sse"
down_revision = "008_self_evolution"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "skill_evaluations",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("proposal_id", sa.String(36), sa.ForeignKey("skill_proposals.id", ondelete="CASCADE"), nullable=False),
        sa.Column("dataset_name", sa.String(200), nullable=False),
        sa.Column("evaluator", sa.String(100), nullable=False, server_default="offline_gate_v1"),
        sa.Column("case_count", sa.Integer(), nullable=False),
        sa.Column("passed", sa.Boolean(), nullable=False, server_default=sa.text("0")),
        sa.Column("metrics", sa.JSON(), nullable=False),
        sa.Column("reason_codes", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_index("ix_skill_evaluations_proposal_id", "skill_evaluations", ["proposal_id"])
    op.create_index("ix_skill_evaluations_passed", "skill_evaluations", ["passed"])
    op.create_index("ix_skill_evaluations_proposal_created", "skill_evaluations", ["proposal_id", "created_at"])

    op.create_table(
        "promotion_decisions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("proposal_id", sa.String(36), sa.ForeignKey("skill_proposals.id", ondelete="CASCADE"), nullable=False),
        sa.Column("evaluation_id", sa.String(36), sa.ForeignKey("skill_evaluations.id", ondelete="SET NULL"), nullable=True),
        sa.Column("reviewer_id", sa.String(36), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("decision", sa.String(20), nullable=False),
        sa.Column("previous_status", sa.String(20), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_index("ix_promotion_decisions_proposal_id", "promotion_decisions", ["proposal_id"])
    op.create_index("ix_promotion_decisions_reviewer_id", "promotion_decisions", ["reviewer_id"])
    op.create_index("ix_promotion_decisions_proposal_created", "promotion_decisions", ["proposal_id", "created_at"])

    op.create_table(
        "agent_run_events",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("run_id", sa.String(36), sa.ForeignKey("agent_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.UniqueConstraint("run_id", "sequence", name="uq_agent_run_events_sequence"),
    )
    op.create_index("ix_agent_run_events_run_id", "agent_run_events", ["run_id"])
    op.create_index("ix_agent_run_events_run_created", "agent_run_events", ["run_id", "created_at"])


def downgrade() -> None:
    op.drop_table("agent_run_events")
    op.drop_table("promotion_decisions")
    op.drop_table("skill_evaluations")
