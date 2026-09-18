"""add durable offline evaluation runs for the compact evolution loop"""

from alembic import op
import sqlalchemy as sa


revision = "018_compact_evolution_runner"
down_revision = "017_structured_llm_proposals"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "offline_evaluation_runs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("idempotency_key", sa.String(100), nullable=False),
        sa.Column("proposal_id", sa.String(36), nullable=False),
        sa.Column("dataset_id", sa.String(36), nullable=False),
        sa.Column("skill_evaluation_id", sa.String(36), nullable=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="queued"),
        sa.Column("case_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("completed_case_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("attempt_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "runner_version",
            sa.String(40),
            nullable=False,
            server_default="agent_graph_runner_v1",
        ),
        sa.Column("metrics", sa.JSON(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_by", sa.String(36), nullable=True),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["proposal_id"], ["skill_proposals.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["dataset_id"], ["evaluation_datasets.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(
            ["skill_evaluation_id"], ["skill_evaluations.id"], ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(["created_by"], ["users.id"], ondelete="SET NULL"),
        sa.UniqueConstraint(
            "idempotency_key", name="uq_offline_evaluation_runs_idempotency_key"
        ),
    )
    op.create_index(
        "ix_offline_evaluation_runs_status_created",
        "offline_evaluation_runs",
        ["status", "created_at"],
    )
    op.create_index(
        "ix_offline_evaluation_runs_proposal_created",
        "offline_evaluation_runs",
        ["proposal_id", "created_at"],
    )
    op.create_index(
        "ix_offline_evaluation_runs_dataset_id", "offline_evaluation_runs", ["dataset_id"]
    )


def downgrade() -> None:
    op.drop_table("offline_evaluation_runs")
