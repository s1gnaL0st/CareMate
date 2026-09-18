"""add immutable versioned evaluation dataset snapshots"""

from alembic import op
import sqlalchemy as sa


revision = "014_evaluation_datasets"
down_revision = "013_skill_composer_tool_drafts"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "evaluation_datasets",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("name", sa.String(120), nullable=False),
        sa.Column("version", sa.String(40), nullable=False),
        sa.Column("suite", sa.String(30), nullable=False),
        sa.Column("fingerprint", sa.String(64), nullable=False),
        sa.Column("case_count", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="frozen"),
        sa.Column("created_by", sa.String(36), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.UniqueConstraint("name", "version", name="uq_evaluation_datasets_name_version"),
        sa.UniqueConstraint("fingerprint", name="uq_evaluation_datasets_fingerprint"),
    )
    op.create_index("ix_evaluation_datasets_suite", "evaluation_datasets", ["suite"])
    op.create_index("ix_evaluation_datasets_status", "evaluation_datasets", ["status"])
    op.create_index("ix_evaluation_datasets_suite_created", "evaluation_datasets", ["suite", "created_at"])
    op.create_table(
        "evaluation_dataset_cases",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("dataset_id", sa.String(36), sa.ForeignKey("evaluation_datasets.id", ondelete="CASCADE"), nullable=False),
        sa.Column("source_candidate_id", sa.String(36), sa.ForeignKey("evaluation_case_candidates.id", ondelete="SET NULL"), nullable=True),
        sa.Column("case_key", sa.String(64), nullable=False),
        sa.Column("prompt_redacted", sa.Text(), nullable=False),
        sa.Column("expected_constraints", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.UniqueConstraint("dataset_id", "source_candidate_id", name="uq_evaluation_dataset_source_case"),
    )
    op.create_index("ix_evaluation_dataset_cases_dataset_id", "evaluation_dataset_cases", ["dataset_id"])
    op.create_index("ix_evaluation_dataset_cases_dataset_created", "evaluation_dataset_cases", ["dataset_id", "created_at"])


def downgrade() -> None:
    op.drop_table("evaluation_dataset_cases")
    op.drop_table("evaluation_datasets")
