"""add capability gaps, challenge cases, and workflow template candidates"""

from alembic import op
import sqlalchemy as sa


revision = "011_evolution_artifacts"
down_revision = "010_memory_candidates"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "capability_gaps",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("fingerprint", sa.String(64), nullable=False),
        sa.Column("category", sa.String(40), nullable=False),
        sa.Column("capability", sa.String(120), nullable=False),
        sa.Column("evidence_redacted", sa.Text(), nullable=False),
        sa.Column("source_failure_ids", sa.JSON(), nullable=True),
        sa.Column("occurrence_count", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("status", sa.String(20), nullable=False, server_default="open"),
        sa.Column("resolution", sa.Text(), nullable=True),
        sa.Column("reviewer_id", sa.String(36), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("reviewed_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now()),
        sa.UniqueConstraint("fingerprint", name="uq_capability_gaps_fingerprint"),
    )
    op.create_index("ix_capability_gaps_fingerprint", "capability_gaps", ["fingerprint"], unique=True)
    op.create_index("ix_capability_gaps_status", "capability_gaps", ["status"])
    op.create_index("ix_capability_gaps_status_category", "capability_gaps", ["status", "category"])

    op.create_table(
        "evaluation_case_candidates",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("fingerprint", sa.String(64), nullable=False),
        sa.Column("source_failure_id", sa.String(36), sa.ForeignKey("failure_cases.id", ondelete="CASCADE"), nullable=False),
        sa.Column("suite", sa.String(30), nullable=False),
        sa.Column("prompt_redacted", sa.Text(), nullable=False),
        sa.Column("expected_constraints", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="staging"),
        sa.Column("reviewer_id", sa.String(36), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("review_reason", sa.Text(), nullable=True),
        sa.Column("reviewed_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.UniqueConstraint("fingerprint", name="uq_evaluation_case_candidates_fingerprint"),
    )
    op.create_index("ix_evaluation_case_candidates_fingerprint", "evaluation_case_candidates", ["fingerprint"], unique=True)
    op.create_index("ix_evaluation_case_candidates_source_failure_id", "evaluation_case_candidates", ["source_failure_id"])
    op.create_index("ix_evaluation_case_candidates_status", "evaluation_case_candidates", ["status"])
    op.create_index("ix_evaluation_case_candidates_status_suite", "evaluation_case_candidates", ["status", "suite"])

    op.create_table(
        "workflow_template_candidates",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("fingerprint", sa.String(64), nullable=False),
        sa.Column("name", sa.String(120), nullable=False),
        sa.Column("intent", sa.String(30), nullable=False),
        sa.Column("version", sa.String(40), nullable=False, server_default="0.1.0"),
        sa.Column("plan_signature", sa.String(40), nullable=False),
        sa.Column("task_shape", sa.JSON(), nullable=False),
        sa.Column("source_experience_ids", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="candidate"),
        sa.Column("reviewer_id", sa.String(36), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("review_reason", sa.Text(), nullable=True),
        sa.Column("reviewed_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.UniqueConstraint("fingerprint", name="uq_workflow_template_candidates_fingerprint"),
    )
    op.create_index("ix_workflow_template_candidates_fingerprint", "workflow_template_candidates", ["fingerprint"], unique=True)
    op.create_index("ix_workflow_template_candidates_status", "workflow_template_candidates", ["status"])
    op.create_index("ix_workflow_template_candidates_status_intent", "workflow_template_candidates", ["status", "intent"])

    op.create_table(
        "workflow_template_evaluations",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("template_id", sa.String(36), sa.ForeignKey("workflow_template_candidates.id", ondelete="CASCADE"), nullable=False),
        sa.Column("dataset_name", sa.String(200), nullable=False),
        sa.Column("case_count", sa.Integer(), nullable=False),
        sa.Column("passed", sa.Boolean(), nullable=False, server_default=sa.text("0")),
        sa.Column("metrics", sa.JSON(), nullable=False),
        sa.Column("reason_codes", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_index("ix_workflow_template_evaluations_template_id", "workflow_template_evaluations", ["template_id"])
    op.create_index("ix_workflow_template_evaluations_passed", "workflow_template_evaluations", ["passed"])
    op.create_index("ix_workflow_template_evaluations_template_created", "workflow_template_evaluations", ["template_id", "created_at"])

    op.create_table(
        "workflow_template_decisions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("template_id", sa.String(36), sa.ForeignKey("workflow_template_candidates.id", ondelete="CASCADE"), nullable=False),
        sa.Column("evaluation_id", sa.String(36), sa.ForeignKey("workflow_template_evaluations.id", ondelete="SET NULL"), nullable=True),
        sa.Column("reviewer_id", sa.String(36), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("decision", sa.String(20), nullable=False),
        sa.Column("previous_status", sa.String(20), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_index("ix_workflow_template_decisions_template_id", "workflow_template_decisions", ["template_id"])
    op.create_index("ix_workflow_template_decisions_template_created", "workflow_template_decisions", ["template_id", "created_at"])


def downgrade() -> None:
    op.drop_table("workflow_template_decisions")
    op.drop_table("workflow_template_evaluations")
    op.drop_table("workflow_template_candidates")
    op.drop_table("evaluation_case_candidates")
    op.drop_table("capability_gaps")
