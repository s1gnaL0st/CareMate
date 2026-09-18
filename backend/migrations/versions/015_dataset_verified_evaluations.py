"""bind offline evaluations to frozen datasets and per-case evidence"""

from alembic import op
import sqlalchemy as sa


revision = "015_dataset_verified_evaluations"
down_revision = "014_evaluation_datasets"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("skill_evaluations", sa.Column("dataset_id", sa.String(36), nullable=True))
    op.add_column("skill_evaluations", sa.Column("dataset_fingerprint", sa.String(64), nullable=True))
    op.add_column(
        "skill_evaluations",
        sa.Column("evidence_mode", sa.String(30), nullable=False, server_default="manual"),
    )
    op.create_foreign_key(
        "fk_skill_evaluations_dataset_id",
        "skill_evaluations",
        "evaluation_datasets",
        ["dataset_id"],
        ["id"],
        ondelete="RESTRICT",
    )
    op.create_index("ix_skill_evaluations_dataset_id", "skill_evaluations", ["dataset_id"])
    op.create_index("ix_skill_evaluations_evidence_mode", "skill_evaluations", ["evidence_mode"])

    op.add_column("workflow_template_evaluations", sa.Column("dataset_id", sa.String(36), nullable=True))
    op.add_column(
        "workflow_template_evaluations", sa.Column("dataset_fingerprint", sa.String(64), nullable=True)
    )
    op.add_column(
        "workflow_template_evaluations",
        sa.Column("evidence_mode", sa.String(30), nullable=False, server_default="manual"),
    )
    op.add_column(
        "workflow_template_evaluations",
        sa.Column("evaluator", sa.String(100), nullable=False, server_default="offline_gate_v1"),
    )
    op.create_foreign_key(
        "fk_workflow_template_evaluations_dataset_id",
        "workflow_template_evaluations",
        "evaluation_datasets",
        ["dataset_id"],
        ["id"],
        ondelete="RESTRICT",
    )
    op.create_index(
        "ix_workflow_template_evaluations_dataset_id", "workflow_template_evaluations", ["dataset_id"]
    )
    op.create_index(
        "ix_workflow_template_evaluations_evidence_mode",
        "workflow_template_evaluations",
        ["evidence_mode"],
    )

    op.create_table(
        "evaluation_case_results",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("skill_evaluation_id", sa.String(36), nullable=True),
        sa.Column("template_evaluation_id", sa.String(36), nullable=True),
        sa.Column("dataset_case_id", sa.String(36), nullable=False),
        sa.Column("case_key", sa.String(64), nullable=False),
        sa.Column("baseline_score", sa.Float(), nullable=False),
        sa.Column("candidate_score", sa.Float(), nullable=False),
        sa.Column("baseline_passed", sa.Boolean(), nullable=False),
        sa.Column("candidate_passed", sa.Boolean(), nullable=False),
        sa.Column("safety_violations", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("route_regression", sa.Boolean(), nullable=False, server_default=sa.text("0")),
        sa.Column("cost_delta", sa.Float(), nullable=False, server_default="0"),
        sa.Column("latency_delta", sa.Float(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.ForeignKeyConstraint(
            ["skill_evaluation_id"], ["skill_evaluations.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["template_evaluation_id"], ["workflow_template_evaluations.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["dataset_case_id"], ["evaluation_dataset_cases.id"], ondelete="RESTRICT"
        ),
        sa.CheckConstraint(
            "(skill_evaluation_id IS NOT NULL AND template_evaluation_id IS NULL) OR "
            "(skill_evaluation_id IS NULL AND template_evaluation_id IS NOT NULL)",
            name="ck_evaluation_case_results_one_parent",
        ),
        sa.UniqueConstraint(
            "skill_evaluation_id", "dataset_case_id", name="uq_evaluation_case_results_skill_case"
        ),
        sa.UniqueConstraint(
            "template_evaluation_id", "dataset_case_id", name="uq_evaluation_case_results_template_case"
        ),
    )
    op.create_index(
        "ix_evaluation_case_results_skill_evaluation_id",
        "evaluation_case_results",
        ["skill_evaluation_id"],
    )
    op.create_index(
        "ix_evaluation_case_results_template_evaluation_id",
        "evaluation_case_results",
        ["template_evaluation_id"],
    )
    op.create_index(
        "ix_evaluation_case_results_dataset_case_id",
        "evaluation_case_results",
        ["dataset_case_id"],
    )
    op.create_index(
        "ix_evaluation_case_results_skill_created",
        "evaluation_case_results",
        ["skill_evaluation_id", "created_at"],
    )
    op.create_index(
        "ix_evaluation_case_results_template_created",
        "evaluation_case_results",
        ["template_evaluation_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_table("evaluation_case_results")

    op.drop_index("ix_workflow_template_evaluations_evidence_mode", table_name="workflow_template_evaluations")
    op.drop_index("ix_workflow_template_evaluations_dataset_id", table_name="workflow_template_evaluations")
    op.drop_constraint(
        "fk_workflow_template_evaluations_dataset_id",
        "workflow_template_evaluations",
        type_="foreignkey",
    )
    op.drop_column("workflow_template_evaluations", "evaluator")
    op.drop_column("workflow_template_evaluations", "evidence_mode")
    op.drop_column("workflow_template_evaluations", "dataset_fingerprint")
    op.drop_column("workflow_template_evaluations", "dataset_id")

    op.drop_index("ix_skill_evaluations_evidence_mode", table_name="skill_evaluations")
    op.drop_index("ix_skill_evaluations_dataset_id", table_name="skill_evaluations")
    op.drop_constraint("fk_skill_evaluations_dataset_id", "skill_evaluations", type_="foreignkey")
    op.drop_column("skill_evaluations", "evidence_mode")
    op.drop_column("skill_evaluations", "dataset_fingerprint")
    op.drop_column("skill_evaluations", "dataset_id")
