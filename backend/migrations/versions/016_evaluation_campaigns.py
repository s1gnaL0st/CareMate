"""add immutable multi-suite evaluation campaigns"""

from alembic import op
import sqlalchemy as sa


revision = "016_evaluation_campaigns"
down_revision = "015_dataset_verified_evaluations"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "evaluation_campaigns",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("proposal_id", sa.String(36), nullable=True),
        sa.Column("template_id", sa.String(36), nullable=True),
        sa.Column("policy_version", sa.String(40), nullable=False, server_default="medical_promotion_v1"),
        sa.Column("required_suites", sa.JSON(), nullable=False),
        sa.Column("evaluation_ids", sa.JSON(), nullable=False),
        sa.Column("dataset_fingerprints", sa.JSON(), nullable=False),
        sa.Column("metrics", sa.JSON(), nullable=False),
        sa.Column("passed", sa.Boolean(), nullable=False, server_default=sa.text("0")),
        sa.Column("reason_codes", sa.JSON(), nullable=True),
        sa.Column("fingerprint", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["proposal_id"], ["skill_proposals.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["template_id"], ["workflow_template_candidates.id"], ondelete="CASCADE"
        ),
        sa.CheckConstraint(
            "(proposal_id IS NOT NULL AND template_id IS NULL) OR "
            "(proposal_id IS NULL AND template_id IS NOT NULL)",
            name="ck_evaluation_campaigns_one_target",
        ),
        sa.UniqueConstraint("fingerprint", name="uq_evaluation_campaigns_fingerprint"),
    )
    op.create_index("ix_evaluation_campaigns_proposal_id", "evaluation_campaigns", ["proposal_id"])
    op.create_index("ix_evaluation_campaigns_template_id", "evaluation_campaigns", ["template_id"])
    op.create_index("ix_evaluation_campaigns_passed", "evaluation_campaigns", ["passed"])
    op.create_index(
        "ix_evaluation_campaigns_proposal_created", "evaluation_campaigns", ["proposal_id", "created_at"]
    )
    op.create_index(
        "ix_evaluation_campaigns_template_created", "evaluation_campaigns", ["template_id", "created_at"]
    )

    op.add_column("promotion_decisions", sa.Column("campaign_id", sa.String(36), nullable=True))
    op.create_foreign_key(
        "fk_promotion_decisions_campaign_id",
        "promotion_decisions",
        "evaluation_campaigns",
        ["campaign_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.add_column("workflow_template_decisions", sa.Column("campaign_id", sa.String(36), nullable=True))
    op.create_foreign_key(
        "fk_workflow_template_decisions_campaign_id",
        "workflow_template_decisions",
        "evaluation_campaigns",
        ["campaign_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    op.drop_constraint(
        "fk_workflow_template_decisions_campaign_id", "workflow_template_decisions", type_="foreignkey"
    )
    op.drop_column("workflow_template_decisions", "campaign_id")
    op.drop_constraint("fk_promotion_decisions_campaign_id", "promotion_decisions", type_="foreignkey")
    op.drop_column("promotion_decisions", "campaign_id")
    op.drop_table("evaluation_campaigns")
