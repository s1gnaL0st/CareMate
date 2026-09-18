"""add structured LLM proposal provenance and lineage"""

from alembic import op
import sqlalchemy as sa


revision = "017_structured_llm_proposals"
down_revision = "016_evaluation_campaigns"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("skill_proposals", sa.Column("content_fingerprint", sa.String(64), nullable=True))
    op.add_column("skill_proposals", sa.Column("parent_proposal_id", sa.String(36), nullable=True))
    op.add_column(
        "skill_proposals",
        sa.Column("generation_mode", sa.String(30), nullable=False, server_default="deterministic"),
    )
    op.add_column("skill_proposals", sa.Column("generator_label", sa.String(100), nullable=True))
    op.add_column("skill_proposals", sa.Column("generation_metadata", sa.JSON(), nullable=True))
    op.create_unique_constraint(
        "uq_skill_proposals_content_fingerprint", "skill_proposals", ["content_fingerprint"]
    )
    op.create_foreign_key(
        "fk_skill_proposals_parent_proposal_id",
        "skill_proposals",
        "skill_proposals",
        ["parent_proposal_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index("ix_skill_proposals_parent_proposal_id", "skill_proposals", ["parent_proposal_id"])


def downgrade() -> None:
    op.drop_index("ix_skill_proposals_parent_proposal_id", table_name="skill_proposals")
    op.drop_constraint("fk_skill_proposals_parent_proposal_id", "skill_proposals", type_="foreignkey")
    op.drop_constraint("uq_skill_proposals_content_fingerprint", "skill_proposals", type_="unique")
    op.drop_column("skill_proposals", "generation_metadata")
    op.drop_column("skill_proposals", "generator_label")
    op.drop_column("skill_proposals", "generation_mode")
    op.drop_column("skill_proposals", "parent_proposal_id")
    op.drop_column("skill_proposals", "content_fingerprint")
