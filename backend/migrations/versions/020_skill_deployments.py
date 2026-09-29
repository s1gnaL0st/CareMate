"""add reversible online skill deployment pointers"""
from alembic import op
import sqlalchemy as sa

revision = "020_skill_deployments"
down_revision = "019_health_memory"
branch_labels = None
depends_on = None

def upgrade() -> None:
    op.create_table(
        "skill_deployments",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("base_skill", sa.String(100), nullable=False),
        sa.Column("proposal_id", sa.String(36), nullable=False),
        sa.Column("previous_proposal_id", sa.String(36), nullable=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="active"),
        sa.Column("deployed_by", sa.String(36), nullable=True),
        # MySQL rejects defaults on TEXT columns; application code supplies
        # an empty reason when omitted.
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("deployed_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["proposal_id"], ["skill_proposals.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["previous_proposal_id"], ["skill_proposals.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["deployed_by"], ["users.id"], ondelete="SET NULL"),
        sa.UniqueConstraint("base_skill", name="uq_skill_deployments_base_skill"),
    )
    op.create_index("ix_skill_deployments_base_skill", "skill_deployments", ["base_skill"])
    op.create_index("ix_skill_deployments_proposal_id", "skill_deployments", ["proposal_id"])
    op.create_index("ix_skill_deployments_status", "skill_deployments", ["status"])

def downgrade() -> None:
    op.drop_table("skill_deployments")
