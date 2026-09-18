"""add user-owned pending memory candidates"""

from alembic import op
import sqlalchemy as sa


revision = "010_memory_candidates"
down_revision = "009_evolution_review_sse"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "memory_candidates",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("kind", sa.String(30), nullable=False),
        sa.Column("value", sa.Text(), nullable=False),
        sa.Column("source", sa.String(50), nullable=False, server_default="user_correction"),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="1"),
        sa.Column("provenance", sa.JSON(), nullable=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("expires_at", sa.DateTime(), nullable=True),
        sa.Column("reviewed_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_index("ix_memory_candidates_user_id", "memory_candidates", ["user_id"])
    op.create_index("ix_memory_candidates_status", "memory_candidates", ["status"])
    op.create_index("ix_memory_candidates_user_status", "memory_candidates", ["user_id", "status"])
    op.create_index("ix_memory_candidates_user_created", "memory_candidates", ["user_id", "created_at"])


def downgrade() -> None:
    op.drop_table("memory_candidates")
