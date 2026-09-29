"""add conservative recent health event memory"""
from alembic import op
import sqlalchemy as sa

revision = "019_health_memory"
down_revision = "018_compact_evolution_runner"
branch_labels = None
depends_on = None

def upgrade() -> None:
    op.create_table(
        "health_events",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(36), nullable=False),
        sa.Column("category", sa.String(50), nullable=False),
        sa.Column("display_name", sa.String(120), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="candidate"),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="0"),
        sa.Column("started_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("last_observed_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("resolved_at", sa.DateTime(), nullable=True),
        sa.Column("confirmation", sa.String(30), nullable=False, server_default="inferred"),
        sa.Column("context_policy", sa.String(30), nullable=False, server_default="relevant_only"),
        sa.Column("resolution_source", sa.String(30), nullable=True),
        # MySQL does not allow defaults on TEXT columns; the ORM supplies the
        # empty-string default when creating health-memory records.
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("metadata_json", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_health_events_user_status_observed", "health_events", ["user_id", "status", "last_observed_at"])
    op.create_index("ix_health_events_user_category", "health_events", ["user_id", "category"])
    op.create_table(
        "health_event_evidence",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("event_id", sa.String(36), nullable=False),
        sa.Column("conversation_id", sa.String(36), nullable=True),
        sa.Column("message_id", sa.String(36), nullable=True),
        sa.Column("source_type", sa.String(30), nullable=False, server_default="user_reported"),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("observed_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("metadata_json", sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(["event_id"], ["health_events.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["message_id"], ["messages.id"], ondelete="SET NULL"),
    )
    op.create_index("ix_health_event_evidence_event_observed", "health_event_evidence", ["event_id", "observed_at"])

def downgrade() -> None:
    op.drop_table("health_event_evidence")
    op.drop_index("ix_health_events_user_category", table_name="health_events")
    op.drop_index("ix_health_events_user_status_observed", table_name="health_events")
    op.drop_table("health_events")
