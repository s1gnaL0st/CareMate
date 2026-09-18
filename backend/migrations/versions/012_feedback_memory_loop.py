"""connect accepted memory and answer feedback to evolution inputs"""

from alembic import op
import sqlalchemy as sa


revision = "012_feedback_memory_loop"
down_revision = "011_evolution_artifacts"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("failure_cases", sa.Column("source_feedback_id", sa.String(36), nullable=True))
    op.create_foreign_key(
        "fk_failure_cases_source_feedback",
        "failure_cases",
        "answer_feedback",
        ["source_feedback_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_unique_constraint(
        "uq_failure_cases_source_feedback_id",
        "failure_cases",
        ["source_feedback_id"],
    )


def downgrade() -> None:
    op.drop_constraint("uq_failure_cases_source_feedback_id", "failure_cases", type_="unique")
    op.drop_constraint("fk_failure_cases_source_feedback", "failure_cases", type_="foreignkey")
    op.drop_column("failure_cases", "source_feedback_id")
