"""add report object metadata and analysis retry counters"""

from alembic import op
import sqlalchemy as sa


revision = "003_report_storage_metadata"
down_revision = "002_complete_p0_tables"
branch_labels = None
depends_on = None


def _columns(table_name: str) -> set[str]:
    return {column["name"] for column in sa.inspect(op.get_bind()).get_columns(table_name)}


def upgrade() -> None:
    report_columns = _columns("medical_reports")
    if "size_bytes" not in report_columns:
        op.add_column("medical_reports", sa.Column("size_bytes", sa.Integer(), nullable=False, server_default="0"))
    if "sha256" not in report_columns:
        op.add_column("medical_reports", sa.Column("sha256", sa.String(length=64), nullable=True))
        op.create_index("ix_medical_reports_user_sha256", "medical_reports", ["user_id", "sha256"], unique=True)

    analysis_columns = _columns("report_analyses")
    if "attempt_count" not in analysis_columns:
        op.add_column("report_analyses", sa.Column("attempt_count", sa.Integer(), nullable=False, server_default="0"))


def downgrade() -> None:
    analysis_columns = _columns("report_analyses")
    if "attempt_count" in analysis_columns:
        op.drop_column("report_analyses", "attempt_count")

    report_columns = _columns("medical_reports")
    if "sha256" in report_columns:
        op.drop_index("ix_medical_reports_user_sha256", table_name="medical_reports")
        op.drop_column("medical_reports", "sha256")
    if "size_bytes" in report_columns:
        op.drop_column("medical_reports", "size_bytes")
