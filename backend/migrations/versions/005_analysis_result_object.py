"""store report analysis output object key"""

from alembic import op
import sqlalchemy as sa


revision = "005_analysis_result_object"
down_revision = "004_chat_run_metrics"
branch_labels = None
depends_on = None


def upgrade() -> None:
    columns = {column["name"] for column in sa.inspect(op.get_bind()).get_columns("report_analyses")}
    if "result_object_key" not in columns:
        op.add_column("report_analyses", sa.Column("result_object_key", sa.String(length=500), nullable=True))


def downgrade() -> None:
    columns = {column["name"] for column in sa.inspect(op.get_bind()).get_columns("report_analyses")}
    if "result_object_key" in columns:
        op.drop_column("report_analyses", "result_object_key")
