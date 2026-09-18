"""add LLM runtime metrics to chat runs"""

from alembic import op
import sqlalchemy as sa


revision = "004_chat_run_metrics"
down_revision = "003_report_storage_metadata"
branch_labels = None
depends_on = None


def _columns(table_name: str) -> set[str]:
    return {column["name"] for column in sa.inspect(op.get_bind()).get_columns(table_name)}


def upgrade() -> None:
    columns = _columns("chat_runs")
    additions = (
        ("model_name", sa.String(length=200), True, None),
        ("input_tokens", sa.Integer(), False, "0"),
        ("output_tokens", sa.Integer(), False, "0"),
        ("estimated_cost_micros", sa.Integer(), False, "0"),
        ("latency_ms", sa.Integer(), True, None),
    )
    for name, column_type, nullable, default in additions:
        if name not in columns:
            op.add_column(
                "chat_runs",
                sa.Column(name, column_type, nullable=nullable, server_default=default),
            )


def downgrade() -> None:
    columns = _columns("chat_runs")
    for name in ("latency_ms", "estimated_cost_micros", "output_tokens", "input_tokens", "model_name"):
        if name in columns:
            op.drop_column("chat_runs", name)
