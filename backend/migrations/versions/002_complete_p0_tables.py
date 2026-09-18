"""add report, medication, insurance and chat-run tables"""
from alembic import op
import sqlalchemy as sa

revision = "002_complete_p0_tables"
down_revision = "001_initial"
branch_labels = None
depends_on = None


def _has_table(name: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    return inspector.has_table(name)


def upgrade() -> None:
    if not _has_table("report_analyses"):
        op.create_table("report_analyses", sa.Column("id", sa.String(36), primary_key=True), sa.Column("report_id", sa.String(36), sa.ForeignKey("medical_reports.id", ondelete="CASCADE"), nullable=False), sa.Column("status", sa.String(30), nullable=False), sa.Column("engine", sa.String(50), nullable=False), sa.Column("result", sa.JSON()), sa.Column("error_message", sa.Text()), sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()), sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now()))
        op.create_index("ix_report_analyses_report_id", "report_analyses", ["report_id"])
        op.create_index("ix_report_analyses_report_created", "report_analyses", ["report_id", "created_at"])
    if not _has_table("medications"):
        op.create_table("medications", sa.Column("id", sa.String(36), primary_key=True), sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False), sa.Column("name", sa.String(200), nullable=False), sa.Column("dosage", sa.String(200), nullable=False), sa.Column("instructions", sa.Text(), nullable=False), sa.Column("is_active", sa.Boolean(), nullable=False), sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()))
        op.create_index("ix_medications_user_id", "medications", ["user_id"])
        op.create_index("ix_medications_user_name", "medications", ["user_id", "name"])
    if not _has_table("medication_tasks"):
        op.create_table("medication_tasks", sa.Column("id", sa.String(36), primary_key=True), sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False), sa.Column("medication_id", sa.String(36), sa.ForeignKey("medications.id", ondelete="CASCADE"), nullable=False), sa.Column("scheduled_at", sa.DateTime(), nullable=False), sa.Column("completed_at", sa.DateTime()), sa.Column("status", sa.String(20), nullable=False))
        op.create_index("ix_medication_tasks_user_id", "medication_tasks", ["user_id"])
        op.create_index("ix_medication_tasks_medication_id", "medication_tasks", ["medication_id"])
        op.create_index("ix_medication_tasks_user_scheduled", "medication_tasks", ["user_id", "scheduled_at"])
    if not _has_table("insurance_accounts"):
        op.create_table("insurance_accounts", sa.Column("id", sa.String(36), primary_key=True), sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False), sa.Column("insurance_type", sa.String(100), nullable=False), sa.Column("region", sa.String(100), nullable=False), sa.Column("personal_balance", sa.Integer(), nullable=False), sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now()), sa.UniqueConstraint("user_id", "insurance_type", name="uq_user_insurance_type"))
        op.create_index("ix_insurance_accounts_user_id", "insurance_accounts", ["user_id"])
    if not _has_table("insurance_transactions"):
        op.create_table("insurance_transactions", sa.Column("id", sa.String(36), primary_key=True), sa.Column("account_id", sa.String(36), sa.ForeignKey("insurance_accounts.id", ondelete="CASCADE"), nullable=False), sa.Column("transaction_type", sa.String(30), nullable=False), sa.Column("amount_cents", sa.Integer(), nullable=False), sa.Column("self_pay_cents", sa.Integer(), nullable=False), sa.Column("reimbursed_cents", sa.Integer(), nullable=False), sa.Column("provider_name", sa.String(200), nullable=False), sa.Column("metadata_json", sa.JSON()), sa.Column("occurred_at", sa.DateTime(), nullable=False))
        op.create_index("ix_insurance_transactions_account_id", "insurance_transactions", ["account_id"])
        op.create_index("ix_insurance_transactions_account_occurred", "insurance_transactions", ["account_id", "occurred_at"])
    if not _has_table("chat_runs"):
        op.create_table("chat_runs", sa.Column("id", sa.String(36), primary_key=True), sa.Column("conversation_id", sa.String(36), sa.ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False), sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False), sa.Column("status", sa.String(20), nullable=False), sa.Column("error_message", sa.Text()), sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()), sa.Column("completed_at", sa.DateTime()))
        op.create_index("ix_chat_runs_conversation_id", "chat_runs", ["conversation_id"])
        op.create_index("ix_chat_runs_user_id", "chat_runs", ["user_id"])
        op.create_index("ix_chat_runs_conversation_created", "chat_runs", ["conversation_id", "created_at"])


def downgrade() -> None:
    for table in ("chat_runs", "insurance_transactions", "insurance_accounts", "medication_tasks", "medications", "report_analyses"):
        if _has_table(table):
            op.drop_table(table)
