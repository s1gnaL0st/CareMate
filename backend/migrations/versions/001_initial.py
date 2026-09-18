"""create initial business tables"""
from alembic import op
import sqlalchemy as sa

revision = "001_initial"
down_revision = None
branch_labels = None
depends_on = None

def upgrade() -> None:
    op.create_table("users", sa.Column("id", sa.String(36), primary_key=True), sa.Column("email", sa.String(255), nullable=False), sa.Column("password_hash", sa.String(255), nullable=False), sa.Column("name", sa.String(100), nullable=False), sa.Column("is_active", sa.Boolean(), nullable=False), sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()), sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now()), sa.UniqueConstraint("email"))
    op.create_index("ix_users_email", "users", ["email"], unique=False)
    op.create_table("user_profiles", sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="CASCADE"), primary_key=True), sa.Column("age", sa.Integer()), sa.Column("medical_history", sa.Text(), nullable=False), sa.Column("elder_mode", sa.Boolean(), nullable=False), sa.Column("region", sa.String(100), nullable=False))
    op.create_table("conversations", sa.Column("id", sa.String(36), primary_key=True), sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False), sa.Column("title", sa.String(200), nullable=False), sa.Column("active_agent", sa.String(50), nullable=False), sa.Column("status", sa.String(20), nullable=False), sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()), sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now()))
    op.create_index("ix_conversations_user_id", "conversations", ["user_id"])
    op.create_index("ix_conversations_user_updated", "conversations", ["user_id", "updated_at"])
    op.create_table("messages", sa.Column("id", sa.String(36), primary_key=True), sa.Column("conversation_id", sa.String(36), sa.ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False), sa.Column("role", sa.String(20), nullable=False), sa.Column("content", sa.Text(), nullable=False), sa.Column("sequence", sa.Integer(), nullable=False), sa.Column("event_metadata", sa.JSON()), sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()))
    op.create_index("ix_messages_conversation_id", "messages", ["conversation_id"])
    op.create_index("ix_messages_conversation_created", "messages", ["conversation_id", "created_at"])
    op.create_index("ix_messages_created_at", "messages", ["created_at"])
    op.create_unique_constraint("uq_message_sequence", "messages", ["conversation_id", "sequence"])
    op.create_table("medical_reports", sa.Column("id", sa.String(36), primary_key=True), sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False), sa.Column("object_key", sa.String(500), nullable=False), sa.Column("original_filename", sa.String(255), nullable=False), sa.Column("content_type", sa.String(100), nullable=False), sa.Column("status", sa.String(30), nullable=False), sa.Column("analysis_result", sa.JSON()), sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()), sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now()))
    op.create_index("ix_medical_reports_user_id", "medical_reports", ["user_id"])
    op.create_index("ix_reports_user_created", "medical_reports", ["user_id", "created_at"])
    op.create_table("report_analyses", sa.Column("id", sa.String(36), primary_key=True), sa.Column("report_id", sa.String(36), sa.ForeignKey("medical_reports.id", ondelete="CASCADE"), nullable=False), sa.Column("status", sa.String(30), nullable=False), sa.Column("engine", sa.String(50), nullable=False), sa.Column("result", sa.JSON()), sa.Column("error_message", sa.Text()), sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()), sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now()))
    op.create_index("ix_report_analyses_report_id", "report_analyses", ["report_id"])
    op.create_index("ix_report_analyses_report_created", "report_analyses", ["report_id", "created_at"])
    op.create_table("medications", sa.Column("id", sa.String(36), primary_key=True), sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False), sa.Column("name", sa.String(200), nullable=False), sa.Column("dosage", sa.String(200), nullable=False), sa.Column("instructions", sa.Text(), nullable=False), sa.Column("is_active", sa.Boolean(), nullable=False), sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()))
    op.create_index("ix_medications_user_id", "medications", ["user_id"])
    op.create_index("ix_medications_user_name", "medications", ["user_id", "name"])
    op.create_table("medication_tasks", sa.Column("id", sa.String(36), primary_key=True), sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False), sa.Column("medication_id", sa.String(36), sa.ForeignKey("medications.id", ondelete="CASCADE"), nullable=False), sa.Column("scheduled_at", sa.DateTime(), nullable=False), sa.Column("completed_at", sa.DateTime()), sa.Column("status", sa.String(20), nullable=False))
    op.create_index("ix_medication_tasks_user_id", "medication_tasks", ["user_id"])
    op.create_index("ix_medication_tasks_medication_id", "medication_tasks", ["medication_id"])
    op.create_index("ix_medication_tasks_user_scheduled", "medication_tasks", ["user_id", "scheduled_at"])
    op.create_table("insurance_accounts", sa.Column("id", sa.String(36), primary_key=True), sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False), sa.Column("insurance_type", sa.String(100), nullable=False), sa.Column("region", sa.String(100), nullable=False), sa.Column("personal_balance", sa.Integer(), nullable=False), sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now()), sa.UniqueConstraint("user_id", "insurance_type", name="uq_user_insurance_type"))
    op.create_index("ix_insurance_accounts_user_id", "insurance_accounts", ["user_id"])
    op.create_table("insurance_transactions", sa.Column("id", sa.String(36), primary_key=True), sa.Column("account_id", sa.String(36), sa.ForeignKey("insurance_accounts.id", ondelete="CASCADE"), nullable=False), sa.Column("transaction_type", sa.String(30), nullable=False), sa.Column("amount_cents", sa.Integer(), nullable=False), sa.Column("self_pay_cents", sa.Integer(), nullable=False), sa.Column("reimbursed_cents", sa.Integer(), nullable=False), sa.Column("provider_name", sa.String(200), nullable=False), sa.Column("metadata_json", sa.JSON()), sa.Column("occurred_at", sa.DateTime(), nullable=False))
    op.create_index("ix_insurance_transactions_account_id", "insurance_transactions", ["account_id"])
    op.create_index("ix_insurance_transactions_account_occurred", "insurance_transactions", ["account_id", "occurred_at"])
    op.create_table("audit_logs", sa.Column("id", sa.String(36), primary_key=True), sa.Column("user_id", sa.String(36)), sa.Column("action", sa.String(100), nullable=False), sa.Column("resource_type", sa.String(50)), sa.Column("resource_id", sa.String(36)), sa.Column("request_id", sa.String(36)), sa.Column("details", sa.JSON()), sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()))
    op.create_index("ix_audit_logs_user_id", "audit_logs", ["user_id"])
    op.create_index("ix_audit_logs_request_id", "audit_logs", ["request_id"])
    op.create_table("chat_runs", sa.Column("id", sa.String(36), primary_key=True), sa.Column("conversation_id", sa.String(36), sa.ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False), sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False), sa.Column("status", sa.String(20), nullable=False), sa.Column("error_message", sa.Text()), sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()), sa.Column("completed_at", sa.DateTime()))
    op.create_index("ix_chat_runs_conversation_id", "chat_runs", ["conversation_id"])
    op.create_index("ix_chat_runs_user_id", "chat_runs", ["user_id"])
    op.create_index("ix_chat_runs_conversation_created", "chat_runs", ["conversation_id", "created_at"])

def downgrade() -> None:
    op.drop_table("chat_runs")
    op.drop_table("audit_logs")
    op.drop_table("insurance_transactions")
    op.drop_table("insurance_accounts")
    op.drop_table("medication_tasks")
    op.drop_table("medications")
    op.drop_table("report_analyses")
    op.drop_table("medical_reports")
    op.drop_table("messages")
    op.drop_table("conversations")
    op.drop_table("user_profiles")
    op.drop_table("users")
