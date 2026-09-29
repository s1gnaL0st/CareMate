"""Business persistence models for the health assistant."""
from datetime import datetime
from uuid import uuid4

from sqlalchemy import Boolean, CheckConstraint, DateTime, Float, ForeignKey, Index, Integer, JSON, String, Text, UniqueConstraint, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sensitive_data import EncryptedText


class Base(DeclarativeBase):
    pass


def uuid_str() -> str:
    return str(uuid4())


class User(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_str)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    name: Mapped[str] = mapped_column(String(100), default="用户")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class UserProfile(Base):
    __tablename__ = "user_profiles"
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    age: Mapped[int | None] = mapped_column(Integer, nullable=True)
    medical_history: Mapped[str] = mapped_column(EncryptedText(), default="")
    elder_mode: Mapped[bool] = mapped_column(Boolean, default=False)
    region: Mapped[str] = mapped_column(String(100), default="")


class Conversation(Base):
    __tablename__ = "conversations"
    __table_args__ = (Index("ix_conversations_user_updated", "user_id", "updated_at"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_str)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    title: Mapped[str] = mapped_column(String(200), default="新对话")
    active_agent: Mapped[str] = mapped_column(String(50), default="advisor_agent")
    status: Mapped[str] = mapped_column(String(20), default="active")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (UniqueConstraint("conversation_id", "sequence", name="uq_message_sequence"), Index("ix_messages_conversation_created", "conversation_id", "created_at"))
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_str)
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id", ondelete="CASCADE"), index=True)
    role: Mapped[str] = mapped_column(String(20))
    content: Mapped[str] = mapped_column(Text)
    sequence: Mapped[int] = mapped_column(Integer)
    event_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), index=True)


class MedicalReport(Base):
    __tablename__ = "medical_reports"
    __table_args__ = (Index("ix_reports_user_created", "user_id", "created_at"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_str)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    object_key: Mapped[str] = mapped_column(String(500))
    original_filename: Mapped[str] = mapped_column(String(255))
    content_type: Mapped[str] = mapped_column(String(100))
    size_bytes: Mapped[int] = mapped_column(Integer)
    sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(30), default="uploaded")
    analysis_result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class ReportAnalysis(Base):
    __tablename__ = "report_analyses"
    __table_args__ = (Index("ix_report_analyses_report_created", "report_id", "created_at"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_str)
    report_id: Mapped[str] = mapped_column(ForeignKey("medical_reports.id", ondelete="CASCADE"), index=True)
    status: Mapped[str] = mapped_column(String(30), default="pending")
    engine: Mapped[str] = mapped_column(String(50), default="vision_llm")
    result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    result_object_key: Mapped[str | None] = mapped_column(String(500), nullable=True)
    attempt_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class Medication(Base):
    __tablename__ = "medications"
    __table_args__ = (Index("ix_medications_user_name", "user_id", "name"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_str)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    dosage: Mapped[str] = mapped_column(String(200), default="")
    instructions: Mapped[str] = mapped_column(Text, default="")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class MedicationTask(Base):
    __tablename__ = "medication_tasks"
    __table_args__ = (Index("ix_medication_tasks_user_scheduled", "user_id", "scheduled_at"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_str)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    medication_id: Mapped[str] = mapped_column(ForeignKey("medications.id", ondelete="CASCADE"), index=True)
    scheduled_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="pending")


class Hospital(Base):
    __tablename__ = "hospitals"
    __table_args__ = (Index("ix_hospitals_city_name", "city", "name"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_str)
    name: Mapped[str] = mapped_column(String(200))
    city: Mapped[str] = mapped_column(String(100), default="")
    address: Mapped[str] = mapped_column(String(300), default="")
    tags: Mapped[list] = mapped_column(JSON, default=list)
    distance_km: Mapped[float | None] = mapped_column(Float, nullable=True)
    phone: Mapped[str] = mapped_column(String(50), default="")
    is_ad: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class HabitGoal(Base):
    __tablename__ = "habit_goals"
    __table_args__ = (Index("ix_habit_goals_user_status", "user_id", "status"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_str)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    title: Mapped[str] = mapped_column(String(200))
    target_value: Mapped[int] = mapped_column(Integer, default=1)
    current_value: Mapped[int] = mapped_column(Integer, default=0)
    unit: Mapped[str] = mapped_column(String(30), default="次")
    color: Mapped[str] = mapped_column(String(30), default="bg-teal-500")
    status: Mapped[str] = mapped_column(String(20), default="active")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class AnswerFeedback(Base):
    __tablename__ = "answer_feedback"
    __table_args__ = (UniqueConstraint("user_id", "message_id", name="uq_feedback_user_message"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_str)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    message_id: Mapped[str | None] = mapped_column(ForeignKey("messages.id", ondelete="SET NULL"), nullable=True, index=True)
    conversation_id: Mapped[str | None] = mapped_column(ForeignKey("conversations.id", ondelete="SET NULL"), nullable=True, index=True)
    rating: Mapped[int] = mapped_column(Integer)
    category: Mapped[str | None] = mapped_column(String(50), nullable=True)
    comment: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class InsuranceAccount(Base):
    __tablename__ = "insurance_accounts"
    __table_args__ = (UniqueConstraint("user_id", "insurance_type", name="uq_user_insurance_type"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_str)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    insurance_type: Mapped[str] = mapped_column(String(100))
    region: Mapped[str] = mapped_column(String(100), default="")
    personal_balance: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class InsuranceTransaction(Base):
    __tablename__ = "insurance_transactions"
    __table_args__ = (Index("ix_insurance_transactions_account_occurred", "account_id", "occurred_at"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_str)
    account_id: Mapped[str] = mapped_column(ForeignKey("insurance_accounts.id", ondelete="CASCADE"), index=True)
    transaction_type: Mapped[str] = mapped_column(String(30))
    amount_cents: Mapped[int] = mapped_column(Integer)
    self_pay_cents: Mapped[int] = mapped_column(Integer, default=0)
    reimbursed_cents: Mapped[int] = mapped_column(Integer, default=0)
    provider_name: Mapped[str] = mapped_column(String(200), default="")
    metadata_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    occurred_at: Mapped[datetime] = mapped_column(DateTime, index=True)


class AuditLog(Base):
    __tablename__ = "audit_logs"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_str)
    user_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    action: Mapped[str] = mapped_column(String(100))
    resource_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    resource_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    request_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    details: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class ChatRun(Base):
    __tablename__ = "chat_runs"
    __table_args__ = (Index("ix_chat_runs_conversation_created", "conversation_id", "created_at"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id", ondelete="CASCADE"), index=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    status: Mapped[str] = mapped_column(String(20), default="pending")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    model_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    estimated_cost_micros: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class AgentRun(Base):
    """Durable control record for one Supervisor AgentLoop execution."""

    __tablename__ = "agent_runs"
    __table_args__ = (Index("ix_agent_runs_user_status", "user_id", "status"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_str)
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id", ondelete="CASCADE"), index=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    status: Mapped[str] = mapped_column(String(30), default="pending", index=True)
    pause_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    plan_signature: Mapped[str] = mapped_column(String(40), default="")
    replan_count: Mapped[int] = mapped_column(Integer, default=0)
    requested_mode: Mapped[str] = mapped_column(String(50), default="general")
    initial_state: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    lease_until: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class AgentTask(Base):
    """Durable task ledger used for task-boundary pause and recovery."""

    __tablename__ = "agent_tasks"
    __table_args__ = (
        UniqueConstraint("run_id", "task_key", name="uq_agent_tasks_run_key"),
        Index("ix_agent_tasks_run_status", "run_id", "status"),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_str)
    run_id: Mapped[str] = mapped_column(ForeignKey("agent_runs.id", ondelete="CASCADE"), index=True)
    task_key: Mapped[str] = mapped_column(String(80))
    agent: Mapped[str] = mapped_column(String(50))
    objective: Mapped[str] = mapped_column(String(500))
    input_slice: Mapped[str] = mapped_column(Text)
    depends_on: Mapped[list | None] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String(30), default="pending", index=True)
    attempt_count: Mapped[int] = mapped_column(Integer, default=0)
    result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    lease_until: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class ExperienceRecord(Base):
    """Redacted evidence from one AgentLoop execution.

    Experience is an offline learning input only.  It is deliberately kept
    separate from the live planner and stores no raw tool arguments.
    """

    __tablename__ = "experience_records"
    __table_args__ = (
        Index("ix_experience_records_user_created", "user_id", "created_at"),
        Index("ix_experience_records_verify_status", "verify_status"),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_str)
    run_id: Mapped[str | None] = mapped_column(ForeignKey("agent_runs.id", ondelete="SET NULL"), nullable=True, index=True)
    user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    intent: Mapped[str] = mapped_column(String(30), default="unknown")
    input_redacted: Mapped[str] = mapped_column(EncryptedText(), default="")
    plan_signature: Mapped[str] = mapped_column(String(40), default="")
    node_names: Mapped[list | None] = mapped_column(JSON, nullable=True)
    tool_calls: Mapped[list | None] = mapped_column(JSON, nullable=True)
    result_summary: Mapped[str] = mapped_column(EncryptedText(), default="")
    verify_status: Mapped[str | None] = mapped_column(String(20), nullable=True)
    verify_issues: Mapped[list | None] = mapped_column(JSON, nullable=True)
    safety_flags: Mapped[list | None] = mapped_column(JSON, nullable=True)
    metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    outcome: Mapped[str] = mapped_column(String(30), default="completed")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class FailureCase(Base):
    """A small, redacted reproducible case for offline evaluation."""

    __tablename__ = "failure_cases"
    __table_args__ = (
        Index("ix_failure_cases_status_severity", "status", "severity"),
        Index("ix_failure_cases_category_created", "category", "created_at"),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_str)
    experience_id: Mapped[str] = mapped_column(ForeignKey("experience_records.id", ondelete="CASCADE"), index=True)
    source_feedback_id: Mapped[str | None] = mapped_column(
        ForeignKey("answer_feedback.id", ondelete="SET NULL"), nullable=True, unique=True
    )
    category: Mapped[str] = mapped_column(String(50))
    severity: Mapped[str] = mapped_column(String(20), default="medium")
    repro_input_redacted: Mapped[str] = mapped_column(EncryptedText(), default="")
    summary: Mapped[str] = mapped_column(EncryptedText(), default="")
    status: Mapped[str] = mapped_column(String(20), default="staging", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class SkillProposal(Base):
    """Offline-generated skill candidate; never enabled by insertion alone."""

    __tablename__ = "skill_proposals"
    __table_args__ = (Index("ix_skill_proposals_status_created", "status", "created_at"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_str)
    base_skill: Mapped[str] = mapped_column(String(100))
    version: Mapped[str] = mapped_column(String(40), default="0.1.0")
    trigger: Mapped[str] = mapped_column(String(500), default="")
    content: Mapped[str] = mapped_column(EncryptedText(), default="")
    content_fingerprint: Mapped[str | None] = mapped_column(String(64), nullable=True, unique=True)
    source_failure_ids: Mapped[list | None] = mapped_column(JSON, nullable=True)
    parent_proposal_id: Mapped[str | None] = mapped_column(
        ForeignKey("skill_proposals.id", ondelete="SET NULL"), nullable=True, index=True
    )
    generation_mode: Mapped[str] = mapped_column(String(30), default="deterministic")
    generator_label: Mapped[str | None] = mapped_column(String(100), nullable=True)
    generation_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="candidate", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class SkillEvaluation(Base):
    """Immutable offline gate result for one skill proposal."""

    __tablename__ = "skill_evaluations"
    __table_args__ = (Index("ix_skill_evaluations_proposal_created", "proposal_id", "created_at"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_str)
    proposal_id: Mapped[str] = mapped_column(ForeignKey("skill_proposals.id", ondelete="CASCADE"), index=True)
    dataset_name: Mapped[str] = mapped_column(String(200))
    dataset_id: Mapped[str | None] = mapped_column(
        ForeignKey("evaluation_datasets.id", ondelete="RESTRICT"), nullable=True, index=True
    )
    dataset_fingerprint: Mapped[str | None] = mapped_column(String(64), nullable=True)
    evidence_mode: Mapped[str] = mapped_column(String(30), default="manual", index=True)
    evaluator: Mapped[str] = mapped_column(String(100), default="offline_gate_v1")
    case_count: Mapped[int] = mapped_column(Integer)
    passed: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    metrics: Mapped[dict] = mapped_column(JSON)
    reason_codes: Mapped[list | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class PromotionDecision(Base):
    """Auditable human decision. Approval does not install or publish code."""

    __tablename__ = "promotion_decisions"
    __table_args__ = (Index("ix_promotion_decisions_proposal_created", "proposal_id", "created_at"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_str)
    proposal_id: Mapped[str] = mapped_column(ForeignKey("skill_proposals.id", ondelete="CASCADE"), index=True)
    evaluation_id: Mapped[str | None] = mapped_column(ForeignKey("skill_evaluations.id", ondelete="SET NULL"), nullable=True)
    campaign_id: Mapped[str | None] = mapped_column(ForeignKey("evaluation_campaigns.id", ondelete="SET NULL"), nullable=True)
    reviewer_id: Mapped[str | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    decision: Mapped[str] = mapped_column(String(20))
    previous_status: Mapped[str] = mapped_column(String(20))
    reason: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class SkillDeployment(Base):
    """Version pointer for online skills; deployment is reversible and auditable."""

    __tablename__ = "skill_deployments"
    __table_args__ = (UniqueConstraint("base_skill", name="uq_skill_deployments_base_skill"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_str)
    base_skill: Mapped[str] = mapped_column(String(100), index=True)
    proposal_id: Mapped[str] = mapped_column(ForeignKey("skill_proposals.id", ondelete="RESTRICT"), index=True)
    previous_proposal_id: Mapped[str | None] = mapped_column(ForeignKey("skill_proposals.id", ondelete="SET NULL"), nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="active", index=True)
    deployed_by: Mapped[str | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    reason: Mapped[str] = mapped_column(Text, default="")
    deployed_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class AgentRunEvent(Base):
    """Durable SSE event ledger used for reconnect replay and de-duplication."""

    __tablename__ = "agent_run_events"
    __table_args__ = (
        UniqueConstraint("run_id", "sequence", name="uq_agent_run_events_sequence"),
        Index("ix_agent_run_events_run_created", "run_id", "created_at"),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_str)
    run_id: Mapped[str] = mapped_column(ForeignKey("agent_runs.id", ondelete="CASCADE"), index=True)
    sequence: Mapped[int] = mapped_column(Integer)
    payload: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class MemoryCandidate(Base):
    """User-owned, reviewable preference candidate; never a medical fact."""

    __tablename__ = "memory_candidates"
    __table_args__ = (
        Index("ix_memory_candidates_user_status", "user_id", "status"),
        Index("ix_memory_candidates_user_created", "user_id", "created_at"),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_str)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(30))
    value: Mapped[str] = mapped_column(EncryptedText())
    source: Mapped[str] = mapped_column(String(50), default="user_correction")
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    provenance: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="pending", index=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class HealthEvent(Base):
    """A user-reported or weakly inferred episode, never a diagnosis."""

    __tablename__ = "health_events"
    __table_args__ = (
        Index("ix_health_events_user_status_observed", "user_id", "status", "last_observed_at"),
        Index("ix_health_events_user_category", "user_id", "category"),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_str)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    category: Mapped[str] = mapped_column(String(50))
    display_name: Mapped[str] = mapped_column(String(120))
    status: Mapped[str] = mapped_column(String(20), default="candidate", index=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    started_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    last_observed_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    confirmation: Mapped[str] = mapped_column(String(30), default="inferred")
    context_policy: Mapped[str] = mapped_column(String(30), default="relevant_only")
    resolution_source: Mapped[str | None] = mapped_column(String(30), nullable=True)
    summary: Mapped[str] = mapped_column(EncryptedText(), default="")
    metadata_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class HealthEventEvidence(Base):
    __tablename__ = "health_event_evidence"
    __table_args__ = (Index("ix_health_event_evidence_event_observed", "event_id", "observed_at"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_str)
    event_id: Mapped[str] = mapped_column(ForeignKey("health_events.id", ondelete="CASCADE"), index=True)
    conversation_id: Mapped[str | None] = mapped_column(ForeignKey("conversations.id", ondelete="SET NULL"), nullable=True)
    message_id: Mapped[str | None] = mapped_column(ForeignKey("messages.id", ondelete="SET NULL"), nullable=True)
    source_type: Mapped[str] = mapped_column(String(30), default="user_reported")
    summary: Mapped[str] = mapped_column(EncryptedText(), default="")
    observed_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    metadata_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class CapabilityGap(Base):
    """Deduplicated offline record of a missing or unreliable capability."""

    __tablename__ = "capability_gaps"
    __table_args__ = (
        UniqueConstraint("fingerprint", name="uq_capability_gaps_fingerprint"),
        Index("ix_capability_gaps_status_category", "status", "category"),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_str)
    fingerprint: Mapped[str] = mapped_column(String(64), unique=True)
    category: Mapped[str] = mapped_column(String(40))
    capability: Mapped[str] = mapped_column(String(120))
    evidence_redacted: Mapped[str] = mapped_column(EncryptedText(), default="")
    source_failure_ids: Mapped[list | None] = mapped_column(JSON, nullable=True)
    occurrence_count: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[str] = mapped_column(String(20), default="open", index=True)
    resolution: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewer_id: Mapped[str | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class EvaluationCaseCandidate(Base):
    """Redacted generated challenge case awaiting dataset review."""

    __tablename__ = "evaluation_case_candidates"
    __table_args__ = (
        UniqueConstraint("fingerprint", name="uq_evaluation_case_candidates_fingerprint"),
        Index("ix_evaluation_case_candidates_status_suite", "status", "suite"),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_str)
    fingerprint: Mapped[str] = mapped_column(String(64), unique=True)
    source_failure_id: Mapped[str | None] = mapped_column(
        ForeignKey("failure_cases.id", ondelete="CASCADE"), nullable=True, index=True
    )
    origin: Mapped[str] = mapped_column(String(30), default="failure_generated", index=True)
    created_by: Mapped[str | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    suite: Mapped[str] = mapped_column(String(30))
    prompt_redacted: Mapped[str] = mapped_column(EncryptedText(), default="")
    expected_constraints: Mapped[dict] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(20), default="staging", index=True)
    reviewer_id: Mapped[str | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    review_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class EvaluationDataset(Base):
    """Immutable, versioned offline dataset assembled from accepted cases."""

    __tablename__ = "evaluation_datasets"
    __table_args__ = (
        UniqueConstraint("name", "version", name="uq_evaluation_datasets_name_version"),
        Index("ix_evaluation_datasets_suite_created", "suite", "created_at"),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_str)
    name: Mapped[str] = mapped_column(String(120))
    version: Mapped[str] = mapped_column(String(40))
    suite: Mapped[str] = mapped_column(String(30), index=True)
    fingerprint: Mapped[str] = mapped_column(String(64), unique=True)
    case_count: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(20), default="frozen", index=True)
    created_by: Mapped[str | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class EvaluationDatasetCase(Base):
    """Snapshot of one accepted candidate, isolated from later source edits."""

    __tablename__ = "evaluation_dataset_cases"
    __table_args__ = (
        UniqueConstraint("dataset_id", "source_candidate_id", name="uq_evaluation_dataset_source_case"),
        Index("ix_evaluation_dataset_cases_dataset_created", "dataset_id", "created_at"),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_str)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("evaluation_datasets.id", ondelete="CASCADE"), index=True)
    source_candidate_id: Mapped[str | None] = mapped_column(
        ForeignKey("evaluation_case_candidates.id", ondelete="SET NULL"), nullable=True
    )
    case_key: Mapped[str] = mapped_column(String(64))
    prompt_redacted: Mapped[str] = mapped_column(EncryptedText())
    expected_constraints: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class WorkflowTemplateCandidate(Base):
    """Task-shape-only template candidate extracted from verified runs."""

    __tablename__ = "workflow_template_candidates"
    __table_args__ = (
        UniqueConstraint("fingerprint", name="uq_workflow_template_candidates_fingerprint"),
        Index("ix_workflow_template_candidates_status_intent", "status", "intent"),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_str)
    fingerprint: Mapped[str] = mapped_column(String(64), unique=True)
    name: Mapped[str] = mapped_column(String(120))
    intent: Mapped[str] = mapped_column(String(30))
    version: Mapped[str] = mapped_column(String(40), default="0.1.0")
    plan_signature: Mapped[str] = mapped_column(String(40))
    task_shape: Mapped[dict] = mapped_column(JSON)
    source_experience_ids: Mapped[list] = mapped_column(JSON)
    source_skill_proposal_ids: Mapped[list | None] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="candidate", index=True)
    reviewer_id: Mapped[str | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    review_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class WorkflowTemplateEvaluation(Base):
    __tablename__ = "workflow_template_evaluations"
    __table_args__ = (Index("ix_workflow_template_evaluations_template_created", "template_id", "created_at"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_str)
    template_id: Mapped[str] = mapped_column(ForeignKey("workflow_template_candidates.id", ondelete="CASCADE"), index=True)
    dataset_name: Mapped[str] = mapped_column(String(200))
    dataset_id: Mapped[str | None] = mapped_column(
        ForeignKey("evaluation_datasets.id", ondelete="RESTRICT"), nullable=True, index=True
    )
    dataset_fingerprint: Mapped[str | None] = mapped_column(String(64), nullable=True)
    evidence_mode: Mapped[str] = mapped_column(String(30), default="manual", index=True)
    evaluator: Mapped[str] = mapped_column(String(100), default="offline_gate_v1")
    case_count: Mapped[int] = mapped_column(Integer)
    passed: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    metrics: Mapped[dict] = mapped_column(JSON)
    reason_codes: Mapped[list | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class EvaluationCaseResult(Base):
    """Immutable per-case evidence for a dataset-verified offline evaluation."""

    __tablename__ = "evaluation_case_results"
    __table_args__ = (
        CheckConstraint(
            "(skill_evaluation_id IS NOT NULL AND template_evaluation_id IS NULL) OR "
            "(skill_evaluation_id IS NULL AND template_evaluation_id IS NOT NULL)",
            name="ck_evaluation_case_results_one_parent",
        ),
        UniqueConstraint(
            "skill_evaluation_id", "dataset_case_id", name="uq_evaluation_case_results_skill_case"
        ),
        UniqueConstraint(
            "template_evaluation_id", "dataset_case_id", name="uq_evaluation_case_results_template_case"
        ),
        Index("ix_evaluation_case_results_skill_created", "skill_evaluation_id", "created_at"),
        Index("ix_evaluation_case_results_template_created", "template_evaluation_id", "created_at"),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_str)
    skill_evaluation_id: Mapped[str | None] = mapped_column(
        ForeignKey("skill_evaluations.id", ondelete="CASCADE"), nullable=True, index=True
    )
    template_evaluation_id: Mapped[str | None] = mapped_column(
        ForeignKey("workflow_template_evaluations.id", ondelete="CASCADE"), nullable=True, index=True
    )
    dataset_case_id: Mapped[str] = mapped_column(
        ForeignKey("evaluation_dataset_cases.id", ondelete="RESTRICT"), index=True
    )
    case_key: Mapped[str] = mapped_column(String(64))
    baseline_score: Mapped[float] = mapped_column(Float)
    candidate_score: Mapped[float] = mapped_column(Float)
    baseline_passed: Mapped[bool] = mapped_column(Boolean)
    candidate_passed: Mapped[bool] = mapped_column(Boolean)
    safety_violations: Mapped[int] = mapped_column(Integer, default=0)
    route_regression: Mapped[bool] = mapped_column(Boolean, default=False)
    cost_delta: Mapped[float] = mapped_column(Float, default=0.0)
    latency_delta: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class EvaluationCampaign(Base):
    """Immutable multi-suite promotion evidence for exactly one candidate."""

    __tablename__ = "evaluation_campaigns"
    __table_args__ = (
        CheckConstraint(
            "(proposal_id IS NOT NULL AND template_id IS NULL) OR "
            "(proposal_id IS NULL AND template_id IS NOT NULL)",
            name="ck_evaluation_campaigns_one_target",
        ),
        UniqueConstraint("fingerprint", name="uq_evaluation_campaigns_fingerprint"),
        Index("ix_evaluation_campaigns_proposal_created", "proposal_id", "created_at"),
        Index("ix_evaluation_campaigns_template_created", "template_id", "created_at"),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_str)
    proposal_id: Mapped[str | None] = mapped_column(
        ForeignKey("skill_proposals.id", ondelete="CASCADE"), nullable=True, index=True
    )
    template_id: Mapped[str | None] = mapped_column(
        ForeignKey("workflow_template_candidates.id", ondelete="CASCADE"), nullable=True, index=True
    )
    policy_version: Mapped[str] = mapped_column(String(40), default="medical_promotion_v1")
    required_suites: Mapped[list] = mapped_column(JSON)
    evaluation_ids: Mapped[list] = mapped_column(JSON)
    dataset_fingerprints: Mapped[dict] = mapped_column(JSON)
    metrics: Mapped[dict] = mapped_column(JSON)
    passed: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    reason_codes: Mapped[list | None] = mapped_column(JSON, nullable=True)
    fingerprint: Mapped[str] = mapped_column(String(64), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class OfflineEvaluationRun(Base):
    """Durable baseline/candidate comparison for one skill proposal."""

    __tablename__ = "offline_evaluation_runs"
    __table_args__ = (
        UniqueConstraint("idempotency_key", name="uq_offline_evaluation_runs_idempotency_key"),
        Index("ix_offline_evaluation_runs_status_created", "status", "created_at"),
        Index("ix_offline_evaluation_runs_proposal_created", "proposal_id", "created_at"),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_str)
    idempotency_key: Mapped[str] = mapped_column(String(100), unique=True)
    proposal_id: Mapped[str] = mapped_column(
        ForeignKey("skill_proposals.id", ondelete="CASCADE"), index=True
    )
    dataset_id: Mapped[str] = mapped_column(
        ForeignKey("evaluation_datasets.id", ondelete="RESTRICT"), index=True
    )
    skill_evaluation_id: Mapped[str | None] = mapped_column(
        ForeignKey("skill_evaluations.id", ondelete="SET NULL"), nullable=True
    )
    status: Mapped[str] = mapped_column(String(20), default="queued", index=True)
    case_count: Mapped[int] = mapped_column(Integer, default=0)
    completed_case_count: Mapped[int] = mapped_column(Integer, default=0)
    attempt_count: Mapped[int] = mapped_column(Integer, default=0)
    runner_version: Mapped[str] = mapped_column(String(40), default="agent_graph_runner_v1")
    metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error_message: Mapped[str | None] = mapped_column(EncryptedText(), nullable=True)
    created_by: Mapped[str | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class WorkflowTemplateDecision(Base):
    __tablename__ = "workflow_template_decisions"
    __table_args__ = (Index("ix_workflow_template_decisions_template_created", "template_id", "created_at"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_str)
    template_id: Mapped[str] = mapped_column(ForeignKey("workflow_template_candidates.id", ondelete="CASCADE"), index=True)
    evaluation_id: Mapped[str | None] = mapped_column(ForeignKey("workflow_template_evaluations.id", ondelete="SET NULL"), nullable=True)
    campaign_id: Mapped[str | None] = mapped_column(ForeignKey("evaluation_campaigns.id", ondelete="SET NULL"), nullable=True)
    reviewer_id: Mapped[str | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    decision: Mapped[str] = mapped_column(String(20))
    previous_status: Mapped[str] = mapped_column(String(20))
    reason: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class ToolInterfaceDraft(Base):
    """Reviewable interface and test draft; insertion never installs a tool."""

    __tablename__ = "tool_interface_drafts"
    __table_args__ = (
        UniqueConstraint("fingerprint", name="uq_tool_interface_drafts_fingerprint"),
        Index("ix_tool_interface_drafts_status_created", "status", "created_at"),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_str)
    fingerprint: Mapped[str] = mapped_column(String(64), unique=True)
    name: Mapped[str] = mapped_column(String(100))
    purpose: Mapped[str] = mapped_column(EncryptedText())
    input_schema: Mapped[dict] = mapped_column(JSON)
    output_schema: Mapped[dict] = mapped_column(JSON)
    test_cases: Mapped[list] = mapped_column(JSON)
    source_gap_ids: Mapped[list] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(20), default="candidate", index=True)
    reviewer_id: Mapped[str | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    review_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class ToolDraftDecision(Base):
    __tablename__ = "tool_draft_decisions"
    __table_args__ = (Index("ix_tool_draft_decisions_draft_created", "draft_id", "created_at"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_str)
    draft_id: Mapped[str] = mapped_column(ForeignKey("tool_interface_drafts.id", ondelete="CASCADE"), index=True)
    reviewer_id: Mapped[str | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    decision: Mapped[str] = mapped_column(String(20))
    previous_status: Mapped[str] = mapped_column(String(20))
    reason: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
