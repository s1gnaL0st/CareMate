"""Application configuration loaded from environment variables."""
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Smart Health Assistant"
    environment: str = "dev"
    api_v1_prefix: str = "/api/v1"
    cors_origins: str = "http://localhost:5173,http://localhost:5174"
    database_url: str = "mysql+asyncmy://health:health@localhost:3306/health_assistant"
    redis_url: str = "redis://localhost:6379/0"
    jwt_secret: str = Field(default="change-me-in-production-please-use-env-0123456789", validation_alias="JWT_SECRET")
    jwt_algorithm: str = "HS256"
    access_token_minutes: int = 60 * 24
    auth_required: bool = True
    max_chat_messages: int = 50
    llm_timeout_seconds: int = 120
    llm_input_cost_per_million_usd: float = 0.0
    llm_output_cost_per_million_usd: float = 0.0
    # Optional per-node clinic model. An empty base URL keeps the existing
    # global model path, so the Qwen service can be enabled independently.
    clinic_llm_enabled: bool = True
    clinic_llm_api_key: str = "dummy"
    clinic_llm_model: str = "grpo-200"
    clinic_llm_base_url: str = ""
    clinic_llm_timeout_seconds: int = 120
    clinic_llm_max_rounds: int = Field(default=8, ge=1, le=16)
    stream_heartbeat_seconds: int = 15
    max_output_chars: int = 8000
    # Shared MySQL/Redis state is required before increasing this value.
    api_workers: int = Field(default=1, ge=1, le=32)
    max_input_chars: int = 16000
    max_image_pixels: int = 20_000_000
    max_report_file_bytes: int = 20 * 1024 * 1024
    ocr_provider: str = "vision"
    mineru_endpoint: str = ""
    paddleocr_endpoint: str = ""
    ocr_timeout_seconds: int = 180
    cache_key_prefix: str = "smart-health"
    cache_version: str = "1"
    cache_default_ttl_seconds: int = 300
    cache_rag_ttl_seconds: int = 900
    rate_limit_requests: int = 30
    rate_limit_window_seconds: int = 60
    lock_timeout_seconds: int = 180
    lock_blocking_timeout_seconds: int = 2
    agent_task_lease_seconds: int = 300
    # Domain-agent execution retries.  A task is attempted once plus this
    # bounded retry count; exhausted tasks remain durably failed (dead-letter)
    # instead of looping forever during replans.
    agent_task_max_retries: int = 2
    agent_task_retry_backoff_seconds: float = 1.0
    tool_timeout_seconds: int = 30
    tool_max_retries: int = 1
    object_storage_endpoint: str = "localhost:9002"
    object_storage_access_key: str = "minio"
    object_storage_secret_key: str = "minio-health-dev"
    object_storage_bucket: str = "health-reports"
    object_storage_secure: bool = False
    report_retention_days: int = 7
    data_encryption_key: str = ""
    # MCP is enabled by default for read-only knowledge tools. If unavailable,
    # domain tools transparently fall back to their local implementations.
    mcp_enabled: bool = True
    mcp_timeout_seconds: float = 30.0
    mcp_openfda_url: str = "https://openfda.caseyjhand.com/mcp"
    mcp_pubmed_url: str = "https://pubmed.caseyhand.com/mcp"
    # Comma-separated reviewers. In non-production environments any signed-in
    # user may exercise the review workflow; production is deny-by-default.
    evolution_reviewer_emails: str = ""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False)

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def evolution_reviewer_email_list(self) -> list[str]:
        return [email.strip().lower() for email in self.evolution_reviewer_emails.split(",") if email.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
