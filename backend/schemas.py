"""Pydantic request and response contracts for the v1 API."""
from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    name: str = Field(default="用户", min_length=1, max_length=100)


class LoginRequest(BaseModel):
    # Accept the development alias ``admin`` as well as a normal email.
    email: str = Field(min_length=1, max_length=255)
    password: str = Field(min_length=1, max_length=128)


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    # The development seed uses the reserved ``.local`` domain. Login
    # responses must still expose it; registration keeps EmailStr validation
    # via UserCreate above.
    email: str
    name: str
    is_active: bool


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse


class ConversationCreate(BaseModel):
    title: str = Field(default="新对话", max_length=200)


class ConversationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    title: str
    active_agent: str
    status: str
    created_at: datetime
    updated_at: datetime


class MessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    role: str
    content: str
    sequence: int
    created_at: datetime


class ReportUploadResponse(BaseModel):
    report_id: str
    analysis_id: str
    status: str
    duplicate: bool = False


class ReportStatusResponse(BaseModel):
    report_id: str
    status: str
    analysis_status: str | None = None
    analysis: dict | None = None
    error: str | None = None
    attempt_count: int = 0


class ReportListItem(BaseModel):
    report_id: str
    original_filename: str
    content_type: str
    size_bytes: int
    status: str
    analysis_status: str | None = None
    created_at: datetime


class HospitalResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    name: str
    city: str
    address: str
    tags: list[str]
    distance_km: float | None = None
    phone: str
    is_ad: bool


class HabitGoalCreate(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    target_value: int = Field(default=1, ge=1, le=100000)
    unit: str = Field(default="次", max_length=30)
    color: str = Field(default="bg-teal-500", max_length=30)


class HabitGoalUpdate(BaseModel):
    current_value: int | None = Field(default=None, ge=0, le=100000)
    status: str | None = Field(default=None, pattern="^(active|completed|archived)$")


class HabitGoalResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    title: str
    target_value: int
    current_value: int
    unit: str
    color: str
    status: str
    created_at: datetime
    updated_at: datetime


class FeedbackCreate(BaseModel):
    message_id: str | None = None
    conversation_id: str | None = None
    rating: int = Field(ge=-1, le=1)
    category: str | None = Field(default=None, max_length=50)
    comment: str | None = Field(default=None, max_length=1000)


class FeedbackResponse(BaseModel):
    id: str
    rating: int
    category: str | None = None
    comment: str | None = None
    created_at: datetime
