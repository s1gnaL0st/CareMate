"""Authenticated wellness directory and habit goal endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from auth import get_current_user
from db import get_db
from evolution import persist_feedback_failure
from models import AnswerFeedback, Conversation, ExperienceRecord, HabitGoal, Hospital, Message, User
from schemas import (
    FeedbackCreate, FeedbackResponse, HabitGoalCreate, HabitGoalResponse, HabitGoalUpdate,
    HospitalResponse,
)

router = APIRouter(prefix="/api/v1", tags=["wellness"])


@router.get("/hospitals", response_model=list[HospitalResponse])
async def list_hospitals(
    city: str = "",
    q: str = "",
    limit: int = 20,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
) -> list[HospitalResponse]:
    limit, offset = min(max(limit, 1), 100), max(offset, 0)
    query = select(Hospital).order_by(Hospital.distance_km.is_(None), Hospital.distance_km, Hospital.name)
    if city.strip():
        query = query.where(Hospital.city == city.strip())
    if q.strip():
        query = query.where(Hospital.name.contains(q.strip()))
    return list(await db.scalars(query.offset(offset).limit(limit)))


@router.get("/habit-goals", response_model=list[HabitGoalResponse])
async def list_habit_goals(user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    result = await db.scalars(select(HabitGoal).where(HabitGoal.user_id == user.id, HabitGoal.status != "archived").order_by(HabitGoal.created_at))
    return list(result)


@router.post("/habit-goals", response_model=HabitGoalResponse, status_code=201)
async def create_habit_goal(payload: HabitGoalCreate, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    goal = HabitGoal(user_id=user.id, **payload.model_dump())
    db.add(goal)
    await db.commit()
    await db.refresh(goal)
    return goal


@router.patch("/habit-goals/{goal_id}", response_model=HabitGoalResponse)
async def update_habit_goal(goal_id: str, payload: HabitGoalUpdate, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    goal = await db.scalar(select(HabitGoal).where(HabitGoal.id == goal_id, HabitGoal.user_id == user.id))
    if goal is None:
        raise HTTPException(status_code=404, detail="健康目标不存在")
    for key, value in payload.model_dump(exclude_unset=True).items():
        setattr(goal, key, value)
    if goal.current_value >= goal.target_value and goal.status == "active":
        goal.status = "completed"
    await db.commit()
    await db.refresh(goal)
    return goal


@router.delete("/habit-goals/{goal_id}", status_code=204)
async def archive_habit_goal(goal_id: str, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    goal = await db.scalar(select(HabitGoal).where(HabitGoal.id == goal_id, HabitGoal.user_id == user.id))
    if goal is None:
        raise HTTPException(status_code=404, detail="健康目标不存在")
    goal.status = "archived"
    await db.commit()


@router.post("/feedback", response_model=FeedbackResponse, status_code=201)
async def create_feedback(payload: FeedbackCreate, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    message = None
    if payload.message_id:
        message = await db.scalar(select(Message).join(Conversation, Message.conversation_id == Conversation.id).where(Message.id == payload.message_id, Conversation.user_id == user.id))
        if message is None:
            raise HTTPException(status_code=404, detail="消息不存在")
    if payload.conversation_id:
        conversation = await db.scalar(select(Conversation).where(Conversation.id == payload.conversation_id, Conversation.user_id == user.id))
        if conversation is None:
            raise HTTPException(status_code=404, detail="会话不存在")
    feedback = AnswerFeedback(user_id=user.id, **payload.model_dump())
    db.add(feedback)
    await db.flush()
    if message is not None and message.role == "assistant":
        run_id = str((message.event_metadata or {}).get("request_id") or "").strip()
        experience = None
        if run_id:
            experience = await db.scalar(
                select(ExperienceRecord).where(ExperienceRecord.run_id == run_id)
            )
        await persist_feedback_failure(
            db,
            feedback_id=feedback.id,
            experience=experience,
            rating=feedback.rating,
            category=feedback.category,
            comment=feedback.comment,
        )
    await db.commit()
    await db.refresh(feedback)
    return feedback
