"""Read-only runtime view of promoted skill versions."""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models import SkillDeployment, SkillProposal


async def active_skill(db: AsyncSession, base_skill: str) -> dict | None:
    deployment = await db.scalar(select(SkillDeployment).where(
        SkillDeployment.base_skill == base_skill,
        SkillDeployment.status == "active",
    ))
    if deployment is None:
        return None
    proposal = await db.get(SkillProposal, deployment.proposal_id)
    if proposal is None or proposal.status != "approved":
        return None
    return {
        "base_skill": proposal.base_skill,
        "version": proposal.version,
        "trigger": proposal.trigger,
        "content": proposal.content,
        "proposal_id": proposal.id,
        "deployment_id": deployment.id,
    }


async def active_skills(db: AsyncSession, limit: int = 32) -> list[dict]:
    rows = list(await db.scalars(select(SkillDeployment).where(
        SkillDeployment.status == "active"
    ).limit(limit)))
    result = []
    for row in rows:
        item = await active_skill(db, row.base_skill)
        if item:
            result.append(item)
    return result
