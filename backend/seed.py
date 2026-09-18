"""Optional development seed data. Never runs as part of application startup."""
import asyncio
import os

from sqlalchemy import select

from auth import hash_password
from db import SessionLocal
from models import User, UserProfile


async def seed() -> None:
    if os.getenv("SEED_DEMO", "false").lower() != "true":
        print("SEED_DEMO is not enabled; no data written")
        return
    email = os.getenv("SEED_EMAIL", "demo@example.com").lower()
    password = os.getenv("SEED_PASSWORD")
    if not password:
        raise SystemExit("SEED_PASSWORD is required when SEED_DEMO=true")
    async with SessionLocal() as db:
        user = await db.scalar(select(User).where(User.email == email))
        if user is None:
            user = User(email=email, password_hash=hash_password(password), name=os.getenv("SEED_NAME", "Demo User"))
            db.add(user)
            await db.flush()
            db.add(UserProfile(user_id=user.id))
            await db.commit()
            print(f"created seed user {email}")
        else:
            print(f"seed user already exists {email}")


if __name__ == "__main__":
    asyncio.run(seed())
