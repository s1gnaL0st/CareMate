## Backend development

1. Copy `.env.example` to `.env` and set the LLM provider credentials.
2. From the repository root run `docker compose up -d mysql redis minio`.
3. From `backend/` run `uv sync` and `uv run alembic upgrade head`.
4. Start the API with `uv run uvicorn main:app --reload --port 8000`.

The versioned API is under `/api/v1`. Compose maps MySQL, Redis and MinIO to
host ports `3307`, `6380`, `9002` and `9003` by default to avoid common local
port conflicts. Override `MYSQL_HOST_PORT`, `REDIS_HOST_PORT`,
`MINIO_API_HOST_PORT` or `MINIO_CONSOLE_HOST_PORT` when needed.

Business data is stored in MySQL; Redis is used for token revocation and chat
idempotency; Chroma remains the RAG vector store. Compose data is bind-mounted
under `infra/data/` so it is visible on the project drive.

Seed data is opt-in and never runs on startup. Set `SEED_DEMO=true` and a local
`SEED_PASSWORD`, then run `uv run python seed.py`.

Useful checks:

```powershell
uv run alembic current
Invoke-RestMethod http://localhost:8000/api/v1/health/live
Invoke-RestMethod http://localhost:8000/api/v1/health/ready
```
