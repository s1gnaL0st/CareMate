# Deployment

## Local Docker stack

From the repository root:

```powershell
docker compose up --build
```

The API is available directly at `http://localhost:8000` and through the Nginx gateway at `http://localhost:8080`. The Vite development server remains available at `http://localhost:5173` and proxies `/api` to port 8000.

The backend runs Alembic migrations before starting. MySQL, Redis, MinIO, the OCR worker, and Nginx have health checks or dependency gates. Persistent development data is under `infra/data/`; do not delete it unless you intend to reset the local stack.

## Production checklist

1. Copy `backend/.env.example` to `backend/.env` and set a long random `JWT_SECRET`, MinIO credentials, encryption key, LLM credentials, database URL, and `CORS_ORIGINS` to the deployed frontend origin(s).
2. Set `ENVIRONMENT=prod`, `AUTH_REQUIRED=true`, and choose `API_WORKERS` after load testing. Multiple API workers require shared MySQL and Redis; do not use process-local state for coordination.
3. Put TLS termination in front of the Nginx service. Forward `X-Forwarded-Proto` and restrict inbound port 8000 so clients use the gateway.
4. Keep `proxy_buffering off` and the 300 second read timeout for `/api/` because chat and vision endpoints stream SSE responses.
5. Back up MySQL with `scripts/db-backup.ps1` (and restore with `scripts/db-restore.ps1`) and back up the MinIO bucket according to the retention policy. Test restores periodically.
6. Docker JSON logs are bounded with `max-size`/`max-file` settings in `docker-compose.yml`. Ship them to a central collector for long-term retention; do not log medical content or uploaded image bytes.

## CI/CD

`.github/workflows/ci.yml` runs backend unit tests and frontend lint/build on every push and pull request. A deployment pipeline should build and scan the backend/worker images, apply migrations as a one-shot release step, then roll the API workers and worker separately behind the existing health checks.

## Operational endpoints

- `GET /api/v1/health/live`: process liveness
- `GET /api/v1/health/ready`: MySQL, Redis, and vector-store readiness
- `GET /metrics`: Prometheus exposition format
