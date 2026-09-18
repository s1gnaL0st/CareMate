# P1 Runtime Guide

P1 separates request handling from long-running report analysis:

```text
browser -> FastAPI -> MySQL + Redis + MinIO
                   -> Arq worker -> vision / MinerU / PaddleOCR
```

## Local Services

Start the API dependencies and Worker with Docker Compose:

```powershell
docker compose up --build
```

The API container applies Alembic migrations before it starts. The Worker uses
Python 3.12 independently from the Python 3.13 API container. It processes up
to four jobs concurrently, retries report failures three times with exponential
backoff, and records `retrying` or `failed` in MySQL.

Use `GET /api/v1/reports/{report_id}` to poll report status. `DELETE` on the
same resource cancels a queued job by removing its durable record and source
object; a subsequently delivered Worker job becomes a no-op.

## OCR Service Contract

The default `OCR_PROVIDER=vision` works for JPEG, PNG, and WebP through the
configured vision LLM. It is suitable for development and for the existing
drug-box and trace-code flows.

For PDF reports configure MinerU:

```text
OCR_PROVIDER=mineru
MINERU_ENDPOINT=http://mineru:port
```

For image-first medical report OCR configure PaddleOCR/PP-Structure:

```text
OCR_PROVIDER=paddleocr
PADDLEOCR_ENDPOINT=http://paddleocr:port
```

Both services must implement `POST /parse` with multipart field `file`. Their
JSON response must contain `text` or `markdown`, and may contain page records:

```json
{
  "markdown": "# Report\n...",
  "pages": [{
    "page": 1,
    "blocks": [{
      "type": "table",
      "text": "...",
      "bbox": [0, 0, 100, 40],
      "confidence": 0.98
    }]
  }]
}
```

The Worker stores the original upload and a JSON analysis result in MinIO. MySQL
stores only object keys, content type, byte size, SHA-256, task status, and the
structured result needed for status retrieval. The scheduled Worker cleanup
removes expired reports according to `REPORT_RETENTION_DAYS`.

## Privacy and Operations

- Generate and set `DATA_ENCRYPTION_KEY` before writing non-empty medical
  history in production. The application uses Fernet encryption for this field.
- Only publish API/frontend ports. Do not expose MySQL, Redis, or MinIO ports
  on an Internet-facing production host.
- Redis holds cache keys, rate-limit counters, distributed locks, idempotency
  keys, and Arq jobs only. It is not a conversation database.
- Set LLM input/output price values if cost estimates are required in
  `chat_runs`; leave them at zero when the provider has no reliable price.
