# Document ingestion and conversational RAG backend

Two REST endpoints built with FastAPI. No UI, LangChain, RetrievalQAChain, FAISS, or Chroma.

Chat and booking extraction use Groq (`openai/gpt-oss-20b`) through its OpenAI-compatible
API. The `openai` Python package is only the HTTP client; no OpenAI account or key is needed.
Embeddings run locally on CPU with FastEmbed (`BAAI/bge-small-en-v1.5`, 384 dimensions).
The first ingestion downloads the model into `data/models`; later requests reuse the cached
model, including after Docker restarts. Internet access is required for that initial download.
Very long chunks may be truncated to the embedding model's token limit; the default 1200
character chunk size is recommended for ordinary English text.

Get a Groq key from https://console.groq.com/keys and put it in `.env` as `GROQ_API_KEY`.
`API_KEY` is a separate secret you choose for authenticating requests to this backend.
Groq's free plan has rate limits: https://console.groq.com/docs/rate-limits.
If you previously ingested with OpenAI, re-upload the documents: the new default Qdrant
collection `documents_bge_small_v1` uses a different embedding model and vector size.

## Run with Docker

1. Copy `.env.example` to `.env` and set `GROQ_API_KEY` and a strong `API_KEY`.
2. Run `docker compose up --build -d`.
3. Open `http://localhost:8000/docs` for the generated API reference.

Redis and Qdrant are private to the Compose network. Docker volumes persist metadata,
vectors, and Redis memory. The API retries startup if Qdrant is still initializing.

For local Python development (Python 3.10+):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
# Configure .env and run Redis and Qdrant at its configured URLs.
uvicorn app.main:app --reload
```

## Document ingestion

`POST /api/v1/documents`, authenticated with `X-API-Key`, accepts multipart fields:

| Field | Default | Meaning |
|---|---|---|
| file | required | PDF or UTF-8 TXT, at most 10 MiB |
| strategy | paragraph | `fixed` or `paragraph` |
| chunk_size | 1200 | Maximum characters per chunk, 200–4000 |
| overlap | 200 | Character overlap, must be smaller than chunk size |

Fixed chunking uses overlapping character windows. Paragraph chunking packs paragraphs
up to the requested size, retaining overlap and splitting oversized paragraphs.
PDF text extraction uses pypdf; scanned documents need OCR before upload.

```bash
curl -X POST http://localhost:8000/api/v1/documents \
  -H 'X-API-Key: YOUR_API_KEY' \
  -F 'file=@handbook.pdf' -F 'strategy=paragraph' -F 'chunk_size=1200' -F 'overlap=200'
```

Returns `201` with `document_id`, filename, strategy and chunk count. Metadata includes
SHA-256, chunk parameters, creation time, and ingestion status in SQLite. Embeddings and
chunk text live in Qdrant. Only `ready` SQL document IDs can be retrieved. A failed upload
is marked `failed`; any partial vectors are inaccessible to retrieval. Uploads are synchronous
and not deduplicated; failed partial vectors may be removed by an operational cleanup job.

## Conversational RAG and booking

`POST /api/v1/chat`, authenticated with `X-API-Key`:

```bash
curl http://localhost:8000/api/v1/chat \
  -H 'Content-Type: application/json' -H 'X-API-Key: YOUR_API_KEY' \
  -d '{"session_id":"ce36981b-b387-4c71-b161-f741e60311a0","message":"What is the interview process?","document_ids":[]}'
```

Generate a random UUID for each conversation and reuse it for follow-ups. An empty
`document_ids` searches all ready documents; otherwise retrieval is restricted to that list.
The response contains the answer, numbered source excerpts, session ID, and optional booking
draft and persisted booking ID.

The custom pipeline loads Redis history → asks the LLM for a standalone query and typed
booking extraction → embeds the query → retrieves five Qdrant results → generates a grounded
answer with citations → saves the latest 20 messages in Redis. Memory expires after 24 hours
of inactivity. A distributed session lock rejects concurrent turns with `409`.

Example booking conversation using the same session ID:

1. “I want to book an interview. My name is Alex.”
2. “My email is alex@example.com.”
3. “January 20, 2027 at 10:00.”
4. Review the returned summary and reply “Yes, confirm.”

The LLM extracts the fields across turns, including relative dates. Application code validates
email, date, time and a future appointment in `BOOKING_TIMEZONE` (default Asia/Kathmandu).
Saving requires an unchanged complete draft and explicit confirmation of a previous summary.
SQL uniqueness prevents two bookings in the same slot and repeated bookings for one session.
Bookings are stored with UTC start times and their original timezone. Confirmation returns a
booking reference; email/calendar delivery and rescheduling are outside this assignment.

## Validation

```bash
python -m pytest -q
python -m ruff check app tests
```

Tests use a real in-memory Qdrant collection and temporary SQLite database, with mocked LLM
and Redis boundaries. They cover chunking, ingestion, failure visibility, follow-up retrieval,
confirmation, booking idempotency and request validation. They need no API credentials.
Live Groq/Redis/Docker verification requires configured services and credentials.

## Design and operational notes

- `config.py` / `schemas.py`: environment configuration and typed contracts.
- `documents.py`: text extraction and two chunking strategies.
- `database.py`: SQL models and async sessions.
- `providers.py`: Groq chat, local FastEmbed, Qdrant and Redis adapters.
- `services.py`: ingestion and custom RAG/booking orchestration.
- `main.py`: exactly two business endpoints, authentication and lifecycle management.

This is a shared-corpus assignment service: every API key holder can search the corpus.
Session UUIDs are bearer secrets; protect them. Before a public multi-user deployment, add
user identity and ownership filters, rate limiting, TLS, migrations, observability and retention
policies. Set reverse-proxy request-body limits as well as the application's file-size check.
The SQL schema is initialized at startup for this assignment; use migrations for evolving it.
Do not change embedding dimensions for an existing collection: create a new collection and
re-ingest documents. Retrieved text is marked untrusted in the answer prompt. Model extraction
is probabilistic, so booking confirmation is also enforced in application code.
