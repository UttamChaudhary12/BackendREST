from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from qdrant_client import AsyncQdrantClient
from sqlalchemy import select

from app.config import Settings
from app.database import Booking, Document
from app.documents import chunk_text, extract_text
from app.main import app, get_services
from app.schemas import BookingDraft, ChatRequest, ChunkStrategy, TurnPlan
from app.services import Services


@pytest.mark.parametrize("strategy", list(ChunkStrategy))
def test_chunks_preserve_content_and_bound_size(strategy):
    text = "alpha " * 100 + "\n\n" + "beta " * 100
    chunks = chunk_text(text, strategy, 200, 30)
    assert len(chunks) > 1
    assert all(0 < len(chunk) <= 200 for chunk in chunks)
    assert "alpha" in chunks[0] and "beta" in chunks[-1]


def test_text_encoding_and_overlap_validation():
    assert extract_text(b"\xef\xbb\xbfhello", ".txt") == "hello"
    with pytest.raises(ValueError):
        chunk_text("abc", ChunkStrategy.fixed, 10, 10)


@pytest.fixture
async def services(tmp_path):
    service = Services(
        Settings(
            groq_api_key="test",
            api_key="secret",
            database_url=f"sqlite+aiosqlite:///{tmp_path / 'test.db'}",
            embedding_dimensions=3,
        )
    )
    await service.vectors.client.close()
    service.vectors.client = AsyncQdrantClient(location=":memory:")
    await service.db.initialize()
    await service.vectors.initialize()
    service.llm.embed = AsyncMock(side_effect=_vectors)
    service.memory.read = AsyncMock(
        return_value={"history": [], "draft": {}, "awaiting_confirmation": False}
    )
    service.memory.save = AsyncMock()
    yield service
    await service.vectors.client.close()
    await service.db.engine.dispose()
    await service.llm.client.close()
    await service.memory.client.aclose()


async def _vectors(texts):
    return [[1.0, 0.0, 0.0] for _ in texts]


async def test_ingestion_and_real_qdrant_retrieval(services):
    result = await services.ingest(
        b"Company interviews last forty minutes.", "guide.txt", ChunkStrategy.fixed, 200, 20
    )
    hits = await services.vectors.search([1.0, 0.0, 0.0], [str(result.document_id)])
    assert hits[0].filename == "guide.txt"
    async with services.db.sessions() as db:
        assert (await db.get(Document, str(result.document_id))).status == "ready"


async def test_failed_ingestion_is_not_ready(services):
    services.llm.embed = AsyncMock(side_effect=RuntimeError("provider unavailable"))
    with pytest.raises(RuntimeError):
        await services.ingest(b"Some content", "guide.txt", ChunkStrategy.fixed, 200, 20)
    async with services.db.sessions() as db:
        assert (await db.scalar(select(Document))).status == "failed"


async def test_booking_requires_prior_summary_then_persists_once(services):
    draft = BookingDraft(name="Alex", email="alex@example.com", date="2099-01-20", time="10:00")
    state = {"history": [], "draft": {}, "awaiting_confirmation": False}

    async def read(_):
        return state

    services.memory.read = AsyncMock(side_effect=read)

    async def save(_, updated):
        state.update(updated)

    services.memory.save = save
    services.llm.plan = AsyncMock(
        return_value=TurnPlan(query="", booking_intent=True, draft=draft, confirmed=True)
    )
    request = ChatRequest(session_id=uuid4(), message="Book the interview")
    first = await services._chat(request)
    assert first.booking_id is None
    assert "Please confirm" in first.answer
    request.message = "Yes, confirm"
    second = await services._chat(request)
    assert second.booking_id is not None
    third = await services._chat(request)
    fourth = await services._chat(request)
    assert third.booking_id is None  # Show summary again before another confirmation.
    assert fourth.booking_id == second.booking_id
    async with services.db.sessions() as db:
        assert len(list((await db.scalars(select(Booking))).all())) == 1


async def test_followup_uses_history_and_rewritten_query(services):
    document = await services.ingest(
        b"Interview duration is 40 minutes", "guide.txt", ChunkStrategy.paragraph, 200, 20
    )
    history = [{"role": "user", "content": "Tell me about interviews"}]
    services.memory.read = AsyncMock(return_value={"history": history, "draft": {}})
    services.llm.plan = AsyncMock(
        return_value=TurnPlan(
            query="How long are interviews?",
            booking_intent=False,
            draft=BookingDraft(),
            confirmed=False,
        )
    )
    services.llm.answer = AsyncMock(return_value="40 minutes [1]")
    response = await services._chat(
        ChatRequest(session_id=uuid4(), message="How long?", document_ids=[document.document_id])
    )
    assert response.sources
    services.llm.embed.assert_awaited_with(["How long are interviews?"])
    assert services.llm.plan.call_args.args[0][0]["content"] == "Tell me about interviews"


def test_http_upload_and_message_validation():
    fake = AsyncMock()
    fake.settings.max_upload_bytes = 10
    app.dependency_overrides[get_services] = lambda: fake
    try:
        client = TestClient(app, raise_server_exceptions=False)
        assert (
            client.post("/api/v1/documents", files={"file": ("a.txt", b"x" * 11)}).status_code
            == 413
        )
        assert (
            client.post(
                "/api/v1/chat", json={"session_id": str(uuid4()), "message": "   "}
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/api/v1/chat", json={"session_id": "invalid", "message": "hello"}
            ).status_code
            == 422
        )
    finally:
        app.dependency_overrides.clear()
