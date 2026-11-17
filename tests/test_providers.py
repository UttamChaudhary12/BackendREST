import json
from unittest.mock import MagicMock

import httpx
import numpy as np
import pytest
from openai import AsyncOpenAI

from app.config import Settings
from app.providers import LLM


async def test_groq_structured_booking_uses_compatible_endpoint():
    llm = LLM(Settings(groq_api_key="groq-test-key", api_key="backend-test-key"))
    await llm.client.close()

    def respond(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://api.groq.com/openai/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer groq-test-key"
        payload = json.loads(request.content)
        assert payload["model"] == "openai/gpt-oss-20b"
        assert payload["response_format"]["json_schema"]["strict"] is True
        return httpx.Response(
            200,
            json={
                "id": "completion-test",
                "object": "chat.completion",
                "created": 0,
                "model": payload["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {
                                    "query": "interview booking",
                                    "booking_intent": True,
                                    "draft": {
                                        "name": "Alex",
                                        "email": None,
                                        "date": None,
                                        "time": None,
                                    },
                                    "confirmed": False,
                                }
                            ),
                        },
                    }
                ],
            },
        )

    llm.client = AsyncOpenAI(
        api_key="groq-test-key",
        base_url="https://api.groq.com/openai/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    )
    try:
        plan = await llm.plan([], "Book an interview, my name is Alex", {}, "2026-11-16")
        assert plan.booking_intent
        assert plan.draft.name == "Alex"
        assert plan.draft.email is None
    finally:
        await llm.client.close()


async def test_local_embeddings_reuse_model_without_remote_embeddings(monkeypatch):
    import fastembed

    model = MagicMock()
    model.embed.side_effect = lambda texts: [np.ones(384) for _ in texts]
    factory = MagicMock(return_value=model)
    monkeypatch.setattr(fastembed, "TextEmbedding", factory)
    llm = LLM(Settings(groq_api_key="test", api_key="test"))
    try:
        assert len((await llm.embed(["hello"]))[0]) == 384
        assert len(await llm.embed(["one", "two"])) == 2
        factory.assert_called_once_with(
            model_name="BAAI/bge-small-en-v1.5", cache_dir="./data/models"
        )
        model.embed.return_value = None
        model.embed.side_effect = lambda texts: [np.ones(3) for _ in texts]
        with pytest.raises(ValueError, match="dimensions"):
            await llm.embed(["wrong dimension"])
    finally:
        await llm.client.close()
