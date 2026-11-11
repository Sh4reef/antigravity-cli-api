"""Unit and integration tests for multi-provider routing in the FastAPI gateway."""

import pytest
from httpx import ASGITransport, AsyncClient

from acpbox.config import VirtualModelConfig
from acpbox.models import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionResponse,
    ChatMessageOutput,
    DeltaMessage,
    ModelItem,
    UsageInfo,
)
from acpbox.providers.base import LLMProvider, ProviderHealthStatus
from acpbox.router import ModelRouter
from acpbox.server import create_app


class MockGenericProvider:
    def __init__(self, provider_id: str, provider_type: str, models: list):
        self.provider_id = provider_id
        self.provider_type = provider_type
        self.available_models = models

    async def start(self):
        pass

    async def stop(self):
        pass

    async def get_models(self):
        return [ModelItem(id=m, owned_by=f"{self.provider_type}:{self.provider_id}") for m in self.available_models]

    async def complete_chat(self, request, upstream_model=None):
        return ChatCompletionResponse(
            id=f"chatcmpl-{self.provider_id}",
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessageOutput(
                        role="assistant",
                        content=f"Response from {self.provider_id} using {upstream_model}",
                    ),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(prompt_tokens=5, completion_tokens=5, total_tokens=10),
        )

    async def stream_chat(self, request, upstream_model=None):
        yield ChatCompletionChunk(
            id=f"chatcmpl-{self.provider_id}",
            created=100,
            model=request.model,
            choices=[ChatCompletionChunkChoice(index=0, delta=DeltaMessage(role="assistant"))],
        )
        yield ChatCompletionChunk(
            id=f"chatcmpl-{self.provider_id}",
            created=100,
            model=request.model,
            choices=[ChatCompletionChunkChoice(index=0, delta=DeltaMessage(content=f"Stream from {self.provider_id}"))],
        )
        yield ChatCompletionChunk(
            id=f"chatcmpl-{self.provider_id}",
            created=100,
            model=request.model,
            choices=[ChatCompletionChunkChoice(index=0, delta=DeltaMessage(), finish_reason="stop")],
        )

    async def health(self):
        return ProviderHealthStatus(
            provider_id=self.provider_id,
            provider_type=self.provider_type,
            healthy=True,
            status="ok",
        )


@pytest.fixture
def multi_provider_router():
    p_agy = MockGenericProvider("agy", "acp", ["gemini-3.8-flash-high"])
    p_opencode = MockGenericProvider("opencode", "acp", ["claude-3-5-sonnet"])
    p_wus = MockGenericProvider("wusrouter", "openai_http", ["wusrouter/qwen3.8-flash:free"])
    p_logfare = MockGenericProvider("logfare", "openai_http", ["glm-5.3-flash"])

    return ModelRouter(
        providers={
            "agy": p_agy,
            "opencode": p_opencode,
            "wusrouter": p_wus,
            "logfare": p_logfare,
        },
        default_provider_id="agy",
        virtual_models={
            "fast-free": VirtualModelConfig(
                provider="wusrouter",
                upstream_model="wusrouter/qwen3.8-flash:free",
            ),
            "reasoning-flash": VirtualModelConfig(
                provider="logfare",
                upstream_model="glm-5.3-flash",
            ),
        },
    )


@pytest.mark.asyncio
async def test_gateway_get_models(multi_provider_router):
    """Verify GET /v1/models aggregates all models from all providers and aliases."""
    app = create_app(router=multi_provider_router)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res = await ac.get("/v1/models")
        assert res.status_code == 200
        data = res.json()
        ids = [m["id"] for m in data["data"]]
        assert "fast-free" in ids
        assert "reasoning-flash" in ids
        assert "wusrouter/wusrouter/qwen3.8-flash:free" in ids or "wusrouter/qwen3.8-flash:free" in ids
        assert "logfare/glm-5.3-flash" in ids or "glm-5.3-flash" in ids
        assert "agy/gemini-3.8-flash-high" in ids or "gemini-3.8-flash-high" in ids
        assert "opencode/claude-3-5-sonnet" in ids or "claude-3-5-sonnet" in ids


@pytest.mark.asyncio
async def test_gateway_health(multi_provider_router):
    """Verify GET /health returns provider breakdown."""
    app = create_app(router=multi_provider_router)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res = await ac.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "healthy"
        assert "providers" in data
        assert "agy" in data["providers"]
        assert "opencode" in data["providers"]
        assert "wusrouter" in data["providers"]
        assert "logfare" in data["providers"]


@pytest.mark.asyncio
async def test_gateway_routing_to_all_providers(multi_provider_router):
    """Verify POST /v1/chat/completions dispatches correctly to each provider."""
    app = create_app(router=multi_provider_router)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # 1. Virtual model -> wusrouter
        r1 = await ac.post("/v1/chat/completions", json={"model": "fast-free", "messages": [{"role": "user", "content": "hi"}]})
        assert r1.status_code == 200
        assert "wusrouter" in r1.json()["choices"][0]["message"]["content"]

        # 2. Virtual model -> logfare
        r2 = await ac.post("/v1/chat/completions", json={"model": "reasoning-flash", "messages": [{"role": "user", "content": "hi"}]})
        assert r2.status_code == 200
        assert "logfare" in r2.json()["choices"][0]["message"]["content"]

        # 3. Namespaced model -> opencode
        r3 = await ac.post("/v1/chat/completions", json={"model": "opencode/claude-3-5-sonnet", "messages": [{"role": "user", "content": "hi"}]})
        assert r3.status_code == 200
        assert "opencode" in r3.json()["choices"][0]["message"]["content"]

        # 4. Default / unmapped model -> agy
        r4 = await ac.post("/v1/chat/completions", json={"model": "gemini-3.8-flash-high", "messages": [{"role": "user", "content": "hi"}]})
        assert r4.status_code == 200
        assert "agy" in r4.json()["choices"][0]["message"]["content"]


@pytest.mark.asyncio
async def test_gateway_streaming_sse(multi_provider_router):
    """Verify SSE streaming delivers chunks with data: prefix and [DONE]."""
    app = create_app(router=multi_provider_router)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res = await ac.post(
            "/v1/chat/completions",
            json={"model": "fast-free", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        )
        assert res.status_code == 200
        assert "text/event-stream" in res.headers["content-type"]
        text = res.text
        assert "data: " in text
        assert "Stream from wusrouter" in text
        assert "data: [DONE]" in text
