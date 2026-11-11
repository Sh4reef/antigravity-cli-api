"""Unit tests for OpenAIHttpProvider using respx to mock upstream HTTP."""

import json
import httpx
import pytest
import respx
from acpbox.config import OpenAIProviderConfig
from acpbox.models import ChatCompletionRequest, ChatMessageInput
from acpbox.providers.base import LLMProvider
from acpbox.providers.openai_http import OpenAIHttpProvider


def test_http_provider_protocol_conformance():
    """Verify OpenAIHttpProvider satisfies LLMProvider protocol."""
    cfg = OpenAIProviderConfig(
        id="wusrouter",
        base_url="https://wusrouter.com/v1",
        api_key="sk-test",
    )
    provider = OpenAIHttpProvider(cfg)
    assert isinstance(provider, LLMProvider)
    assert provider.provider_id == "wusrouter"
    assert provider.provider_type == "openai_http"


@pytest.mark.asyncio
@respx.mock
async def test_http_provider_model_discovery():
    """Verify model discovery queries /models and caches results."""
    cfg = OpenAIProviderConfig(
        id="logfare",
        base_url="https://logfare.ai/v1",
        api_key="lfu-test",
        discover_models=True,
    )
    provider = OpenAIHttpProvider(cfg)

    respx.get("https://logfare.ai/v1/models").mock(
        return_value=httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {"id": "glm-5.3-flash", "owned_by": "zhipu"},
                    {"id": "deepseek-v4.1-flash", "owned_by": "deepseek"},
                ],
            },
        )
    )

    models = await provider.get_models()
    assert len(models) == 2
    assert models[0].id == "glm-5.3-flash"
    assert models[0].owned_by == "openai:logfare"
    assert models[1].id == "deepseek-v4.1-flash"
    assert models[1].owned_by == "openai:logfare"


@pytest.mark.asyncio
@respx.mock
async def test_http_provider_fixed_models_bypasses_discovery():
    """Verify get_models returns only fixed models without querying upstream /models."""
    cfg = OpenAIProviderConfig(
        id="wusrouter",
        base_url="https://wusrouter.com/v1",
        api_key="sk-test",
        models=["wusrouter/qwen3.8-flash:free"],
    )
    provider = OpenAIHttpProvider(cfg)
    assert provider.has_fixed_models is True

    # Note: no respx mock route defined for /models. If get_models attempts to query /models,
    # respx will raise an unmocked request error.
    models = await provider.get_models()
    assert len(models) == 1
    assert models[0].id == "wusrouter/qwen3.8-flash:free"
    assert models[0].owned_by == "openai:wusrouter"


@pytest.mark.asyncio
@respx.mock
async def test_http_provider_complete_chat():
    """Verify non-streaming chat completions are correctly deserialized."""
    cfg = OpenAIProviderConfig(
        id="wusrouter",
        base_url="https://wusrouter.com/v1",
        api_key="sk-test",
    )
    provider = OpenAIHttpProvider(cfg)

    respx.post("https://wusrouter.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chatcmpl-test-123",
                "object": "chat.completion",
                "created": 1794365000,
                "model": "wusrouter/qwen3.8-flash:free",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello from upstream!"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 4,
                    "total_tokens": 14,
                },
            },
        )
    )

    req = ChatCompletionRequest(
        model="wusrouter/qwen3.8-flash:free",
        messages=[ChatMessageInput(role="user", content="hello")],
        stream=False,
    )

    res = await provider.complete_chat(req)
    assert res.id == "chatcmpl-test-123"
    assert res.choices[0].message.content == "Hello from upstream!"
    assert res.usage.total_tokens == 14


@pytest.mark.asyncio
@respx.mock
async def test_http_provider_stream_chat_sse():
    """Verify streaming chat completions parses and normalizes SSE lines."""
    cfg = OpenAIProviderConfig(
        id="logfare",
        base_url="https://logfare.ai/v1",
        api_key="lfu-test",
    )
    provider = OpenAIHttpProvider(cfg)

    sse_content = (
        'data: {"id":"c1","created":100,"choices":[{"index":0,"delta":{"role":"assistant"}}]}\n\n'
        'data: {"id":"c1","created":100,"choices":[{"index":0,"delta":{"reasoning":"I should think"}}]}\n\n'
        'data: {"id":"c1","created":100,"choices":[{"index":0,"delta":{"content":"Hi "}}]}\n\n'
        'data: {"id":"c1","created":100,"choices":[{"index":0,"delta":{"content":"there"}}]}\n\n'
        'data: {"id":"c1","created":100,"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
        "data: [DONE]\n\n"
    )

    respx.post("https://logfare.ai/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            content=sse_content.encode("utf-8"),
        )
    )

    req = ChatCompletionRequest(
        model="glm-5.3-flash",
        messages=[ChatMessageInput(role="user", content="hello")],
        stream=True,
    )

    chunks = []
    async for chunk in provider.stream_chat(req):
        chunks.append(chunk)

    assert len(chunks) == 5
    assert chunks[0].choices[0].delta.role == "assistant"
    # Verify 'reasoning' was normalized to reasoning_content
    assert chunks[1].choices[0].delta.reasoning_content == "I should think"
    assert chunks[2].choices[0].delta.content == "Hi "
    assert chunks[3].choices[0].delta.content == "there"
    assert chunks[4].choices[0].finish_reason == "stop"


@pytest.mark.asyncio
@respx.mock
async def test_http_provider_health():
    """Verify health returns healthy when /models responds."""
    cfg = OpenAIProviderConfig(
        id="wusrouter",
        base_url="https://wusrouter.com/v1",
        api_key="sk-test",
    )
    provider = OpenAIHttpProvider(cfg)

    respx.get("https://wusrouter.com/v1/models").mock(
        return_value=httpx.Response(200, json={"data": []})
    )

    st = await provider.health()
    assert st.provider_id == "wusrouter"
    assert st.healthy is True
    assert st.status == "reachable"
    assert "latency_ms" in st.details
