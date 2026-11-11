"""Live integration tests for external OpenAI-compatible HTTP providers (WusRouter & Logfare)."""

import pytest
from acpbox.config import OpenAIProviderConfig
from acpbox.models import ChatCompletionRequest, ChatMessageInput
from acpbox.providers.openai_http import OpenAIHttpProvider

WUSROUTER_KEY = "sk-ws-a21ab55a1fa24eaea1b66eae78be7ebf400a056221c05449b5e7a1d371919de5"
LOGFARE_KEY = "lfu_ZVox-kb49mhC_IFODKHcrbbhJChlWxAo"


@pytest.mark.asyncio
async def test_live_wusrouter_complete_chat():
    """Verifies non-streaming completion against live WusRouter API."""
    cfg = OpenAIProviderConfig(
        id="wusrouter",
        base_url="https://wusrouter.com/v1",
        api_key=WUSROUTER_KEY,
        discover_models=False,
        models=["wusrouter/qwen3.8-flash:free"],
    )
    provider = OpenAIHttpProvider(cfg)
    try:
        req = ChatCompletionRequest(
            model="wusrouter/qwen3.8-flash:free",
            messages=[ChatMessageInput(role="user", content="Answer with just one word: pong")],
            max_tokens=10,
            stream=False,
        )
        res = await provider.complete_chat(req)
        assert res.id is not None
        assert len(res.choices) > 0
        assert res.choices[0].message.content is not None
        assert len(res.choices[0].message.content.strip()) > 0
    finally:
        await provider.stop()


@pytest.mark.asyncio
async def test_live_wusrouter_stream_chat():
    """Verifies SSE streaming against live WusRouter API."""
    cfg = OpenAIProviderConfig(
        id="wusrouter",
        base_url="https://wusrouter.com/v1",
        api_key=WUSROUTER_KEY,
        discover_models=False,
    )
    provider = OpenAIHttpProvider(cfg)
    try:
        req = ChatCompletionRequest(
            model="wusrouter/qwen3.8-flash:free",
            messages=[ChatMessageInput(role="user", content="Count from 1 to 3")],
            max_tokens=15,
            stream=True,
        )
        chunks = []
        async for chunk in provider.stream_chat(req):
            chunks.append(chunk)

        assert len(chunks) > 0
        content_pieces = [c.choices[0].delta.content for c in chunks if c.choices[0].delta.content]
        full_text = "".join(content_pieces)
        assert len(full_text.strip()) > 0
    finally:
        await provider.stop()


@pytest.mark.asyncio
async def test_live_logfare_complete_chat():
    """Verifies non-streaming completion against live Logfare.ai API."""
    cfg = OpenAIProviderConfig(
        id="logfare",
        base_url="https://logfare.ai/v1",
        api_key=LOGFARE_KEY,
        discover_models=False,
        models=["glm-5.3-flash"],
    )
    provider = OpenAIHttpProvider(cfg)
    try:
        req = ChatCompletionRequest(
            model="glm-5.3-flash",
            messages=[ChatMessageInput(role="user", content="Reply 'hello'")],
            max_tokens=15,
            stream=False,
        )
        res = await provider.complete_chat(req)
        assert res.id is not None
        assert len(res.choices) > 0
        assert res.choices[0].message.content is not None
    finally:
        await provider.stop()


@pytest.mark.asyncio
async def test_live_logfare_stream_chat():
    """Verifies SSE streaming against live Logfare.ai API."""
    cfg = OpenAIProviderConfig(
        id="logfare",
        base_url="https://logfare.ai/v1",
        api_key=LOGFARE_KEY,
        discover_models=False,
    )
    provider = OpenAIHttpProvider(cfg)
    try:
        req = ChatCompletionRequest(
            model="glm-5.3-flash",
            messages=[ChatMessageInput(role="user", content="Say pong")],
            max_tokens=15,
            stream=True,
        )
        chunks = []
        async for chunk in provider.stream_chat(req):
            chunks.append(chunk)

        assert len(chunks) > 0
        has_content_or_reasoning = any(
            c.choices[0].delta.content or c.choices[0].delta.reasoning_content
            for c in chunks
        )
        assert has_content_or_reasoning is True
    finally:
        await provider.stop()
