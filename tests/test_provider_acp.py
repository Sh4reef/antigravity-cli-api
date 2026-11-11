"""Unit tests for GenericACPProvider."""

import pytest
from acpbox.config import ACPProviderConfig
from acpbox.models import ChatCompletionRequest, ChatMessageInput
from acpbox.providers.acp import GenericACPProvider
from acpbox.providers.base import LLMProvider


def test_provider_protocol_conformance():
    """Verify GenericACPProvider satisfies LLMProvider protocol."""
    cfg = ACPProviderConfig(
        id="opencode",
        command=["opencode", "acp"],
        models=["claude-3-5-sonnet", "gpt-4o"],
    )
    provider = GenericACPProvider(cfg)
    assert isinstance(provider, LLMProvider)
    assert provider.provider_id == "opencode"
    assert provider.provider_type == "acp"


@pytest.mark.asyncio
async def test_acp_provider_get_models():
    """Verify get_models returns tagged ModelItem instances."""
    cfg = ACPProviderConfig(
        id="opencode",
        command=["opencode", "acp"],
        models=["claude-3-5-sonnet", "gpt-4o"],
    )
    provider = GenericACPProvider(cfg)
    models = await provider.get_models()
    assert len(models) == 2
    assert models[0].id == "claude-3-5-sonnet"
    assert models[0].owned_by == "acp:opencode"
    assert models[1].id == "gpt-4o"
    assert models[1].owned_by == "acp:opencode"


@pytest.mark.asyncio
async def test_acp_provider_stream_chat():
    """Verify stream_chat emits role, reasoning, content, and stop chunks."""
    cfg = ACPProviderConfig(
        id="agy",
        command=["agy", "acp", "--stdio"],
        models=["gemini-3.8-flash-high"],
    )
    provider = GenericACPProvider(cfg)

    # Mock stream_turn
    async def mock_stream_turn(prompt, model=None):
        yield (None, "Analyzing problem...", None)
        yield ("Hello ", None, None)
        yield ("world!", None, None)
        yield (None, None, {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})

    provider.stream_turn = mock_stream_turn

    req = ChatCompletionRequest(
        model="gemini-3.8-flash-high",
        messages=[ChatMessageInput(role="user", content="say hello")],
        stream=True,
    )

    chunks = []
    async for chunk in provider.stream_chat(req):
        chunks.append(chunk)

    assert len(chunks) == 5  # role + thought + text1 + text2 + stop
    assert chunks[0].choices[0].delta.role == "assistant"
    assert chunks[1].choices[0].delta.reasoning_content == "Analyzing problem..."
    assert chunks[2].choices[0].delta.content == "Hello "
    assert chunks[3].choices[0].delta.content == "world!"
    assert chunks[4].choices[0].finish_reason == "stop"


@pytest.mark.asyncio
async def test_acp_provider_complete_chat():
    """Verify complete_chat returns non-streaming response with text and usage."""
    cfg = ACPProviderConfig(
        id="agy",
        command=["agy", "acp", "--stdio"],
        models=["gemini-3.8-flash-high"],
    )
    provider = GenericACPProvider(cfg)

    async def mock_stream_turn(prompt, model=None):
        yield ("Combined ", None, None)
        yield ("result.", None, None)
        yield (None, None, {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12})

    provider.stream_turn = mock_stream_turn

    req = ChatCompletionRequest(
        model="gemini-3.8-flash-high",
        messages=[ChatMessageInput(role="user", content="compute")],
        stream=False,
    )

    res = await provider.complete_chat(req)
    assert res.choices[0].message.content == "Combined result."
    assert res.choices[0].finish_reason == "stop"
    assert res.usage.prompt_tokens == 8
    assert res.usage.completion_tokens == 4
    assert res.usage.total_tokens == 12


@pytest.mark.asyncio
async def test_acp_provider_health():
    """Verify health reports status correctly."""
    cfg = ACPProviderConfig(id="test_acp", command=["agy", "acp"])
    provider = GenericACPProvider(cfg)
    st = await provider.health()
    assert st.provider_id == "test_acp"
    assert st.provider_type == "acp"
    assert st.healthy is False
    assert st.status == "stopped"


def test_acp_provider_fixed_models_flag():
    """Verify has_fixed_models is True when models is non-empty, False when empty."""
    cfg_fixed = ACPProviderConfig(id="agy", models=["gemini-3.8-flash-high"])
    provider_fixed = GenericACPProvider(cfg_fixed)
    assert provider_fixed.has_fixed_models is True
    assert provider_fixed.available_models == ["gemini-3.8-flash-high"]

    cfg_dynamic = ACPProviderConfig(id="agy", models=[])
    provider_dynamic = GenericACPProvider(cfg_dynamic)
    assert provider_dynamic.has_fixed_models is False
    assert provider_dynamic.available_models == []
