"""Tests for the OpenAI-compatible HTTP Gateway exposing agy acp."""

import json
import pytest
from httpx import ASGITransport, AsyncClient

from acpbox.backend import AgyACPBackend
from acpbox.server import create_app


def test_agy_backend_command_flags():
    """Verifies that agy acp is launched with --disable-telemetry and required flags."""
    backend = AgyACPBackend(
        disable_telemetry=True,
        dangerously_skip_permissions=True,
        agent="code-reviewer",
        add_dir=["/tmp/workspace"],
    )

    cmd = backend.command
    assert "agy" in cmd[0]
    assert "acp" in cmd[1]
    assert "--stdio" in cmd
    assert "--disable-telemetry" in cmd
    assert "--dangerously-skip-permissions" in cmd
    assert "--agent" in cmd and "code-reviewer" in cmd
    assert "--add-dir" in cmd and "/tmp/workspace" in cmd


@pytest.mark.asyncio
async def test_get_models_endpoint():
    """Verifies GET /v1/models returns the list of available models."""
    backend = AgyACPBackend()
    backend.available_models = ["gemini-3.8-flash-high", "claude-sonnet-4-6"]

    app = create_app(backend=backend)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res = await ac.get("/v1/models")
        assert res.status_code == 200
        data = res.json()
        assert data["object"] == "list"
        model_ids = [m["id"] for m in data["data"]]
        assert "gemini-3.8-flash-high" in model_ids
        assert "claude-sonnet-4-6" in model_ids


@pytest.mark.asyncio
async def test_chat_completions_non_streaming():
    """Verifies POST /v1/chat/completions returns standard OpenAI JSON."""
    backend = AgyACPBackend()

    async def mock_stream_turn(prompt):
        yield ("Hello ", None, None)
        yield ("world from agy!", None, None)
        yield (None, None, {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17})

    backend.stream_turn = mock_stream_turn

    app = create_app(backend=backend)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        req_payload = {
            "model": "gemini-3.8-flash-high",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
        }
        res = await ac.post("/v1/chat/completions", json=req_payload)
        assert res.status_code == 200
        data = res.json()
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"]["content"] == "Hello world from agy!"
        assert data["choices"][0]["finish_reason"] == "stop"
        assert data["usage"]["total_tokens"] == 17


@pytest.mark.asyncio
async def test_chat_completions_streaming():
    """Verifies POST /v1/chat/completions with stream=True returns SSE stream."""
    backend = AgyACPBackend()

    async def mock_stream_turn(prompt):
        yield (None, "Thinking about greeting...", None)
        yield ("Hello ", None, None)
        yield ("there!", None, None)
        yield (None, None, {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14})

    backend.stream_turn = mock_stream_turn

    app = create_app(backend=backend)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        req_payload = {
            "model": "gemini-3.8-flash-high",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }
        res = await ac.post("/v1/chat/completions", json=req_payload)
        assert res.status_code == 200
        assert "text/event-stream" in res.headers["content-type"]

        lines = res.text.strip().splitlines()
        data_lines = [l for l in lines if l.startswith("data: ")]
        assert len(data_lines) >= 4  # role chunk + thought chunk + text chunk 1 + text chunk 2 + finish + [DONE]
        assert data_lines[-1] == "data: [DONE]"

        # Check content assembly
        assembled_content = ""
        assembled_thought = ""
        for dl in data_lines[:-1]:
            payload = json.loads(dl[6:])
            delta = payload["choices"][0]["delta"]
            if delta.get("content"):
                assembled_content += delta["content"]
            if delta.get("reasoning_content"):
                assembled_thought += delta["reasoning_content"]

        assert assembled_content == "Hello there!"
        assert assembled_thought == "Thinking about greeting..."
