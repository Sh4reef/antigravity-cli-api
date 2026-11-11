"""Live end-to-end integration test verifying unified gateway with real backends."""

import os
import pytest
from httpx import ASGITransport, AsyncClient

from acpbox.config import load_config
from acpbox.router import ModelRouter
from acpbox.server import create_app


@pytest.mark.asyncio
async def test_live_unified_gateway_end_to_end():
    """Validates unified gateway loading config.yaml and routing to real providers."""
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    config = load_config(config_path)

    router = ModelRouter.from_config(config)
    await router.start_all()
    try:
        app = create_app(router=router)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            # 1. Health check
            health_res = await client.get("/health")
            assert health_res.status_code == 200
            health_data = health_res.json()
            assert health_data["status"] == "healthy"
            assert "wusrouter" in health_data["providers"]
            assert "logfare" in health_data["providers"]

            # 2. Aggregated models check
            models_res = await client.get("/v1/models")
            assert models_res.status_code == 200
            models_data = models_res.json()
            model_ids = [m["id"] for m in models_data["data"]]

            # Check virtual model aliases
            assert "fast-free" in model_ids
            assert "reasoning-flash" in model_ids

            # 3. Non-streaming completion via virtual model alias (wusrouter)
            req_payload = {
                "model": "fast-free",
                "messages": [{"role": "user", "content": "Return the single word: OK"}],
                "max_tokens": 10,
                "stream": False,
            }
            comp_res = await client.post("/v1/chat/completions", json=req_payload)
            assert comp_res.status_code == 200
            comp_data = comp_res.json()
            assert comp_data["object"] == "chat.completion"
            assert len(comp_data["choices"]) > 0
            assert comp_data["choices"][0]["message"]["content"] is not None

            # 4. Streaming SSE completion via virtual model alias (logfare)
            stream_payload = {
                "model": "reasoning-flash",
                "messages": [{"role": "user", "content": "Say hello"}],
                "max_tokens": 15,
                "stream": True,
            }
            stream_res = await client.post("/v1/chat/completions", json=stream_payload)
            assert stream_res.status_code == 200
            assert "text/event-stream" in stream_res.headers["content-type"]
            assert "data: " in stream_res.text
            assert "data: [DONE]" in stream_res.text
    finally:
        await router.stop_all()
