"""Unit tests for ModelRouter."""

import pytest
from fastapi import HTTPException

from acpbox.config import GatewayConfig, VirtualModelConfig
from acpbox.models import ModelItem
from acpbox.providers.base import LLMProvider, ProviderHealthStatus
from acpbox.router import ModelRouter


class DummyProvider:
    def __init__(self, provider_id: str, provider_type: str = "openai_http", models: list = None):
        self.provider_id = provider_id
        self.provider_type = provider_type
        self.available_models = models or []
        self.started = False
        self.stopped = False

    async def start(self):
        self.started = True

    async def stop(self):
        self.stopped = True

    async def get_models(self):
        return [ModelItem(id=m, owned_by=self.provider_id) for m in self.available_models]

    async def stream_chat(self, request, upstream_model=None):
        yield None

    async def complete_chat(self, request, upstream_model=None):
        return None

    async def health(self):
        return ProviderHealthStatus(
            provider_id=self.provider_id,
            provider_type=self.provider_type,
            healthy=True,
            status="ok",
        )


def test_router_resolve_virtual_model():
    """Verify virtual model alias routes to mapped provider and upstream model."""
    p_wus = DummyProvider("wusrouter", models=["wusrouter/qwen3.8-flash:free"])
    p_agy = DummyProvider("agy", models=["gemini-3.8-flash-high"])

    router = ModelRouter(
        providers={"wusrouter": p_wus, "agy": p_agy},
        default_provider_id="agy",
        virtual_models={
            "fast-free": VirtualModelConfig(
                provider="wusrouter", upstream_model="wusrouter/qwen3.8-flash:free"
            )
        },
    )

    provider, upstream_model = router.resolve("fast-free")
    assert provider.provider_id == "wusrouter"
    assert upstream_model == "wusrouter/qwen3.8-flash:free"


def test_router_resolve_namespaced_prefix():
    """Verify provider/model prefix routes correctly."""
    p_logfare = DummyProvider("logfare", models=["glm-5.3-flash"])
    p_opencode = DummyProvider("opencode", models=["default"])

    router = ModelRouter(
        providers={"logfare": p_logfare, "opencode": p_opencode},
        default_provider_id="opencode",
    )

    provider, upstream_model = router.resolve("logfare/glm-5.3-flash")
    assert provider.provider_id == "logfare"
    assert upstream_model == "glm-5.3-flash"

    provider2, upstream_model2 = router.resolve("opencode:custom-model")
    assert provider2.provider_id == "opencode"
    assert upstream_model2 == "custom-model"


def test_router_resolve_catalog_match():
    """Verify direct model name match in available_models routes to owning provider."""
    p_wus = DummyProvider("wusrouter", models=["wus-unique-model"])
    p_agy = DummyProvider("agy", models=["gemini-3.8-flash-high"])

    router = ModelRouter(
        providers={"wusrouter": p_wus, "agy": p_agy},
        default_provider_id="agy",
    )

    provider, upstream_model = router.resolve("wus-unique-model")
    assert provider.provider_id == "wusrouter"
    assert upstream_model == "wus-unique-model"


def test_router_resolve_default_fallback():
    """Verify unmapped model falls back to default provider."""
    p_agy = DummyProvider("agy", models=["gemini-3.8-flash-high"])
    router = ModelRouter(
        providers={"agy": p_agy},
        default_provider_id="agy",
    )

    provider, upstream_model = router.resolve("unmapped-model")
    assert provider.provider_id == "agy"
    assert upstream_model == "unmapped-model"


@pytest.mark.asyncio
async def test_router_list_models_aggregation():
    """Verify list_models aggregates models from providers and virtual models."""
    p_wus = DummyProvider("wusrouter", models=["qwen"])
    p_agy = DummyProvider("agy", models=["gemini"])

    router = ModelRouter(
        providers={"wusrouter": p_wus, "agy": p_agy},
        default_provider_id="agy",
        virtual_models={
            "my-fast": VirtualModelConfig(provider="wusrouter", upstream_model="qwen")
        },
    )

    models = await router.list_models()
    model_ids = [m.id for m in models]
    assert "my-fast" in model_ids
    assert "wusrouter/qwen" in model_ids
    assert "qwen" in model_ids
    assert "agy/gemini" in model_ids
    assert "gemini" in model_ids


@pytest.mark.asyncio
async def test_router_lifecycle_and_health():
    """Verify start_all, stop_all, and aggregated health."""
    p_wus = DummyProvider("wusrouter")
    p_agy = DummyProvider("agy")

    router = ModelRouter(providers={"wusrouter": p_wus, "agy": p_agy})
    await router.start_all()
    assert p_wus.started is True
    assert p_agy.started is True

    health = await router.health()
    assert health["status"] == "healthy"
    assert "wusrouter" in health["providers"]
    assert "agy" in health["providers"]

    await router.stop_all()
    assert p_wus.stopped is True
    assert p_agy.stopped is True
