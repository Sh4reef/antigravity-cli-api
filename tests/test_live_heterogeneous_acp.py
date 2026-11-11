"""Live integration tests verifying heterogeneous ACP agents (agy and opencode)."""

import shutil
import pytest
from acpbox.config import ACPProviderConfig
from acpbox.providers.acp import GenericACPProvider


@pytest.mark.asyncio
async def test_live_agy_acp(tmp_path):
    """Verifies that GenericACPProvider can launch and handshake with real 'agy acp'."""
    if not shutil.which("agy"):
        pytest.skip("'agy' binary not in PATH")

    cfg = ACPProviderConfig(
        id="agy",
        command=["agy", "acp", "--stdio", "--disable-telemetry", "--dangerously-skip-permissions"],
        cwd=str(tmp_path),
    )
    provider = GenericACPProvider(cfg)
    try:
        await provider.start()
        assert provider.session_id is not None
        assert len(provider.session_id) > 0
        assert provider.proc is not None
        assert provider.proc.returncode is None

        models = await provider.get_models()
        assert len(models) > 0
        assert any(m.owned_by == "acp:agy" for m in models)

        health = await provider.health()
        assert health.healthy is True
        assert health.status == "running"
    finally:
        await provider.stop()


@pytest.mark.asyncio
async def test_live_opencode_acp(tmp_path):
    """Verifies that GenericACPProvider can launch and handshake with real 'opencode acp'."""
    if not shutil.which("opencode"):
        pytest.skip("'opencode' binary not in PATH")

    cfg = ACPProviderConfig(
        id="opencode",
        command=["opencode", "acp"],
        cwd=str(tmp_path),
    )
    provider = GenericACPProvider(cfg)
    try:
        await provider.start()
        assert provider.session_id is not None
        assert len(provider.session_id) > 0
        assert provider.proc is not None
        assert provider.proc.returncode is None

        models = await provider.get_models()
        assert len(models) > 0
        assert any(m.owned_by == "acp:opencode" for m in models)

        health = await provider.health()
        assert health.healthy is True
        assert health.status == "running"
    finally:
        await provider.stop()
