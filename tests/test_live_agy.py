"""Integration test verifying communication between AgyACPBackend and live 'agy acp'."""

import shutil
import pytest
from acpbox.backend import AgyACPBackend


@pytest.mark.asyncio
async def test_live_agy_acp_handshake(tmp_path):
    """Verifies that AgyACPBackend starts live 'agy acp', performs handshake, and discovers models."""
    if not shutil.which("agy"):
        pytest.skip("'agy' command is not available in PATH")

    backend = AgyACPBackend(
        cwd=str(tmp_path),
        disable_telemetry=True,
        dangerously_skip_permissions=True,
    )

    try:
        await backend.start()
        assert backend.session_id is not None
        assert len(backend.session_id) > 0
        assert len(backend.available_models) > 0
        assert any("gemini" in m or "claude" in m for m in backend.available_models)
    finally:
        await backend.stop()
