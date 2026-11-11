"""Unit tests for agy-acpbox multi-provider configuration loading."""

import os
from acpbox.config import Settings, expand_env_vars, load_config


def test_default_config_fallback():
    """Verify load_config without a file creates default agy ACP provider with port 8001."""
    cfg = load_config()
    assert cfg.gateway.port == 8001
    assert cfg.gateway.default_provider == "agy"
    assert len(cfg.acp_providers) == 1
    assert cfg.acp_providers[0].id == "agy"
    assert "--disable-telemetry" in cfg.acp_providers[0].command
    assert len(cfg.openai_providers) == 0
    assert len(cfg.virtual_models) == 0


def test_expand_env_vars():
    """Verify expand_env_vars correctly expands existing, default, and missing env vars."""
    os.environ["TEST_ACP_KEY"] = "sk-test-12345"
    try:
        text = "key: ${TEST_ACP_KEY}, fallback: ${NONEXISTENT_VAR:-my_default}, empty: ${ANOTHER_MISSING}"
        expanded = expand_env_vars(text)
        assert "key: sk-test-12345" in expanded
        assert "fallback: my_default" in expanded
        assert "empty: " in expanded
    finally:
        del os.environ["TEST_ACP_KEY"]


def test_load_yaml_config(tmp_path):
    """Verify loading a complete YAML multi-provider config file."""
    yaml_content = """
gateway:
  host: "0.0.0.0"
  port: 8001
  default_provider: "agy"

acp_providers:
  - id: "agy"
    command: ["agy", "acp", "--stdio"]
    models: ["gemini-3.8-flash-high"]
  - id: "opencode"
    command: ["opencode", "acp"]
    models: ["default"]

openai_providers:
  - id: "wusrouter"
    base_url: "https://wusrouter.com/v1"
    api_key: "sk-ws-test"
    default_model: "wusrouter/qwen3.8-flash:free"
  - id: "logfare"
    base_url: "https://logfare.ai/v1"
    api_key: "lfu-test"
    default_model: "glm-5.3-flash"

virtual_models:
  "fast-free":
    provider: "wusrouter"
    upstream_model: "wusrouter/qwen3.8-flash:free"
"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml_content, encoding="utf-8")

    cfg = load_config(str(config_file))
    assert cfg.gateway.host == "0.0.0.0"
    assert cfg.gateway.port == 8001
    assert len(cfg.acp_providers) == 2
    assert cfg.acp_providers[0].id == "agy"
    assert cfg.acp_providers[1].id == "opencode"
    assert len(cfg.openai_providers) == 2
    assert cfg.openai_providers[0].id == "wusrouter"
    assert cfg.openai_providers[0].base_url == "https://wusrouter.com/v1"
    assert cfg.openai_providers[1].id == "logfare"
    assert "fast-free" in cfg.virtual_models
    assert cfg.virtual_models["fast-free"].provider == "wusrouter"


def test_load_config_with_custom_settings():
    """Verify load_config honors custom Settings instance when path is None."""
    custom_s = Settings(
        port=9090,
        agent="special-agent",
        disable_telemetry=False,
    )
    cfg = load_config(settings=custom_s)
    assert cfg.gateway.port == 9090
    assert cfg.acp_providers[0].agent == "special-agent"
    assert cfg.acp_providers[0].disable_telemetry is False
    assert "--disable-telemetry" not in cfg.acp_providers[0].command
