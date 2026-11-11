"""Configuration settings and schema for agy-acpbox OpenAI Gateway."""

import os
import re
import shutil
from typing import Any, Dict, List, Optional
import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Legacy/Single-backend settings loaded from environment or CLI."""

    model_config = SettingsConfigDict(
        env_prefix="ACPBOX_",
        env_file=".env",
        extra="ignore",
    )

    host: str = Field(default="127.0.0.1", description="Host address to bind HTTP gateway")
    port: int = Field(default=8001, description="Port to bind HTTP gateway")
    disable_telemetry: bool = Field(default=True, description="Enforce --disable-telemetry in agy acp")
    dangerously_skip_permissions: bool = Field(
        default=True, description="Auto-approve tool permissions in agy acp"
    )
    agent: Optional[str] = Field(default=None, description="Default agent persona for agy acp")
    workspace: Optional[str] = Field(default=None, description="Primary workspace root directory for agy acp")
    add_dir: List[str] = Field(default_factory=list, description="Workspace directories to include")
    log_level: str = Field(default="INFO", description="Server log level")
    pid_file: Optional[str] = Field(default=None, description="Path to PID file for background mode")
    log_file: Optional[str] = Field(default=None, description="Path to log file for background mode")
    background: bool = Field(default=False, description="Run server in background daemon mode")
    config_file: Optional[str] = Field(default=None, description="Path to YAML multi-provider config file")


# Multi-Provider Configuration Schemas
class ACPProviderConfig(BaseModel):
    id: str = Field(description="Unique provider identifier, e.g. 'agy', 'opencode'")
    command: List[str] = Field(
        default_factory=lambda: ["agy", "acp", "--stdio"],
        description="Command and flags to spawn the ACP CLI over stdio",
    )
    cwd: Optional[str] = Field(default=None, description="Working directory for this ACP agent process")
    env: Dict[str, str] = Field(default_factory=dict, description="Custom environment variables for this process")
    disable_telemetry: bool = Field(default=True, description="Enforce --disable-telemetry if supported")
    dangerously_skip_permissions: bool = Field(
        default=True, description="Enforce --dangerously-skip-permissions if supported"
    )
    agent: Optional[str] = Field(default=None, description="Agent persona name for agy acp")
    add_dir: List[str] = Field(default_factory=list, description="Additional directories for agy acp")
    models: List[str] = Field(default_factory=list, description="Explicit models provided by this ACP agent")


class OpenAIProviderConfig(BaseModel):
    id: str = Field(description="Unique provider identifier, e.g. 'wusrouter', 'logfare'")
    base_url: str = Field(description="Base URL of OpenAI-compatible API, e.g. 'https://wusrouter.com/v1'")
    api_key: Optional[str] = Field(default=None, description="Bearer token or API key for upstream authentication")
    headers: Dict[str, str] = Field(default_factory=dict, description="Additional custom HTTP headers")
    models: List[str] = Field(default_factory=list, description="Static model list if known")
    discover_models: bool = Field(default=True, description="Automatically discover models via GET /v1/models")
    default_model: Optional[str] = Field(default=None, description="Default model for this provider if unspecified")
    timeout: float = Field(default=60.0, description="HTTP request timeout in seconds")


class VirtualModelConfig(BaseModel):
    provider: str = Field(description="Target provider ID")
    upstream_model: str = Field(description="Actual model ID on the target provider")


class GatewaySettings(BaseModel):
    host: str = Field(default="127.0.0.1", description="Gateway host address")
    port: int = Field(default=8001, description="Gateway port")
    default_provider: str = Field(default="agy", description="Default provider ID for un-prefixed models")
    log_level: str = Field(default="info", description="Logging level")


class GatewayConfig(BaseModel):
    gateway: GatewaySettings = Field(default_factory=GatewaySettings)
    acp_providers: List[ACPProviderConfig] = Field(default_factory=list)
    openai_providers: List[OpenAIProviderConfig] = Field(default_factory=list)
    virtual_models: Dict[str, VirtualModelConfig] = Field(default_factory=dict)


def expand_env_vars(content: str) -> str:
    """Expand environment variables in format ${VAR} or ${VAR:-default}."""
    pattern = re.compile(r"\$\{([A-Za-z0-9_]+)(?::-([^}]*))?\}")

    def _replace(match: re.Match) -> str:
        var_name = match.group(1)
        default_val = match.group(2) if match.group(2) is not None else ""
        return os.environ.get(var_name, default_val)

    return pattern.sub(_replace, content)


def find_config_file(explicit_path: Optional[str] = None) -> Optional[str]:
    """Finds configuration file by checking explicit path, env vars, cwd, and user config dir."""
    if explicit_path:
        if os.path.isfile(explicit_path):
            return os.path.abspath(explicit_path)
        return None

    # 1. Environment variables
    for env_key in ("AGY_ACPBOX_CONFIG", "ACPBOX_CONFIG", "ACPBOX_CONFIG_FILE"):
        val = os.environ.get(env_key)
        if val and os.path.isfile(val):
            return os.path.abspath(val)

    # 2. Current working directory
    for fname in ("config.yaml", "config.yml"):
        candidate = os.path.join(os.getcwd(), fname)
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)

    # 3. User config directory
    user_config = os.path.expanduser("~/.config/agy-acpbox/config.yaml")
    if os.path.isfile(user_config):
        return os.path.abspath(user_config)

    return None


def load_config(
    path: Optional[str] = None,
    settings: Optional[Settings] = None,
    auto_discover: bool = False,
) -> GatewayConfig:
    """Loads GatewayConfig from YAML file (if provided or discovered) or builds default from Settings."""
    resolved_path = None
    if path:
        resolved_path = find_config_file(path)
    elif settings and settings.config_file:
        resolved_path = find_config_file(settings.config_file)
    elif auto_discover:
        resolved_path = find_config_file()

    if resolved_path and os.path.isfile(resolved_path):
        with open(resolved_path, "r", encoding="utf-8") as f:
            raw_content = f.read()

        expanded_content = expand_env_vars(raw_content)
        data = yaml.safe_load(expanded_content) or {}
        return GatewayConfig.model_validate(data)

    # Fallback to single/default provider settings (defaulting to agy acp)
    s = settings or Settings()
    base_cmd = ["agy", "acp", "--stdio"]
    if s.disable_telemetry:
        base_cmd.append("--disable-telemetry")
    if s.dangerously_skip_permissions:
        base_cmd.append("--dangerously-skip-permissions")
    if s.agent:
        base_cmd.extend(["--agent", s.agent])
    for d in s.add_dir:
        base_cmd.extend(["--add-dir", d])

    default_acp = ACPProviderConfig(
        id="agy",
        command=base_cmd,
        cwd=s.workspace,
        disable_telemetry=s.disable_telemetry,
        dangerously_skip_permissions=s.dangerously_skip_permissions,
        agent=s.agent,
        add_dir=s.add_dir,
        models=[
            "gemini-3.8-flash-high",
            "gemini-3.8-flash-medium",
            "gemini-3.8-flash-low",
            "claude-sonnet-4-6",
            "claude-opus-4-6-thinking",
            "gpt-oss-120b-medium",
        ],
    )

    return GatewayConfig(
        gateway=GatewaySettings(
            host=s.host,
            port=s.port,
            default_provider="agy",
            log_level=s.log_level.lower(),
        ),
        acp_providers=[default_acp],
        openai_providers=[],
        virtual_models={},
    )
