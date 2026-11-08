"""Configuration settings for agy-acpbox OpenAI Gateway."""

from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ACPBOX_",
        env_file=".env",
        extra="ignore",
    )

    host: str = Field(default="127.0.0.1", description="Host address to bind HTTP gateway")
    port: int = Field(default=8000, description="Port to bind HTTP gateway")
    disable_telemetry: bool = Field(default=True, description="Enforce --disable-telemetry in agy acp")
    dangerously_skip_permissions: bool = Field(
        default=True, description="Auto-approve tool permissions in agy acp"
    )
    agent: Optional[str] = Field(default=None, description="Default agent persona for agy acp")
    workspace: Optional[str] = Field(default=None, description="Primary workspace root directory for agy acp")
    add_dir: List[str] = Field(default_factory=list, description="Workspace directories to include")
    log_level: str = Field(default="INFO", description="Server log level")
