"""Base protocol and types for LLM Providers in agy-acpbox."""

from typing import Any, AsyncGenerator, Dict, List, Literal, Optional, Protocol, runtime_checkable
from pydantic import BaseModel, Field

from acpbox.models import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ModelItem,
)


class ProviderHealthStatus(BaseModel):
    provider_id: str
    provider_type: Literal["acp", "openai_http"]
    healthy: bool
    status: str
    details: Dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol implemented by both local ACP agents and remote OpenAI-compatible HTTP providers."""

    provider_id: str
    provider_type: Literal["acp", "openai_http"]

    async def start(self) -> None:
        """Initialize provider (launch subprocess or establish HTTP connection)."""
        ...

    async def stop(self) -> None:
        """Gracefully terminate subprocess or close HTTP client."""
        ...

    async def get_models(self) -> List[ModelItem]:
        """Return list of models supported by this provider."""
        ...

    async def stream_chat(
        self, request: ChatCompletionRequest, upstream_model: Optional[str] = None
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        """Stream chat completions as Server-Sent Event (SSE) chunks."""
        ...

    async def complete_chat(
        self, request: ChatCompletionRequest, upstream_model: Optional[str] = None
    ) -> ChatCompletionResponse:
        """Generate a complete, non-streaming chat completion."""
        ...

    async def health(self) -> ProviderHealthStatus:
        """Report live health status of the provider."""
        ...
