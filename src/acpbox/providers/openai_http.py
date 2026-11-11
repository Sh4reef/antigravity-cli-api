"""OpenAI-compatible HTTP provider proxying requests to external REST/SSE endpoints."""

import asyncio
import json
import time
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional
import httpx
from fastapi import HTTPException

from acpbox.config import OpenAIProviderConfig
from acpbox.models import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessageOutput,
    DeltaMessage,
    ModelItem,
    UsageInfo,
)
from acpbox.providers.base import ProviderHealthStatus


class OpenAIHttpProvider:
    """Provider adapter for external OpenAI-compatible HTTP APIs (e.g., wusrouter, logfare, ollama)."""

    def __init__(self, config: OpenAIProviderConfig):
        self.config = config
        self.provider_id = config.id
        self.provider_type: Literal["openai_http"] = "openai_http"

        # Normalize base_url (strip trailing slashes)
        self.base_url = config.base_url.rstrip("/")
        self.api_key = config.api_key
        self.default_model = config.default_model
        self.timeout = config.timeout
        self.discover_models = config.discover_models

        # Build default headers
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(config.headers)
        self.headers = headers

        self.client: Optional[httpx.AsyncClient] = None
        self.available_models: List[str] = list(config.models)

    async def start(self) -> None:
        """Initializes the HTTP client session and performs initial model discovery if enabled."""
        if self.client is None or self.client.is_closed:
            self.client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=self.headers,
                timeout=self.timeout,
            )

        if self.discover_models:
            try:
                await self.get_models()
            except Exception:
                pass  # Gracefully proceed even if upstream is temporarily unreachable on boot

    async def stop(self) -> None:
        """Closes the underlying HTTP client session."""
        if self.client and not self.client.is_closed:
            await self.client.aclose()
        self.client = None

    def _ensure_client(self) -> httpx.AsyncClient:
        if self.client is None or self.client.is_closed:
            self.client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=self.headers,
                timeout=self.timeout,
            )
        return self.client

    async def get_models(self) -> List[ModelItem]:
        """Discovers models from upstream GET /models or returns configured models."""
        client = self._ensure_client()
        if self.discover_models:
            try:
                # Try relative paths
                resp = await client.get("/models")
                if resp.status_code == 200:
                    data = resp.json()
                    models_data = data.get("data", [])
                    discovered = [
                        m["id"]
                        for m in models_data
                        if isinstance(m, dict) and "id" in m
                    ]
                    if discovered:
                        for dm in discovered:
                            if dm not in self.available_models:
                                self.available_models.append(dm)
            except Exception:
                pass

        if not self.available_models:
            fallback = self.default_model or "default"
            self.available_models = [fallback]

        return [
            ModelItem(id=m, owned_by=f"openai:{self.provider_id}")
            for m in self.available_models
        ]

    def _build_payload(
        self, request: ChatCompletionRequest, upstream_model: Optional[str] = None
    ) -> Dict[str, Any]:
        """Constructs upstream OpenAI chat completions request body."""
        target_model = upstream_model or self.default_model or request.model
        payload: Dict[str, Any] = {
            "model": target_model,
            "messages": [m.model_dump(exclude_none=True) for m in request.messages],
            "stream": request.stream,
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.user is not None:
            payload["user"] = request.user
        return payload

    async def complete_chat(
        self, request: ChatCompletionRequest, upstream_model: Optional[str] = None
    ) -> ChatCompletionResponse:
        """Executes a non-streaming completion request against upstream HTTP endpoint."""
        client = self._ensure_client()
        payload = self._build_payload(request, upstream_model=upstream_model)
        payload["stream"] = False

        resp = None
        last_exc = None
        for attempt in range(3):
            try:
                resp = await client.post("/chat/completions", json=payload)
                break
            except (httpx.ConnectError, httpx.ReadError, httpx.TimeoutException) as e:
                last_exc = e
                await asyncio.sleep(0.5 * (attempt + 1))

        if resp is None:
            raise HTTPException(
                status_code=502,
                detail=f"Failed to connect to upstream provider '{self.provider_id}': {last_exc}",
            )

        if resp.status_code != 200:
            raise HTTPException(
                status_code=resp.status_code,
                detail=f"Upstream provider '{self.provider_id}' returned {resp.status_code}: {resp.text}",
            )

        data = resp.json()

        # Parse choices and normalize reasoning content if present
        choices = []
        raw_choices = data.get("choices", [])
        for rc in raw_choices:
            msg = rc.get("message", {})
            content = msg.get("content") or ""
            # Handle providers like logfare that return 'reasoning'
            reasoning = msg.get("reasoning") or msg.get("reasoning_content")
            if reasoning and not content:
                content = f"[Reasoning]\n{reasoning}"

            choices.append(
                ChatCompletionChoice(
                    index=rc.get("index", 0),
                    message=ChatMessageOutput(
                        role=msg.get("role", "assistant"),
                        content=content,
                    ),
                    finish_reason=rc.get("finish_reason", "stop") or "stop",
                )
            )

        if not choices:
            choices = [
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessageOutput(role="assistant", content=""),
                    finish_reason="stop",
                )
            ]

        raw_usage = data.get("usage", {})
        usage = UsageInfo(
            prompt_tokens=raw_usage.get("prompt_tokens", 0),
            completion_tokens=raw_usage.get("completion_tokens", 0),
            total_tokens=raw_usage.get("total_tokens", 0),
        )

        return ChatCompletionResponse(
            id=data.get("id", f"chatcmpl-{self.provider_id}"),
            model=request.model,
            choices=choices,
            usage=usage,
        )

    async def stream_chat(
        self, request: ChatCompletionRequest, upstream_model: Optional[str] = None
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        """Proxies streaming completion chunks, normalizing SSE events into ChatCompletionChunk."""
        client = self._ensure_client()
        payload = self._build_payload(request, upstream_model=upstream_model)
        payload["stream"] = True

        resp = None
        last_exc = None
        for attempt in range(3):
            try:
                req = client.build_request("POST", "/chat/completions", json=payload)
                resp = await client.send(req, stream=True)
                break
            except (httpx.ConnectError, httpx.ReadError, httpx.TimeoutException) as e:
                last_exc = e
                await asyncio.sleep(0.5 * (attempt + 1))

        if resp is None:
            raise HTTPException(
                status_code=502,
                detail=f"Failed to connect to upstream provider '{self.provider_id}': {last_exc}",
            )

        if resp.status_code != 200:
            error_body = await resp.aread()
            await resp.aclose()
            raise HTTPException(
                status_code=resp.status_code,
                detail=f"Upstream provider '{self.provider_id}' returned {resp.status_code}: {error_body.decode('utf-8', errors='replace')}",
            )

        try:
            async for raw_line in resp.aiter_lines():
                line = raw_line.strip()
                if not line:
                    continue

                if line.startswith("data: "):
                    data_str = line[6:].strip()
                elif line.startswith("data:"):
                    data_str = line[5:].strip()
                else:
                    continue

                if data_str == "[DONE]":
                    break

                try:
                    chunk_json = json.loads(data_str)
                except Exception:
                    continue

                chunk_id = chunk_json.get("id", f"chatcmpl-{self.provider_id}")
                created = chunk_json.get("created", int(time.time()))

                choices = []
                for c in chunk_json.get("choices", []):
                    delta_obj = c.get("delta", {})
                    role = delta_obj.get("role")
                    content = delta_obj.get("content")
                    # Normalize reasoning (e.g. logfare uses 'reasoning', deepseek uses 'reasoning_content')
                    reasoning = delta_obj.get("reasoning_content") or delta_obj.get("reasoning")
                    finish_reason = c.get("finish_reason")

                    choices.append(
                        ChatCompletionChunkChoice(
                            index=c.get("index", 0),
                            delta=DeltaMessage(
                                role=role,
                                content=content,
                                reasoning_content=reasoning,
                            ),
                            finish_reason=finish_reason,
                        )
                    )

                if choices:
                    yield ChatCompletionChunk(
                        id=chunk_id,
                        created=created,
                        model=request.model,
                        choices=choices,
                    )
        except (httpx.ReadError, httpx.RemoteProtocolError):
            pass
        finally:
            await resp.aclose()

    async def health(self) -> ProviderHealthStatus:
        """Pings upstream models endpoint to assess connectivity and measure latency."""
        client = self._ensure_client()
        t0 = time.perf_counter()
        healthy = False
        status_msg = "unreachable"
        latency_ms = None

        try:
            resp = await client.get("/models")
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            if resp.status_code in (200, 401, 403):
                # Server responded (even if auth is checked per-route)
                healthy = True
                status_msg = "reachable"
            else:
                status_msg = f"HTTP {resp.status_code}"
        except Exception as e:
            status_msg = str(e)

        return ProviderHealthStatus(
            provider_id=self.provider_id,
            provider_type="openai_http",
            healthy=healthy,
            status=status_msg,
            details={
                "base_url": self.base_url,
                "latency_ms": latency_ms,
                "models_count": len(self.available_models),
            },
        )
