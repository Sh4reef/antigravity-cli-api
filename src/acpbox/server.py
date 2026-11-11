"""FastAPI HTTP Server exposing OpenAI-compatible endpoints backed by ModelRouter."""

import json
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from acpbox.backend import AgyACPBackend
from acpbox.config import GatewayConfig, load_config
from acpbox.models import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessageOutput,
    DeltaMessage,
    ModelItem,
    ModelListResponse,
    UsageInfo,
)
from acpbox.providers.base import LLMProvider, ProviderHealthStatus
from acpbox.router import ModelRouter


class _LegacyBackendAdapter:
    """Wraps an existing AgyACPBackend instance to conform to LLMProvider."""

    def __init__(self, backend: AgyACPBackend):
        self.backend = backend
        self.provider_id = "agy"
        self.provider_type = "acp"

    @property
    def available_models(self) -> List[str]:
        return self.backend.available_models

    async def start(self) -> None:
        await self.backend.start()

    async def stop(self) -> None:
        await self.backend.stop()

    async def get_models(self) -> List[ModelItem]:
        models = [ModelItem(id=m) for m in self.backend.available_models]
        if not models:
            models = [ModelItem(id="default"), ModelItem(id="gemini-3.8-flash-high")]
        return models

    def _format_messages_to_prompt(self, request: ChatCompletionRequest) -> str:
        prompt_lines = []
        for msg in request.messages:
            content = msg.content if isinstance(msg.content, str) else json.dumps(msg.content)
            if msg.role == "system":
                prompt_lines.append(f"[System Instruction]\n{content}")
            elif msg.role == "user":
                prompt_lines.append(f"[User]\n{content}")
            elif msg.role == "assistant":
                prompt_lines.append(f"[Assistant]\n{content}")
            else:
                prompt_lines.append(f"[{msg.role}]\n{content}")
        return "\n\n".join(prompt_lines)

    async def stream_chat(
        self, request: ChatCompletionRequest, upstream_model: Optional[str] = None
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        import time, uuid
        req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created_ts = int(time.time())
        full_prompt = self._format_messages_to_prompt(request)
        target_model = upstream_model or request.model

        yield ChatCompletionChunk(
            id=req_id,
            created=created_ts,
            model=request.model,
            choices=[ChatCompletionChunkChoice(index=0, delta=DeltaMessage(role="assistant"))],
        )

        async for text, thought, usage in self.backend.stream_turn(full_prompt, model=target_model):
            if thought:
                yield ChatCompletionChunk(
                    id=req_id,
                    created=created_ts,
                    model=request.model,
                    choices=[ChatCompletionChunkChoice(index=0, delta=DeltaMessage(reasoning_content=thought))],
                )
            if text:
                yield ChatCompletionChunk(
                    id=req_id,
                    created=created_ts,
                    model=request.model,
                    choices=[ChatCompletionChunkChoice(index=0, delta=DeltaMessage(content=text))],
                )

        yield ChatCompletionChunk(
            id=req_id,
            created=created_ts,
            model=request.model,
            choices=[ChatCompletionChunkChoice(index=0, delta=DeltaMessage(), finish_reason="stop")],
        )

    async def complete_chat(
        self, request: ChatCompletionRequest, upstream_model: Optional[str] = None
    ) -> ChatCompletionResponse:
        import time, uuid
        req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created_ts = int(time.time())
        full_prompt = self._format_messages_to_prompt(request)
        target_model = upstream_model or request.model

        accumulated_text = []
        final_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        async for text, thought, usage in self.backend.stream_turn(full_prompt, model=target_model):
            if text:
                accumulated_text.append(text)
            if usage:
                final_usage = usage

        return ChatCompletionResponse(
            id=req_id,
            created=created_ts,
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessageOutput(role="assistant", content="".join(accumulated_text)),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(**final_usage),
        )

    async def health(self) -> ProviderHealthStatus:
        is_running = self.backend.proc is not None and self.backend.proc.returncode is None
        return ProviderHealthStatus(
            provider_id="agy",
            provider_type="acp",
            healthy=is_running,
            status="running" if is_running else "stopped",
            details={
                "session_id": self.backend.session_id,
                "models": self.backend.available_models,
            },
        )


def create_app(
    backend: Optional[AgyACPBackend] = None,
    router: Optional[ModelRouter] = None,
    config: Optional[GatewayConfig] = None,
) -> FastAPI:
    if router is not None:
        app_router = router
    elif backend is not None:
        adapter = _LegacyBackendAdapter(backend)
        app_router = ModelRouter(providers={"agy": adapter}, default_provider_id="agy")
    elif config is not None:
        app_router = ModelRouter.from_config(config)
    else:
        default_cfg = load_config()
        app_router = ModelRouter.from_config(default_cfg)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            await app_router.start_all()
        except Exception:
            pass
        yield
        await app_router.stop_all()

    app = FastAPI(
        title="agy-acpbox Universal Gateway",
        description="Unified OpenAI-Compatible HTTP Server routing across ACP Agents and OpenAI Providers",
        version="0.2.0",
        lifespan=lifespan,
    )

    # Enable CORS for browser-based web UIs
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health():
        health_info = await app_router.health()
        agy_p = app_router.providers.get("agy")
        session_id = getattr(agy_p, "session_id", None)
        if hasattr(agy_p, "backend"):
            session_id = getattr(agy_p.backend, "session_id", None)

        return {
            "status": health_info["status"],
            "session_id": session_id,
            "running": health_info["status"] == "healthy",
            **health_info,
        }

    @app.get("/v1/models", response_model=ModelListResponse)
    @app.get("/models", response_model=ModelListResponse)
    async def list_models():
        models = await app_router.list_models()
        return ModelListResponse(data=models)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        provider, upstream_model = app_router.resolve(request.model)

        # 1. Streaming response
        if request.stream:
            async def event_generator() -> AsyncGenerator[str, None]:
                async for chunk in provider.stream_chat(request, upstream_model=upstream_model):
                    yield f"data: {chunk.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(event_generator(), media_type="text/event-stream")

        # 2. Non-streaming response
        return await provider.complete_chat(request, upstream_model=upstream_model)

    return app
