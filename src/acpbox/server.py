"""FastAPI HTTP Server exposing OpenAI-compatible endpoints backed by agy acp."""

import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from acpbox.backend import AgyACPBackend
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


def create_app(backend: Optional[AgyACPBackend] = None) -> FastAPI:
    app_backend = backend or AgyACPBackend()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            await app_backend.start()
        except Exception:
            pass
        yield
        await app_backend.stop()

    app = FastAPI(
        title="agy-acpbox OpenAI Gateway",
        description="Exposes 'agy acp' over OpenAI-compatible HTTP endpoints (/v1/chat/completions)",
        version="0.1.0",
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
        return {
            "status": "healthy",
            "session_id": app_backend.session_id,
            "running": app_backend.proc is not None and app_backend.proc.returncode is None,
        }

    @app.get("/v1/models", response_model=ModelListResponse)
    @app.get("/models", response_model=ModelListResponse)
    async def list_models():
        models = [ModelItem(id=m) for m in app_backend.available_models]
        if not models:
            models = [ModelItem(id="default"), ModelItem(id="gemini-3.8-flash-high")]
        return ModelListResponse(data=models)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        # Format messages into prompt text
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

        full_prompt = "\n\n".join(prompt_lines)

        req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created_ts = int(time.time())

        # 1. Streaming response
        if request.stream:
            async def event_generator() -> AsyncGenerator[str, None]:
                # Send initial role chunk
                first_chunk = ChatCompletionChunk(
                    id=req_id,
                    created=created_ts,
                    model=request.model,
                    choices=[ChatCompletionChunkChoice(index=0, delta=DeltaMessage(role="assistant"))],
                )
                yield f"data: {first_chunk.model_dump_json()}\n\n"

                async for text, thought, usage in app_backend.stream_turn(full_prompt, model=request.model):
                    if thought:
                        thought_chunk = ChatCompletionChunk(
                            id=req_id,
                            created=created_ts,
                            model=request.model,
                            choices=[ChatCompletionChunkChoice(index=0, delta=DeltaMessage(reasoning_content=thought))],
                        )
                        yield f"data: {thought_chunk.model_dump_json()}\n\n"

                    if text:
                        text_chunk = ChatCompletionChunk(
                            id=req_id,
                            created=created_ts,
                            model=request.model,
                            choices=[ChatCompletionChunkChoice(index=0, delta=DeltaMessage(content=text))],
                        )
                        yield f"data: {text_chunk.model_dump_json()}\n\n"

                # Final finish chunk
                end_chunk = ChatCompletionChunk(
                    id=req_id,
                    created=created_ts,
                    model=request.model,
                    choices=[ChatCompletionChunkChoice(index=0, delta=DeltaMessage(), finish_reason="stop")],
                )
                yield f"data: {end_chunk.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(event_generator(), media_type="text/event-stream")

        # 2. Non-streaming response
        accumulated_text = []
        final_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        async for text, thought, usage in app_backend.stream_turn(full_prompt, model=request.model):
            if text:
                accumulated_text.append(text)
            if usage:
                final_usage = usage

        complete_content = "".join(accumulated_text)

        return ChatCompletionResponse(
            id=req_id,
            created=created_ts,
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessageOutput(role="assistant", content=complete_content),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(**final_usage),
        )

    return app
