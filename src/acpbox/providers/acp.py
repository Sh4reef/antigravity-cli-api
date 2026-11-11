"""Generic ACP (Agent Client Protocol) provider running CLI agents over stdio."""

import asyncio
import json
import os
import shutil
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional, Tuple

from acpbox.config import ACPProviderConfig
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


class GenericACPProvider:
    """Provider adapter for any ACP-compliant agent CLI (agy acp, opencode acp, etc.)."""

    def __init__(self, config: ACPProviderConfig):
        self.config = config
        self.provider_id = config.id
        self.provider_type: Literal["acp"] = "acp"
        self.cwd = os.path.abspath(config.cwd or os.getcwd())
        self.command = list(config.command)
        self.env = {**os.environ, **config.env}

        self.proc: Optional[asyncio.subprocess.Process] = None
        self.session_id: Optional[str] = None
        self.current_model: Optional[str] = None
        self.has_fixed_models: bool = bool(config.models)
        self.available_models: List[str] = list(config.models)

        self._req_counter = 0
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        """Launches the ACP process and performs initialize and session/new handshakes."""
        cmd_bin = self.command[0]
        if not shutil.which(cmd_bin):
            raise RuntimeError(f"Binary '{cmd_bin}' not found in PATH for provider '{self.provider_id}'.")

        self.proc = await asyncio.create_subprocess_exec(
            *self.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.cwd,
            env=self.env,
        )

        # 1. initialize handshake
        self._req_counter += 1
        init_req = {
            "jsonrpc": "2.0",
            "id": self._req_counter,
            "method": "initialize",
            "params": {
                "protocolVersion": 1,
                "clientCapabilities": {
                    "fs": {"readTextFile": True, "writeTextFile": True},
                    "terminal": {"create": True},
                },
                "clientInfo": {"name": f"agy-acpbox-{self.provider_id}", "version": "0.2.0"},
            },
        }
        await self._send_json(init_req)
        init_res = await self._read_json_response(self._req_counter)
        if "error" in init_res:
            raise RuntimeError(f"Initialize failed for provider '{self.provider_id}': {init_res['error']}")

        # 2. session/new handshake
        self._req_counter += 1
        new_req = {
            "jsonrpc": "2.0",
            "id": self._req_counter,
            "method": "session/new",
            "params": {
                "cwd": self.cwd,
                "mcpServers": [],
            },
        }
        await self._send_json(new_req)
        new_res = await self._read_json_response(self._req_counter)
        if "error" in new_res:
            raise RuntimeError(f"session/new failed for provider '{self.provider_id}': {new_res['error']}")

        result = new_res.get("result", {})
        self.session_id = result.get("sessionId")
        self.current_model = result.get("models", {}).get("currentModelId")

        # If models list was explicitly defined in config, do not append dynamically discovered models
        if not self.has_fixed_models:
            # Extract available models if provided by agent (agy format: models.availableModels)
            models_data = result.get("models", {}).get("availableModels", [])
            if models_data:
                discovered = [m.get("modelId") for m in models_data if "modelId" in m]
                for dm in discovered:
                    if dm not in self.available_models:
                        self.available_models.append(dm)
            else:
                # Extract available models if provided by agent (opencode format: configOptions with id=='model')
                config_options = result.get("configOptions", [])
                for opt in config_options:
                    if opt.get("id") == "model":
                        if not self.current_model and opt.get("currentValue"):
                            self.current_model = opt.get("currentValue")
                        for option in opt.get("options", []):
                            val = option.get("value")
                            if val and val not in self.available_models:
                                self.available_models.append(val)

            if not self.available_models:
                self.available_models = ["default"]

    async def _ensure_model(self, model: Optional[str]) -> None:
        """Switches the ACP session model if different from the currently active model."""
        if not model or model == "default" or model == self.current_model:
            return
        self._req_counter += 1
        req_id = self._req_counter
        await self._send_json({
            "jsonrpc": "2.0",
            "id": req_id,
            "method": "session/set_model",
            "params": {
                "sessionId": self.session_id,
                "modelId": model,
            },
        })
        res = await self._read_json_response(req_id)
        if "error" not in res:
            self.current_model = model

    async def _send_json(self, data: Dict[str, Any]) -> None:
        if not self.proc or not self.proc.stdin:
            raise RuntimeError(f"Backend process for '{self.provider_id}' is not running")
        line = json.dumps(data) + "\n"
        self.proc.stdin.write(line.encode("utf-8"))
        await self.proc.stdin.drain()

    async def _read_json_response(self, expected_id: int) -> Dict[str, Any]:
        """Reads stream lines until response with expected_id is found, answering client requests."""
        while True:
            if not self.proc or not self.proc.stdout:
                raise RuntimeError(f"Process output stream unavailable for '{self.provider_id}'")
            line_bytes = await self.proc.stdout.readline()
            if not line_bytes:
                raise RuntimeError(f"Unexpected EOF from ACP agent '{self.provider_id}'")
            line = line_bytes.decode("utf-8").strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except Exception:
                continue

            # Check if agent sent a client request (e.g. fs/read_text_file)
            if "method" in data and "id" in data:
                await self._handle_client_request(data)
                continue

            if data.get("id") == expected_id:
                return data

    async def _handle_client_request(self, req: Dict[str, Any]) -> None:
        """Handles agent-initiated requests like reading/writing a file."""
        req_id = req["id"]
        method = req["method"]
        params = req.get("params", {})

        if method == "fs/read_text_file":
            path = params.get("path", "")
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                await self._send_json({"jsonrpc": "2.0", "id": req_id, "result": {"content": content}})
            except Exception as e:
                await self._send_json({"jsonrpc": "2.0", "id": req_id, "error": {"code": -32603, "message": str(e)}})

        elif method == "fs/write_text_file":
            path = params.get("path", "")
            content = params.get("content", "")
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
                await self._send_json({"jsonrpc": "2.0", "id": req_id, "result": {}})
            except Exception as e:
                await self._send_json({"jsonrpc": "2.0", "id": req_id, "error": {"code": -32603, "message": str(e)}})
        else:
            await self._send_json({"jsonrpc": "2.0", "id": req_id, "result": {}})

    async def stream_turn(
        self, prompt_text: str, model: Optional[str] = None
    ) -> AsyncGenerator[Tuple[Optional[str], Optional[str], Optional[Dict[str, int]]], None]:
        """Submits prompt to ACP agent and yields (text_delta, thought_delta, usage_dict)."""
        async with self._lock:
            if not self.proc or self.proc.returncode is not None:
                await self.start()

            if model:
                await self._ensure_model(model)

            self._req_counter += 1
            turn_req_id = self._req_counter

            prompt_payload = {
                "jsonrpc": "2.0",
                "id": turn_req_id,
                "method": "session/prompt",
                "params": {
                    "sessionId": self.session_id,
                    "prompt": [{"type": "text", "text": prompt_text}],
                },
            }
            await self._send_json(prompt_payload)

            while True:
                if not self.proc or not self.proc.stdout:
                    break
                line_bytes = await self.proc.stdout.readline()
                if not line_bytes:
                    break
                line = line_bytes.decode("utf-8").strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except Exception:
                    continue

                # Handle client requests if any
                if "method" in data and "id" in data:
                    await self._handle_client_request(data)
                    continue

                # Handle streaming update notifications
                if data.get("method") == "session/update":
                    update = data.get("params", {}).get("update", {})
                    update_type = update.get("sessionUpdate")

                    if update_type == "agent_message_chunk":
                        content_obj = update.get("content", {})
                        text = content_obj.get("text", "")
                        if text:
                            yield (text, None, None)

                    elif update_type == "agent_thought_chunk":
                        thought = update.get("thought", "")
                        if thought:
                            yield (None, thought, None)

                # Check if final response is reached
                if data.get("id") == turn_req_id:
                    res = data.get("result", {})
                    raw_usage = res.get("usage", {})
                    usage_info = {
                        "prompt_tokens": raw_usage.get("inputTokens", 0),
                        "completion_tokens": raw_usage.get("outputTokens", 0),
                        "total_tokens": raw_usage.get("totalTokens", 0),
                    }
                    yield (None, None, usage_info)
                    break

    def _format_messages_to_prompt(self, request: ChatCompletionRequest) -> str:
        """Converts OpenAI messages array into flat prompt string for ACP prompt."""
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

    async def get_models(self) -> List[ModelItem]:
        """Returns list of ModelItem supported by this ACP agent."""
        return [
            ModelItem(id=m, owned_by=f"acp:{self.provider_id}")
            for m in self.available_models
        ]

    async def stream_chat(
        self, request: ChatCompletionRequest, upstream_model: Optional[str] = None
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        """Streams ChatCompletionChunk items compliant with OpenAI SSE standard."""
        full_prompt = self._format_messages_to_prompt(request)
        target_model = upstream_model or request.model
        req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created_ts = int(time.time())

        # 1. Initial role chunk
        yield ChatCompletionChunk(
            id=req_id,
            created=created_ts,
            model=request.model,
            choices=[ChatCompletionChunkChoice(index=0, delta=DeltaMessage(role="assistant"))],
        )

        # 2. Stream tokens from ACP turn
        async for text, thought, usage in self.stream_turn(full_prompt, model=target_model):
            if thought:
                yield ChatCompletionChunk(
                    id=req_id,
                    created=created_ts,
                    model=request.model,
                    choices=[
                        ChatCompletionChunkChoice(
                            index=0, delta=DeltaMessage(reasoning_content=thought)
                        )
                    ],
                )
            if text:
                yield ChatCompletionChunk(
                    id=req_id,
                    created=created_ts,
                    model=request.model,
                    choices=[
                        ChatCompletionChunkChoice(
                            index=0, delta=DeltaMessage(content=text)
                        )
                    ],
                )

        # 3. Final finish chunk
        yield ChatCompletionChunk(
            id=req_id,
            created=created_ts,
            model=request.model,
            choices=[
                ChatCompletionChunkChoice(
                    index=0, delta=DeltaMessage(), finish_reason="stop"
                )
            ],
        )

    async def complete_chat(
        self, request: ChatCompletionRequest, upstream_model: Optional[str] = None
    ) -> ChatCompletionResponse:
        """Non-streaming complete chat response."""
        full_prompt = self._format_messages_to_prompt(request)
        target_model = upstream_model or request.model
        req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created_ts = int(time.time())

        accumulated_text = []
        final_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        async for text, thought, usage in self.stream_turn(full_prompt, model=target_model):
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

    async def health(self) -> ProviderHealthStatus:
        """Reports health status of the subprocess."""
        is_running = self.proc is not None and self.proc.returncode is None
        return ProviderHealthStatus(
            provider_id=self.provider_id,
            provider_type="acp",
            healthy=is_running,
            status="running" if is_running else "stopped",
            details={
                "pid": self.proc.pid if is_running else None,
                "session_id": self.session_id,
                "current_model": self.current_model,
                "available_models": self.available_models,
            },
        )

    async def stop(self) -> None:
        """Terminates the ACP child process."""
        if self.proc:
            try:
                self.proc.terminate()
                await asyncio.wait_for(self.proc.wait(), timeout=2.0)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
            self.proc = None
            self.session_id = None
            self.current_model = None
