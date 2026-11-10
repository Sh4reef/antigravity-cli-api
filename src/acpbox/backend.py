"""Backend process manager running 'agy acp' over stdio."""

import asyncio
import json
import os
import shutil
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple


class AgyACPBackend:
    def __init__(
        self,
        command: Optional[List[str]] = None,
        cwd: Optional[str] = None,
        disable_telemetry: bool = True,
        dangerously_skip_permissions: bool = True,
        agent: Optional[str] = None,
        add_dir: Optional[List[str]] = None,
    ):
        self.cwd = os.path.abspath(cwd or os.getcwd())
        self.disable_telemetry = disable_telemetry
        self.dangerously_skip_permissions = dangerously_skip_permissions
        self.agent = agent
        self.add_dir = add_dir or []

        if command:
            self.command = command
        else:
            base_cmd = ["agy", "acp", "--stdio"]
            if self.disable_telemetry:
                base_cmd.append("--disable-telemetry")
            if self.dangerously_skip_permissions:
                base_cmd.append("--dangerously-skip-permissions")
            if self.agent:
                base_cmd.extend(["--agent", self.agent])
            for d in self.add_dir:
                base_cmd.extend(["--add-dir", d])
            self.command = base_cmd

        self.proc: Optional[asyncio.subprocess.Process] = None
        self.session_id: Optional[str] = None
        self.current_model: Optional[str] = None
        self.available_models: List[str] = [
            "gemini-3.8-flash-high",
            "gemini-3.8-flash-medium",
            "gemini-3.8-flash-low",
            "claude-sonnet-4-6",
            "claude-opus-4-6-thinking",
            "gpt-oss-120b-medium",
        ]
        self._req_counter = 0
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        """Launches the agy acp process and performs initialize and session/new handshakes."""
        if not shutil.which(self.command[0]):
            raise RuntimeError(f"Binary '{self.command[0]}' not found in PATH.")

        self.proc = await asyncio.create_subprocess_exec(
            *self.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.cwd,
        )

        # 1. initialize
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
                "clientInfo": {"name": "agy-acpbox-gateway", "version": "0.1.0"},
            },
        }
        await self._send_json(init_req)
        init_res = await self._read_json_response(self._req_counter)
        if "error" in init_res:
            raise RuntimeError(f"Initialize failed: {init_res['error']}")

        # 2. session/new
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
            raise RuntimeError(f"session/new failed: {new_res['error']}")

        result = new_res.get("result", {})
        self.session_id = result.get("sessionId")
        self.current_model = result.get("models", {}).get("currentModelId")

        # Extract available models if provided by agy acp
        models_data = result.get("models", {}).get("availableModels", [])
        if models_data:
            self.available_models = [m.get("modelId") for m in models_data if "modelId" in m]

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
            raise RuntimeError("Backend process not running")
        line = json.dumps(data) + "\n"
        self.proc.stdin.write(line.encode("utf-8"))
        await self.proc.stdin.drain()

    async def _read_json_response(self, expected_id: int) -> Dict[str, Any]:
        """Reads stream lines until the response with expected_id is found, answering client requests on the fly."""
        while True:
            if not self.proc or not self.proc.stdout:
                raise RuntimeError("Process output stream unavailable")
            line_bytes = await self.proc.stdout.readline()
            if not line_bytes:
                raise RuntimeError("Unexpected EOF from agy acp")
            line = line_bytes.decode("utf-8").strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except Exception:
                continue

            # Check if agy acp sent a client request (e.g. fs/read_text_file)
            if "method" in data and "id" in data:
                await self._handle_client_request(data)
                continue

            if data.get("id") == expected_id:
                return data

    async def _handle_client_request(self, req: Dict[str, Any]) -> None:
        """Handles agent-initiated requests like reading a file or running a command."""
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
        """
        Submits prompt to agy acp and yields tuples of:
        (text_delta, thought_delta, usage_dict)
        """
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

            usage_info: Optional[Dict[str, int]] = None

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

    async def stop(self) -> None:
        """Terminates the agy acp process."""
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
