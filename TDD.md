# Technical Design Document (TDD)

## Project: `agy-acpbox` — OpenAI-Compatible API Gateway for `agy acp`
**Status:** Approved & Verified  
**Version:** 2.0.0 (Gateway Architecture)  
**Author:** AI Systems Architecture Team  
**Companion Document:** [PRD.md](file:///Users/syarif/Projects/agy-acpbox/PRD.md)

---

## 1. System Overview

`agy-acpbox` is a high-performance, asynchronous Python HTTP service built on **FastAPI** and **Uvicorn** that bridges the **OpenAI REST API** (`/v1/chat/completions` and `/v1/models`) to the **Google Antigravity ACP stdio daemon** (`agy acp`).

```
┌────────────────────────────────────────────────────────────────────────┐
│                   OpenAI-Compatible Client Applications                │
│                                                                        │
│   Continue.dev        Cursor (Custom API)        Open WebUI / cURL     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ HTTP POST /v1/chat/completions
                                    │ HTTP GET /v1/models
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                         agy-acpbox Gateway                             │
│                                                                        │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │                 FastAPI App & CORS Middleware                  │   │
│   └───────────────────────────────┬────────────────────────────────┘   │
│                                   │ SSE / JSON Router                  │
│                                   ▼                                    │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │                   OpenAI Data Models (Pydantic)                │   │
│   └───────────────────────────────┬────────────────────────────────┘   │
│                                   │ Stream Processor                   │
│                                   ▼                                    │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │               AgyACPBackend Subprocess Manager                 │   │
│   │    - Spawns 'agy acp --stdio --disable-telemetry'              │   │
│   │    - Handles initialize & session/new handshakes               │   │
│   │    - Translates session/update chunks to OpenAI SSE deltas     │   │
│   │    - Resolves client fs requests (read_text_file) locally      │   │
│   └───────────────────────────────┬────────────────────────────────┘   │
└───────────────────────────────────┼────────────────────────────────────┘
                                    │ JSON-RPC 2.0 (stdin / stdout pipes)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   Antigravity Daemon (agy acp)                         │
│                                                                        │
│   --stdio  --disable-telemetry  --dangerously-skip-permissions         │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Component Design

### 2.1 Process Manager (`acpbox.backend.AgyACPBackend`)
The process manager controls the lifecycle of the `agy acp` subprocess:

```python
class AgyACPBackend:
    def __init__(
        self,
        cwd: Optional[str] = None,
        disable_telemetry: bool = True,
        dangerously_skip_permissions: bool = True,
        agent: Optional[str] = None,
        add_dir: Optional[List[str]] = None,
    ): ...
```

- **Command Construction:**
  Ensures flags are appended in order:
  `["agy", "acp", "--stdio", "--disable-telemetry", "--dangerously-skip-permissions"]`
- **Handshake Sequence:**
  1. Sends `initialize` with `clientCapabilities` (`fs: {readTextFile: true, writeTextFile: true}`).
  2. Sends `session/new` with `cwd` and receives `sessionId` + list of `availableModels`.
- **Client Request Mediation:**
  When `agy acp` emits a client request (such as `fs/read_text_file`), `AgyACPBackend` reads the file locally and returns the response line immediately so the agent never blocks.
- **Turn Stream Generator (`stream_turn`):**
  Sends `session/prompt` with `prompt: [{"type": "text", "text": prompt_text}]` and yields tuples of:
  `(text_delta, thought_delta, usage_dict)`

### 2.2 OpenAI Data Models (`acpbox.models`)
- `ChatCompletionRequest`: Validates incoming messages, `stream`, `model`, `temperature`.
- `ChatCompletionResponse`: Formats non-streaming assistant reply and `usage` info.
- `ChatCompletionChunk`: Formats SSE stream deltas (`content` and `reasoning_content`).
- `ModelListResponse`: Formats `/v1/models` list.

### 2.3 HTTP Server (`acpbox.server`)
Built with FastAPI and lifespan context manager:
- **`GET /v1/models`**: Returns `ModelListResponse`.
- **`POST /v1/chat/completions`**:
  - Formats multi-turn chat messages into conversation context.
  - If `stream == True`, returns a `StreamingResponse(media_type="text/event-stream")` yielding `data: {JSON}\n\n` chunks and `data: [DONE]\n\n`.
  - If `stream == False`, gathers tokens and returns `ChatCompletionResponse`.
- **`GET /health`**: Returns subprocess running state and active session ID.

### 2.4 CLI Interface (`acpbox.cli`)
- Single command startup:
  `uv run agy-acpbox --host 127.0.0.1 --port 8000`
- Configurable flags for host, port, telemetry, permissions, agent, and workspace dirs.

---

## 3. Directory Layout

```text
src/acpbox/
├── __init__.py           # Package exports
├── __main__.py           # Direct python -m acpbox entrypoint
├── backend.py            # AgyACPBackend subprocess runner
├── cli.py                # Typer CLI definition
├── config.py             # Pydantic Settings
├── models.py             # OpenAI API Pydantic schemas
└── server.py             # FastAPI HTTP application

tests/
├── test_cli.py           # CLI argument parsing tests
├── test_gateway.py       # Endpoints & SSE streaming tests
└── test_live_agy.py      # Live handshake test against agy acp
```

---

## 4. Verification Checklist

- [x] Streamlined codebase to Gateway Mode
- [x] Enforced `--disable-telemetry` by default
- [x] Verified non-streaming completion JSON
- [x] Verified streaming SSE chunks with reasoning content
- [x] Verified live handshake against `agy acp`
- [x] 100% test pass rate across test suite
