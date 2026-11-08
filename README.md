# agy-acpbox

> **Bidirectional Bridge between Agent Client Protocol (ACP) and OpenAI-Compatible APIs**

`agy-acpbox` operates in two directions:
1. **OpenAI Gateway Mode (`serve`):** Exposes Antigravity (`agy acp`) as standard **OpenAI-compatible HTTP endpoints** (`/v1/chat/completions` and `/v1/models`) with `--disable-telemetry`. Any OpenAI-compatible client (Continue.dev, Open WebUI, Cursor, LangChain, curl) can now talk to Antigravity.
2. **ACP Server Mode (`stdio`):** Bridges code editors and IDEs (Zed, JetBrains) implementing the [Agent Client Protocol (ACP)](https://agentclientprotocol.com) to external **OpenAI-Compatible LLM endpoints** (Ollama, vLLM, DeepSeek, LocalAI, OpenAI).

---

## Key Features

- **OpenAI-Compatible Gateway:** Exposes `agy acp` via HTTP endpoints (`GET /v1/models`, `POST /v1/chat/completions`) with full streaming Server-Sent Events (SSE) support and thought extraction (`reasoning_content`).
- **Telemetry Protection:** Automatically applies `--disable-telemetry` when launching `agy acp`.
- **Full ACP v1.x Compliance:** Implements `initialize`, `session/new`, `session/prompt`, and `session/cancel` over standard input/output (`stdio`) JSON-RPC 2.0.
- **Drop-in Parity with `agy acp`:** Supports CLI flags matching Antigravity's reference server (`--stdio`, `--add-dir`, `--agent`, `--dangerously-skip-permissions`, `--log-file`).
- **Autonomous ReAct Agent Loop:** Handles multi-step recursive function/tool execution turns with self-correcting error handling.

---

## Installation & Setup

- Python 3.11+
- Package manager: [`uv`](https://docs.astral.sh/uv/) or `pip`

```bash
# Clone the repository
git clone https://github.com/your-org/agy-acpbox.git
cd agy-acpbox

# Install dependencies with uv
uv sync
```

---

## Usage

### Mode 1: Expose `agy acp` as OpenAI-Compatible HTTP Server

Start the gateway server on port 8000 (telemetry disabled by default):

```bash
uv run agy-acpbox serve --host 127.0.0.1 --port 8000 --disable-telemetry --dangerously-skip-permissions
```

Now any OpenAI client can connect to `http://127.0.0.1:8000/v1`:

#### Test with cURL (Non-Streaming):
```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-3.8-flash-high",
    "messages": [{"role": "user", "content": "Explain binary search in one sentence."}]
  }'
```

#### Test with cURL (Streaming SSE):
```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-3.8-flash-high",
    "messages": [{"role": "user", "content": "Write a python quicksort function."}],
    "stream": true
  }'
```

#### List Available Models:
```bash
curl http://127.0.0.1:8000/v1/models
```

#### Use with Python `openai` SDK:
```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="dummy")

response = client.chat.completions.create(
    model="gemini-3.8-flash-high",
    messages=[{"role": "user", "content": "Hello agy acp!"}],
    stream=True,
)

for chunk in response:
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="", flush=True)
```

---

### Mode 2: Run as ACP stdio Server (for Zed / JetBrains)

Point your ACP IDE to an external OpenAI-compatible model:

```bash
# Using a local Ollama instance with Qwen 2.5 Coder
uv run agy-acpbox \
  --base-url "http://localhost:11434/v1" \
  --model "qwen2.5-coder:32b" \
  --api-key "ollama"
```

In Zed's `settings.json`:
```json
{
  "agent": {
    "custom_agents": [
      {
        "name": "Local Qwen (ACP)",
        "command": "uvx",
        "args": [
          "agy-acpbox",
          "--base-url", "http://localhost:11434/v1",
          "--model", "qwen2.5-coder:32b",
          "--api-key", "ollama"
        ]
      }
    ]
  }
}
```

---

## Testing & Conformance

`agy-acpbox` includes an automated differential test suite comparing behavior directly against Antigravity's reference ACP command (`agy acp`):

```bash
# Run all tests (unit + ReAct loop + differential parity with agy acp + HTTP gateway)
uv run pytest -v
```

---

## Documents

- 📄 **[PRD.md](PRD.md):** Product Requirements Document
- 📄 **[TDD.md](TDD.md):** Technical Design Document
