"""Command Line Interface for agy-acpbox: OpenAI-Compatible API Gateway for agy acp."""

from typing import List, Optional
import typer
import uvicorn

from acpbox.backend import AgyACPBackend
from acpbox.config import Settings
from acpbox.server import create_app

app = typer.Typer(
    help="agy-acpbox: OpenAI-Compatible API Gateway for Antigravity (agy acp)",
    add_completion=False,
)


@app.command()
def main(
    host: str = typer.Option("127.0.0.1", "--host", help="Host address to bind HTTP server"),
    port: int = typer.Option(8000, "--port", help="Port to bind HTTP server"),
    disable_telemetry: bool = typer.Option(
        True, "--disable-telemetry", help="Enforce --disable-telemetry in agy acp"
    ),
    dangerously_skip_permissions: bool = typer.Option(
        True, "--dangerously-skip-permissions", help="Auto-approve tool requests in agy acp"
    ),
    agent: Optional[str] = typer.Option(None, "--agent", help="Agent persona for agy acp session"),
    workspace: Optional[str] = typer.Option(
        None, "--workspace", "--cwd", help="Primary workspace directory for agy acp (defaults to current directory)"
    ),
    add_dir: List[str] = typer.Option([], "--add-dir", help="Additional workspace directories to include"),
    log_level: str = typer.Option("info", "--log-level", help="Log level (debug, info, warning, error)"),
):
    """Start an OpenAI-compatible HTTP server exposing 'agy acp' on /v1/chat/completions."""
    settings = Settings(
        host=host,
        port=port,
        disable_telemetry=disable_telemetry,
        dangerously_skip_permissions=dangerously_skip_permissions,
        agent=agent,
        workspace=workspace,
        add_dir=add_dir,
        log_level=log_level,
    )

    backend = AgyACPBackend(
        cwd=settings.workspace,
        disable_telemetry=settings.disable_telemetry,
        dangerously_skip_permissions=settings.dangerously_skip_permissions,
        agent=settings.agent,
        add_dir=settings.add_dir,
    )
    api_app = create_app(backend=backend)

    typer.echo(f"Starting agy-acpbox OpenAI Gateway on http://{settings.host}:{settings.port}")
    typer.echo(f"Telemetry disabled: {settings.disable_telemetry} | Auto-approve permissions: {settings.dangerously_skip_permissions}")
    typer.echo(f"Endpoints: /v1/chat/completions , /v1/models , /health")

    uvicorn.run(api_app, host=settings.host, port=settings.port, log_level=settings.log_level.lower())


if __name__ == "__main__":
    app()
