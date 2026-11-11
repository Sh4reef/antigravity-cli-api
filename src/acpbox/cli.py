"""Command Line Interface for agy-acpbox: OpenAI-Compatible API Gateway for ACP and OpenAI Providers."""

import atexit
import os
from pathlib import Path
from typing import List, Optional
import typer
import uvicorn

from acpbox.autostart import (
    get_service_status,
    install_service,
    uninstall_service,
)
from acpbox.backend import AgyACPBackend
from acpbox.config import Settings, load_config
from acpbox.daemon import (
    check_status,
    get_default_log_path,
    get_default_pid_path,
    remove_pid,
    start_daemon,
    stop_daemon,
    write_pid,
)
from acpbox.router import ModelRouter
from acpbox.server import create_app

app = typer.Typer(
    help="agy-acpbox: OpenAI-Compatible API Gateway for ACP Agents and External OpenAI Providers",
    add_completion=False,
)


def _build_forward_args(
    host: str,
    port: int,
    disable_telemetry: bool,
    dangerously_skip_permissions: bool,
    agent: Optional[str],
    workspace: Optional[str],
    add_dir: List[str],
    log_level: str,
    pid_file: Optional[Path],
    log_file: Optional[Path],
    config_file: Optional[Path] = None,
) -> List[str]:
    """Build list of CLI arguments to forward to background worker or autostart service."""
    args = [
        "--host", host,
        "--port", str(port),
        "--log-level", log_level,
    ]
    if config_file:
        args.extend(["--config", str(config_file)])
    if not disable_telemetry:
        args.append("--no-disable-telemetry")
    if not dangerously_skip_permissions:
        args.append("--no-dangerously-skip-permissions")
    if agent:
        args.extend(["--agent", agent])
    if workspace:
        args.extend(["--workspace", workspace])
    for d in add_dir:
        args.extend(["--add-dir", d])
    if pid_file:
        args.extend(["--pid-file", str(pid_file)])
    if log_file:
        args.extend(["--log-file", str(log_file)])
    return args


@app.command()
def main(
    host: str = typer.Option("127.0.0.1", "--host", help="Host address to bind HTTP server"),
    port: int = typer.Option(8001, "--port", help="Port to bind HTTP server"),
    config: Optional[Path] = typer.Option(
        None, "-c", "--config", help="Path to YAML multi-provider config file"
    ),
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
    background: bool = typer.Option(
        False, "-d", "-b", "--background", "--daemon", help="Run server in the background (detached process)"
    ),
    stop: bool = typer.Option(False, "--stop", help="Stop running background server"),
    status: bool = typer.Option(False, "--status", help="Check status of background server"),
    install_autostart: bool = typer.Option(
        False, "--install-autostart", "--autostart", help="Install and enable automatic startup on boot/login"
    ),
    uninstall_autostart: bool = typer.Option(
        False, "--uninstall-autostart", help="Uninstall and disable automatic startup service"
    ),
    autostart_status: bool = typer.Option(
        False, "--autostart-status", help="Check auto-start service status"
    ),
    pid_file: Optional[Path] = typer.Option(
        None, "--pid-file", help="PID file path for background process"
    ),
    log_file: Optional[Path] = typer.Option(
        None, "--log-file", help="Log file path for background process"
    ),
    worker: bool = typer.Option(
        False, "--worker", hidden=True, help="Internal worker flag for background daemon"
    ),
):
    """Start an OpenAI-compatible HTTP server routing ACP agents and external OpenAI providers."""
    effective_pid_path = pid_file or get_default_pid_path()
    effective_log_path = log_file or get_default_log_path()

    # Mode 1: Stop running daemon
    if stop:
        success, msg = stop_daemon(effective_pid_path)
        if success:
            typer.echo(f"[OK] {msg}")
            raise typer.Exit(code=0)
        else:
            typer.echo(f"[ERROR] {msg}", err=True)
            raise typer.Exit(code=1)

    # Mode 2: Status check
    if status:
        info = check_status(effective_pid_path, host=host, port=port)
        if info["running"]:
            typer.echo(f"[RUNNING] agy-acpbox is active (PID: {info['pid']})")
            typer.echo(f"          Endpoint: {info.get('endpoint')}")
            typer.echo(f"          PID File: {info.get('pid_file')}")
            typer.echo(f"          Health:   {'healthy' if info.get('healthy') else 'unresponsive'}")
            health_data = info.get("health_data") or {}
            providers = health_data.get("providers") or {}
            if providers:
                typer.echo(f"          Providers ({len(providers)}):")
                for pid, pdata in providers.items():
                    typer.echo(f"            - {pid} [{pdata.get('provider_type', 'unknown')}]: {pdata.get('status')}")
            raise typer.Exit(code=0)
        else:
            typer.echo(f"[STOPPED] agy-acpbox is not running ({info.get('status')})")
            if "message" in info:
                typer.echo(f"          {info['message']}")
            raise typer.Exit(code=1)

    # Mode 3: Install autostart service
    if install_autostart:
        forward_args = _build_forward_args(
            host=host,
            port=port,
            disable_telemetry=disable_telemetry,
            dangerously_skip_permissions=dangerously_skip_permissions,
            agent=agent,
            workspace=workspace,
            add_dir=add_dir,
            log_level=log_level,
            pid_file=pid_file,
            log_file=log_file,
            config_file=config,
        )
        success, msg = install_service(forward_args, effective_log_path)
        if success:
            typer.echo(f"[OK] {msg}")
            raise typer.Exit(code=0)
        else:
            typer.echo(f"[ERROR] {msg}", err=True)
            raise typer.Exit(code=1)

    # Mode 4: Uninstall autostart service
    if uninstall_autostart:
        success, msg = uninstall_service()
        if success:
            typer.echo(f"[OK] {msg}")
            raise typer.Exit(code=0)
        else:
            typer.echo(f"[ERROR] {msg}", err=True)
            raise typer.Exit(code=1)

    # Mode 5: Autostart status check
    if autostart_status:
        st = get_service_status()
        typer.echo(f"Platform:  {st['platform']} ({st['mechanism']})")
        typer.echo(f"Installed: {st['installed']}")
        typer.echo(f"Active:    {st['loaded']}")
        if st["path"]:
            typer.echo(f"Path:      {st['path']}")
        raise typer.Exit(code=0 if st["installed"] else 1)

    # Mode 6: Background daemon spawn
    if background:
        forward_args = _build_forward_args(
            host=host,
            port=port,
            disable_telemetry=disable_telemetry,
            dangerously_skip_permissions=dangerously_skip_permissions,
            agent=agent,
            workspace=workspace,
            add_dir=add_dir,
            log_level=log_level,
            pid_file=pid_file,
            log_file=log_file,
            config_file=config,
        )
        typer.echo(f"Starting agy-acpbox in background on http://{host}:{port}...")
        success, msg, pid = start_daemon(
            cmd_args=forward_args,
            log_file=effective_log_path,
            pid_file=effective_pid_path,
            host=host,
            port=port,
        )
        if success:
            typer.echo(f"[OK] {msg}")
            typer.echo(f"     Endpoint: http://{host}:{port}")
            typer.echo(f"     PID File: {effective_pid_path}")
            typer.echo(f"     Logs:     {effective_log_path}")
            typer.echo("\nTo check status: uv run agy-acpbox --status")
            typer.echo("To stop server:  uv run agy-acpbox --stop")
            raise typer.Exit(code=0)
        else:
            typer.echo(f"[ERROR] {msg}", err=True)
            raise typer.Exit(code=1)

    # Mode 7: Foreground Server (or internal worker process)
    if config:
        gateway_config = load_config(str(config))
        # CLI options override config defaults if explicitly given
        target_host = host if host != "127.0.0.1" else gateway_config.gateway.host
        target_port = port if port != 8001 else gateway_config.gateway.port
        target_log_level = log_level if log_level != "info" else gateway_config.gateway.log_level
        router = ModelRouter.from_config(gateway_config)
        api_app = create_app(router=router)
    else:
        settings = Settings(
            host=host,
            port=port,
            disable_telemetry=disable_telemetry,
            dangerously_skip_permissions=dangerously_skip_permissions,
            agent=agent,
            workspace=workspace,
            add_dir=add_dir,
            log_level=log_level,
            pid_file=str(effective_pid_path),
            log_file=str(effective_log_path),
            background=worker,
        )
        gateway_config = load_config(settings=settings)
        target_host = settings.host
        target_port = settings.port
        target_log_level = settings.log_level.lower()
        router = ModelRouter.from_config(gateway_config)
        api_app = create_app(router=router)

    if worker:
        write_pid(effective_pid_path, os.getpid())
        atexit.register(remove_pid, effective_pid_path)

    typer.echo(f"Starting agy-acpbox Universal Gateway on http://{target_host}:{target_port}")
    typer.echo(f"Registered providers: {list(router.providers.keys())}")
    typer.echo(f"Endpoints: /v1/chat/completions , /v1/models , /health")

    try:
        uvicorn.run(api_app, host=target_host, port=target_port, log_level=target_log_level)
    finally:
        if worker:
            remove_pid(effective_pid_path)


if __name__ == "__main__":
    app()
