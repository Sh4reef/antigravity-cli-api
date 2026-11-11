"""Daemon process management for agy-acpbox."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def get_default_dir() -> Path:
    """Return default directory for agy-acpbox runtime files."""
    home_dir = Path.home() / ".agy-acpbox"
    home_dir.mkdir(parents=True, exist_ok=True)
    return home_dir


def get_default_pid_path() -> Path:
    """Return default path to PID file."""
    return get_default_dir() / "agy-acpbox.pid"


def get_default_log_path() -> Path:
    """Return default path to daemon log file."""
    return get_default_dir() / "agy-acpbox.log"


def is_process_alive(pid: int) -> bool:
    """Check if process with given PID is currently active."""
    if pid <= 0:
        return False
    try:
        # If this process is the parent, waitpid with WNOHANG reaps it if it exited
        res_pid, _ = os.waitpid(pid, os.WNOHANG)
        if res_pid == pid:
            return False
    except (ChildProcessError, OSError):
        pass

    try:
        os.kill(pid, 0)
    except OSError:
        return False

    # Check for zombie / defunct processes
    try:
        res = subprocess.run(["ps", "-o", "state=", "-p", str(pid)], capture_output=True, text=True)
        if res.returncode == 0:
            state = res.stdout.strip()
            if state.startswith("Z"):
                return False
    except Exception:
        pass

    return True


def read_pid(pid_file: Path) -> Optional[int]:
    """Read PID from PID file if it exists and is valid."""
    if not pid_file.is_file():
        return None
    try:
        content = pid_file.read_text().strip()
        if content.isdigit():
            return int(content)
        return None
    except Exception:
        return None


def write_pid(pid_file: Path, pid: int) -> None:
    """Write PID to PID file."""
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(f"{pid}\n")


def remove_pid(pid_file: Path) -> None:
    """Remove PID file if it exists."""
    try:
        if pid_file.is_file():
            pid_file.unlink()
    except OSError:
        pass


def check_health(host: str, port: int, timeout: float = 5.0) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Query /health endpoint to check server responsiveness."""
    # When host is 0.0.0.0 or 127.0.0.1, query 127.0.0.1
    target_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    url = f"http://{target_host}:{port}/health"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "agy-acpbox-daemon-healthcheck"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                import json
                data = json.loads(resp.read().decode("utf-8"))
                return True, data
    except Exception:
        pass
    return False, None


def start_daemon(
    cmd_args: List[str],
    log_file: Path,
    pid_file: Path,
    host: str,
    port: int,
    healthcheck_timeout: float = 15.0,
) -> Tuple[bool, str, Optional[int]]:
    """Spawn agy-acpbox in background as a detached daemon process.
    
    Returns (success, message, pid).
    """
    # Check if already running
    existing_pid = read_pid(pid_file)
    if existing_pid is not None:
        if is_process_alive(existing_pid):
            return False, f"Server already running with PID {existing_pid}. Use --stop first.", existing_pid
        # PID file is stale, clean it
        remove_pid(pid_file)

    log_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.parent.mkdir(parents=True, exist_ok=True)

    # Spawn child detached process
    # sys.executable -m acpbox <args...> --worker
    child_cmd = [sys.executable, "-m", "acpbox"] + cmd_args + ["--worker"]

    with open(log_file, "a") as log_fp:
        proc = subprocess.Popen(
            child_cmd,
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,  # setsid on POSIX
        )

    pid = proc.pid
    write_pid(pid_file, pid)

    # Wait for server to become responsive or exit early
    deadline = time.time() + healthcheck_timeout
    started = False

    while time.time() < deadline:
        # Check if child died early
        if not is_process_alive(pid):
            remove_pid(pid_file)
            # Read last lines of log to explain failure
            err_snippet = ""
            if log_file.is_file():
                try:
                    lines = log_file.read_text().splitlines()
                    err_snippet = "\n".join(lines[-10:])
                except Exception:
                    pass
            return False, f"Process exited prematurely.\nLog tail:\n{err_snippet}", None

        ok, _ = check_health(host=host, port=port, timeout=3.0)
        if ok:
            started = True
            break
        time.sleep(0.3)

    if not started:
        # Check if process is still running even if healthcheck didn't respond in time
        if is_process_alive(pid):
            return True, f"Process spawned with PID {pid}, but healthcheck timed out after {healthcheck_timeout}s.", pid
        remove_pid(pid_file)
        return False, "Failed to start daemon (healthcheck timeout and process terminated).", None

    return True, f"Server started in background (PID: {pid}).", pid


def stop_daemon(pid_file: Path, timeout: float = 5.0) -> Tuple[bool, str]:
    """Gracefully terminate the background daemon process."""
    pid = read_pid(pid_file)
    if pid is None:
        return False, f"No PID file found at {pid_file}. Server does not appear to be running."

    if not is_process_alive(pid):
        remove_pid(pid_file)
        return True, f"Cleaned stale PID file {pid_file} (PID {pid} was not running)."

    # Send SIGTERM for graceful shutdown
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError as err:
        return False, f"Failed to send SIGTERM to PID {pid}: {err}"

    deadline = time.time() + timeout
    while time.time() < deadline:
        if not is_process_alive(pid):
            remove_pid(pid_file)
            return True, f"Server with PID {pid} stopped gracefully."
        time.sleep(0.2)

    # Force kill if still alive
    if is_process_alive(pid):
        try:
            os.kill(pid, signal.SIGKILL)
            time.sleep(0.2)
        except OSError:
            pass

    remove_pid(pid_file)
    if not is_process_alive(pid):
        return True, f"Server with PID {pid} terminated forcibly (SIGKILL)."
    return False, f"Failed to terminate PID {pid}."


def check_status(pid_file: Path, host: str, port: int) -> Dict[str, Any]:
    """Inspect daemon liveness and health status."""
    pid = read_pid(pid_file)
    if pid is None:
        return {"status": "STOPPED", "running": False, "pid": None, "pid_file": str(pid_file)}

    alive = is_process_alive(pid)
    if not alive:
        return {
            "status": "STALE_PID",
            "running": False,
            "pid": pid,
            "pid_file": str(pid_file),
            "message": "PID file exists but process is not running.",
        }

    healthy, health_data = check_health(host=host, port=port, timeout=5.0)
    return {
        "status": "RUNNING" if healthy else "UNRESPONSIVE",
        "running": True,
        "pid": pid,
        "pid_file": str(pid_file),
        "endpoint": f"http://{host}:{port}",
        "healthy": healthy,
        "health_data": health_data,
    }
