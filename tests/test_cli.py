"""Tests for agy-acpbox CLI."""

import os
import subprocess
import sys


def test_cli_help():
    """Validates that agy-acpbox --help displays gateway options including background & autostart."""
    env = {**os.environ, "COLUMNS": "200"}
    res = subprocess.run([sys.executable, "-m", "acpbox", "--help"], capture_output=True, text=True, env=env)
    assert res.returncode == 0
    assert "--host" in res.stdout
    assert "--port" in res.stdout
    assert "--disable-telemetry" in res.stdout
    assert "--dangerously-skip-permissions" in res.stdout
    assert "--agent" in res.stdout
    assert "--workspace" in res.stdout
    assert "--add-dir" in res.stdout
    assert "--background" in res.stdout or "-d" in res.stdout
    assert "--stop" in res.stdout
    assert "--status" in res.stdout
    assert "--install-autostart" in res.stdout
    assert "--uninstall-autostart" in res.stdout
    assert "--autostart-status" in res.stdout
    assert "--pid-file" in res.stdout
    assert "--log-file" in res.stdout


def test_cli_rejects_unknown_flags():
    """Validates that unknown flags are rejected."""
    res = subprocess.run([sys.executable, "-m", "acpbox", "--unknown-flag"], capture_output=True, text=True)
    assert res.returncode != 0


def test_cli_status_when_stopped(tmp_path):
    """Validates that --status returns exit code 1 when no server is running."""
    dummy_pid_file = tmp_path / "nonexistent.pid"
    res = subprocess.run(
        [sys.executable, "-m", "acpbox", "--status", "--pid-file", str(dummy_pid_file)],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 1
    assert "not running" in res.stdout or "STOPPED" in res.stdout


def test_cli_stop_when_not_running(tmp_path):
    """Validates that --stop reports an error when no server is running."""
    dummy_pid_file = tmp_path / "nonexistent.pid"
    res = subprocess.run(
        [sys.executable, "-m", "acpbox", "--stop", "--pid-file", str(dummy_pid_file)],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 1
    assert "No PID file found" in res.stderr


def test_cli_background_lifecycle(tmp_path):
    """Validates CLI flags --background, --status, and --stop in an end-to-end flow."""
    port = 8966
    pid_file = tmp_path / "cli_bg.pid"
    log_file = tmp_path / "cli_bg.log"

    # 1. Start background daemon via CLI
    start_res = subprocess.run(
        [
            sys.executable, "-m", "acpbox",
            "--background",
            "--port", str(port),
            "--host", "127.0.0.1",
            "--pid-file", str(pid_file),
            "--log-file", str(log_file),
        ],
        capture_output=True,
        text=True,
    )
    assert start_res.returncode == 0
    assert "[OK]" in start_res.stdout
    assert "started in background" in start_res.stdout

    try:
        # 2. Check status via CLI
        status_res = subprocess.run(
            [
                sys.executable, "-m", "acpbox",
                "--status",
                "--port", str(port),
                "--host", "127.0.0.1",
                "--pid-file", str(pid_file),
            ],
            capture_output=True,
            text=True,
        )
        assert status_res.returncode == 0
        assert "[RUNNING]" in status_res.stdout
        assert "healthy" in status_res.stdout

    finally:
        # 3. Stop via CLI
        stop_res = subprocess.run(
            [
                sys.executable, "-m", "acpbox",
                "--stop",
                "--pid-file", str(pid_file),
            ],
            capture_output=True,
            text=True,
        )
        assert stop_res.returncode == 0
        assert "[OK]" in stop_res.stdout
