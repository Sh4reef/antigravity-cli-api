"""Tests for agy-acpbox daemon process manager."""

import os
from pathlib import Path
import pytest

from acpbox.daemon import (
    check_status,
    is_process_alive,
    read_pid,
    remove_pid,
    start_daemon,
    stop_daemon,
    write_pid,
)


def test_pid_file_helpers(tmp_path: Path):
    """Test reading, writing, and removing PID file."""
    pid_file = tmp_path / "test.pid"
    assert read_pid(pid_file) is None

    write_pid(pid_file, 12345)
    assert read_pid(pid_file) == 12345

    remove_pid(pid_file)
    assert read_pid(pid_file) is None
    # Verify idempotent removal
    remove_pid(pid_file)


def test_is_process_alive():
    """Test process liveness detection."""
    assert is_process_alive(os.getpid()) is True
    # Negative PID or impossible PID
    assert is_process_alive(-1) is False
    assert is_process_alive(9999999) is False


def test_daemon_lifecycle_background_run(tmp_path: Path):
    """Test launching, querying status, and stopping a background daemon instance."""
    port = 8943
    pid_file = tmp_path / "daemon.pid"
    log_file = tmp_path / "daemon.log"

    # 1. Start daemon
    success, msg, pid = start_daemon(
        cmd_args=[
            "--port", str(port),
            "--host", "127.0.0.1",
            "--pid-file", str(pid_file),
            "--log-file", str(log_file),
        ],
        log_file=log_file,
        pid_file=pid_file,
        host="127.0.0.1",
        port=port,
        healthcheck_timeout=6.0,
    )

    assert success is True
    assert pid is not None
    assert is_process_alive(pid) is True
    assert pid_file.is_file()

    try:
        # 2. Check status
        status = check_status(pid_file, host="127.0.0.1", port=port)
        assert status["running"] is True
        assert status["pid"] == pid
        assert status["healthy"] is True

        # 3. Prevent duplicate start
        dup_success, dup_msg, dup_pid = start_daemon(
            cmd_args=["--port", str(port)],
            log_file=log_file,
            pid_file=pid_file,
            host="127.0.0.1",
            port=port,
        )
        assert dup_success is False
        assert "already running" in dup_msg
        assert dup_pid == pid

    finally:
        # 4. Stop daemon
        stop_success, stop_msg = stop_daemon(pid_file, timeout=5.0)
        assert stop_success is True
        assert not pid_file.is_file()
        assert is_process_alive(pid) is False


def test_daemon_cleans_stale_pid_file(tmp_path: Path):
    """Test that stop_daemon cleans stale PID if process already exited."""
    pid_file = tmp_path / "stale.pid"
    # Write PID that definitely does not exist
    write_pid(pid_file, 9999999)

    success, msg = stop_daemon(pid_file)
    assert success is True
    assert "Cleaned stale PID" in msg
    assert not pid_file.is_file()
