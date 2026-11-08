"""Tests for agy-acpbox CLI."""

import subprocess
import sys


def test_cli_help():
    """Validates that agy-acpbox --help displays gateway options."""
    res = subprocess.run([sys.executable, "-m", "acpbox", "--help"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "--host" in res.stdout
    assert "--port" in res.stdout
    assert "--disable-telemetry" in res.stdout
    assert "--dangerously-skip-permissions" in res.stdout
    assert "--agent" in res.stdout
    assert "--workspace" in res.stdout
    assert "--add-dir" in res.stdout


def test_cli_rejects_unknown_flags():
    """Validates that unknown flags are rejected."""
    res = subprocess.run([sys.executable, "-m", "acpbox", "--unknown-flag"], capture_output=True, text=True)
    assert res.returncode != 0
