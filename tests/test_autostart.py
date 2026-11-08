"""Tests for agy-acpbox autostart service generator and manager."""

from pathlib import Path
import pytest

from acpbox.autostart import (
    detect_platform,
    generate_linux_service,
    generate_macos_plist,
    get_service_status,
)


def test_detect_platform():
    """Verify platform detection returns supported string."""
    plat = detect_platform()
    assert plat in ("macos", "linux", "unsupported")


def test_generate_macos_plist(tmp_path: Path):
    """Verify macOS launchd plist structure."""
    log_file = tmp_path / "test.log"
    plist_xml = generate_macos_plist(
        python_bin="/usr/local/bin/python3",
        args=["--host", "127.0.0.1", "--port", "8000"],
        log_file=log_file,
        env_path="/opt/bin:/usr/bin",
    )

    assert "<?xml version=" in plist_xml
    assert "<key>Label</key>" in plist_xml
    assert "<string>com.antigravity.agy-acpbox</string>" in plist_xml
    assert "<key>RunAtLoad</key>" in plist_xml
    assert "<true/>" in plist_xml
    assert "<key>KeepAlive</key>" in plist_xml
    assert "<string>/usr/local/bin/python3</string>" in plist_xml
    assert "<string>acpbox</string>" in plist_xml
    assert "<string>--port</string>" in plist_xml
    assert "<string>8000</string>" in plist_xml
    assert str(log_file) in plist_xml
    assert "/opt/bin:/usr/bin" in plist_xml


def test_generate_linux_service(tmp_path: Path):
    """Verify Linux systemd unit file structure."""
    log_file = tmp_path / "test.log"
    service_ini = generate_linux_service(
        python_bin="/usr/bin/python3",
        args=["--host", "0.0.0.0", "--port", "8080"],
        log_file=log_file,
        env_path="/usr/local/bin:/usr/bin",
    )

    assert "[Unit]" in service_ini
    assert "[Service]" in service_ini
    assert "ExecStart=/usr/bin/python3 -m acpbox --host 0.0.0.0 --port 8080" in service_ini
    assert "Restart=always" in service_ini
    assert f"StandardOutput=append:{log_file}" in service_ini
    assert 'Environment="PATH=/usr/local/bin:/usr/bin"' in service_ini
    assert "WantedBy=default.target" in service_ini


def test_get_service_status():
    """Verify get_service_status returns expected keys."""
    st = get_service_status()
    assert "platform" in st
    assert "mechanism" in st
    assert "installed" in st
    assert "loaded" in st
