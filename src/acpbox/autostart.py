"""Reboot auto-start service manager for agy-acpbox (macOS launchd & Linux systemd)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

MACOS_LABEL = "com.antigravity.agy-acpbox"
SYSTEMD_SERVICE_NAME = "agy-acpbox.service"


def detect_platform() -> Literal["macos", "linux", "unsupported"]:
    """Detect current operating system platform for autostart services."""
    if sys.platform == "darwin":
        return "macos"
    if sys.platform.startswith("linux"):
        return "linux"
    return "unsupported"


def get_macos_plist_path() -> Path:
    """Return path for user LaunchAgent plist."""
    return Path.home() / "Library" / "LaunchAgents" / f"{MACOS_LABEL}.plist"


def get_linux_service_path() -> Path:
    """Return path for user systemd service file."""
    return Path.home() / ".config" / "systemd" / "user" / SYSTEMD_SERVICE_NAME


def generate_macos_plist(
    python_bin: str,
    args: List[str],
    log_file: Path,
    env_path: Optional[str] = None,
) -> str:
    """Generate macOS launchd plist XML."""
    effective_path = env_path or os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")
    
    # ProgramArguments array
    prog_args = [python_bin, "-m", "acpbox"] + args
    prog_args_xml = "\n".join(f"        <string>{arg}</string>" for arg in prog_args)
    
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{MACOS_LABEL}</string>
    <key>ProgramArguments</key>
    <array>
{prog_args_xml}
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>{log_file}</string>
    <key>StandardErrorPath</key>
    <string>{log_file}</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>{effective_path}</string>
    </dict>
</dict>
</plist>
"""


def generate_linux_service(
    python_bin: str,
    args: List[str],
    log_file: Path,
    env_path: Optional[str] = None,
) -> str:
    """Generate Linux systemd user service unit file."""
    effective_path = env_path or os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")
    exec_args = " ".join([f'"{arg}"' if " " in arg else arg for arg in [python_bin, "-m", "acpbox"] + args])

    return f"""[Unit]
Description=agy-acpbox OpenAI Gateway for Antigravity
After=network.target

[Service]
Type=simple
ExecStart={exec_args}
Restart=always
RestartSec=5
StandardOutput=append:{log_file}
StandardError=append:{log_file}
Environment="PATH={effective_path}"

[Install]
WantedBy=default.target
"""


def install_service(
    args: List[str],
    log_file: Path,
    python_bin: Optional[str] = None,
) -> Tuple[bool, str]:
    """Install and enable user autostart service for current platform."""
    plat = detect_platform()
    py_exec = python_bin or sys.executable

    if plat == "macos":
        plist_path = get_macos_plist_path()
        plist_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Unload previous if exists
        if plist_path.is_file():
            subprocess.run(["launchctl", "unload", "-w", str(plist_path)], capture_output=True)

        content = generate_macos_plist(py_exec, args, log_file)
        plist_path.write_text(content)

        res = subprocess.run(["launchctl", "load", "-w", str(plist_path)], capture_output=True, text=True)
        if res.returncode == 0:
            return True, f"macOS LaunchAgent successfully installed and loaded at {plist_path}"
        return False, f"Failed to load LaunchAgent with launchctl: {res.stderr or res.stdout}"

    elif plat == "linux":
        service_path = get_linux_service_path()
        service_path.parent.mkdir(parents=True, exist_ok=True)

        content = generate_linux_service(py_exec, args, log_file)
        service_path.write_text(content)

        subprocess.run(["systemctl", "--user", "daemon-reload"], capture_output=True)
        res = subprocess.run(["systemctl", "--user", "enable", "--now", SYSTEMD_SERVICE_NAME], capture_output=True, text=True)
        if res.returncode == 0:
            return True, f"Linux systemd user service successfully installed and enabled at {service_path}"
        return False, f"Failed to enable systemd user service: {res.stderr or res.stdout}"

    return False, f"Autostart is not supported on this platform ({sys.platform})."


def uninstall_service() -> Tuple[bool, str]:
    """Disable and remove user autostart service for current platform."""
    plat = detect_platform()

    if plat == "macos":
        plist_path = get_macos_plist_path()
        if not plist_path.is_file():
            return True, "No macOS LaunchAgent file found to uninstall."

        subprocess.run(["launchctl", "unload", "-w", str(plist_path)], capture_output=True)
        try:
            plist_path.unlink()
        except OSError as err:
            return False, f"Failed to delete plist {plist_path}: {err}"
        return True, f"Successfully unloaded and removed LaunchAgent from {plist_path}."

    elif plat == "linux":
        service_path = get_linux_service_path()
        if not service_path.is_file():
            return True, "No Linux systemd service file found to uninstall."

        subprocess.run(["systemctl", "--user", "disable", "--now", SYSTEMD_SERVICE_NAME], capture_output=True)
        subprocess.run(["systemctl", "--user", "daemon-reload"], capture_output=True)
        try:
            service_path.unlink()
        except OSError as err:
            return False, f"Failed to delete service file {service_path}: {err}"
        return True, f"Successfully disabled and removed systemd service from {service_path}."

    return False, f"Autostart is not supported on this platform ({sys.platform})."


def get_service_status() -> Dict[str, Any]:
    """Check autostart service configuration and status."""
    plat = detect_platform()

    if plat == "macos":
        plist_path = get_macos_plist_path()
        installed = plist_path.is_file()
        loaded = False
        if installed:
            res = subprocess.run(["launchctl", "list", MACOS_LABEL], capture_output=True, text=True)
            loaded = (res.returncode == 0)

        return {
            "platform": "macos",
            "mechanism": "launchd (LaunchAgent)",
            "installed": installed,
            "loaded": loaded,
            "path": str(plist_path),
        }

    elif plat == "linux":
        service_path = get_linux_service_path()
        installed = service_path.is_file()
        active = False
        if installed:
            res = subprocess.run(["systemctl", "--user", "is-active", SYSTEMD_SERVICE_NAME], capture_output=True, text=True)
            active = (res.stdout.strip() == "active")

        return {
            "platform": "linux",
            "mechanism": "systemd (--user)",
            "installed": installed,
            "loaded": active,
            "path": str(service_path),
        }

    return {
        "platform": sys.platform,
        "mechanism": "unsupported",
        "installed": False,
        "loaded": False,
        "path": None,
    }
