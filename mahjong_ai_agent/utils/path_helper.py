"""Path helper utility for robust cross-environment resource resolution.

Seamlessly handles path resolution for:
1. Standard Python development environment (relative to project root).
2. PyInstaller bundled executable environment (using sys._MEIPASS).
3. External writable file storage (logs, output, calibration) next to the executable.
"""

from __future__ import annotations
import os
import sys
from typing import Optional


def is_frozen() -> bool:
    """Returns True if running inside a PyInstaller or frozen binary environment."""
    return getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")


def get_base_dir() -> str:
    """
    Returns the read-only base directory for bundled static assets and configurations.
    In frozen environment: sys._MEIPASS (temporary unpacked directory).
    In dev environment: mahjong_ai_agent root directory.
    """
    if is_frozen():
        return getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    # Dev: utils/.. -> mahjong_ai_agent
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def get_app_dir() -> str:
    """
    Returns the writable directory where the executable resides or dev project root.
    Use this for writing persistent files like output/, logs, and user configs.
    """
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def get_resource_path(*rel_path: str) -> str:
    """
    Resolves static bundled resource path (e.g. assets, default config).
    """
    joined = os.path.join(*rel_path) if rel_path else ""
    if os.path.isabs(joined):
        return joined
    return os.path.normpath(os.path.join(get_base_dir(), joined))


def get_writable_path(*rel_path: str, ensure_parent: bool = True) -> str:
    """
    Resolves a writable path next to the application binary or project root.
    Optionally ensures the parent directory exists.
    """
    joined = os.path.join(*rel_path) if rel_path else ""
    if os.path.isabs(joined):
        full_path = joined
    else:
        full_path = os.path.normpath(os.path.join(get_app_dir(), joined))

    if ensure_parent:
        parent = os.path.dirname(full_path)
        if parent and not os.path.exists(parent):
            try:
                os.makedirs(parent, exist_ok=True)
            except Exception:
                pass
    return full_path


def get_config_path(filename: str = "strategy_config.yaml") -> str:
    """
    Finds configuration file:
    1. First checks for user-customized config in writable app directory (<app_dir>/config/<filename>).
    2. Falls back to bundled default configuration (<_MEIPASS>/config/<filename>).
    """
    user_config = get_writable_path("config", filename, ensure_parent=False)
    if os.path.isfile(user_config):
        return user_config
    bundled_config = get_resource_path("config", filename)
    if os.path.isfile(bundled_config):
        return bundled_config
    # Fallback to user_config path even if not yet created
    return user_config


def get_assets_path(*subpaths: str) -> str:
    """Returns absolute path to static assets directory or a file within it."""
    return get_resource_path("assets", *subpaths)


def get_output_path(*subpaths: str) -> str:
    """Returns absolute path to writable output directory or a file within it."""
    path = get_writable_path("output", *subpaths, ensure_parent=True)
    return path
