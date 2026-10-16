"""Automated PyInstaller Build Script for Mahjong AI Assistant Standalone EXE.

Builds a self-contained, standalone Windows executable (MahjongAI_Assistant.exe)
from gui.py with all bundled assets, configs, and hidden imports.
"""

from __future__ import annotations
import argparse
import os
import shutil
import sys
from typing import List

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def get_pyinstaller_args(
    entry_point: str = "gui.py",
    output_name: str = "MahjongAI_Assistant",
    noconsole: bool = True,
    onefile: bool = True,
    icon_path: str = "",
) -> List[str]:
    """Generates the argument list for PyInstaller."""
    config_dir = os.path.join(PROJECT_ROOT, "config")
    assets_dir = os.path.join(PROJECT_ROOT, "assets")

    args = [
        os.path.join(PROJECT_ROOT, entry_point),
        f"--name={output_name}",
        "--clean",
        f"--distpath={os.path.join(PROJECT_ROOT, 'dist')}",
        f"--workpath={os.path.join(PROJECT_ROOT, 'build')}",
        f"--specpath={PROJECT_ROOT}",
    ]

    # Onefile mode
    if onefile:
        args.append("--onefile")

    # Windowed mode (no console window)
    if noconsole:
        args.append("--noconsole")

    # Bundled data directories (Windows uses ';' separator)
    sep = os.pathsep
    if os.path.exists(config_dir):
        args.append(f"--add-data={config_dir}{sep}config")
    if os.path.exists(assets_dir):
        args.append(f"--add-data={assets_dir}{sep}assets")

    # Hidden imports required by dependencies and dynamic imports
    hidden_imports = [
        "asyncio",
        "websockets",
        "websockets.legacy",
        "websockets.legacy.client",
        "websockets.client",
        "pyautogui",
        "PIL",
        "PIL.Image",
        "PIL.ImageGrab",
        "PIL.ImageStat",
        "PIL.ImageDraw",
        "yaml",
        "ctypes",
        "ctypes.wintypes",
        "tkinter",
        "tkinter.ttk",
        "tkinter.messagebox",
        "tkinter.scrolledtext",
        "pydantic",
        "queue",
        "threading",
        "json",
        "urllib",
        "urllib.request",
        "utils",
        "utils.path_helper",
        "core",
        "core.engine",
        "core.state_manager",
        "core.strategy_modifier",
        "core.opponent_profiler",
        "io_adapter",
        "io_adapter.action_controller",
        "io_adapter.live_sniffer",
        "io_adapter.majsoul_bridge",
        "io_adapter.vision_hand_reader",
        "io_adapter.input_simulator",
        "io_adapter.stream_client",
        "tools.calibrate_ui",
    ]

    for hi in hidden_imports:
        args.append(f"--hidden-import={hi}")

    # Optional icon
    if icon_path and os.path.exists(icon_path):
        args.append(f"--icon={icon_path}")

    return args


def build(noconsole: bool = True, onefile: bool = True, output_name: str = "MahjongAI_Assistant") -> bool:
    """Executes the PyInstaller build process."""
    print("=" * 60)
    print(" Mahjong AI Assistant - PyInstaller Build Script")
    print("=" * 60)
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Target EXE   : {output_name}.exe")
    print(f"No Console   : {noconsole}")
    print(f"Onefile Mode : {onefile}")
    print("-" * 60)

    try:
        import PyInstaller.__main__
    except ImportError:
        print("[ERROR] PyInstaller is not installed. Please run: pip install pyinstaller")
        return False

    args = get_pyinstaller_args(
        entry_point="gui.py",
        output_name=output_name,
        noconsole=noconsole,
        onefile=onefile,
    )

    print(f"[Build] Executing PyInstaller with {len(args)} arguments...")
    try:
        PyInstaller.__main__.run(args)
    except Exception as e:
        print(f"[ERROR] PyInstaller build failed: {e}")
        return False

    exe_path = os.path.join(PROJECT_ROOT, "dist", f"{output_name}.exe")
    if os.path.isfile(exe_path):
        size_mb = os.path.getsize(exe_path) / (1024 * 1024)
        print("=" * 60)
        print(f" BUILD SUCCESSFUL!")
        print(f" Executable path: {exe_path}")
        print(f" Size: {size_mb:.2f} MB")
        print("=" * 60)
        return True
    else:
        print(f"[ERROR] Build finished but expected executable not found at: {exe_path}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Build Standalone EXE for Mahjong AI Assistant")
    parser.add_argument("--console", action="store_true", help="Keep console window visible for debugging")
    parser.add_argument("--onedir", action="store_true", help="Build as directory instead of single file")
    parser.add_argument("--name", default="MahjongAI_Assistant", help="Name of output executable")
    args = parser.parse_args()

    success = build(
        noconsole=not args.console,
        onefile=not args.onedir,
        output_name=args.name,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
