"""Build script to compile bootstrap_launcher.py into a minimal standalone EXE (MajanAI_Launcher.exe).

Uses PyInstaller with strictly minimal standard library dependencies (urllib, zipfile, tkinter, json, subprocess)
to produce a tiny (<10MB) launcher executable that friends can run without installing Python.
"""

from __future__ import annotations
import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAUNCHER_SCRIPT = os.path.join(PROJECT_ROOT, "launcher", "bootstrap_launcher.py")
DIST_DIR = os.path.join(PROJECT_ROOT, "dist")
BUILD_DIR = os.path.join(PROJECT_ROOT, "build", "launcher")


def build_launcher(output_name: str = "MajanAI_Launcher", noconsole: bool = True) -> bool:
    print("=" * 60)
    print(" MajanAI - Lightweight Bootstrap Launcher Builder")
    print("=" * 60)
    print(f"Source script : {LAUNCHER_SCRIPT}")
    print(f"Target EXE    : {output_name}.exe")
    print(f"Dist dir      : {DIST_DIR}")
    print("=" * 60)

    try:
        import PyInstaller.__main__
    except ImportError:
        print("[ERROR] PyInstaller is not installed. Please install it via: pip install pyinstaller")
        return False

    args = [
        LAUNCHER_SCRIPT,
        f"--name={output_name}",
        "--clean",
        "--onefile",
        f"--distpath={DIST_DIR}",
        f"--workpath={BUILD_DIR}",
        f"--specpath={os.path.join(PROJECT_ROOT, 'launcher')}",
    ]

    if noconsole:
        args.append("--noconsole")

    # Exclude heavy optional libraries to keep the EXE footprint under 10MB
    excludes = [
        "torch",
        "scipy",
        "pandas",
        "matplotlib",
        "numpy",
        "cv2",
        "PIL",
        "pydantic",
        "websockets",
        "pytest",
        "yaml",
        "IPython",
        "pyautogui",
    ]
    for ex in excludes:
        args.append(f"--exclude-module={ex}")

    # Optional icon if present
    icon_candidate = os.path.join(PROJECT_ROOT, "assets", "app_icon.ico")
    if os.path.isfile(icon_candidate):
        args.append(f"--icon={icon_candidate}")

    print(f"[Build] Running PyInstaller with args: {args}...")
    try:
        PyInstaller.__main__.run(args)
    except Exception as e:
        print(f"[ERROR] PyInstaller build failed: {e}")
        return False

    exe_path = os.path.join(DIST_DIR, f"{output_name}.exe")
    if os.path.isfile(exe_path):
        size_mb = os.path.getsize(exe_path) / (1024 * 1024)
        print("=" * 60)
        print(" BUILD SUCCESSFUL!")
        print(f" Launcher Path: {exe_path}")
        print(f" Launcher Size: {size_mb:.2f} MB")
        print("=" * 60)
        return True
    else:
        print(f"[ERROR] Target executable not found at: {exe_path}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Build lightweight MajanAI Launcher EXE")
    parser.add_argument("--console", action="store_true", help="Keep console window visible")
    parser.add_argument("--name", default="MajanAI_Launcher", help="Output EXE name")
    args = parser.parse_args()

    success = build_launcher(output_name=args.name, noconsole=not args.console)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
