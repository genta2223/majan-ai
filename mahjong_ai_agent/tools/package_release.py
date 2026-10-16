"""Packaging script to create GitHub Releases asset (`majan-ai-core.zip`).

Packages:
1. Windows Official Python Embeddable package (amd64) configured with site-packages
2. Minimal core dependencies (websockets, Pillow, PyYAML, pydantic, numpy, requests)
3. Latest mahjong_ai_agent application source code in `app/`
4. Release metadata `version.txt`
"""

from __future__ import annotations
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from typing import List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIST_DIR = os.path.join(PROJECT_ROOT, "dist")

# Windows Official Python Embeddable Package URL
PYTHON_EMBED_VERSION = "3.11.9"
PYTHON_EMBED_URL = f"https://www.python.org/ftp/python/{PYTHON_EMBED_VERSION}/python-{PYTHON_EMBED_VERSION}-embed-amd64.zip"
GET_PIP_URL = "https://bootstrap.pypa.io/get-pip.py"

CORE_REQUIREMENTS = [
    "websockets>=12.0",
    "Pillow>=10.0.0",
    "PyYAML>=6.0",
    "pydantic>=2.0.0",
    "numpy>=1.24.0",
]


def download_file(url: str, dest_path: str, desc: str = ""):
    """Downloads a file with basic progress display."""
    print(f"[Package] Downloading {desc or url}...")
    req = urllib.request.Request(url, headers={"User-Agent": "MajanAI-Packager/1.0"})
    with urllib.request.urlopen(req, timeout=30.0) as resp, open(dest_path, "wb") as f:
        shutil.copyfileobj(resp, f)
    print(f"[Package] Saved to {dest_path} ({os.path.getsize(dest_path) // 1024} KB)")


def setup_embed_python(target_runtime_dir: str):
    """
    Downloads and prepares official Windows Embeddable Python.
    Uncomments 'import site' in python3xx._pth to enable site-packages loading.
    """
    os.makedirs(target_runtime_dir, exist_ok=True)
    temp_zip = os.path.join(target_runtime_dir, f"python-{PYTHON_EMBED_VERSION}-embed.zip")

    if not os.path.exists(os.path.join(target_runtime_dir, "python.exe")):
        download_file(PYTHON_EMBED_URL, temp_zip, f"Python {PYTHON_EMBED_VERSION} Embeddable")
        print(f"[Package] Extracting Python embeddable into {target_runtime_dir}...")
        with zipfile.ZipFile(temp_zip, "r") as zf:
            zf.extractall(target_runtime_dir)
        try:
            os.remove(temp_zip)
        except Exception:
            pass

    # Find and patch python3xx._pth
    pth_files = [f for f in os.listdir(target_runtime_dir) if f.endswith("._pth")]
    for pth_file in pth_files:
        full_pth = os.path.join(target_runtime_dir, pth_file)
        with open(full_pth, "r", encoding="utf-8") as f:
            content = f.read()

        # Enable 'import site' and ensure Lib/site-packages and . are present
        lines = content.splitlines()
        new_lines = []
        for line in lines:
            if line.strip() == "#import site":
                new_lines.append("import site")
            else:
                new_lines.append(line)

        if "Lib/site-packages" not in new_lines:
            new_lines.append("Lib/site-packages")
        if "." not in new_lines:
            new_lines.append(".")

        with open(full_pth, "w", encoding="utf-8") as f:
            f.write("\n".join(new_lines) + "\n")
        print(f"[Package] Patched {pth_file} to enable site-packages and 'import site'.")

    # Install pip and core requirements into embed python
    python_exe = os.path.join(target_runtime_dir, "python.exe")
    site_packages = os.path.join(target_runtime_dir, "Lib", "site-packages")
    os.makedirs(site_packages, exist_ok=True)

    get_pip_path = os.path.join(target_runtime_dir, "get-pip.py")
    if not os.path.exists(os.path.join(target_runtime_dir, "Scripts", "pip.exe")):
        try:
            download_file(GET_PIP_URL, get_pip_path, "get-pip.py")
            print(f"[Package] Bootstrapping pip into embed Python...")
            subprocess.run([python_exe, get_pip_path, "--no-warn-script-location"], check=True)
            if os.path.exists(get_pip_path):
                os.remove(get_pip_path)
        except Exception as e:
            print(f"[Package] Warning: get-pip failed ({e}). Attempting direct wheel installation...")

    # Install requirements
    pip_exe = os.path.join(target_runtime_dir, "Scripts", "pip.exe")
    if os.path.exists(pip_exe):
        print(f"[Package] Installing core dependencies via pip: {CORE_REQUIREMENTS}...")
        cmd = [pip_exe, "install", "--no-warn-script-location"] + CORE_REQUIREMENTS
        subprocess.run(cmd, check=True)
    else:
        # Fallback: copy site-packages from current active environment if possible
        print("[Package] Using current Python environment packages as fallback...")
        import importlib.util
        for pkg in ["websockets", "PIL", "yaml", "pydantic", "numpy"]:
            spec = importlib.util.find_spec(pkg)
            if spec and spec.submodule_search_locations:
                src_loc = spec.submodule_search_locations[0]
                dst_loc = os.path.join(site_packages, os.path.basename(src_loc))
                if not os.path.exists(dst_loc):
                    shutil.copytree(src_loc, dst_loc)
                    print(f"[Package] Copied {pkg} into embed site-packages.")


def copy_app_source(target_app_dir: str):
    """Copies application source files into app/ excluding temp and test files."""
    os.makedirs(target_app_dir, exist_ok=True)

    include_dirs = ["core", "io_adapter", "utils", "tools", "config", "assets"]
    include_files = [
        "gui.py",
        "main.py",
        "player_profiler_dialog.py",
        "paifu_pipeline.py",
        "requirements.txt",
        "README.md",
    ]

    for d in include_dirs:
        src = os.path.join(PROJECT_ROOT, d)
        dst = os.path.join(target_app_dir, d)
        if os.path.exists(src):
            shutil.copytree(
                src,
                dst,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", "output"),
                dirs_exist_ok=True,
            )
            print(f"[Package] Bundled directory: {d}/")

    for f in include_files:
        src = os.path.join(PROJECT_ROOT, f)
        dst = os.path.join(target_app_dir, f)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"[Package] Bundled file: {f}")


def make_core_zip(staging_dir: str, output_zip_path: str):
    """Compresses the staging directory into output_zip_path."""
    print(f"[Package] Creating {output_zip_path}...")
    with zipfile.ZipFile(output_zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for root, dirs, files in os.walk(staging_dir):
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, staging_dir)
                zf.write(full_path, rel_path)

    size_mb = os.path.getsize(output_zip_path) / (1024 * 1024)
    print("=" * 60)
    print(f" PACKAGE COMPLETE: {output_zip_path}")
    print(f" Size: {size_mb:.2f} MB")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Package Mahjong AI Assistant for GitHub Releases")
    parser.add_argument("--tag", default="v1.0.0", help="Release version tag (e.g. v1.0.0)")
    parser.add_argument("--output", default="majan-ai-core.zip", help="Output ZIP file name")
    args = parser.parse_args()

    os.makedirs(DIST_DIR, exist_ok=True)
    staging_dir = os.path.join(DIST_DIR, "release_staging")
    if os.path.exists(staging_dir):
        shutil.rmtree(staging_dir, ignore_errors=True)
    os.makedirs(staging_dir, exist_ok=True)

    runtime_dir = os.path.join(staging_dir, "runtime")
    app_dir = os.path.join(staging_dir, "app")
    version_file = os.path.join(staging_dir, "version.txt")

    # 1. Version file
    with open(version_file, "w", encoding="utf-8") as f:
        f.write(args.tag.strip())

    # 2. Embeddable Python runtime
    setup_embed_python(runtime_dir)

    # 3. Application source code
    copy_app_source(app_dir)

    # 4. Create ZIP
    out_zip = os.path.join(DIST_DIR, args.output)
    make_core_zip(staging_dir, out_zip)


if __name__ == "__main__":
    main()
