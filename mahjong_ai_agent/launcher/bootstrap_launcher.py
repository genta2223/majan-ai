"""Mahjong AI Assistant - Bootstrap Launcher & Auto-Updater.

A lightweight launcher executable (<10 MB) that checks GitHub Releases for the latest
application files and Embeddable Python runtime, downloads/extracts them if needed,
and launches the main application (`app/gui.py`).
"""

from __future__ import annotations
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import zipfile
from typing import Optional, Tuple, Dict, Any

# Target GitHub Repository configuration
GITHUB_REPO = "genta2223/majan-ai"
GITHUB_API_LATEST = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
DEFAULT_ASSET_NAME = "majan-ai-core.zip"


def get_base_dir() -> str:
    """Returns the base directory where this executable / script is located."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


BASE_DIR = get_base_dir()
RUNTIME_DIR = os.path.join(BASE_DIR, "runtime")
APP_DIR = os.path.join(BASE_DIR, "app")
VERSION_FILE = os.path.join(BASE_DIR, "version.txt")
TEMP_ZIP = os.path.join(BASE_DIR, "_update_temp.zip")


def get_local_version() -> str:
    """Reads the installed version tag from version.txt if present."""
    if os.path.exists(VERSION_FILE):
        try:
            with open(VERSION_FILE, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception:
            pass
    return ""


def set_local_version(tag: str) -> None:
    """Saves the installed version tag into version.txt."""
    try:
        with open(VERSION_FILE, "w", encoding="utf-8") as f:
            f.write(tag.strip())
    except Exception as e:
        print(f"[Launcher] Warning: Could not save version.txt: {e}")


def is_local_ready() -> bool:
    """Checks whether the Python runtime and application entrypoint exist locally."""
    python_exe = os.path.join(RUNTIME_DIR, "python.exe")
    # Also support runtime/bin/python or standard windows path
    entry_gui = os.path.join(APP_DIR, "gui.py")
    entry_main = os.path.join(APP_DIR, "main.py")
    return os.path.isfile(python_exe) and (os.path.isfile(entry_gui) or os.path.isfile(entry_main))


def fetch_latest_release_info(timeout: float = 6.0) -> Optional[Dict[str, Any]]:
    """
    Queries GitHub Releases API for the latest release metadata.
    Returns None if offline or rate limited.
    """
    try:
        req = urllib.request.Request(
            GITHUB_API_LATEST,
            headers={
                "User-Agent": "MajanAI-Launcher/1.0",
                "Accept": "application/vnd.github.v3+json",
            }
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                raw_data = resp.read().decode("utf-8")
                return json.loads(raw_data)
    except Exception as e:
        print(f"[Launcher] GitHub Releases check failed ({e}). Running in offline/fallback mode.")
    return None


def find_download_asset(release_data: Dict[str, Any]) -> Tuple[Optional[str], int, str]:
    """
    Finds the appropriate ZIP asset download URL, size in bytes, and asset name.
    """
    assets = release_data.get("assets", [])
    # 1. Exact match for majan-ai-core.zip
    for asset in assets:
        name = asset.get("name", "")
        if name.lower() == DEFAULT_ASSET_NAME.lower():
            return asset.get("browser_download_url"), asset.get("size", 0), name

    # 2. Match any asset ending in .zip containing 'core' or 'majan'
    for asset in assets:
        name = asset.get("name", "")
        if name.lower().endswith(".zip") and ("core" in name.lower() or "majan" in name.lower()):
            return asset.get("browser_download_url"), asset.get("size", 0), name

    # 3. Fallback: Any zip asset
    for asset in assets:
        name = asset.get("name", "")
        if name.lower().endswith(".zip"):
            return asset.get("browser_download_url"), asset.get("size", 0), name

    # 4. Fallback: zipball_url
    zipball = release_data.get("zipball_url")
    if zipball:
        return zipball, 0, "source.zip"

    return None, 0, ""


def launch_application() -> bool:
    """
    Executes the main application using the local Embeddable Python runtime.
    Sets PYTHONPATH to app directory to guarantee smooth module discovery.
    """
    python_exe = os.path.join(RUNTIME_DIR, "python.exe")
    if not os.path.isfile(python_exe):
        # Fallback to system python if runtime/python.exe missing
        python_exe = sys.executable

    entry_point = os.path.join(APP_DIR, "gui.py")
    if not os.path.isfile(entry_point):
        entry_point = os.path.join(APP_DIR, "main.py")

    if not os.path.isfile(entry_point):
        return False

    env = os.environ.copy()
    env["PYTHONPATH"] = APP_DIR + (os.pathsep + env.get("PYTHONPATH", "") if "PYTHONPATH" in env else "")

    cmd = [python_exe, entry_point]
    print(f"[Launcher] Launching application: {' '.join(cmd)}")

    try:
        # Launch detached so the launcher can exit cleanly
        creationflags = 0
        if sys.platform == "win32":
            # DETACHED_PROCESS = 0x00000008, CREATE_NEW_PROCESS_GROUP = 0x00000200
            creationflags = 0x00000008 | 0x00000200

        subprocess.Popen(
            cmd,
            cwd=APP_DIR if os.path.exists(APP_DIR) else BASE_DIR,
            env=env,
            creationflags=creationflags,
            close_fds=True if sys.platform != "win32" else False,
        )
        return True
    except Exception as e:
        print(f"[Launcher] Error starting application: {e}")
        return False


class LauncherUI:
    """
    Lightweight Tkinter GUI displaying download progress and status messages.
    Falls back gracefully to stdout if graphical environment is unavailable.
    """

    def __init__(self):
        self.root = None
        self.lbl_status = None
        self.lbl_progress = None
        self.progress_bar = None
        self.is_cancelled = False
        self.use_gui = True

        try:
            import tkinter as tk
            from tkinter import ttk

            self.tk = tk
            self.ttk = ttk
            self._init_window()
        except Exception as e:
            print(f"[Launcher] Tkinter unavailable ({e}). Using console progress.")
            self.use_gui = False

    def _init_window(self):
        self.root = self.tk.Tk()
        self.root.title("麻雀AI アシスタント - ランチャー")
        self.root.geometry("460x180")
        self.root.resizable(False, False)

        # Center on screen
        self.root.eval('tk::PlaceWindow . center')

        # Modern styling
        bg_color = "#1e1e2e"
        fg_color = "#cdd6f4"
        accent_color = "#89b4fa"

        self.root.configure(bg=bg_color)

        style = self.ttk.Style()
        style.theme_use('clam')
        style.configure("Custom.Horizontal.TProgressbar",
                        background=accent_color,
                        troughcolor="#313244",
                        bordercolor=bg_color,
                        lightcolor=accent_color,
                        darkcolor=accent_color)

        title_lbl = self.tk.Label(
            self.root,
            text="🀄 麻雀AI アシスタント 起動システム",
            font=("Meiryo", 12, "bold"),
            bg=bg_color,
            fg=accent_color,
        )
        title_lbl.pack(pady=(15, 5))

        self.lbl_status = self.tk.Label(
            self.root,
            text="環境をチェックしています...",
            font=("Meiryo", 9),
            bg=bg_color,
            fg=fg_color,
            wraplength=420,
        )
        self.lbl_status.pack(pady=5)

        self.progress_bar = self.ttk.Progressbar(
            self.root,
            style="Custom.Horizontal.TProgressbar",
            orient="horizontal",
            length=400,
            mode="determinate",
        )
        self.progress_bar.pack(pady=5)

        self.lbl_progress = self.tk.Label(
            self.root,
            text="準備中...",
            font=("Meiryo", 8),
            bg=bg_color,
            fg="#a6adc8",
        )
        self.lbl_progress.pack(pady=(0, 10))

        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _on_closing(self):
        self.is_cancelled = True
        if self.root:
            self.root.destroy()
        sys.exit(0)

    def set_status(self, text: str):
        print(f"[Launcher] {text}")
        if self.use_gui and self.lbl_status and self.root:
            self.lbl_status.config(text=text)
            self.root.update_idletasks()

    def set_progress(self, percent: float, detail: str = ""):
        if self.use_gui and self.progress_bar and self.root:
            self.progress_bar["value"] = percent
            if detail and self.lbl_progress:
                self.lbl_progress.config(text=detail)
            self.root.update_idletasks()
        else:
            if int(percent) % 20 == 0:
                print(f"[Launcher] Progress: {percent:.1f}% ({detail})")

    def show_error(self, title: str, message: str):
        print(f"[Launcher ERROR] {title}: {message}")
        if self.use_gui:
            from tkinter import messagebox
            messagebox.showerror(title, message)
        else:
            input("Press Enter to exit...")

    def close(self):
        if self.use_gui and self.root:
            try:
                self.root.destroy()
            except Exception:
                pass


def download_with_progress(url: str, dest_path: str, ui: LauncherUI, total_size: int = 0) -> bool:
    """
    Streams file download with realtime progress updates to the UI.
    """
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "MajanAI-Launcher/1.0"}
        )
        with urllib.request.urlopen(req, timeout=30.0) as resp:
            content_length = resp.headers.get("Content-Length")
            if content_length:
                total_size = int(content_length)

            chunk_size = 64 * 1024  # 64 KB
            downloaded = 0
            start_time = time.time()

            with open(dest_path, "wb") as f:
                while True:
                    if ui.is_cancelled:
                        return False
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    elapsed = max(time.time() - start_time, 0.001)
                    speed_mb = (downloaded / (1024 * 1024)) / elapsed

                    if total_size > 0:
                        pct = (downloaded / total_size) * 100.0
                        detail = f"{downloaded / (1024*1024):.1f}MB / {total_size / (1024*1024):.1f}MB ({pct:.1f}%) - {speed_mb:.2f}MB/s"
                    else:
                        pct = 50.0
                        detail = f"{downloaded / (1024*1024):.1f}MB ダウンロード完了 - {speed_mb:.2f}MB/s"

                    ui.set_progress(pct, detail)

            return True
    except Exception as e:
        ui.set_status(f"ダウンロード失敗: {e}")
        return False


def extract_zip(zip_path: str, extract_to: str, ui: LauncherUI) -> bool:
    """
    Extracts the downloaded ZIP archive into the destination directory.
    """
    try:
        ui.set_status("パッケージを展開中...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            namelist = zf.namelist()
            total_files = len(namelist)

            for idx, member in enumerate(namelist):
                if ui.is_cancelled:
                    return False
                zf.extract(member, extract_to)
                if idx % 20 == 0:
                    pct = (idx / total_files) * 100.0
                    ui.set_progress(pct, f"展開中: {idx}/{total_files} ファイル")

        ui.set_progress(100.0, "展開完了")
        return True
    except Exception as e:
        ui.set_status(f"展開エラー: {e}")
        return False


def run_bootstrap_flow(ui: LauncherUI):
    """
    Executes the main bootstrap workflow:
    1. Check local environment.
    2. Query GitHub Releases for updates.
    3. Download and extract if new version or missing.
    4. Start application.
    """
    local_ready = is_local_ready()
    local_version = get_local_version()

    ui.set_status("最新バージョンを確認しています...")
    release_info = fetch_latest_release_info(timeout=5.0)

    need_download = False
    target_tag = ""
    download_url = None
    expected_size = 0

    if release_info:
        target_tag = release_info.get("tag_name", "")
        download_url, expected_size, asset_name = find_download_asset(release_info)

        if not local_ready:
            print(f"[Launcher] Local runtime/app missing. Initial download required.")
            need_download = True
        elif target_tag and local_version != target_tag:
            print(f"[Launcher] Update available: Local {local_version or 'unknown'} -> Remote {target_tag}")
            need_download = True
        else:
            print(f"[Launcher] Local version {local_version} is up to date.")
    else:
        # Offline or GitHub API unavailable
        if local_ready:
            ui.set_status("オフラインモード: 既存のローカル環境で起動します。")
            time.sleep(1.0)
        else:
            ui.show_error(
                "接続エラー",
                "初回起動に必要なファイルが見つかりません。\n"
                "インターネット接続を確認してから再度起動してください。"
            )
            ui.close()
            return

    if need_download:
        if not download_url:
            if local_ready:
                ui.set_status("最新リリースアセットが見つかりません。既存環境で起動します。")
            else:
                ui.show_error(
                    "リリース取得失敗",
                    f"GitHub リポジトリ ({GITHUB_REPO}) に配布用ZIPが見つかりませんでした。\n"
                    "管理者に確認してください。"
                )
                ui.close()
                return
        else:
            ui.set_status(f"最新バージョン ({target_tag}) をダウンロード中...")
            success = download_with_progress(download_url, TEMP_ZIP, ui, total_size=expected_size)
            if not success:
                if local_ready:
                    ui.set_status("ダウンロードに失敗しました。既存環境で起動します。")
                else:
                    ui.show_error("ダウンロード失敗", "最新データの取得に失敗しました。")
                    ui.close()
                    return
            else:
                # Extract archive
                extract_ok = extract_zip(TEMP_ZIP, BASE_DIR, ui)
                # Cleanup temp file
                if os.path.exists(TEMP_ZIP):
                    try:
                        os.remove(TEMP_ZIP)
                    except Exception:
                        pass

                if extract_ok:
                    set_local_version(target_tag)
                    ui.set_status("更新が正常に完了しました！")
                else:
                    if not local_ready:
                        ui.show_error("展開失敗", "ZIPファイルの展開に失敗しました。")
                        ui.close()
                        return

    # Ready to launch
    ui.set_status("🀄 麻雀AI アシスタントを起動しています...")
    time.sleep(0.8)
    started = launch_application()
    if not started:
        ui.show_error("起動エラー", "アプリケーション (`app/gui.py`) の起動に失敗しました。")

    ui.close()


def main():
    ui = LauncherUI()

    # Run the bootstrap worker in a background thread while Tkinter runs on main thread
    if ui.use_gui:
        worker_thread = threading.Thread(target=run_bootstrap_flow, args=(ui,), daemon=True)
        worker_thread.start()
        ui.root.mainloop()
    else:
        run_bootstrap_flow(ui)


if __name__ == "__main__":
    main()
