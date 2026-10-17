"""GUI Launcher & Always-on-Top HUD for Mahjong Soul AI Assist.

Provides a compact, dark-themed Tkinter interface for:
1. Launching Google Chrome in remote debugging mode.
2. Safe real-time stdout/stderr redirection and exception tracking.
3. Live status and high-visibility AI decision HUD.
4. Paifu URL importing & Opponent Profiling (Exploit Mode).
"""

import asyncio
import os
import queue
import subprocess
import sys
import threading
import time
import traceback
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
from typing import Any, Dict, List, Optional

# Add mahjong_ai_agent directory to sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from core.opponent_profiler import OpponentProfiler, PlayerProfile
from io_adapter.action_controller import format_tile_jp
from io_adapter.live_sniffer import LiveSniffer
from io_adapter.paifu_downloader import extract_paifu_uuids, parse_paifu_url
from main import MahjongAIAgent
from player_profiler_dialog import PlayerProfilerDialog
from utils.debug_tracer import trace_log, get_tracer


def find_chrome_executable() -> Optional[str]:
    """Find the path to the standard Google Chrome executable on Windows."""
    candidates = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
        os.path.expandvars(r"%PROGRAMFILES%\Google\Chrome\Application\chrome.exe"),
        os.path.expandvars(r"%PROGRAMFILES(X86)%\Google\Chrome\Application\chrome.exe"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


class TextRedirector:
    """Thread-safe stream redirector that tees output to terminal and a queue."""

    def __init__(self, log_queue: queue.Queue, is_err: bool = False, original_stream=None) -> None:
        self.log_queue = log_queue
        self.is_err = is_err
        self.original_stream = original_stream

    def write(self, s: str) -> None:
        if self.original_stream:
            try:
                self.original_stream.write(s)
                self.original_stream.flush()
            except Exception:
                pass
        if s:
            self.log_queue.put((s, self.is_err))

    def flush(self) -> None:
        if self.original_stream:
            try:
                self.original_stream.flush()
            except Exception:
                pass


class MahjongHUDApp:
    """Tkinter-based always-on-top HUD and launcher for Mahjong Soul AI."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("雀魂 AI HUD & エクスプロイトランチャー")
        self.root.geometry("450x820")
        self.root.minsize(420, 750)
        self.root.configure(bg="#1a1c23")

        # Topmost flag
        self.is_topmost = tk.BooleanVar(value=True)
        self.root.attributes("-topmost", True)

        # Logging queue
        self.log_queue: queue.Queue = queue.Queue()
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        sys.stdout = TextRedirector(self.log_queue, is_err=False, original_stream=self._orig_stdout)
        sys.stderr = TextRedirector(self.log_queue, is_err=True, original_stream=self._orig_stderr)

        # Assist state
        self.output_mode_var = tk.StringVar(value="text_only")
        self.port_var = tk.StringVar(value="9222")
        self.is_exploit_mode = tk.BooleanVar(value=False)
        self.is_running = False

        self.agent_thread: Optional[threading.Thread] = None
        self.agent_instance: Optional[MahjongAIAgent] = None
        self.sniffer_instance: Optional[LiveSniffer] = None
        self.agent_loop: Optional[asyncio.AbstractEventLoop] = None

        # Opponent profiler & paifu sniffer state
        self.active_profiler = OpponentProfiler()
        self.active_custom_exploit: Optional[Dict[str, Any]] = None
        self.paifu_sniffer: Optional[Any] = None

        self._init_styles()
        self._build_ui()
        self._poll_log_queue()
        self._start_watchdog()

    def _init_styles(self) -> None:
        """Initialize color palette."""
        self.c_bg = "#1a1c23"
        self.c_card = "#242731"
        self.c_card_border = "#333846"
        self.c_text = "#e2e8f0"
        self.c_text_dim = "#94a3b8"
        self.c_accent = "#38bdf8"
        self.c_green = "#4ade80"
        self.c_yellow = "#facc15"
        self.c_red = "#f87171"
        self.c_purple = "#c084fc"
        self.c_btn = "#3b82f6"
        self.c_btn_stop = "#ef4444"

    def _build_ui(self) -> None:
        """Construct the HUD interface."""
        # Top Header Bar
        header = tk.Frame(self.root, bg=self.c_bg, padx=12, pady=6)
        header.pack(fill=tk.X)

        title_lbl = tk.Label(
            header,
            text="🀄 雀魂 AI ASSIST HUD",
            font=("Meiryo", 12, "bold"),
            fg=self.c_text,
            bg=self.c_bg,
        )
        title_lbl.pack(side=tk.LEFT)

        topmost_chk = tk.Checkbutton(
            header,
            text="常に最前面",
            variable=self.is_topmost,
            command=self._toggle_topmost,
            font=("Meiryo", 9),
            fg=self.c_accent,
            bg=self.c_bg,
            selectcolor=self.c_card,
            activebackground=self.c_bg,
            activeforeground=self.c_accent,
        )
        topmost_chk.pack(side=tk.RIGHT)

        # Control Panel Frame
        ctrl_frame = tk.LabelFrame(
            self.root,
            text=" ⚙ 起動・制御 ",
            font=("Meiryo", 9, "bold"),
            fg=self.c_text_dim,
            bg=self.c_card,
            padx=10,
            pady=6,
        )
        ctrl_frame.pack(fill=tk.X, padx=12, pady=3)

        # Row 1: Chrome launch button & port
        row1 = tk.Frame(ctrl_frame, bg=self.c_card)
        row1.pack(fill=tk.X, pady=2)

        self.btn_chrome = tk.Button(
            row1,
            text="🚀 じゃんたまを起動",
            command=self._launch_chrome,
            font=("Meiryo", 9, "bold"),
            bg="#2563eb",
            fg="#ffffff",
            activebackground="#1d4ed8",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=8,
            pady=4,
            cursor="hand2",
        )
        self.btn_chrome.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))

        port_lbl = tk.Label(
            row1, text="Port:", font=("Meiryo", 8), fg=self.c_text_dim, bg=self.c_card
        )
        port_lbl.pack(side=tk.LEFT)
        port_entry = tk.Entry(
            row1,
            textvariable=self.port_var,
            width=5,
            font=("Meiryo", 8),
            bg=self.c_bg,
            fg=self.c_text,
            insertbackground=self.c_text,
            relief=tk.FLAT,
        )
        port_entry.pack(side=tk.LEFT, padx=(2, 0))

        # Row 2: Mode selection radiobuttons
        row2 = tk.Frame(ctrl_frame, bg=self.c_card)
        row2.pack(fill=tk.X, pady=4)

        mode_lbl = tk.Label(
            row2, text="動作モード:", font=("Meiryo", 9), fg=self.c_text_dim, bg=self.c_card
        )
        mode_lbl.pack(side=tk.LEFT, padx=(0, 6))

        rb_text = tk.Radiobutton(
            row2,
            text="テキストのみ (安全)",
            variable=self.output_mode_var,
            value="text_only",
            font=("Meiryo", 9),
            fg=self.c_green,
            bg=self.c_card,
            selectcolor=self.c_bg,
            activebackground=self.c_card,
            activeforeground=self.c_green,
        )
        rb_text.pack(side=tk.LEFT, padx=(0, 6))

        rb_mouse = tk.Radiobutton(
            row2,
            text="半自動マウス誘導",
            variable=self.output_mode_var,
            value="semi_auto",
            font=("Meiryo", 9),
            fg=self.c_yellow,
            bg=self.c_card,
            selectcolor=self.c_bg,
            activebackground=self.c_card,
            activeforeground=self.c_yellow,
        )
        rb_mouse.pack(side=tk.LEFT)

        # Row 3: Assist Toggle Button
        self.btn_assist = tk.Button(
            ctrl_frame,
            text="▶ アシスト開始",
            command=self._toggle_assist,
            font=("Meiryo", 10, "bold"),
            bg="#059669",
            fg="#ffffff",
            activebackground="#047857",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            pady=4,
            cursor="hand2",
        )
        self.btn_assist.pack(fill=tk.X, pady=(2, 0))

        # Status Display Frame
        status_frame = tk.Frame(self.root, bg=self.c_bg, padx=12, pady=2)
        status_frame.pack(fill=tk.X)

        self.lbl_conn_status = tk.Label(
            status_frame,
            text="● 待機中 (Chrome未接続)",
            font=("Meiryo", 9, "bold"),
            fg=self.c_text_dim,
            bg=self.c_bg,
            anchor=tk.W,
        )
        self.lbl_conn_status.pack(fill=tk.X)

        self.lbl_round_info = tk.Label(
            status_frame,
            text="対局情報: 待機中",
            font=("Meiryo", 9),
            fg=self.c_text_dim,
            bg=self.c_bg,
            anchor=tk.W,
        )
        self.lbl_round_info.pack(fill=tk.X)

        # AI Recommendation Box (Hero Area)
        rec_card = tk.Frame(
            self.root,
            bg=self.c_card,
            highlightbackground=self.c_card_border,
            highlightthickness=1,
            padx=12,
            pady=8,
        )
        rec_card.pack(fill=tk.X, padx=12, pady=4)

        rec_header = tk.Label(
            rec_card,
            text="AI 推奨アクション",
            font=("Meiryo", 9, "bold"),
            fg=self.c_accent,
            bg=self.c_card,
        )
        rec_header.pack(anchor=tk.W)

        self.lbl_rec_main = tk.Label(
            rec_card,
            text="対局開始を待機しています...",
            font=("Meiryo", 14, "bold"),
            fg=self.c_text,
            bg=self.c_card,
            pady=6,
            wraplength=410,
        )
        self.lbl_rec_main.pack(fill=tk.X)

        self.lbl_rec_sub = tk.Label(
            rec_card,
            text="期待値: -- | 向聴数: --",
            font=("Meiryo", 9),
            fg=self.c_text_dim,
            bg=self.c_card,
        )
        self.lbl_rec_sub.pack(fill=tk.X)

        self.lbl_rec_detail = tk.Label(
            rec_card,
            text="",
            font=("Consolas", 8),
            fg=self.c_text_dim,
            bg=self.c_card,
            wraplength=410,
            justify=tk.LEFT,
        )
        self.lbl_rec_detail.pack(fill=tk.X, pady=(2, 0))

        # Paifu Packet Sniffer & Exploit Adaptation Section
        exploit_frame = tk.LabelFrame(
            self.root,
            text=" 📊 牌譜パケットスニッファー & 相手癖分析 ",
            font=("Meiryo", 9, "bold"),
            fg=self.c_purple,
            bg=self.c_card,
            padx=10,
            pady=6,
        )
        exploit_frame.pack(fill=tk.X, padx=12, pady=4)

        self.lbl_sniffer_guide = tk.Label(
            exploit_frame,
            text="雀魂の牌譜画面で「詳細」をクリックするとパケットを直接キャプチャします（最大3件）。",
            font=("Meiryo", 8),
            fg=self.c_text_dim,
            bg=self.c_card,
            anchor=tk.W,
            wraplength=410,
            justify=tk.LEFT,
        )
        self.lbl_sniffer_guide.pack(fill=tk.X)

        exp_btn_row = tk.Frame(exploit_frame, bg=self.c_card)
        exp_btn_row.pack(fill=tk.X, pady=(4, 2))
        exp_btn_row.columnconfigure(0, weight=1)
        exp_btn_row.columnconfigure(1, weight=1)

        self.btn_sniffer = tk.Button(
            exp_btn_row,
            text="📡 取り込み開始",
            command=self._toggle_sniffer,
            font=("Meiryo", 8, "bold"),
            bg="#7c3aed",
            fg="#ffffff",
            activebackground="#6d28d9",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=4,
            pady=3,
            cursor="hand2",
        )
        self.btn_sniffer.grid(row=0, column=0, sticky="ew", padx=(0, 3))

        self.btn_finish_sniffer = tk.Button(
            exp_btn_row,
            text="👥 完了・分析",
            command=self._finish_and_show_profiler,
            font=("Meiryo", 8, "bold"),
            bg="#2563eb",
            fg="#ffffff",
            activebackground="#1d4ed8",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=4,
            pady=3,
            cursor="hand2",
        )
        self.btn_finish_sniffer.grid(row=0, column=1, sticky="ew", padx=(3, 0))

        self.chk_exploit = tk.Checkbutton(
            exp_btn_row,
            text="癖エクスプロイト",
            variable=self.is_exploit_mode,
            font=("Meiryo", 8, "bold"),
            fg=self.c_purple,
            bg=self.c_card,
            selectcolor=self.c_bg,
            activebackground=self.c_card,
            activeforeground=self.c_purple,
        )
        self.chk_exploit.grid(row=1, column=0, columnspan=2, sticky="w", pady=(3, 0))

        self.lbl_exploit_summary = tk.Label(
            exploit_frame,
            text="待機中 (「牌譜取り込み準備」を押して監視を開始してください)",
            font=("Meiryo", 8),
            fg=self.c_text_dim,
            bg=self.c_card,
            anchor=tk.W,
            wraplength=410,
            justify=tk.LEFT,
        )
        self.lbl_exploit_summary.pack(fill=tk.X, pady=(2, 0))

        # Real-time Log Area
        log_frame = tk.LabelFrame(
            self.root,
            text=" 📜 リアルタイムログ ",
            font=("Meiryo", 9),
            fg=self.c_text_dim,
            bg=self.c_card,
            padx=6,
            pady=4,
        )
        log_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 6))

        self.txt_log = scrolledtext.ScrolledText(
            log_frame,
            font=("Consolas", 8),
            bg=self.c_bg,
            fg="#94a3b8",
            insertbackground="#ffffff",
            relief=tk.FLAT,
            height=6,
        )
        self.txt_log.pack(fill=tk.BOTH, expand=True)

        # Text tags for colorized logs
        self.txt_log.tag_config("error", foreground="#f87171")
        self.txt_log.tag_config("highlight", foreground="#38bdf8")
        self.txt_log.tag_config("rec", foreground="#4ade80")
        self.txt_log.tag_config("exploit", foreground="#c084fc")

        self.txt_log.insert(tk.END, "雀魂 AI HUD & エクスプロイトランチャーを起動しました。\n")

    def _toggle_topmost(self) -> None:
        """Toggle always-on-top window attribute."""
        self.root.attributes("-topmost", self.is_topmost.get())

    def _poll_log_queue(self) -> None:
        """Drain the log queue periodically to keep the UI snappy."""
        try:
            while True:
                msg, is_err = self.log_queue.get_nowait()
                tag = "error" if is_err else None
                if not tag:
                    if "[AI RECOMMEND]" in msg:
                        tag = "rec"
                    elif "[Exploit" in msg:
                        tag = "exploit"
                    elif "[CDPSniffer]" in msg or "[STREAM EVENT]" in msg:
                        tag = "highlight"

                self.txt_log.insert(tk.END, msg, tag)
                self.txt_log.see(tk.END)
                self.log_queue.task_done()
        except queue.Empty:
            pass
        finally:
            self.root.after(80, self._poll_log_queue)

    def _log(self, message: str) -> None:
        """Log standard text via sys.stdout redirector."""
        print(message)

    def _log_err(self, message: str) -> None:
        """Log error text via sys.stderr redirector."""
        sys.stderr.write(message + "\n")

    def _launch_chrome(self) -> None:
        """Launch Google Chrome with remote debugging flags."""
        chrome_path = find_chrome_executable()
        if not chrome_path:
            messagebox.showerror(
                "Chromeが見つかりません",
                "標準パスに Google Chrome (chrome.exe) が見つかりませんでした。\n"
                "Chromeがインストールされているか確認してください。",
            )
            return

        user_data_dir = r"C:\ChromeDebugTemp"
        os.makedirs(user_data_dir, exist_ok=True)
        port = self.port_var.get().strip() or "9222"

        cmd = [
            chrome_path,
            f"--remote-debugging-port={port}",
            f"--user-data-dir={user_data_dir}",
            "https://game.mahjongsoul.com/",
        ]

        try:
            subprocess.Popen(cmd)
            self._log(f"[Chrome] 起動成功: Port {port}, Profile {user_data_dir}")
            self.lbl_conn_status.config(
                text=f"● じゃんたま起動済 (Port: {port})", fg=self.c_yellow
            )
        except Exception as e:
            messagebox.showerror("Chrome起動エラー", f"Chromeの起動に失敗しました:\n{e}")
            self._log_err(f"[ERROR] Chrome起動失敗: {e}")

    def _open_profiler_dialog(self) -> None:
        """Opens the standalone Player Profiler analysis window."""
        try:
            dialog = PlayerProfilerDialog(self.root, self.active_profiler)
            dialog.focus_set()
        except Exception as e:
            self._log_err(f"[Profiler Dialog Error] {e}")

    def _toggle_sniffer(self) -> None:
        """Starts or stops the passive WebSocket paifu sniffer on the active MahjongSoul tab."""
        from tools.paifu_to_mjai import PaifuSniffer

        if self.paifu_sniffer and self.paifu_sniffer.is_running:
            self.paifu_sniffer.stop()
            self.btn_sniffer.config(text="📡 取り込み開始", bg="#7c3aed")
            self.lbl_exploit_summary.config(text="スニッファーを停止しました。", fg=self.c_text_dim)
            self._log("[Paifu Sniffer] スニッファーを手動停止しました。")
            return

        port = int(self.port_var.get().strip() or "9222")
        self.paifu_sniffer = PaifuSniffer(cdp_port=port, max_games=3)

        def _on_captured(rec, count, max_count, events):
            def _ui_update():
                uuid = rec.get("uuid", "unknown")
                self._log(f"[Paifu Sniffer] 牌譜パケットキャプチャ成功 ({count}/{max_count}件): {uuid}")
                if events:
                    self.active_profiler.analyze_game_events(events)
                    self._log(f"[Profiler] 対戦相手プロファイル更新完了 (解析イベント数: {len(events)})")
                self.lbl_exploit_summary.config(
                    text=f"1件取得完了: {uuid[:24]}...（取得中: {count}/{max_count}件）",
                    fg=self.c_green
                )
                self._update_exploit_summary()
            self.root.after(0, _ui_update)

        def _on_status(status_msg):
            def _ui_status():
                self.lbl_exploit_summary.config(text=status_msg, fg=self.c_yellow)
                self._log(status_msg)
            self.root.after(0, _ui_status)

        def _on_finished(records):
            def _ui_finish():
                self.btn_sniffer.config(text="📡 取り込み開始", bg="#7c3aed")
                count = len(records)
                if count > 0:
                    self.lbl_exploit_summary.config(
                        text=f"取得完了: {count}件の牌譜を取り込みました。プロファイラーを表示します。",
                        fg=self.c_green
                    )
                    self._log(f"[Paifu Sniffer] 合計 {count} 件の牌譜を取得完了。プロファイラーを開きます。")
                    self._open_profiler_dialog()
                else:
                    self.lbl_exploit_summary.config(
                        text="スニッファー待機終了 (取得データなし)",
                        fg=self.c_text_dim
                    )
            self.root.after(0, _ui_finish)

        def _on_error(err_msg):
            def _ui_err():
                self.btn_sniffer.config(text="📡 取り込み開始", bg="#7c3aed")
                self.lbl_exploit_summary.config(text=err_msg, fg=self.c_red)
                self._log_err(f"[Paifu Sniffer Error] {err_msg}")
            self.root.after(0, _ui_err)

        self.btn_sniffer.config(text="⏳ 監視中...", bg="#b91c1c")
        self.lbl_exploit_summary.config(
            text="牌譜画面の『詳細』をクリックしてください（取得中: 0/3件）",
            fg=self.c_yellow
        )
        self._log(f"[Paifu Sniffer] 既存タブ監視開始 (Port: {port})。雀魂の牌譜一覧で「詳細」をクリックしてください。")

        self.paifu_sniffer.start(
            on_captured=_on_captured,
            on_status=_on_status,
            on_finished=_on_finished,
            on_error=_on_error
        )

    def _finish_and_show_profiler(self) -> None:
        """Stops sniffer if running and opens the PlayerProfilerDialog."""
        if self.paifu_sniffer and self.paifu_sniffer.is_running:
            self.paifu_sniffer.stop()
            self.btn_sniffer.config(text="📡 取り込み開始", bg="#7c3aed")

        self._update_exploit_summary()
        self._open_profiler_dialog()

    def _update_exploit_summary(self) -> None:
        """Summarizes player profiles and enables exploit tactics if applicable."""
        profiles = self.active_profiler.profiles
        if not profiles:
            return

        summary_lines = []
        best_exploit = None
        for name, p in profiles.items():
            if p.total_rounds >= 1:
                badges_str = " ".join([f"[{b}]" for b in p.badges])
                summary_lines.append(
                    f"【{name}】{badges_str} | 和了:{p.hora_rate*100:.0f}% 放銃:{p.deal_in_rate*100:.0f}% 鳴き:{p.open_rate*100:.0f}%"
                )
                mod = self.active_profiler.get_exploit_modifiers(name)
                if "exploit_note" in mod and not best_exploit:
                    best_exploit = mod

        if best_exploit:
            self.active_custom_exploit = best_exploit
            summary_lines.append(f"★ 推奨エクスプロイト: {best_exploit['exploit_note']}")
            self.is_exploit_mode.set(True)
        else:
            self.active_custom_exploit = {
                "defense_scale_mult": 1.0,
                "danger_penalty_mult": 1.0,
                "attack_scale_mult": 1.0,
                "exploit_note": "バランス型対戦相手: 標準戦術を適用",
            }

        result_text = "\n".join(summary_lines)
        self.lbl_exploit_summary.config(text=result_text, fg=self.c_green)

    def _toggle_assist(self) -> None:
        """Start or stop the Mahjong AI assist agent loop."""
        if not self.is_running:
            self._start_assist()
        else:
            self._stop_assist()

    def _start_assist(self) -> None:
        """Start CDP sniffer and AI loop in a daemon thread."""
        port_str = self.port_var.get().strip() or "9222"
        try:
            port = int(port_str)
        except ValueError:
            messagebox.showerror("エラー", "ポート番号が不正です。")
            return

        mode = self.output_mode_var.get()
        self.is_running = True
        self.btn_assist.config(text="⏹ アシスト停止", bg=self.c_btn_stop)
        self.lbl_conn_status.config(text=f"● 接続中 (Port: {port})...", fg=self.c_yellow)
        self._log(f"[Assist] 開始: mode={mode}, port={port}")

        def _run_worker():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self.agent_loop = loop

            from utils.path_helper import get_config_path
            config_path = get_config_path("strategy_config.yaml")

            try:
                agent = MahjongAIAgent(
                    config_path=config_path,
                    my_actor_id=0,
                    apply_sleep=True,
                    output_mode=mode,
                )
                if mode == "semi_auto":
                    agent.config.setdefault("mouse_control", {})["enable_pyautogui"] = True
                    agent.action_controller.mouse_cfg["enable_pyautogui"] = True

                # Wire up recommendation and status callbacks
                agent.action_controller.on_recommendation_callback = self._on_recommendation
                agent.status_callback = self._on_agent_status

                # Exploit Mode integration
                if self.is_exploit_mode.get():
                    if self.active_profiler and self.active_profiler.profiles:
                        agent.modifier.profiler = self.active_profiler
                    if self.active_custom_exploit:
                        agent.modifier.custom_exploit = self.active_custom_exploit
                        self._log(f"[Exploit Mode] 適用中: {self.active_custom_exploit.get('exploit_note')}")

                self.agent_instance = agent

                sniffer = LiveSniffer(
                    mode="cdp",
                    cdp_port=port,
                    my_seat=0,
                )
                self.sniffer_instance = sniffer

                # On CDP WebSocket connection success
                sniffer.on_connected_callback = lambda: self.root.after(
                    0,
                    lambda: self.lbl_conn_status.config(
                        text="● 接続完了 (雀魂タブ監視中)", fg=self.c_green
                    ),
                )

                self._log(f"[CDPSniffer] ポート {port} の雀魂ブラウザへアタッチを開始します...")

                async def _async_runner():
                    sniffer_task = asyncio.create_task(sniffer.start())
                    stream_task = asyncio.create_task(agent.run_stream(sniffer.stream_events()))
                    done, pending = await asyncio.wait(
                        [sniffer_task, stream_task],
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    for t in done:
                        if t.exception():
                            raise t.exception()
                    for p in pending:
                        p.cancel()

                loop.run_until_complete(_async_runner())
            except asyncio.CancelledError:
                pass
            except Exception as e:
                tb = traceback.format_exc()
                self._log_err(f"[ERROR in Agent Loop]\n{tb}")
            finally:
                if self.sniffer_instance:
                    self.sniffer_instance.stop()
                self.root.after(0, self._on_agent_stopped)

        self.agent_thread = threading.Thread(target=_run_worker, daemon=True)
        self.agent_thread.start()

    def _stop_assist(self) -> None:
        """Request the agent loop to stop."""
        self._log("[Assist] 停止要求を送信中...")
        if self.sniffer_instance:
            self.sniffer_instance.stop()
        if self.agent_loop and self.agent_loop.is_running():
            self.agent_loop.call_soon_threadsafe(self.agent_loop.stop)
        self._on_agent_stopped()

    def _on_agent_stopped(self) -> None:
        """Update UI state when agent stops."""
        self.is_running = False
        self.btn_assist.config(text="▶ アシスト開始", bg="#059669")
        self.lbl_conn_status.config(text="● 停止中", fg=self.c_text_dim)
        self._log("[Assist] アシストが停止しました。")

    def _on_agent_status(self, status_info: Dict[str, Any]) -> None:
        """Handle status update from the agent (invoked in thread)."""

        def _do_update():
            try:
                round_info = status_info.get("round_info", "")
                seat_name = status_info.get("seat_name", "")
                event_type = status_info.get("event_type", "")
                trace_log("GUI_STATUS_UPDATE", f"Round: {round_info} Seat: {seat_name} Event: {event_type}")
                if round_info:
                    text = f"対局中: {round_info} | 自席: {seat_name}"
                    self.lbl_round_info.config(text=text, fg=self.c_accent)
                    self.lbl_conn_status.config(text="● 対局中 (監視同期中)", fg=self.c_green)
                elif event_type in ("hora", "ryukyoku"):
                    self.lbl_round_info.config(text="局終了 / 次局待機中", fg=self.c_text_dim)
            except Exception as e:
                tb = traceback.format_exc()
                trace_log("GUI_STATUS_ERR", f"Exception in _on_agent_status: {e}\n{tb}")
                self._log_err(f"[GUI Status Error] {e}\n{tb}")

        self.root.after(0, _do_update)

    def _on_recommendation(self, rec: Dict[str, Any]) -> None:
        """Handle AI recommendation update (invoked in thread)."""

        def _do_update():
            try:
                action_type = rec.get("type") or rec.get("action_type")
                target = rec.get("target") or rec.get("tile")
                score = rec.get("score") if rec.get("score") is not None else rec.get("expected_value", 0.0)
                shanten = rec.get("shanten")
                top_candidates = rec.get("top_candidates", [])

                target_tile = str(target) if target and str(target) != "None" else ""
                trace_log("GUI_REC_UPDATE", f"Action: {action_type} Target: {target_tile} Score: {score}")

                if action_type in ("dahai", "reach"):
                    jp_name = format_tile_jp(target_tile, include_unicode=True) if target_tile else "？"
                    reach_suffix = " [立直]" if action_type == "reach" else ""
                    self.lbl_rec_main.config(
                        text=f"推奨打牌: {jp_name}{reach_suffix}",
                        fg=self.c_accent,
                    )
                    shanten_str = f"{shanten}シャンテン" if shanten is not None and shanten > 0 else ("テンパイ" if shanten == 0 else "--")
                    self.lbl_rec_sub.config(
                        text=f"期待値: {score:+.2f} | 向聴数: {shanten_str}",
                        fg=self.c_text,
                    )

                    # Candidates list
                    cand_texts = []
                    for i, c in enumerate(top_candidates[:3], start=1):
                        c_tile = c.get("tile", "")
                        cand_tile = format_tile_jp(c_tile, include_unicode=False) if c_tile else "？"
                        c_act = c.get("action_type", "")
                        c_reach = "[立直]" if c_act == "reach" else ""
                        cand_ev = c.get("modified_q") or c.get("expected_value", 0.0)
                        cand_texts.append(f"{i}位:{cand_tile}{c_reach}({cand_ev:+.1f})")
                    self.lbl_rec_detail.config(
                        text=" | ".join(cand_texts) if cand_texts else ""
                    )

                    self._log(
                        f"[AI RECOMMEND] 推奨打牌: {jp_name}{reach_suffix} | 期待値: {score:+.2f} | {shanten_str}"
                    )

                elif action_type in ("chi", "pon", "kan", "hora"):
                    action_names = {
                        "chi": "チー",
                        "pon": "ポン",
                        "kan": "カン",
                        "hora": "ロン",
                    }
                    action_name = action_names.get(action_type, action_type.upper())
                    target_str = (
                        f" ({format_tile_jp(str(target), include_unicode=True)})"
                        if target
                        else ""
                    )
                    self.lbl_rec_main.config(
                        text=f"副露判断: {action_name}推奨{target_str}",
                        fg=self.c_yellow,
                    )
                    self.lbl_rec_sub.config(
                        text=f"期待値: {score:+.2f}",
                        fg=self.c_text,
                    )
                    self.lbl_rec_detail.config(text="")
                    self._log(
                        f"[AI RECOMMEND] 副露判断: {action_name}推奨{target_str} | 期待値: {score:+.2f}"
                    )

                elif action_type in ("pass", "none"):
                    self.lbl_rec_main.config(
                        text="副露判断: スルー推奨 (パス)",
                        fg=self.c_green,
                    )
                    self.lbl_rec_sub.config(text="期待値: +0.00", fg=self.c_text_dim)
                    self.lbl_rec_detail.config(text="")
                    self._log("[AI RECOMMEND] 副露判断: スルー推奨 (パス)")
            except Exception as e:
                tb = traceback.format_exc()
                trace_log("GUI_REC_ERR", f"Exception in _on_recommendation: {e}\n{tb}")
                self._log_err(f"[GUI Rec Error] {e}\n{tb}")

        self.root.after(0, _do_update)

    def _start_watchdog(self) -> None:
        """Periodic watchdog timer to detect unresponsive loops or frozen states."""
        if not self.is_running:
            self.root.after(3000, self._start_watchdog)
            return

        try:
            tracer = get_tracer()
            idle_sec = tracer.get_idle_seconds()
            if idle_sec > 10.0:
                # If idle for > 10 seconds while running, dump current state
                state_dump = "No active state manager"
                if self.agent_instance and self.agent_instance.state_mgr:
                    state_dump = self.agent_instance.state_mgr.dump_state()
                trace_log("WATCHDOG_HEARTBEAT", f"No packets for {idle_sec:.1f}s | LastEvent={tracer.last_event_type} | {state_dump}")
        except Exception as e:
            trace_log("WATCHDOG_ERR", f"Watchdog error: {e}")
        finally:
            self.root.after(3000, self._start_watchdog)

    def close(self) -> None:
        """Cleanup redirectors and close root."""
        sys.stdout = self._orig_stdout
        sys.stderr = self._orig_stderr
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    app = MahjongHUDApp(root)
    root.protocol("WM_DELETE_WINDOW", app.close)
    root.mainloop()


if __name__ == "__main__":
    main()
