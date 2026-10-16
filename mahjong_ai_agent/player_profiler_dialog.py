"""Player Profiler Dialog for inspecting individual opponent habits and exploit tactics."""

from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from typing import Dict, Optional, Any

from core.opponent_profiler import OpponentProfiler, PlayerProfile


class PlayerProfilerDialog(tk.Toplevel):
    """
    Dedicated dark-themed popup window for visualizing individual player
    habits, push/fold tendencies, meld preferences, and exploit countermeasures.
    """

    def __init__(self, parent: tk.Tk, profiler: OpponentProfiler, initial_player: Optional[str] = None):
        super().__init__(parent)
        self.profiler = profiler
        self.title("📊 プレイヤー傾向・癖分析 (対戦相手プロファイル)")
        self.geometry("640x720")
        self.minsize(580, 620)
        self.configure(bg="#1a1c23")

        # Color palette (matching main HUD dark mode)
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
        self.c_badge_bg = "#312e81"

        self.selected_player_var = tk.StringVar()
        self.profiles = profiler.profiles

        self._build_ui()

        # Select initial player or first available
        player_names = list(self.profiles.keys())
        if initial_player and initial_player in player_names:
            self._select_player(initial_player)
        elif player_names:
            self._select_player(player_names[0])
        else:
            self._show_empty_state()

    def _build_ui(self) -> None:
        """Builds dialog layout."""
        # Top Header Bar
        header = tk.Frame(self, bg=self.c_bg, padx=16, pady=12)
        header.pack(fill=tk.X)

        title_lbl = tk.Label(
            header,
            text="👥 プレイヤー傾向・癖分析 (対局相手プロファイラー)",
            font=("Meiryo", 13, "bold"),
            fg=self.c_text,
            bg=self.c_bg,
        )
        title_lbl.pack(side=tk.LEFT)

        count_lbl = tk.Label(
            header,
            text=f"登録プレイヤー: {len(self.profiles)} 名",
            font=("Meiryo", 9),
            fg=self.c_accent,
            bg=self.c_bg,
        )
        count_lbl.pack(side=tk.RIGHT)

        # Main Split Body
        body = tk.Frame(self, bg=self.c_bg, padx=12, pady=4)
        body.pack(fill=tk.BOTH, expand=True)

        # Left Column: Player Selector List
        left_col = tk.Frame(body, bg=self.c_card, width=180, highlightbackground=self.c_card_border, highlightthickness=1)
        left_col.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10), pady=4)
        left_col.pack_propagate(False)

        list_header = tk.Label(
            left_col,
            text="対戦相手一覧",
            font=("Meiryo", 9, "bold"),
            fg=self.c_text_dim,
            bg=self.c_card,
            pady=6,
        )
        list_header.pack(fill=tk.X)

        self.player_listbox = tk.Listbox(
            left_col,
            font=("Meiryo", 9),
            bg=self.c_bg,
            fg=self.c_text,
            selectbackground="#4338ca",
            selectforeground="#ffffff",
            relief=tk.FLAT,
            highlightthickness=0,
            activestyle="none",
        )
        self.player_listbox.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.player_listbox.bind("<<ListboxSelect>>", self._on_listbox_select)

        # Populate listbox
        for name, p in self.profiles.items():
            self.player_listbox.insert(tk.END, f"{p.display_name} ({p.total_rounds}局)")

        # Right Column: Profile Detail Cards
        self.right_col = tk.Frame(body, bg=self.c_bg)
        self.right_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, pady=4)

        # Player Header Card
        self.card_header = tk.Frame(self.right_col, bg=self.c_card, padx=14, pady=10, highlightbackground=self.c_card_border, highlightthickness=1)
        self.card_header.pack(fill=tk.X, pady=(0, 8))

        self.lbl_p_name = tk.Label(self.card_header, text="--", font=("Meiryo", 14, "bold"), fg=self.c_text, bg=self.c_card)
        self.lbl_p_name.pack(anchor=tk.W)

        self.lbl_p_meta = tk.Label(self.card_header, text="総対局: -- 局", font=("Meiryo", 9), fg=self.c_text_dim, bg=self.c_card)
        self.lbl_p_meta.pack(anchor=tk.W, pady=(2, 6))

        # Badges Container
        self.badge_frame = tk.Frame(self.card_header, bg=self.c_card)
        self.badge_frame.pack(anchor=tk.W, fill=tk.X)

        # Section 1: Basic Stats Grid
        self.sec1 = tk.LabelFrame(self.right_col, text=" 📈 基本スタッツ ", font=("Meiryo", 9, "bold"), fg=self.c_accent, bg=self.c_card, padx=10, pady=8)
        self.sec1.pack(fill=tk.X, pady=(0, 8))

        self.stats_grid = tk.Frame(self.sec1, bg=self.c_card)
        self.stats_grid.pack(fill=tk.X)
        self.stats_grid.columnconfigure(0, weight=1)
        self.stats_grid.columnconfigure(1, weight=1)
        self.stats_grid.columnconfigure(2, weight=1)

        self.lbl_stat_hora = self._make_stat_box(self.stats_grid, 0, 0, "和了率", "--", self.c_green)
        self.lbl_stat_dealin = self._make_stat_box(self.stats_grid, 0, 1, "放銃率", "--", self.c_red)
        self.lbl_stat_riichi = self._make_stat_box(self.stats_grid, 0, 2, "立直率", "--", self.c_accent)
        self.lbl_stat_open = self._make_stat_box(self.stats_grid, 1, 0, "副露率 (鳴き)", "--", self.c_yellow)
        self.lbl_stat_score = self._make_stat_box(self.stats_grid, 1, 1, "平均打点", "--", self.c_text)
        self.lbl_stat_aggr = self._make_stat_box(self.stats_grid, 1, 2, "危険牌押し率", "--", self.c_purple)

        # Section 2: Habit & Tendency Profile
        self.sec2 = tk.LabelFrame(self.right_col, text=" 🔍 癖・プレイスタイル分析 ", font=("Meiryo", 9, "bold"), fg=self.c_yellow, bg=self.c_card, padx=12, pady=8)
        self.sec2.pack(fill=tk.X, pady=(0, 8))

        self.lbl_tend_call = self._make_tendency_row(self.sec2, "鳴き傾向:", "--")
        self.lbl_tend_push = self._make_tendency_row(self.sec2, "押し引き傾向:", "--")
        self.lbl_tend_early = self._make_tendency_row(self.sec2, "序盤の傾向:", "--")

        # Section 3: AI Exploit Strategy Countermeasure
        self.sec3 = tk.LabelFrame(self.right_col, text=" ⚔ 対戦相手エクスプロイト指南 ", font=("Meiryo", 9, "bold"), fg=self.c_purple, bg=self.c_card, padx=12, pady=8)
        self.sec3.pack(fill=tk.X, pady=(0, 8))

        self.lbl_exploit_advice = tk.Label(
            self.sec3,
            text="--",
            font=("Meiryo", 9),
            fg=self.c_text,
            bg=self.c_card,
            justify=tk.LEFT,
            wraplength=380,
        )
        self.lbl_exploit_advice.pack(anchor=tk.W, fill=tk.X)

        # Bottom Close Button
        btn_bar = tk.Frame(self, bg=self.c_bg, padx=12, pady=6)
        btn_bar.pack(fill=tk.X)
        btn_close = tk.Button(
            btn_bar,
            text="閉じる",
            command=self.destroy,
            font=("Meiryo", 9, "bold"),
            bg="#333846",
            fg=self.c_text,
            activebackground="#475569",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=16,
            pady=4,
            cursor="hand2",
        )
        btn_close.pack(side=tk.RIGHT)

    def _make_stat_box(self, parent: tk.Frame, row: int, col: int, title: str, init_val: str, color: str) -> tk.Label:
        box = tk.Frame(parent, bg=self.c_bg, padx=6, pady=4, highlightbackground=self.c_card_border, highlightthickness=1)
        box.grid(row=row, column=col, padx=4, pady=3, sticky="nsew")

        t_lbl = tk.Label(box, text=title, font=("Meiryo", 8), fg=self.c_text_dim, bg=self.c_bg)
        t_lbl.pack(anchor=tk.W)

        v_lbl = tk.Label(box, text=init_val, font=("Meiryo", 11, "bold"), fg=color, bg=self.c_bg)
        v_lbl.pack(anchor=tk.W)
        return v_lbl

    def _make_tendency_row(self, parent: tk.Frame, label_text: str, init_val: str) -> tk.Label:
        row = tk.Frame(parent, bg=self.c_card)
        row.pack(fill=tk.X, pady=2)

        lbl = tk.Label(row, text=label_text, font=("Meiryo", 9, "bold"), fg=self.c_text_dim, bg=self.c_card, width=12, anchor=tk.W)
        lbl.pack(side=tk.LEFT)

        val = tk.Label(row, text=init_val, font=("Meiryo", 9), fg=self.c_text, bg=self.c_card, anchor=tk.W)
        val.pack(side=tk.LEFT, fill=tk.X, expand=True)
        return val

    def _on_listbox_select(self, event) -> None:
        selection = self.player_listbox.curselection()
        if not selection:
            return
        idx = selection[0]
        player_names = list(self.profiles.keys())
        if idx < len(player_names):
            self._select_player(player_names[idx])

    def _select_player(self, name: str) -> None:
        """Renders the full profile for the chosen player."""
        self.selected_player_var.set(name)
        prof: Optional[PlayerProfile] = self.profiles.get(name)
        if not prof:
            return

        # Header
        self.lbl_p_name.config(text=f"【{prof.display_name}】 ({prof.archetype})")
        self.lbl_p_meta.config(
            text=f"総対局: {prof.total_games}半荘 / {prof.total_rounds}局 | リーチ: {prof.riichi_count}回 | 和了: {prof.hora_count}回 | 放銃: {prof.deal_in_count}回"
        )

        # Clear old badges
        for child in self.badge_frame.winfo_children():
            child.destroy()

        # Render Badges
        is_cpu = "CPU" in prof.name or "cpu" in prof.name.lower()
        if is_cpu:
            cpu_lbl = tk.Label(
                self.badge_frame,
                text=" 🤖 CPU対局枠 ",
                font=("Meiryo", 8, "bold"),
                fg="#ffffff",
                bg="#64748b",
                padx=4,
                pady=1,
            )
            cpu_lbl.pack(side=tk.LEFT, padx=(0, 6))

        for badge_text in prof.badges:
            badge_color = "#3b82f6"
            if "鳴き" in badge_text or "喰いタン" in badge_text or "役牌" in badge_text:
                badge_color = "#d97706"
            elif "ベタオリ" in badge_text or "低放銃" in badge_text:
                badge_color = "#059669"
            elif "無筋押し" in badge_text or "放銃注意" in badge_text:
                badge_color = "#dc2626"
            elif "門前" in badge_text:
                badge_color = "#7c3aed"

            b_lbl = tk.Label(
                self.badge_frame,
                text=f" {badge_text} ",
                font=("Meiryo", 8, "bold"),
                fg="#ffffff",
                bg=badge_color,
                padx=4,
                pady=1,
            )
            b_lbl.pack(side=tk.LEFT, padx=(0, 6))

        # Basic Stats
        self.lbl_stat_hora.config(text=f"{prof.hora_rate * 100:.1f}%")
        self.lbl_stat_dealin.config(text=f"{prof.deal_in_rate * 100:.1f}%")
        self.lbl_stat_riichi.config(text=f"{prof.riichi_rate * 100:.1f}%")
        self.lbl_stat_open.config(text=f"{prof.open_rate * 100:.1f}%")
        self.lbl_stat_score.config(text=f"{int(prof.avg_hora_score):,}点" if prof.avg_hora_score > 0 else "--")
        self.lbl_stat_aggr.config(text=f"{prof.aggressiveness * 100:.1f}%")

        # Tendencies
        self.lbl_tend_call.config(
            text=f"{prof.call_tendency} (役牌鳴き:{prof.yakuhai_call_count}回 / 喰いタン:{prof.tanyao_call_count}回 / 早巡鳴き:{prof.early_call_count}回)"
        )
        self.lbl_tend_push.config(
            text=f"{prof.push_fold_tendency} (対リーチベタオリ率: {prof.fold_rate * 100:.1f}%)"
        )
        self.lbl_tend_early.config(
            text=f"{prof.early_game_tendency} (序盤字牌切出率: {(prof.early_honor_discards / max(1, prof.early_total_discards)) * 100:.1f}%)"
        )

        # AI Exploit Advice
        mod = self.profiler.get_exploit_modifiers(name)
        exploit_note = mod.get("exploit_note", "")

        advice_lines = []
        if prof.aggressiveness >= 0.60:
            advice_lines.append("⚡ 超攻撃的プッシャー: 相手のリーチや仕掛けは愚形・安手の可能性が高く、過度に降りすぎず押し返すのが有効です。")
        elif prof.fold_rate >= 0.65 or prof.aggressiveness <= 0.35:
            advice_lines.append("🛡 鉄壁ベタオリ派: リーチ時は高打点・好形の本手である確率が高いため、徹底的に現物や安牌で降りるのが賢明です。")
        else:
            advice_lines.append("⚖ バランス型: 標準的な攻守判断の打ち手です。一般的な牌効率と危険度評価で対応してください。")

        if prof.open_rate >= 0.40:
            advice_lines.append("⏩ 早仕掛け多用: 鳴きが早く手番の進行速度が早いため、序盤から字牌や安牌を意識して残してください。")
        elif prof.open_rate <= 0.18:
            advice_lines.append("💎 門前重視型: 鳴きが少なく打点が高い傾向があります。テンパイ時のリーチに要警戒です。")

        if exploit_note:
            advice_lines.append(f"★ 推奨パラメタ調整: {exploit_note}")

        self.lbl_exploit_advice.config(text="\n".join(advice_lines))

    def _show_empty_state(self) -> None:
        """Handles empty state when no paifu records have been imported yet."""
        self.lbl_p_name.config(text="牌譜データがありません")
        self.lbl_p_meta.config(text="牌譜URLまたはUUIDを入力して『牌譜を取り込んで解析』を実行してください。")
        self.lbl_exploit_advice.config(text="過去牌譜を取り込むと、ここに各プレイヤーの癖・傾向が可視化されます。")
