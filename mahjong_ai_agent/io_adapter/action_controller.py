"""Action controller for human-like timing latency management, PC Majsoul mouse control, and HUD dashboard."""

from __future__ import annotations
import asyncio
import math
import random
import time
from typing import Dict, Any, Optional, Tuple, List, Callable
from pydantic import BaseModel, Field

from core.engine import ActionCandidate
from core.state_manager import MahjongStateManager, normalize_tile, is_valid_mahjong_tile


class ScreenPoint(BaseModel):
    x: int
    y: int


class MouseStep(BaseModel):
    """Represents a discrete step in the human-like mouse interaction sequence."""
    step_type: str                   # "approach", "hover", "mouse_down", "mouse_up", "step_wait", "rest_retreat"
    target_point: ScreenPoint
    duration_sec: float = 0.0
    trajectory_sample_count: int = 0
    note: str = ""


class OutputAction(BaseModel):
    """Structured decision output sent to game client or visual assistant."""
    action_type: str                         # "dahai", "reach", "chi", "pon", "kan", "hora", "none"
    tile: Optional[str] = None
    target_slot_index: Optional[int] = None
    screen_coordinates: Optional[ScreenPoint] = None
    thinking_time_sec: float = 0.0
    computation_time_ms: float = 0.0
    reasoning_summary: str = ""
    output_mode: str = "auto"
    top_candidates: List[Dict[str, Any]] = Field(default_factory=list)
    mouse_steps: List[MouseStep] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


TILE_JP_NAMES = {
    # Manzu
    "1m": "1萬", "2m": "2萬", "3m": "3萬", "4m": "4萬", "5m": "5萬",
    "6m": "6萬", "7m": "7萬", "8m": "8萬", "9m": "9萬",
    "0m": "赤5萬", "5mr": "赤5萬",
    # Pinzu
    "1p": "1筒", "2p": "2筒", "3p": "3筒", "4p": "4筒", "5p": "5筒",
    "6p": "6筒", "7p": "7筒", "8p": "8筒", "9p": "9筒",
    "0p": "赤5筒", "5pr": "赤5筒",
    # Souzu
    "1s": "1索", "2s": "2索", "3s": "3索", "4s": "4索", "5s": "5索",
    "6s": "6索", "7s": "7索", "8s": "8索", "9s": "9索",
    "0s": "赤5索", "5sr": "赤5索",
    # Winds (mjai & numeric liqi & aliases)
    "E": "東", "1z": "東", "ton": "東", "higashi": "東",
    "S": "南", "2z": "南", "nan": "南", "minami": "南",
    "W": "西", "3z": "西", "sha": "西", "nishi": "西",
    "N": "北", "4z": "北", "pei": "北", "kita": "北",
    # Dragons (mjai & numeric liqi & aliases)
    "P": "白", "5z": "白", "haku": "白", "shiro": "白",
    "F": "發", "6z": "發", "hatsu": "發",
    "C": "中", "7z": "中", "chun": "中", "naka": "中",
}

MAHJONG_UNICODE_TILES = {
    "1m": "🀇", "2m": "🀈", "3m": "🀉", "4m": "🀊", "5m": "🀋",
    "6m": "🀌", "7m": "🀍", "8m": "🀎", "9m": "🀏", "0m": "🀋", "5mr": "🀋",
    "1p": "🀙", "2p": "🀚", "3p": "🀛", "4p": "🀜", "5p": "🀝",
    "6p": "🀞", "7p": "🀟", "8p": "🀠", "9p": "🀡", "0p": "🀝", "5pr": "🀝",
    "1s": "🀐", "2s": "🀑", "3s": "🀒", "4s": "🀓", "5s": "🀔",
    "6s": "🀕", "7s": "🀖", "8s": "🀗", "9s": "🀘", "0s": "🀔", "5sr": "🀔",
    "E": "🀀", "1z": "🀀", "ton": "🀀",
    "S": "🀁", "2z": "🀁", "nan": "🀁",
    "W": "🀂", "3z": "🀂", "sha": "🀂",
    "N": "🀃", "4z": "🀃", "pei": "🀃",
    "P": "🀆", "5z": "🀆", "haku": "🀆",
    "F": "🀅", "6z": "🀅", "hatsu": "🀅",
    "C": "🀄", "7z": "🀄", "chun": "🀄",
}


def format_tile_jp(tile: Optional[str], include_unicode: bool = False) -> str:
    """Format tile symbol to friendly Japanese notation (e.g. '北 (N)', '8筒 (8p)', '🀃 北 (N)')."""
    if not tile or tile.strip() in ("", "?", "None"):
        return "？"
    tile = tile.strip()
    jp_name = TILE_JP_NAMES.get(tile)
    if jp_name is None:
        # Check lowercase
        jp_name = TILE_JP_NAMES.get(tile.lower(), tile)

    uni = f"{MAHJONG_UNICODE_TILES.get(tile, '')} " if include_unicode and tile in MAHJONG_UNICODE_TILES else ""
    return f"{uni}{jp_name} ({tile})"


def format_tile_label(tile: Optional[str], include_unicode: bool = False) -> str:
    """
    Format tile code (mjai/liqi/honors) into friendly user-facing label:
    e.g. '1z' -> '東 (1z)', 'E' -> '東 (E)', '7z' -> '中 (7z)', 'C' -> '中 (C)', '1m' -> '1萬 (1m)'
    """
    if not tile or tile.strip() in ("", "?", "None"):
        return "？"
    tile = tile.strip()
    jp_name = TILE_JP_NAMES.get(tile)
    if jp_name is None:
        jp_name = TILE_JP_NAMES.get(tile.lower(), tile)
    uni = f"{MAHJONG_UNICODE_TILES.get(tile, '')} " if include_unicode and tile in MAHJONG_UNICODE_TILES else ""
    if jp_name and jp_name != tile:
        return f"{uni}{jp_name} ({tile})"
    return f"{uni}{tile}"


def generate_cubic_bezier_trajectory(
    p0: ScreenPoint,
    p3: ScreenPoint,
    num_samples: int = 15
) -> List[ScreenPoint]:
    """
    Generate smooth human-like mouse curve trajectory using cubic Bezier curve.
    B(t) = (1-t)^3 * P0 + 3(1-t)^2 * t * P1 + 3(1-t) * t^2 * P2 + t^3 * P3
    """
    dx = p3.x - p0.x
    dy = p3.y - p0.y
    dist = math.hypot(dx, dy)

    # If already close, straight line
    if dist < 10 or num_samples <= 2:
        return [p0, p3]

    # Generate control points with subtle human arc variation
    perp_x = -dy / dist
    perp_y = dx / dist
    arc_factor = random.uniform(-0.15, 0.15) * dist

    p1_x = int(p0.x + dx * 0.3 + perp_x * arc_factor)
    p1_y = int(p0.y + dy * 0.3 + perp_y * arc_factor)
    p2_x = int(p0.x + dx * 0.7 + perp_x * arc_factor * 0.7)
    p2_y = int(p0.y + dy * 0.7 + perp_y * arc_factor * 0.7)

    trajectory: List[ScreenPoint] = []
    for i in range(num_samples):
        t = i / (num_samples - 1)
        inv = 1.0 - t
        bx = (inv**3 * p0.x) + (3 * inv**2 * t * p1_x) + (3 * inv * t**2 * p2_x) + (t**3 * p3.x)
        by = (inv**3 * p0.y) + (3 * inv**2 * t * p1_y) + (3 * inv * t**2 * p2_y) + (t**3 * p3.y)
        trajectory.append(ScreenPoint(x=int(bx), y=int(by)))

    return trajectory


class ActionController:
    """
    Controls emission of decisions:
    1. Computes human-like realistic thinking delay (1.0s - 3.0s Gaussian bounded).
    2. Calculates screen click coordinates for tiles or buttons.
    3. Handles PC Majsoul mouse mechanics: Approach -> Hover delay -> Single Click -> Rest Retreat.
    4. Renders live TUI HUD dashboard in assistant mode.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.output_mode = config.get("output_mode", "dashboard")
        self.timing_cfg = config.get("timing", {})
        self.screen_cfg = config.get("screen_mapping", {})
        self.mouse_cfg = config.get("mouse_control", {})

        # Latency parameters
        self.mean_latency = self.timing_cfg.get("mean_latency_sec", 1.8)
        self.std_dev = self.timing_cfg.get("std_dev_sec", 0.4)
        self.min_latency = self.timing_cfg.get("min_latency_sec", 1.0)
        self.max_latency = self.timing_cfg.get("max_latency_sec", 3.0)

        # Contextual fast response timings
        self.fast_tsumogiri_sec = self.timing_cfg.get("fast_tsumogiri_sec", 0.8)
        self.riichi_auto_sec = self.timing_cfg.get("riichi_auto_discard_sec", 0.4)
        self.urgent_cut_sec = self.timing_cfg.get("urgent_cut_sec", 0.6)

        # Virtual cursor tracker (starts in safe rest area)
        self.virtual_cursor: ScreenPoint = self._get_random_rest_point()

        # Cache of previous recommendation to manage freshness
        self._last_recommendation: Optional[OutputAction] = None
        self.top_candidates: List[Dict[str, Any]] = []

        # Optional GUI / HUD recommendation callback hook
        self.on_recommendation_callback: Optional[Callable[[Dict[str, Any]], None]] = None

    def has_active_recommendation(self) -> bool:
        """Check if an AI recommendation is currently active and computed."""
        return self._last_recommendation is not None

    def clear_recommendation(self, notify: bool = True, is_self_turn: bool = False) -> None:
        """Clear cached previous recommendations to prevent stale UI state when turn ends or new draw arrives."""
        self._last_recommendation = None
        self.top_candidates = []
        if notify and self.on_recommendation_callback:
            try:
                rec_text = "🤔 思考中 (自手番)..." if is_self_turn else "相手番 (待機中...)"
                self.on_recommendation_callback({
                    "action_type": "thinking" if is_self_turn else "cleared",
                    "status": "thinking" if is_self_turn else "waiting",
                    "recommendation_text": rec_text,
                    "top_candidates": []
                })
            except Exception as e:
                print(f"[ActionController] Clear callback error: {e}")

    def _get_random_rest_point(self) -> ScreenPoint:
        """Returns a random safe point in the neutral rest zone away from hand tiles."""
        rest_area = self.mouse_cfg.get("rest_area", {})
        x_min = rest_area.get("x_min", 1450)
        x_max = rest_area.get("x_max", 1750)
        y_min = rest_area.get("y_min", 650)
        y_max = rest_area.get("y_max", 800)
        return ScreenPoint(
            x=random.randint(x_min, x_max),
            y=random.randint(y_min, y_max)
        )

    def calculate_thinking_delay(
        self,
        candidate: ActionCandidate,
        state_mgr: MahjongStateManager,
        computation_time_ms: float
    ) -> float:
        """
        Calculate target total delay (sec) using Gaussian distribution,
        adjusted for context (e.g., fast discard if already in riichi, tsumogiri, or pass).
        """
        my_p = state_mgr.my_player

        # 1. If self is in Riichi, human players usually discard immediately
        if my_p and my_p.is_riichi and candidate.action_type == "dahai":
            target = random.gauss(self.riichi_auto_sec, 0.08)
            return max(0.25, round(target, 2))

        # 2. Simple pass on other's discard
        if candidate.action_type == "none":
            target = random.gauss(self.urgent_cut_sec, 0.12)
            return max(0.3, min(1.2, round(target, 2)))

        # 3. Tsumogiri can be faster unless it was a tough decision
        if candidate.is_tsumogiri:
            base_mean = self.fast_tsumogiri_sec
            std = 0.15
        else:
            base_mean = self.mean_latency
            std = self.std_dev

        # Sample from Gaussian distribution
        sampled = random.gauss(base_mean, std)
        bounded = max(self.min_latency, min(self.max_latency, sampled))

        # Subtract already elapsed computation time
        comp_sec = computation_time_ms / 1000.0
        remaining = max(0.0, bounded - comp_sec)
        return round(remaining, 2)

    def calculate_screen_coordinates(
        self,
        candidate: ActionCandidate,
        state_mgr: MahjongStateManager
    ) -> Tuple[Optional[int], Optional[ScreenPoint]]:
        """
        Calculates hand slot index (0..13) and pixel coordinates.
        For hand tiles in PC Majsoul, targets slightly above center (y - 15) to hit popup area.
        """
        action = candidate.action_type
        button_pos = self.screen_cfg.get("button_positions", {})
        hand_area = self.screen_cfg.get("hand_area", {})
        tsumo_slot = self.screen_cfg.get("tsumo_slot", {})

        # Normalize action aliases to config button keys
        btn_key = None
        if action == "chi":
            btn_key = "chi"
        elif action == "pon":
            btn_key = "pon"
        elif action in ("kan", "daiminkan", "ankan", "kankan"):
            btn_key = "kan"
        elif action == "hora":
            # If target != self (Ron), look for "ron" or "hora"; if target == self (Tsumo), look for "tsumo" or "hora"
            target = candidate.metadata.get("target")
            if target is not None and target != state_mgr.my_actor_id:
                btn_key = "ron" if "ron" in button_pos else "hora"
            else:
                btn_key = "tsumo" if "tsumo" in button_pos else "hora"
        elif action in ("none", "pass", "skip"):
            # Only point to pass button if this was a response to an enemy discard (target is set)
            if candidate.metadata.get("target") is not None:
                btn_key = "pass" if "pass" in button_pos else ("skip" if "skip" in button_pos else "none")

        if btn_key and btn_key in button_pos:
            coords = button_pos[btn_key]
            return None, ScreenPoint(x=coords[0], y=coords[1])

        # Dahai tile coordinates
        if action in ("dahai", "reach"):
            my_p = state_mgr.my_player
            if not my_p or not candidate.tile:
                return None, None

            # Check if tsumogiri (14th slot)
            if candidate.is_tsumogiri and tsumo_slot:
                tx = tsumo_slot.get("x", 1530)
                ty = tsumo_slot.get("y", 980) - 15  # Slightly elevated for popup
                return 13, ScreenPoint(x=tx, y=ty)

            # Match tile position in tehai
            matching_tile = my_p.get_matching_tile_in_hand(candidate.tile)
            slot_idx = -1
            if matching_tile:
                for idx, t in enumerate(my_p.tehai):
                    if t == matching_tile:
                        slot_idx = idx
                        break

            if slot_idx == -1:
                print(f"[ActionController ALERT] Discard candidate '{candidate.tile}' not found in player hand {my_p.tehai}!")
                slot_idx = max(0, len(my_p.tehai) - 1)

            start_x = hand_area.get("start_x", 430)
            slot_w = hand_area.get("slot_width", 75)
            y_pos = hand_area.get("y", 980) - 15  # Aim slightly above center for reliable popup hit

            pixel_x = int(start_x + slot_idx * slot_w + (slot_w / 2))
            return slot_idx, ScreenPoint(x=pixel_x, y=y_pos)

        return None, None

    def plan_mouse_interaction_sequence(
        self,
        target_point: ScreenPoint,
        is_riichi_sequence: bool = False,
        riichi_button_point: Optional[ScreenPoint] = None,
        is_semi_auto: bool = False
    ) -> List[MouseStep]:
        """
        Plans the complete PC Majsoul mouse interaction sequence.
        In semi_auto mode:
        1. Bezier Approach with random jitter (+-5~8px)
        2. Hover to trigger tile popup
        3. semi_auto_wait: pauses on the tile without auto-clicking, waiting for user confirmation!
        In auto_click mode:
        4 steps: Approach -> Hover -> Single Click -> Rest Retreat.
        """
        jitter = self.mouse_cfg.get("jitter_px", 6)
        hover_min = self.mouse_cfg.get("hover_delay_min", 0.08)
        hover_max = self.mouse_cfg.get("hover_delay_max", 0.18)
        click_hold_min = self.mouse_cfg.get("click_hold_min", 0.04)
        click_hold_max = self.mouse_cfg.get("click_hold_max", 0.09)
        rest_min = self.mouse_cfg.get("rest_escape_min_sec", 0.2)
        rest_max = self.mouse_cfg.get("rest_escape_max_sec", 0.5)

        steps: List[MouseStep] = []
        curr_pos = self.virtual_cursor

        # If Riichi: Phase 1 is click Riichi button, Phase 2 is hover/click discard tile
        if is_riichi_sequence and riichi_button_point:
            # 1a. Approach Riichi button
            btn_jit_x = riichi_button_point.x + random.randint(-jitter, jitter)
            btn_jit_y = riichi_button_point.y + random.randint(-jitter, jitter)
            btn_target = ScreenPoint(x=btn_jit_x, y=btn_jit_y)
            traj1 = generate_cubic_bezier_trajectory(curr_pos, btn_target, num_samples=12)
            steps.append(MouseStep(
                step_type="approach",
                target_point=btn_target,
                duration_sec=0.25,
                trajectory_sample_count=len(traj1),
                note="Approach Riichi button"
            ))
            # 1b. Hover button
            steps.append(MouseStep(
                step_type="hover",
                target_point=btn_target,
                duration_sec=round(random.uniform(hover_min, hover_max), 3),
                note="Hover Riichi button"
            ))
            # 1c. Single click Riichi button
            steps.append(MouseStep(
                step_type="mouse_down",
                target_point=btn_target,
                duration_sec=0.0,
                note="MouseDown Riichi button"
            ))
            hold_btn = round(random.uniform(click_hold_min, click_hold_max), 3)
            steps.append(MouseStep(
                step_type="mouse_up",
                target_point=btn_target,
                duration_sec=hold_btn,
                note="MouseUp Riichi button"
            ))
            # 1d. Riichi step wait
            step_wait = self.mouse_cfg.get("riichi_step_wait", 0.35)
            steps.append(MouseStep(
                step_type="step_wait",
                target_point=btn_target,
                duration_sec=step_wait,
                note="Wait for Riichi discard prompt"
            ))
            curr_pos = btn_target

        # Step 1: Bezier Approach to target tile
        jit_x = target_point.x + random.randint(-jitter, jitter)
        jit_y = target_point.y + random.randint(-jitter, jitter)
        jittered_target = ScreenPoint(x=jit_x, y=jit_y)
        traj = generate_cubic_bezier_trajectory(curr_pos, jittered_target, num_samples=16)

        steps.append(MouseStep(
            step_type="approach",
            target_point=jittered_target,
            duration_sec=0.28,
            trajectory_sample_count=len(traj),
            note=f"Bezier curve approach from ({curr_pos.x},{curr_pos.y}) to ({jit_x},{jit_y})"
        ))

        # Step 2: Hover Delay (allow tile to slide upward and human visual confirmation)
        hover_dur = round(random.uniform(hover_min, hover_max), 3)
        steps.append(MouseStep(
            step_type="hover",
            target_point=jittered_target,
            duration_sec=hover_dur,
            note="Hover delay for tile popup animation"
        ))

        # If Semi-Auto Mode: STOP HERE! Wait for human confirmation/click
        if is_semi_auto:
            steps.append(MouseStep(
                step_type="semi_auto_wait",
                target_point=jittered_target,
                duration_sec=0.0,
                note="[SEMI-AUTO] Hovering and paused. Waiting for user manual click or space confirmation."
            ))
            self.virtual_cursor = jittered_target
            return steps

        # Step 3: Single Click (MouseDown -> Hold -> MouseUp)
        steps.append(MouseStep(
            step_type="mouse_down",
            target_point=jittered_target,
            duration_sec=0.0,
            note="MouseDown single click"
        ))
        click_dur = round(random.uniform(click_hold_min, click_hold_max), 3)
        steps.append(MouseStep(
            step_type="mouse_up",
            target_point=jittered_target,
            duration_sec=click_dur,
            note=f"MouseUp release after {click_dur*1000:.1f}ms hold"
        ))

        # Step 4: Neutral Rest Position Retreat
        rest_pt = self._get_random_rest_point()
        traj_rest = generate_cubic_bezier_trajectory(jittered_target, rest_pt, num_samples=12)
        rest_dur = round(random.uniform(rest_min, rest_max), 3)
        steps.append(MouseStep(
            step_type="rest_retreat",
            target_point=rest_pt,
            duration_sec=rest_dur,
            trajectory_sample_count=len(traj_rest),
            note=f"Escape cursor to safe rest area ({rest_pt.x},{rest_pt.y}) to prevent unwanted popups"
        ))

        self.virtual_cursor = rest_pt
        return steps

    async def execute_action(
        self,
        candidate: ActionCandidate,
        state_mgr: MahjongStateManager,
        comp_time_ms: float,
        apply_sleep: bool = True,
        all_candidates: Optional[List[ActionCandidate]] = None
    ) -> OutputAction:
        """
        Executes human-like delay, builds mouse interaction sequence or renders HUD dashboard.
        Guarantees that candidate.tile is valid for dahai/reach, never falling back to '?' or None.
        """
        # Safety Guard: Ensure candidate.tile is physically present in player's hand if dahai / reach
        if candidate.action_type in ("dahai", "reach"):
            my_p = state_mgr.my_player
            if my_p:
                valid_hand_tiles = [t for t in my_p.tehai if is_valid_mahjong_tile(t)]
                if valid_hand_tiles:
                    matching_tile = my_p.get_matching_tile_in_hand(candidate.tile) if candidate.tile else None
                    if not matching_tile:
                        old_tile = candidate.tile
                        candidate.tile = valid_hand_tiles[-1]
                        print(f"[ActionController ALERT] Phantom discard candidate '{old_tile}' not in hand! Forcibly replaced with actual tile '{candidate.tile}'.")
                    else:
                        candidate.tile = matching_tile
                else:
                    if not candidate.tile or candidate.tile in ("?", "None"):
                        candidate.tile = "1z"
            else:
                if not candidate.tile or candidate.tile in ("?", "None"):
                    candidate.tile = "1z"

        delay_sec = self.calculate_thinking_delay(candidate, state_mgr, comp_time_ms)
        is_text_only = (self.output_mode == "text_only")

        if is_text_only:
            slot_idx = None
            coords = None
            mouse_steps = []
        else:
            slot_idx, coords = self.calculate_screen_coordinates(candidate, state_mgr)

        # Human-like thinking sleep
        if apply_sleep and delay_sec > 0:
            await asyncio.sleep(delay_sec)

        # Prepare top candidates summary
        top_list: List[Dict[str, Any]] = []
        if all_candidates:
            best_q = all_candidates[0].modified_q_value
            for c in all_candidates[:3]:
                # Estimate win probability from shanten and ukeire
                c_shanten = c.shanten_after
                c_ukeire = c.metadata.get("ukeire", 0) if isinstance(c.metadata, dict) else 0
                if c_shanten == -1:
                    win_prob = 100
                elif c_shanten == 0:
                    win_prob = min(85, max(30, int(35 + c_ukeire * 3.5)))
                elif c_shanten == 1:
                    win_prob = min(60, max(15, int(15 + c_ukeire * 2.0)))
                elif c_shanten == 2:
                    win_prob = min(40, max(8, int(8 + c_ukeire * 1.2)))
                else:
                    win_prob = max(3, int(15 - c_shanten * 3))

                top_list.append({
                    "action_type": c.action_type,
                    "tile": c.tile,
                    "tile_jp": format_tile_jp(c.tile, include_unicode=True),
                    "modified_q": c.modified_q_value,
                    "raw_q": c.raw_q_value,
                    "danger": c.danger_score,
                    "shanten": c.shanten_after,
                    "win_prob": win_prob,
                    "diff_q": round(c.modified_q_value - best_q, 1),
                })

        # Mouse interaction sequence planning (if coordinates exist and not in text_only mode)
        mouse_steps: List[MouseStep] = []
        if coords is not None and not is_text_only:
            is_riichi = (candidate.action_type == "reach")
            btn_riichi = None
            if is_riichi:
                r_pos = self.screen_cfg.get("button_positions", {}).get("riichi", [1100, 850])
                btn_riichi = ScreenPoint(x=r_pos[0], y=r_pos[1])

            is_semi_auto = (self.output_mode == "semi_auto")
            mouse_steps = self.plan_mouse_interaction_sequence(
                target_point=coords,
                is_riichi_sequence=is_riichi,
                riichi_button_point=btn_riichi,
                is_semi_auto=is_semi_auto
            )

            # Live OS execution only if explicitly enabled (run in thread to prevent blocking event loop)
            if self.mouse_cfg.get("enable_pyautogui", False):
                await asyncio.to_thread(self._dispatch_os_mouse, mouse_steps)

        # Build user-friendly Japanese text recommendation
        act = candidate.action_type
        tile_str = format_tile_jp(candidate.tile)
        tile_uni = format_tile_jp(candidate.tile, include_unicode=True)

        if act in ("dahai", "reach"):
            sh = candidate.shanten_after
            sh_str = "和了" if sh == -1 else ("テンパイ" if sh == 0 else f"{sh}シャンテン")
            reach_tag = " [立直]" if act == "reach" else ""
            rec_text = f"[AI RECOMMEND] 推奨打牌: {tile_str}{reach_tag} | 期待値: {candidate.modified_q_value:+.1f} | 向聴数: {sh_str}"
        elif act in ("none", "pass"):
            rec_text = f"[AI RECOMMEND] 副露判断: スルー推奨 (パス) | 期待値: {candidate.modified_q_value:+.1f}"
        elif act == "pon":
            rec_text = f"[AI RECOMMEND] 副露判断: ポン推奨 ({tile_str}) | 期待値: {candidate.modified_q_value:+.1f}"
        elif act == "chi":
            rec_text = f"[AI RECOMMEND] 副露判断: チー推奨 ({tile_str}) | 期待値: {candidate.modified_q_value:+.1f}"
        elif act in ("kan", "daiminkan"):
            rec_text = f"[AI RECOMMEND] 副露判断: カン推奨 ({tile_str}) | 期待値: {candidate.modified_q_value:+.1f}"
        elif act == "hora":
            rec_text = f"[AI RECOMMEND] 和了判断: ロン和了推奨 ({tile_str}) | 期待値: {candidate.modified_q_value:+.1f}"
        else:
            rec_text = f"[AI RECOMMEND] 推奨アクション: {act.upper()} {tile_str}"

        if is_text_only:
            print(rec_text)

        # Build reasoning summary with factor breakdown
        ctx = candidate.metadata.get("strategy_context", {})
        breakdown = candidate.metadata.get("breakdown", {})
        rank = ctx.get("rank", "?")
        is_ooras = "Ooras" if ctx.get("is_ooras") else "Regular"
        bd_str = " ".join([f"{k}:{v:+}" if isinstance(v, (int, float)) else f"{k}:{v}" for k, v in breakdown.items()])
        summary = (
            f"Action: {candidate.action_type.upper()} {candidate.tile or ''} "
            f"| Raw Q: {candidate.raw_q_value:.2f} -> Mod Q: {candidate.modified_q_value:.2f} "
            f"| Danger: {candidate.danger_score:.2f} | Rank: {rank} ({is_ooras})"
        )
        if bd_str:
            summary += f" | [{bd_str}]"

        output = OutputAction(
            action_type=candidate.action_type,
            tile=candidate.tile,
            target_slot_index=slot_idx,
            screen_coordinates=coords,
            thinking_time_sec=delay_sec,
            computation_time_ms=comp_time_ms,
            reasoning_summary=summary,
            output_mode=self.output_mode,
            top_candidates=top_list,
            mouse_steps=mouse_steps,
            metadata={
                "shanten_after": candidate.shanten_after,
                "is_tsumogiri": candidate.is_tsumogiri,
                "strategy_context": ctx,
                "breakdown": breakdown,
                "recommendation_text": rec_text,
                "tile_display": tile_uni,
            }
        )

        self._last_recommendation = output
        self.top_candidates = top_list

        # Notify callback if registered (for GUI HUD)
        if self.on_recommendation_callback:
            try:
                round_st = state_mgr.round_state
                cb_payload = {
                    "type": candidate.action_type,
                    "target": candidate.tile,
                    "score": round(candidate.modified_q_value, 2),
                    "action_type": candidate.action_type,
                    "tile": candidate.tile,
                    "tile_display": tile_uni,
                    "recommendation_text": rec_text,
                    "shanten": candidate.shanten_after,
                    "shanten_str": "和了" if candidate.shanten_after == -1 else ("テンパイ" if candidate.shanten_after == 0 else f"{candidate.shanten_after}シャンテン"),
                    "expected_value": round(candidate.modified_q_value, 2),
                    "danger": round(candidate.danger_score, 2),
                    "top_candidates": top_list,
                    "round_info": f"{round_st.bakaze}{round_st.kyoku}局 {round_st.honba}本場" if round_st else "対局中",
                    "seat_name": ["東家", "南家", "西家", "北家"][state_mgr.my_actor_id] if 0 <= state_mgr.my_actor_id <= 3 else f"P{state_mgr.my_actor_id}",
                    "output_action": output
                }
                self.on_recommendation_callback(cb_payload)
            except Exception as e:
                print(f"[ActionController] Callback error: {e}")

        if self.output_mode in ("dashboard", "semi_auto") and (candidate.action_type != "none" or coords is not None):
            self.render_dashboard(output, state_mgr)

        return output

    def render_dashboard(self, action: OutputAction, state_mgr: MahjongStateManager) -> None:
        """Renders rich TUI HUD dashboard in assistant mode."""
        round_st = state_mgr.round_state
        my_p = state_mgr.my_player

        round_name = f"{round_st.bakaze}{round_st.kyoku}局 {round_st.honba}本場" if round_st else "N/A"
        rank_str = f"Rank {my_p.rank} ({my_p.score} pts)" if my_p else "N/A"
        ooras_tag = "[OORAS]" if round_st and round_st.is_ooras() else "[NORMAL]"
        active_riichis = state_mgr.get_active_riichi_actors()
        riichi_tag = f"ALERT: P{active_riichis} RIICHI" if active_riichis else "SAFE (No Riichi)"

        coords_str = f"({action.screen_coordinates.x}, {action.screen_coordinates.y})" if action.screen_coordinates else "N/A"

        print("\n+----------------------------------------------------------------------+")
        print(f"|  MAJSOUL AI HUD ASSISTANT  |  {round_name:<14}  |  {ooras_tag} {riichi_tag:<18}|")
        print("+----------------------------------------------------------------------+")
        print(f"|  Self: {rank_str:<22} | Latency: {action.thinking_time_sec:.2f}s | Comp: {action.computation_time_ms:.2f}ms  |")
        print("+----------------------------------------------------------------------+")
        print("|  TOP 3 RECOMMENDED CHOICES:                                          |")
        if action.top_candidates:
            for rank_idx, cand in enumerate(action.top_candidates[:3], 1):
                star = ">>>" if rank_idx == 1 else "   "
                act_str = f"{cand['action_type'].upper()} {cand['tile'] or ''}"
                print(f"| {star} #{rank_idx} {act_str:<12} | Mod Q: {cand['modified_q']:>6.2f} | Raw Q: {cand['raw_q']:>5.2f} | Danger: {cand['danger']:>4.2f} |")
        else:
            print(f"|  Primary: {action.action_type.upper()} {action.tile or ''} (Coords: {coords_str})")
        print("+----------------------------------------------------------------------+")
        if action.mouse_steps:
            print(f"|  Mouse Action Sequence ({len(action.mouse_steps)} steps planned):                           |")
            for step in action.mouse_steps:
                print(f"|   * [{step.step_type:<12}] -> ({step.target_point.x:>4}, {step.target_point.y:>4}) {step.note:<34} |")
            if any(s.step_type == "semi_auto_wait" for s in action.mouse_steps):
                print("|  >>> [SEMI-AUTO PAUSE] Cursor hovered. Click or press Space to confirm. <<< |")
            print("+----------------------------------------------------------------------+")

    def _dispatch_os_mouse(self, mouse_steps: List[MouseStep]) -> None:
        """Dispatches mouse steps to OS via pyautogui if installed and enabled."""
        try:
            import pyautogui
            for step in mouse_steps:
                if step.step_type in ("approach", "rest_retreat"):
                    pyautogui.moveTo(step.target_point.x, step.target_point.y, duration=step.duration_sec)
                elif step.step_type == "hover":
                    time.sleep(step.duration_sec)
                elif step.step_type == "mouse_down":
                    pyautogui.mouseDown(button="left")
                elif step.step_type == "mouse_up":
                    time.sleep(step.duration_sec)
                    pyautogui.mouseUp(button="left")
                elif step.step_type == "step_wait":
                    time.sleep(step.duration_sec)
                elif step.step_type == "semi_auto_wait":
                    # Semi-auto assist: Cursor is hovering, pause for human confirmation
                    semi_cfg = self.config.get("semi_auto", {})
                    timeout = semi_cfg.get("hover_timeout_sec", 15.0)
                    print(f"\n[Semi-Auto Assist] Cursor is poised over recommendation. Pausing (up to {timeout}s)...")
                    # Short pause to maintain hover animation
                    time.sleep(min(0.5, timeout))
        except ImportError:
            pass
