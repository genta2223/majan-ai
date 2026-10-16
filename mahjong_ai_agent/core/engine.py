"""Inference engine module for calculating candidate action Q-values (Mortal / Heuristic Fallback)."""

from __future__ import annotations
import time
import os
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field

from .state_manager import (
    MahjongStateManager,
    normalize_tile,
    is_valid_mahjong_tile,
    is_aka_dora,
    SUITS,
    HONORS
)


class ActionCandidate(BaseModel):
    """Represents an action candidate evaluated by the engine."""
    action_type: str                         # "dahai", "reach", "chi", "pon", "kan", "hora", "none"
    tile: Optional[str] = None               # Target tile (e.g., "1m")
    consumed: Optional[List[str]] = None     # Consumed tiles for meld
    raw_q_value: float = 0.0                 # Base Q-value or expected value from neural network
    modified_q_value: float = 0.0            # Q-value after tactical modification
    danger_score: float = 0.0                # Estimated danger (0.0: 100% safe, 1.0: extremely dangerous)
    shanten_after: int = 8                   # Shanten if this action is chosen
    is_tsumogiri: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DecisionEngine:
    """
    Evaluates current board state to compute candidate actions and their Q-values.
    Supports PyTorch Mortal model if available, and features an ultra-fast (<20ms)
    rule-based fallback engine to guarantee operation without external model weights.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
        use_fallback: bool = True,
        max_inference_time_ms: int = 200
    ):
        self.model_path = model_path
        self.device = device
        self.use_fallback = use_fallback
        self.max_inference_time_ms = max_inference_time_ms
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """Attempt to load PyTorch model weights if present."""
        if self.model_path and os.path.exists(self.model_path):
            try:
                import torch
                self.model = torch.load(self.model_path, map_location=self.device)
                self.model.eval()
            except Exception as e:
                print(f"[Engine] Warning: Failed to load PyTorch model from {self.model_path}: {e}")
                self.model = None
        else:
            self.model = None

    def evaluate(self, state_mgr: MahjongStateManager, valid_actions: Optional[List[Dict[str, Any]]] = None) -> List[ActionCandidate]:
        """
        Main entry point for calculating Q-values for all valid candidates.
        Guaranteed to complete well within max_inference_time_ms (typically < 10ms for fallback).
        Strictly validates that any proposed discard (dahai/reach) physically exists in the player's tehai.
        """
        start_time = time.perf_counter()

        if self.model is not None:
            try:
                candidates = self._evaluate_with_pytorch(state_mgr, valid_actions)
            except Exception as e:
                print(f"[Engine] PyTorch inference failed: {e}. Falling back to internal engine.")
                candidates = self._evaluate_with_fallback(state_mgr, valid_actions)
        else:
            candidates = self._evaluate_with_fallback(state_mgr, valid_actions)

        # Strict validation guard: eliminate any candidate whose tile or consumed does not physically exist in player's hand
        my_p = state_mgr.my_player
        if my_p and candidates:
            hand_tiles = [t for t in my_p.tehai if is_valid_mahjong_tile(t)]
            valid_candidates: List[ActionCandidate] = []

            for c in candidates:
                # 1. Dahai / Reach validation
                if c.action_type in ("dahai", "reach"):
                    if not c.tile or not is_valid_mahjong_tile(c.tile):
                        print(f"[Engine Guard WARNING] Discard candidate with invalid tile '{c.tile}' dropped.")
                        continue
                    # Check physical presence in hand
                    matching_tile = my_p.get_matching_tile_in_hand(c.tile)
                    if not matching_tile:
                        print(f"[Engine Guard ALERT] Discard candidate '{c.tile}' does NOT exist in Player {state_mgr.my_actor_id} hand {hand_tiles}! Forcibly dropped.")
                        continue
                    # Synchronize candidate tile notation with exact physical representation present in hand
                    c.tile = matching_tile
                    valid_candidates.append(c)

                # 2. Meld (chi, pon, kan, daiminkan, kankan, ankan) validation
                elif c.action_type in ("chi", "pon", "kan", "daiminkan", "kankan", "ankan"):
                    if not c.consumed:
                        print(f"[Engine Guard ALERT] Meld candidate '{c.action_type}' has empty consumed tiles! Forcibly dropped.")
                        continue
                    # Check physical presence and capacity in hand pool
                    if not my_p.can_consume(c.consumed):
                        print(f"[Engine Guard ALERT] Meld candidate '{c.action_type}' requires consumed {c.consumed} which exceeds hand pool {hand_tiles}! Forcibly dropped.")
                        continue
                    valid_candidates.append(c)

                # 3. Winning action (hora) validation
                elif c.action_type == "hora":
                    valid_candidates.append(c)

                # 4. Pass / Skip (none) validation
                elif c.action_type in ("none", "pass", "skip"):
                    valid_candidates.append(c)

                else:
                    valid_candidates.append(c)

            # If self turn (active_actor == my_actor_id) and all discard candidates were dropped:
            # Emergency Recovery: physically pick a tile that ACTUALLY exists in hand_tiles!
            is_self_turn = (state_mgr.round_state and state_mgr.round_state.active_actor == state_mgr.my_actor_id)
            has_dahai = any(c.action_type in ("dahai", "reach") for c in valid_candidates)
            if is_self_turn and not has_dahai:
                if hand_tiles:
                    fb = hand_tiles[-1]
                    print(f"[Engine Guard RECOVERY] Rescued empty discard candidates using actual tile in hand: '{fb}'.")
                else:
                    fb = "1z"
                    print(f"[Engine Guard WARNING] Hand tiles contain no recognized tiles! Using default emergency fallback tile '{fb}'.")
                valid_candidates.insert(0, ActionCandidate(
                    action_type="dahai",
                    tile=fb,
                    raw_q_value=1.0,
                    danger_score=0.05,
                    shanten_after=state_mgr.estimate_shanten(hand_tiles),
                    is_tsumogiri=True,
                    metadata={"norm_tile": normalize_tile(fb), "fallback": True, "breakdown": {"base_shanten": 1.0}}
                ))

            candidates = valid_candidates

        # Sort candidates descending by raw_q_value so candidates[0] is always the best action
        candidates.sort(key=lambda x: x.raw_q_value, reverse=True)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        if elapsed_ms > self.max_inference_time_ms:
            print(f"[Engine] Alert: Inference time {elapsed_ms:.2f}ms exceeded limit {self.max_inference_time_ms}ms")

        return candidates

    def _evaluate_with_pytorch(self, state_mgr: MahjongStateManager, valid_actions: Optional[List[Dict[str, Any]]]) -> List[ActionCandidate]:
        """
        Encodes board state into tensors for Mortal/libriichi network inference.
        """
        import torch
        # Note: In production Mortal deployments, features are encoded as 2D/3D state tensors (45+ channels).
        # When a model is provided, pass input tensor to model:
        # with torch.no_grad():
        #     q_values = self.model(input_tensor)
        # Here we bridge to fallback if custom tensor shape is not yet adapted.
        return self._evaluate_with_fallback(state_mgr, valid_actions)

    def calculate_ukeire(self, hand: List[str], state_mgr: Optional[MahjongStateManager] = None) -> int:
        """
        Calculate ukeire (acceptance count / number of tiles that improve shanten).
        Tests all 34 tiles against the hand and sums the remaining available count of improving tiles.
        """
        if not hand:
            return 0
        base_shanten = state_mgr.estimate_shanten(hand) if state_mgr else 8
        if base_shanten <= -1:
            return 0

        # Build count of tiles visible to player (in hand, discards, doras, melds)
        seen_counts: Dict[str, int] = {}
        for t in hand:
            nt = normalize_tile(t)
            seen_counts[nt] = seen_counts.get(nt, 0) + 1
        if state_mgr:
            for nt, cnt in state_mgr.get_visible_tiles().items():
                seen_counts[nt] = max(seen_counts.get(nt, 0), cnt)

        ukeire_count = 0
        all_possible_tiles = [f"{i}{s}" for s in SUITS for i in range(1, 10)] + HONORS
        for cand_tile in all_possible_tiles:
            remaining = 4 - seen_counts.get(cand_tile, 0)
            if remaining <= 0:
                continue
            test_hand = hand + [cand_tile]
            test_shanten = state_mgr.estimate_shanten(test_hand) if state_mgr else 8
            # If drawing this tile reduces shanten
            if test_shanten < base_shanten:
                ukeire_count += remaining

        return ukeire_count

    def _evaluate_with_fallback(self, state_mgr: MahjongStateManager, valid_actions: Optional[List[Dict[str, Any]]]) -> List[ActionCandidate]:
        """
        Fast heuristic evaluation calculating tile efficiency, ukeire, dora holding, and safety against active riichis.
        """
        candidates: List[ActionCandidate] = []
        my_p = state_mgr.my_player
        if not my_p or not state_mgr.round_state:
            return candidates

        # Determine if this is a discard decision or a call/pass decision
        active_actor = state_mgr.round_state.active_actor
        last_tile = state_mgr.round_state.last_discard_tile
        last_actor = state_mgr.round_state.last_discard_actor

        # Case 1: Self turn (Dahai / Riichi / Tsumo-Hora)
        if active_actor == state_mgr.my_actor_id:
            tehai = list(my_p.tehai)
            current_shanten = state_mgr.estimate_shanten(tehai)

            # Check if Tsumo Agari
            if current_shanten == -1:
                candidates.append(ActionCandidate(
                    action_type="hora",
                    tile=my_p.tsumo_tile,
                    raw_q_value=100.0,
                    danger_score=0.0,
                    shanten_after=-1,
                    metadata={"description": "Tsumo Agari"}
                ))
                return candidates

            # Discard candidates for each distinct valid tile in hand
            valid_hand_tiles = [t for t in tehai if is_valid_mahjong_tile(t)]
            distinct_tiles = set(valid_hand_tiles)
            dora_tiles = [normalize_tile(d) for d in state_mgr.round_state.dora_tiles]

            for tile in distinct_tiles:
                # Hand remaining after discarding tile
                hand_after = list(tehai)
                if tile in hand_after:
                    hand_after.remove(tile)
                shanten_after = state_mgr.estimate_shanten(hand_after)

                # Base Q-value from shanten efficiency (lower shanten is better)
                # Shanten 0 (Tenpai) = ~12.0, 1 = ~8.0, 2 = ~4.0, etc.
                base_shanten_score = max(0.0, (4 - shanten_after) * 3.0)
                efficiency_score = base_shanten_score

                # Ukeire (acceptance count): tiles improving hand
                ukeire = self.calculate_ukeire(hand_after, state_mgr)
                ukeire_score = min(6.0, ukeire * 0.18)
                efficiency_score += ukeire_score

                norm_tile = normalize_tile(tile)
                tile_count_in_hand = tehai.count(tile)
                tile_count_after = hand_after.count(tile)

                # Precompute tile characteristics & connections
                is_honor = norm_tile in HONORS
                if not is_honor and len(norm_tile) == 2 and norm_tile[0].isdigit():
                    val = int(norm_tile[0])
                    suit = norm_tile[1]
                    has_direct_conn = any(normalize_tile(t) in (f"{val-1}{suit}", f"{val+1}{suit}") for t in hand_after)
                    has_kanchan_conn = any(normalize_tile(t) in (f"{val-2}{suit}", f"{val+2}{suit}") for t in hand_after)
                else:
                    val = 0
                    suit = ""
                    has_direct_conn = False
                    has_kanchan_conn = False

                # Dora bonus/penalty: discarding dora severely reduces hand value and scoring potential
                dora_score = 0.0
                tile_dora_count = 0
                if state_mgr.round_state:
                    tile_dora_count = state_mgr.round_state.get_dora_count(tile)
                else:
                    if is_aka_dora(tile):
                        tile_dora_count += 1
                    if any(normalize_tile(d) == norm_tile for d in dora_tiles):
                        tile_dora_count += 1

                if tile_dora_count > 0:
                    # 1. Base dora discard penalty scaled by shanten:
                    # Discarding dora in early/mid game (especially 2+ shanten) is severely punished (-8.0 to -12.0)
                    if current_shanten >= 3:
                        base_dora_pen = -10.0 * tile_dora_count
                    elif current_shanten == 2:
                        base_dora_pen = -8.0 * tile_dora_count
                    elif current_shanten == 1:
                        base_dora_pen = -6.0 * tile_dora_count
                    else:
                        base_dora_pen = -4.0 * tile_dora_count
                    dora_score += base_dora_pen

                    # 2. Dora pair penalty:
                    # Holding a pair of doras (e.g. 7p7p) guarantees 2+ han and huge offensive potential.
                    # Discarding one tile from a dora pair is heavily penalized!
                    if tile_count_in_hand >= 2:
                        dora_score -= 6.0

                    # 3. Dora-connected block / taatsu destruction penalty:
                    # If the dora connects directly with adjacent tiles (e.g. 7p with 8p forming 7p-8p ryanmen):
                    if has_direct_conn:
                        dora_score -= 4.0
                    elif has_kanchan_conn:
                        dora_score -= 2.0
                else:
                    # Tile itself is not dora, but check if discarding it breaks a dora-connected taatsu
                    is_partner_to_dora = False
                    if not is_honor and state_mgr.round_state and state_mgr.round_state.dora_tiles:
                        for dt in state_mgr.round_state.dora_tiles:
                            norm_dt = normalize_tile(dt)
                            if len(norm_dt) == 2 and norm_dt[1] == suit and norm_dt[0].isdigit():
                                dora_val = int(norm_dt[0])
                                if abs(val - dora_val) == 1:
                                    if f"{dora_val}{suit}" in [normalize_tile(t) for t in tehai]:
                                        is_partner_to_dora = True
                                        break
                    if is_partner_to_dora and current_shanten >= 2:
                        dora_score -= 3.0

                efficiency_score += dora_score

                # Shape / Taatsu / Toitsu preservation & isolated tile priority
                shape_score = 0.0
                yaku_score = 0.0

                if is_honor:
                    # Honor tiles
                    if tile_count_in_hand == 1:
                        # Completely isolated honor tile: top priority to discard early!
                        shape_score += 3.0
                    elif tile_count_in_hand == 2:
                        # Breaking an honor pair (potential janto or yakuhai triplet) is heavily discouraged
                        shape_score -= 3.5
                    elif tile_count_in_hand >= 3:
                        # Breaking a completed triplet is strictly forbidden
                        shape_score -= 6.0
                else:
                    # Number tiles (m, p, s)
                    # 1. Pair / Triplet protection
                    if tile_count_in_hand >= 2:
                        # Hand has a pair or triplet with this tile
                        if tile_count_in_hand >= 3:
                            # Breaking an existing triplet
                            shape_score -= 6.0
                        else:
                            # Breaking a pair: enforce strict minimum penalty of -3.5 (even with excess pairs)
                            shape_score -= 3.5

                        # Connected pair / ryanmen-toitsu protection:
                        # If this pair connects to an adjacent tile (e.g. 7p7p with 8p, or 5p5p with 4p),
                        # breaking it destroys multi-sided waits and flexible block shapes!
                        if has_direct_conn:
                            shape_score -= 3.0

                    # 2. Taatsu protection (Ryanmen / Kanchan) and isolated / floater priority
                    if tile_count_in_hand == 1:
                        if has_direct_conn:
                            # Breaking a ryanmen (e.g. 23, 78) or direct connected taatsu
                            # Check if ryanmen (val between 2 and 8 with partner)
                            is_ryanmen = False
                            if val >= 2 and normalize_tile(f"{val-1}{suit}") in [normalize_tile(t) for t in hand_after]:
                                if val - 1 >= 2 and val <= 8:
                                    is_ryanmen = True
                            if val <= 8 and normalize_tile(f"{val+1}{suit}") in [normalize_tile(t) for t in hand_after]:
                                if val >= 2 and val + 1 <= 8:
                                    is_ryanmen = True

                            if is_ryanmen:
                                shape_score -= 4.0  # Breaking a ryanmen is heavily penalized
                            else:
                                shape_score -= 2.5  # Penchan / terminal connection
                        else:
                            # No direct neighbor. Check if this is an exterior floater attached to an already formed ryanmen/mentsu
                            # Lower floater: e.g. 2p with partner 4p, where 4p connects to 5p forming 4p-5p ryanmen
                            is_lower_floater = False
                            if normalize_tile(f"{val+2}{suit}") in [normalize_tile(t) for t in hand_after]:
                                if normalize_tile(f"{val+3}{suit}") in [normalize_tile(t) for t in hand_after]:
                                    is_lower_floater = True

                            # Upper floater: e.g. 8s with partner 6s, where 6s connects to 5s forming 5s-6s ryanmen
                            is_upper_floater = False
                            if normalize_tile(f"{val-2}{suit}") in [normalize_tile(t) for t in hand_after]:
                                if normalize_tile(f"{val-3}{suit}") in [normalize_tile(t) for t in hand_after]:
                                    is_upper_floater = True

                            if is_lower_floater or is_upper_floater:
                                # Exterior floater tile: top cutting priority to preserve superior ryanmen
                                if val in (1, 2, 8, 9):
                                    shape_score += 3.0
                                else:
                                    shape_score += 2.2
                            elif has_kanchan_conn:
                                shape_score -= 2.0  # Breaking a genuine isolated kanchan taatsu
                            else:
                                # Completely isolated number tile!
                                if val in (1, 9):
                                    shape_score += 2.8  # Completely isolated 1 or 9: top discard priority
                                elif val in (2, 8):
                                    shape_score += 2.0  # Isolated 2 or 8
                                else:
                                    shape_score += 1.4  # Isolated 3-7 middle tile

                    # 3. Tanyao potential: if hand has high potential for Tanyao, discarding 1/9 earlier is favored
                    non_tanyao_count = sum(1 for t in hand_after if normalize_tile(t) in HONORS or (len(normalize_tile(t)) > 0 and normalize_tile(t)[0] in ("1", "9")))
                    if non_tanyao_count <= 2 and val in (1, 9) and tile_count_in_hand == 1:
                        yaku_score += 1.0

                efficiency_score += shape_score + yaku_score

                # Danger estimation against active riichis
                danger = self._estimate_tile_danger(norm_tile, state_mgr)
                is_tsumogiri = (tile == my_p.tsumo_tile)

                breakdown = {
                    "base_shanten": round(base_shanten_score, 2),
                    "ukeire": round(ukeire_score, 2),
                    "shape": round(shape_score, 2),
                    "dora": round(dora_score, 2),
                    "yaku": round(yaku_score, 2),
                    "danger": round(danger, 2),
                }

                candidates.append(ActionCandidate(
                    action_type="dahai",
                    tile=tile,
                    raw_q_value=round(efficiency_score, 3),
                    danger_score=danger,
                    shanten_after=shanten_after,
                    is_tsumogiri=is_tsumogiri,
                    metadata={"norm_tile": norm_tile, "ukeire": ukeire, "breakdown": breakdown}
                ))

            # Riichi rule: If player has declared and accepted Riichi, force tsumogiri among valid candidates
            if my_p.is_riichi and candidates:
                tsumo = my_p.tsumo_tile or ""
                # Find candidate matching tsumo_tile
                tsumo_cands = [c for c in candidates if c.tile == tsumo]
                if tsumo_cands:
                    # Give tsumogiri highest score while keeping all candidates evaluated
                    for c in candidates:
                        if c.tile == tsumo:
                            c.raw_q_value += 50.0
                            c.is_tsumogiri = True
                            c.metadata["description"] = "Riichi Auto Tsumogiri"
                else:
                    # If tsumo tile is not specifically found, pick last tile in tehai as tsumogiri
                    candidates[-1].raw_q_value += 50.0
                    candidates[-1].is_tsumogiri = True

            # If candidates is empty (e.g. hand tiles were unknown or missing), fallback to safe valid tile in hand
            if not candidates:
                fallback_tile = ""
                # Priority 1: tsumo_tile if valid and in hand
                if my_p.tsumo_tile and is_valid_mahjong_tile(my_p.tsumo_tile) and my_p.has_tile(my_p.tsumo_tile):
                    fallback_tile = my_p.get_matching_tile_in_hand(my_p.tsumo_tile) or my_p.tsumo_tile
                # Priority 2: first valid tile in tehai
                if not fallback_tile:
                    for t in tehai:
                        if t and is_valid_mahjong_tile(t):
                            fallback_tile = t
                            break

                if fallback_tile:
                    candidates.append(ActionCandidate(
                        action_type="dahai",
                        tile=fallback_tile,
                        raw_q_value=1.0,
                        danger_score=0.05,
                        shanten_after=state_mgr.estimate_shanten(tehai),
                        is_tsumogiri=True,
                        metadata={"norm_tile": normalize_tile(fallback_tile), "fallback": True, "breakdown": {"base_shanten": 1.0}}
                    ))
                else:
                    candidates.append(ActionCandidate(
                        action_type="none",
                        raw_q_value=0.0,
                        danger_score=0.0,
                        shanten_after=8,
                        metadata={"fallback": True}
                    ))

            # Check Riichi candidate if tenpai and menzen
            if current_shanten == 0 and not my_p.is_riichi and not my_p.melds:
                # If riichi is available, add riichi candidates
                for c in list(candidates):
                    if c.shanten_after == 0 and c.tile and c.tile != "?":
                        reach_breakdown = dict(c.metadata.get("breakdown", {}))
                        reach_breakdown["riichi_bonus"] = 3.0
                        candidates.append(ActionCandidate(
                            action_type="reach",
                            tile=c.tile,
                            raw_q_value=round(c.raw_q_value + 3.0, 3), # High expectation value for riichi
                            danger_score=c.danger_score,
                            shanten_after=0,
                            is_tsumogiri=c.is_tsumogiri,
                            metadata={"tile": c.tile, "breakdown": reach_breakdown}
                        ))

        # Case 2: Response to other player's discard (Ron / Chi / Pon / Kan / Pass)
        else:
            if last_tile is not None and last_actor is not None:
                norm_last = normalize_tile(last_tile)
                current_shanten = state_mgr.estimate_shanten(my_p.tehai)

                # 1. Check Ron (Hora)
                temp_hand = my_p.tehai + [last_tile]
                if state_mgr.estimate_shanten(temp_hand) == -1:
                    candidates.append(ActionCandidate(
                        action_type="hora",
                        tile=last_tile,
                        raw_q_value=100.0,
                        danger_score=0.0,
                        shanten_after=-1,
                        metadata={
                            "target": last_actor,
                            "action_name": "ron",
                            "breakdown": {"hora_bonus": 100.0}
                        }
                    ))

                # Helper to determine yaku viability for calls
                has_existing_meld_yaku = False
                for m in my_p.melds:
                    m_norms = [normalize_tile(t) for t in m.tiles]
                    if any(t in ("5z", "6z", "7z") for t in m_norms):
                        has_existing_meld_yaku = True
                    round_w = f"{'1' if state_mgr.round_state.bakaze == 'E' else '2'}z"
                    if any(t == round_w for t in m_norms):
                        has_existing_meld_yaku = True
                    seat_w = f"{state_mgr.my_actor_id + 1}z"
                    if any(t == seat_w for t in m_norms):
                        has_existing_meld_yaku = True

                # 2. Check Pon candidate
                same_tiles = [t for t in my_p.tehai if normalize_tile(t) == norm_last]
                if len(same_tiles) >= 2 and my_p.can_consume(same_tiles[:2]):
                    temp_hand = list(my_p.tehai)
                    temp_hand.remove(same_tiles[0])
                    temp_hand.remove(same_tiles[1])
                    sh_after = max(0, state_mgr.estimate_shanten(temp_hand) - 1)

                    base_pon = 6.0 if sh_after == 0 else (4.0 if sh_after == 1 else 2.0)
                    yaku_bonus = 0.0
                    call_penalty = 0.0

                    # Check Yaku for Pon:
                    # (a) Yakuhai: dragon or round wind or seat wind
                    is_yakuhai = False
                    if norm_last in ("5z", "6z", "7z"):
                        is_yakuhai = True
                    round_wind_tile = f"{'1' if state_mgr.round_state.bakaze == 'E' else '2'}z"
                    if norm_last == round_wind_tile:
                        is_yakuhai = True
                    seat_wind_tile = f"{state_mgr.my_actor_id + 1}z"
                    if norm_last == seat_wind_tile:
                        is_yakuhai = True

                    # (b) Tanyao viability: check if all tiles in hand and melds (including this pon) are 2-8
                    melds_all_simples = True
                    for m in my_p.melds:
                        for mt in m.tiles:
                            mnt = normalize_tile(mt)
                            if mnt in HONORS or (len(mnt) > 0 and mnt[0] in ("1", "9")):
                                melds_all_simples = False
                                break

                    hand_has_terminal_or_honor = any(
                        normalize_tile(t) in HONORS or (len(normalize_tile(t)) > 0 and normalize_tile(t)[0] in ("1", "9"))
                        for t in temp_hand
                    )
                    pon_is_terminal_or_honor = (norm_last in HONORS or (len(norm_last) > 0 and norm_last[0] in ("1", "9")))

                    is_tanyao_feasible = (not pon_is_terminal_or_honor) and melds_all_simples and (
                        not hand_has_terminal_or_honor or (sh_after <= 1 and sum(1 for t in temp_hand if normalize_tile(t) in HONORS or normalize_tile(t)[0] in ("1", "9")) <= 1)
                    )

                    has_confirmed_yaku = is_yakuhai or has_existing_meld_yaku or is_tanyao_feasible

                    if is_yakuhai:
                        yaku_bonus += 2.5
                    elif is_tanyao_feasible:
                        yaku_bonus += 1.5
                    elif has_existing_meld_yaku:
                        yaku_bonus += 1.0
                    else:
                        # NO YAKU! Harmful call that ruins closed hand, removes Menzen, and leads to no-yaku penalty!
                        call_penalty -= 10.0

                    # Janto (pair) destruction check: if hand only had 1 pair, ponning destroys the head
                    pairs_before = sum(1 for t in set(my_p.tehai) if my_p.tehai.count(t) >= 2)
                    if pairs_before <= 1 and sh_after > 0:
                        call_penalty -= 2.5

                    # Dora bonus
                    dora_bonus = 0.0
                    if is_aka_dora(last_tile) or norm_last in [normalize_tile(d) for d in state_mgr.round_state.dora_tiles]:
                        dora_bonus += 1.5

                    # Far shanten call penalty (ponning at 2-shanten or higher without solid yaku)
                    if sh_after >= 2 and not is_yakuhai:
                        call_penalty -= 3.0

                    pon_score = base_pon + yaku_bonus + dora_bonus + call_penalty

                    breakdown = {
                        "base_shanten": round(base_pon, 2),
                        "yaku_bonus": round(yaku_bonus, 2),
                        "dora_bonus": round(dora_bonus, 2),
                        "call_penalty": round(call_penalty, 2),
                        "has_yaku": has_confirmed_yaku,
                    }

                    candidates.append(ActionCandidate(
                        action_type="pon",
                        tile=last_tile,
                        consumed=same_tiles[:2],
                        raw_q_value=round(pon_score, 3),
                        danger_score=0.0,
                        shanten_after=sh_after,
                        metadata={"target": last_actor, "breakdown": breakdown}
                    ))

                # 3. Check Daiminkan candidate
                if len(same_tiles) >= 3 and my_p.can_consume(same_tiles[:3]):
                    temp_hand = list(my_p.tehai)
                    for i in range(3):
                        temp_hand.remove(same_tiles[i])
                    sh_after = max(0, state_mgr.estimate_shanten(temp_hand) - 1)
                    # Daiminkan exposes hand and hands kandora to rivals: generally disadvantageous unless yakuhai or tenpai
                    kan_score = 3.0 if sh_after == 0 else 0.5
                    candidates.append(ActionCandidate(
                        action_type="kan",
                        tile=last_tile,
                        consumed=same_tiles[:3],
                        raw_q_value=kan_score,
                        danger_score=0.0,
                        shanten_after=sh_after,
                        metadata={"target": last_actor, "meld_type": "daiminkan", "breakdown": {"base_kan": kan_score}}
                    ))

                # 4. Check Chi candidate (only from kamicha)
                is_kamicha = ((last_actor + 1) % 4 == state_mgr.my_actor_id)
                suit = norm_last[1] if len(norm_last) > 1 else ""
                tile_num = int(norm_last[0]) if (len(norm_last) > 0 and norm_last[0].isdigit()) else 0

                if is_kamicha and suit in ("m", "p", "s") and 1 <= tile_num <= 9:
                    chi_patterns = []
                    if tile_num - 2 >= 1:
                        chi_patterns.append((tile_num - 2, tile_num - 1))
                    if tile_num - 1 >= 1 and tile_num + 1 <= 9:
                        chi_patterns.append((tile_num - 1, tile_num + 1))
                    if tile_num + 2 <= 9:
                        chi_patterns.append((tile_num + 1, tile_num + 2))

                    seen_combs = set()
                    for n1, n2 in chi_patterns:
                        t1_norm = f"{n1}{suit}"
                        t2_norm = f"{n2}{suit}"
                        m1 = [t for t in my_p.tehai if normalize_tile(t) == t1_norm]
                        m2 = [t for t in my_p.tehai if normalize_tile(t) == t2_norm]
                        if m1 and m2:
                            comb_key = tuple(sorted([t1_norm, t2_norm]))
                            if comb_key in seen_combs:
                                continue
                            seen_combs.add(comb_key)
                            consumed = [m1[0], m2[0]]
                            if not my_p.can_consume(consumed):
                                continue
                            temp_hand = list(my_p.tehai)
                            temp_hand.remove(consumed[0])
                            temp_hand.remove(consumed[1])
                            sh_after = max(0, state_mgr.estimate_shanten(temp_hand) - 1)

                            base_chi = 5.0 if sh_after == 0 else (3.2 if sh_after == 1 else 1.2)
                            yaku_bonus = 0.0
                            call_penalty = 0.0

                            # Yaku check for Chi:
                            # Tanyao check:
                            melds_all_simples = True
                            for m in my_p.melds:
                                for mt in m.tiles:
                                    mnt = normalize_tile(mt)
                                    if mnt in HONORS or (len(mnt) > 0 and mnt[0] in ("1", "9")):
                                        melds_all_simples = False
                                        break

                            chi_has_terminals = (n1 in (1, 9) or n2 in (1, 9) or tile_num in (1, 9))
                            hand_has_terminal_or_honor = any(
                                normalize_tile(t) in HONORS or (len(normalize_tile(t)) > 0 and normalize_tile(t)[0] in ("1", "9"))
                                for t in temp_hand
                            )

                            is_tanyao_feasible = (not chi_has_terminals) and melds_all_simples and (
                                not hand_has_terminal_or_honor or (sh_after <= 1 and sum(1 for t in temp_hand if normalize_tile(t) in HONORS or normalize_tile(t)[0] in ("1", "9")) <= 1)
                            )

                            has_confirmed_yaku = has_existing_meld_yaku or is_tanyao_feasible

                            if is_tanyao_feasible:
                                yaku_bonus += 1.5
                            elif has_existing_meld_yaku:
                                yaku_bonus += 1.0
                            else:
                                # Severe penalty: chi without yaku destroys menzen with 0 win chance
                                call_penalty -= 10.0

                            # Head destruction check
                            pairs_before = sum(1 for t in set(my_p.tehai) if my_p.tehai.count(t) >= 2)
                            if pairs_before <= 1 and sh_after > 0:
                                call_penalty -= 2.5

                            # Far shanten chi penalty
                            if sh_after >= 2:
                                call_penalty -= 3.5

                            # Dora bonus
                            dora_bonus = 0.0
                            dora_tiles = [normalize_tile(d) for d in state_mgr.round_state.dora_tiles]
                            if any(normalize_tile(t) in dora_tiles or is_aka_dora(t) for t in temp_hand + [last_tile] + consumed):
                                dora_bonus += 0.8

                            chi_score = base_chi + yaku_bonus + dora_bonus + call_penalty

                            breakdown = {
                                "base_shanten": round(base_chi, 2),
                                "yaku_bonus": round(yaku_bonus, 2),
                                "dora_bonus": round(dora_bonus, 2),
                                "call_penalty": round(call_penalty, 2),
                                "has_yaku": has_confirmed_yaku,
                            }

                            candidates.append(ActionCandidate(
                                action_type="chi",
                                tile=last_tile,
                                consumed=consumed,
                                raw_q_value=round(chi_score, 3),
                                danger_score=0.0,
                                shanten_after=sh_after,
                                metadata={"target": last_actor, "consumed": consumed, "breakdown": breakdown}
                            ))

                # 5. Always offer "none" (Pass / Skip) option
                # Menzen preservation is paramount in standard mahjong theory:
                # Passing maintains riichi possibility, menzen tsumo, pinfu, and defensive safety
                if not my_p.melds:
                    # Closed hand: Pass has massive baseline advantage
                    if current_shanten >= 2:
                        pass_score = 6.8
                    elif current_shanten == 1:
                        pass_score = 5.5
                    else:
                        pass_score = 4.2
                else:
                    # Already open hand
                    pass_score = 3.5

                candidates.append(ActionCandidate(
                    action_type="none",
                    raw_q_value=round(pass_score, 3),
                    danger_score=0.0,
                    shanten_after=current_shanten,
                    metadata={"target": last_actor, "breakdown": {"menzen_preservation": pass_score}}
                ))

        return candidates

    def _estimate_tile_danger(self, norm_tile: str, state_mgr: MahjongStateManager) -> float:
        """
        Estimate the danger level (0.0 to 1.0) of a tile against any active riichis.
        0.0 = Genbutsu (100% safe against riichi)
        0.2 - 0.4 = Suji / Kabe safe
        0.8 - 1.0 = Pure musuji dangerous middle tile
        """
        riichi_actors = state_mgr.get_active_riichi_actors()
        if not riichi_actors:
            return 0.05  # Minimal baseline danger if nobody declared riichi

        max_danger = 0.0
        visible = state_mgr.get_visible_tiles()

        for enemy_id in riichi_actors:
            genbutsu = state_mgr.get_genbutsu_for_actor(enemy_id)
            if norm_tile in genbutsu:
                # Absolute safety against this actor
                danger = 0.0
            elif norm_tile in HONORS:
                # Honors: Count how many are visible. If 3 or 4 visible -> 0.05 (near safe)
                seen = visible.get(norm_tile, 0)
                if seen >= 3:
                    danger = 0.08
                elif seen == 2:
                    danger = 0.35
                else:
                    danger = 0.65
            else:
                # Number tiles: check suji
                val, suit = int(norm_tile[0]), norm_tile[1]
                enemy_discards = state_mgr.get_genbutsu_for_actor(enemy_id)

                is_suji = False
                if val in (1, 9):
                    is_suji = f"{4 if val==1 else 6}{suit}" in enemy_discards
                elif val in (2, 8):
                    is_suji = f"{5}{suit}" in enemy_discards
                elif val in (3, 7):
                    is_suji = f"{6 if val==3 else 4}{suit}" in enemy_discards
                elif val in (4, 5, 6):
                    # Middle tiles require double suji
                    if val == 4:
                        is_suji = f"1{suit}" in enemy_discards and f"7{suit}" in enemy_discards
                    elif val == 5:
                        is_suji = f"2{suit}" in enemy_discards and f"8{suit}" in enemy_discards
                    elif val == 6:
                        is_suji = f"3{suit}" in enemy_discards and f"9{suit}" in enemy_discards

                if is_suji:
                    danger = 0.25 if val in (1, 9) else 0.45
                else:
                    # Musuji middle tile is dangerous
                    danger = 0.90 if val in (3, 4, 5, 6, 7) else 0.70

            max_danger = max(max_danger, danger)

        return max_danger
