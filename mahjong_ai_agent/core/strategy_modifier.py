"""Tactical modification layer for dynamically adjusting Q-values based on rank, score differentials, and round situation."""

from __future__ import annotations
from typing import List, Dict, Any, Optional
import os
import copy

from .state_manager import MahjongStateManager
from .engine import ActionCandidate
from .opponent_profiler import OpponentProfiler


class StrategyModifier:
    """
    Dynamically adjusts action Q-values (attack vs. defense) according to:
    1. Current player rank (1st to 4th)
    2. Score margins against 1st place and 4th place
    3. Round state (e.g. Ooras, All-Last, Dealer status)
    4. Active riichi threats, self shanten, and opponent behavioral exploit profiles
    """

    def __init__(self, config: Dict[str, Any], profiler: Optional[OpponentProfiler] = None):
        self.config = config
        self.strat_cfg = config.get("strategy", {})
        self.rank_multipliers = self.strat_cfg.get("rank_multipliers", {})
        self.ooras_cfg = self.strat_cfg.get("ooras", {})
        self.riichi_cfg = self.strat_cfg.get("riichi_response", {})

        prof_path = config.get("opponent_profiles_path")
        self.profiler = profiler or (OpponentProfiler(prof_path) if prof_path and os.path.exists(prof_path) else None)
        self.custom_exploit: Optional[Dict[str, float]] = None

    def modify_candidates(
        self,
        candidates: List[ActionCandidate],
        state_mgr: MahjongStateManager
    ) -> List[ActionCandidate]:
        """
        Apply strategy modifications to candidates and return modified list sorted by modified_q_value descending.
        """
        if not candidates:
            return []

        my_p = state_mgr.my_player
        if not my_p or not state_mgr.round_state:
            # Fallback to unmodified raw values
            for c in candidates:
                c.modified_q_value = c.raw_q_value
            return sorted(candidates, key=lambda x: x.modified_q_value, reverse=True)

        my_rank = my_p.rank
        is_ooras = state_mgr.round_state.is_ooras()
        is_dealer = my_p.is_dealer
        active_riichis = state_mgr.get_active_riichi_actors()
        current_shanten = state_mgr.estimate_shanten(my_p.tehai)

        # 1. Base weights by rank
        rank_settings = self.rank_multipliers.get(my_rank, {"defense_scale": 1.0, "attack_scale": 1.0})
        defense_scale = rank_settings.get("defense_scale", 1.0) * self.strat_cfg.get("defense_bias", 1.0)
        attack_scale = rank_settings.get("attack_scale", 1.0) * self.strat_cfg.get("attack_bias", 1.0)

        # 2. Score differential tuning
        scores = [p.score for p in state_mgr.round_state.players]
        top_score = max(scores)
        last_score = min(scores)
        gap_to_top = top_score - my_p.score
        gap_to_last = my_p.score - last_score

        # If dangerously close to last place (< 4000 pts ahead of 4th), heighten defense
        if my_rank in (2, 3) and gap_to_last < 4000 and gap_to_last > 0:
            defense_scale *= 1.35
            attack_scale *= 0.85

        # 3. Ooras (Final round) tactics
        if is_ooras:
            if my_rank == 1:
                # 1st place in Ooras: Prioritize safety above all else. Avoid dealing in at any cost.
                ooras_def_mult = self.ooras_cfg.get("top_heavy_defense", 2.2)
                defense_scale *= ooras_def_mult
                attack_scale *= 0.50
            elif my_rank == 4:
                # 4th place in Ooras: Must push for reverse/escape. Aggression multiplier.
                ooras_atk_mult = self.ooras_cfg.get("last_push_aggression", 2.5)
                attack_scale *= ooras_atk_mult
                defense_scale *= 0.50
            elif is_dealer:
                # Dealer in Ooras: Needs renchan if not 1st
                defense_scale *= 0.85
                attack_scale *= self.ooras_cfg.get("oya_renchan_bias", 1.3)

        # 4. Opponent Behavioral Exploit Profiling & Riichi Push-Fold Judgment
        exploit_note = ""
        riichi_penalty_weight = self.riichi_cfg.get("danger_penalty_weight", 1.5)

        # Apply custom exploit (e.g. from GUI paifu analysis) if active
        if self.custom_exploit:
            defense_scale *= self.custom_exploit.get("defense_scale_mult", 1.0)
            attack_scale *= self.custom_exploit.get("attack_scale_mult", 1.0)
            riichi_penalty_weight *= self.custom_exploit.get("danger_penalty_mult", 1.0)
            exploit_note = self.custom_exploit.get("exploit_note", "")
        elif active_riichis and self.profiler:
            primary_riichi_actor = active_riichis[0]
            if primary_riichi_actor < len(state_mgr.round_state.players):
                enemy_name = state_mgr.round_state.players[primary_riichi_actor].name
                if enemy_name:
                    exploit = self.profiler.get_exploit_modifiers(enemy_name)
                    defense_scale *= exploit.get("defense_scale_mult", 1.0)
                    attack_scale *= exploit.get("attack_scale_mult", 1.0)
                    riichi_penalty_weight *= exploit.get("danger_penalty_mult", 1.0)
                    exploit_note = exploit.get("exploit_note", "")

        shanten_push_threshold = self.riichi_cfg.get("push_shanten_threshold", 1)

        modified_list: List[ActionCandidate] = []
        for cand in candidates:
            c = copy.deepcopy(cand)
            mod_q = c.raw_q_value * attack_scale

            # Always reward winning directly
            if c.action_type == "hora":
                c.modified_q_value = 999.0
                modified_list.append(c)
                continue

            # Apply danger penalty if someone is riichi
            if active_riichis:
                # If hand is far from tenpai (shanten > threshold), apply heavy fold penalty to any danger
                if current_shanten > shanten_push_threshold:
                    fold_factor = 2.5
                elif current_shanten == 1:
                    fold_factor = 1.3
                else:
                    # Tenpai (shanten 0): Can consider pushing safe-ish tiles or high value
                    fold_factor = 0.8

                danger_cost = (c.danger_score ** 1.5) * 12.0 * riichi_penalty_weight * defense_scale * fold_factor
                mod_q -= danger_cost

                # Bonus for 100% safe tiles (Genbutsu) when folding
                if c.danger_score == 0.0 and current_shanten > 0:
                    mod_q += 3.5 * defense_scale

            # Riichi tactical adjustment:
            # If 1st in Ooras, prefer dama (silent tenpai) unless necessary
            if c.action_type == "reach":
                if is_ooras and my_rank == 1:
                    mod_q -= 5.0 # Dama preferred to avoid risking 1000 stick and stuck riichi
                elif is_ooras and my_rank == 4:
                    mod_q += 4.0 # Riichi for bonus han / reverse

            # Call (chi, pon, kan) vs Pass tactical adjustment:
            if c.action_type in ("chi", "pon", "kan"):
                bd = c.metadata.get("breakdown", {})
                has_yaku = bd.get("has_yaku", True)
                if not has_yaku:
                    # Enforce that no-yaku calls can NEVER beat pass
                    mod_q -= 15.0 * defense_scale

                if active_riichis:
                    if current_shanten > 0:
                        mod_q -= 8.0 * defense_scale  # Opening hand without tenpai against Riichi is dangerous
                if is_ooras and my_rank == 4 and has_yaku:
                    mod_q += 3.0  # 4th in Ooras aggressively calls only if yaku is confirmed
                elif is_ooras and my_rank == 1 and c.action_type == "kan":
                    mod_q -= 5.0  # 1st in Ooras avoids kan to prevent handing kandora to opponents

            elif c.action_type == "none":
                if active_riichis and current_shanten > 0:
                    mod_q += 2.5 * defense_scale  # Passing preserves closed hand defense against riichi

            c.modified_q_value = round(mod_q, 3)
            c.metadata["strategy_context"] = {
                "rank": my_rank,
                "is_ooras": is_ooras,
                "attack_scale": round(attack_scale, 2),
                "defense_scale": round(defense_scale, 2),
                "active_riichis": len(active_riichis),
                "current_shanten": current_shanten,
                "exploit_note": exploit_note,
            }
            modified_list.append(c)

        # Sort highest modified Q-value first
        modified_list.sort(key=lambda x: x.modified_q_value, reverse=True)
        return modified_list

    def select_best_action(
        self,
        candidates: List[ActionCandidate],
        state_mgr: MahjongStateManager
    ) -> Optional[ActionCandidate]:
        """Convenience method to retrieve the single highest rated action."""
        modified = self.modify_candidates(candidates, state_mgr)
        return modified[0] if modified else None
