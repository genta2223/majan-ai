"""Opponent Profiling & Tactical Adaptation module for analyzing player habits from paifu logs."""

from __future__ import annotations
import json
import os
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

import re
from core.state_manager import normalize_tile, HONORS


def clean_player_name(raw_name: str) -> str:
    """
    Normalizes player account nicknames by stripping seating annotations
    (e.g. ' (東家)', ' (南家)', ' (西家)', ' (北家)', '(南家)') and leading/trailing whitespace.
    Returns authentic account name alone (e.g. 'ブロコフ').
    Leaves CPU slots (e.g. 'CPUプレイヤー (西家)') distinct per seat to avoid collision.
    """
    if not raw_name:
        return ""
    name = str(raw_name).strip()
    if name.startswith("CPU") or "CPUプレイヤー" in name or name.startswith("Player_Seat"):
        return name
    name = re.sub(r'\s*[\(（][東西南北]家[\)）]$', '', name).strip()
    return name


class PlayerProfile(BaseModel):
    """Statistical profile and behavioral archetype of an individual opponent."""
    name: str
    seat_name: str = ""             # "東家", "南家", "西家", "北家"
    total_games: int = 0
    total_rounds: int = 0
    hora_count: int = 0             # Number of winning hands (Hora)
    hora_scores_sum: int = 0        # Sum of winning points
    riichi_count: int = 0
    call_count: int = 0             # Number of melds (chi, pon, kan)
    rounds_with_call: int = 0
    rounds_with_riichi: int = 0
    deal_in_count: int = 0          # Number of times dealt into Ron
    aggressiveness_pushes: int = 0  # Number of un-safe tiles pushed against enemy riichi
    aggressiveness_total_turns: int = 0 # Turns played while enemy riichi was active
    fold_turns: int = 0             # Turns discarded safe tiles (genbutsu) against enemy riichi

    # Processed game UUIDs to prevent double-counting between live and historical logs
    recorded_game_uuids: List[str] = Field(default_factory=list)

    # Detailed habit counts
    yakuhai_call_count: int = 0     # Honor tile melds
    tanyao_call_count: int = 0      # All-simples melds (numbers 2-8)
    early_call_count: int = 0       # Melds made within turns 1-6
    early_honor_discards: int = 0   # Honor discards within turns 1-6
    early_total_discards: int = 0   # Total discards within turns 1-6

    # Raw observed metrics (0.0 to 1.0 or points)
    hora_rate: float = 0.0          # Hora rounds / total rounds
    deal_in_rate: float = 0.0       # Deal in rounds / total rounds
    riichi_rate: float = 0.0        # Riichi rounds / total rounds
    open_rate: float = 0.0          # Melded rounds / total rounds
    avg_hora_score: float = 0.0     # Average score per win
    aggressiveness: float = 0.50    # Un-safe push rate against riichi
    fold_rate: float = 0.50         # Safe tile fold rate against riichi
    safety_hoarding: float = 0.50   # Hoards honors / genbutsu until late game

    # Bayesian smoothed metrics (prior N=10, standard player baseline)
    # Prevents extreme 0% / 100% distortion when sample size is small
    smoothed_hora_rate: float = 0.22
    smoothed_deal_in_rate: float = 0.12
    smoothed_riichi_rate: float = 0.18
    smoothed_open_rate: float = 0.30
    smoothed_aggressiveness: float = 0.25
    smoothed_fold_rate: float = 0.50

    # Current Live Session Stats (real-time stream within current match)
    live_rounds: int = 0
    live_hora_count: int = 0
    live_deal_in_count: int = 0
    live_riichi_count: int = 0
    live_call_count: int = 0
    live_rounds_with_call: int = 0
    live_rounds_with_riichi: int = 0
    live_aggressiveness_pushes: int = 0
    live_aggressiveness_turns: int = 0
    live_fold_turns: int = 0

    # Tendency profile & badges
    archetype: str = "Balanced"
    call_tendency: str = "バランス型仕掛け"       # 役牌速攻型 / 喰いタン多用 / 遠い鳴き多め / 門前重視
    push_fold_tendency: str = "バランス型押し引き"   # 鉄壁ベタオリ / 超攻撃的プッシャー / やや押し気味 / 安全重視
    early_game_tendency: str = "標準的進行"      # 速攻字牌整理型 / 字牌抱え守備型 / 標準的進行
    badges: List[str] = Field(default_factory=list)

    @property
    def display_name(self) -> str:
        """Returns authentic clean account nickname without permanent seating tag."""
        return clean_player_name(self.name)

    def reset_live_stats(self) -> None:
        """Resets the in-match live session statistics."""
        self.live_rounds = 0
        self.live_hora_count = 0
        self.live_deal_in_count = 0
        self.live_riichi_count = 0
        self.live_call_count = 0
        self.live_rounds_with_call = 0
        self.live_rounds_with_riichi = 0
        self.live_aggressiveness_pushes = 0
        self.live_aggressiveness_turns = 0
        self.live_fold_turns = 0

    def get_effective_metrics(self, live_weight: float = 0.35) -> Dict[str, float]:
        """
        Blends historical smoothed stats with current live session stats (30% - 40% weight).
        Allows AI to quickly adapt to 'usually conservative, but hyper-aggressive today'.
        """
        w = min(0.45, max(0.20, live_weight))

        if self.live_rounds == 0 and self.live_aggressiveness_turns == 0:
            return {
                "hora_rate": self.smoothed_hora_rate,
                "deal_in_rate": self.smoothed_deal_in_rate,
                "riichi_rate": self.smoothed_riichi_rate,
                "open_rate": self.smoothed_open_rate,
                "aggressiveness": self.smoothed_aggressiveness,
                "fold_rate": self.smoothed_fold_rate,
            }

        # Live rates
        lr = max(1, self.live_rounds)
        live_open = self.live_rounds_with_call / lr
        live_hora = self.live_hora_count / lr
        live_deal_in = self.live_deal_in_count / lr
        live_riichi = self.live_rounds_with_riichi / lr

        if self.live_aggressiveness_turns > 0:
            live_aggr = self.live_aggressiveness_pushes / self.live_aggressiveness_turns
            live_fold = self.live_fold_turns / self.live_aggressiveness_turns
        else:
            live_aggr = self.smoothed_aggressiveness
            live_fold = self.smoothed_fold_rate

        return {
            "hora_rate": round((1.0 - w) * self.smoothed_hora_rate + w * live_hora, 3),
            "deal_in_rate": round((1.0 - w) * self.smoothed_deal_in_rate + w * live_deal_in, 3),
            "riichi_rate": round((1.0 - w) * self.smoothed_riichi_rate + w * live_riichi, 3),
            "open_rate": round((1.0 - w) * self.smoothed_open_rate + w * live_open, 3),
            "aggressiveness": round((1.0 - w) * self.smoothed_aggressiveness + w * live_aggr, 3),
            "fold_rate": round((1.0 - w) * self.smoothed_fold_rate + w * live_fold, 3),
        }

    def compute_metrics(self) -> None:
        """Calculates normalized percentages, Bayesian smoothed rates, behavioral archetype, and badges."""
        # Raw observed percentages
        if self.total_rounds > 0:
            self.hora_rate = round(min(1.0, self.hora_count / self.total_rounds), 3)
            self.open_rate = round(min(1.0, self.rounds_with_call / self.total_rounds), 3)
            self.riichi_rate = round(min(1.0, self.rounds_with_riichi / self.total_rounds), 3)
            self.deal_in_rate = round(min(1.0, self.deal_in_count / self.total_rounds), 3)
        else:
            self.hora_rate = 0.0
            self.open_rate = 0.0
            self.riichi_rate = 0.0
            self.deal_in_rate = 0.0

        if self.hora_count > 0:
            self.avg_hora_score = round(self.hora_scores_sum / self.hora_count, 1)
        else:
            self.avg_hora_score = 0.0

        if self.aggressiveness_total_turns > 0:
            self.aggressiveness = round(
                self.aggressiveness_pushes / self.aggressiveness_total_turns, 3
            )
            self.fold_rate = round(
                self.fold_turns / self.aggressiveness_total_turns, 3
            )
        else:
            self.aggressiveness = 0.50
            self.fold_rate = 0.50

        # Bayesian smoothing: virtual sample size N_prior = 10 rounds
        # Standard average player prior:
        # Hora: 22%, Deal-in: 12%, Riichi: 18%, Open: 30%, Aggr: 25%, Fold: 50%
        n_prior = 10.0
        self.smoothed_hora_rate = round((self.hora_count + n_prior * 0.22) / (self.total_rounds + n_prior), 3)
        self.smoothed_deal_in_rate = round((self.deal_in_count + n_prior * 0.12) / (self.total_rounds + n_prior), 3)
        self.smoothed_riichi_rate = round((self.rounds_with_riichi + n_prior * 0.18) / (self.total_rounds + n_prior), 3)
        self.smoothed_open_rate = round((self.rounds_with_call + n_prior * 0.30) / (self.total_rounds + n_prior), 3)

        aggr_prior = 5.0
        self.smoothed_aggressiveness = round(
            (self.aggressiveness_pushes + aggr_prior * 0.25) / (self.aggressiveness_total_turns + aggr_prior), 3
        )
        self.smoothed_fold_rate = round(
            (self.fold_turns + aggr_prior * 0.50) / (self.aggressiveness_total_turns + aggr_prior), 3
        )

        # 1. Call Tendency
        if self.open_rate >= 0.40 and self.tanyao_call_count > self.yakuhai_call_count:
            self.call_tendency = "喰いタン多用"
        elif self.open_rate >= 0.35 and self.yakuhai_call_count >= self.tanyao_call_count:
            self.call_tendency = "役牌速攻型"
        elif self.early_call_count >= 2:
            self.call_tendency = "遠い鳴き多め"
        elif self.open_rate <= 0.18:
            self.call_tendency = "門前重視"
        else:
            self.call_tendency = "バランス型仕掛け"

        # 2. Push/Fold Tendency
        if self.fold_rate >= 0.65 or self.aggressiveness <= 0.32:
            self.push_fold_tendency = "鉄壁ベタオリ"
        elif self.aggressiveness >= 0.60:
            self.push_fold_tendency = "超攻撃的プッシャー"
        elif self.aggressiveness <= 0.40:
            self.push_fold_tendency = "安全重視"
        else:
            self.push_fold_tendency = "バランス型押し引き"

        # 3. Early Game Tendency
        if self.early_total_discards > 0:
            honor_ratio = self.early_honor_discards / self.early_total_discards
            if honor_ratio >= 0.45:
                self.early_game_tendency = "速攻字牌整理型"
            elif honor_ratio <= 0.15:
                self.early_game_tendency = "字牌抱え守備型"
            else:
                self.early_game_tendency = "標準的進行"
        else:
            self.early_game_tendency = "標準的進行"

        # 4. Archetype
        if self.aggressiveness >= 0.60 and self.deal_in_rate >= 0.15:
            self.archetype = "Berserker Pusher"
        elif self.aggressiveness <= 0.35 and self.deal_in_rate <= 0.08:
            self.archetype = "Iron Wall Defense"
        elif self.open_rate >= 0.45:
            self.archetype = "Speed Call Rusher"
        elif self.riichi_rate >= 0.30:
            self.archetype = "Heavy Riichi Slugger"
        else:
            self.archetype = "Balanced"

        # 5. Badges
        b_list = []
        if self.open_rate >= 0.35:
            b_list.append("鳴きが早い")
        elif self.open_rate <= 0.18:
            b_list.append("重厚門前派")

        if self.fold_rate >= 0.65 or self.aggressiveness <= 0.32:
            b_list.append("ベタオリしやすい")
        elif self.aggressiveness >= 0.60:
            b_list.append("無筋押し多め")

        if self.tanyao_call_count >= 2:
            b_list.append("喰いタン多用")
        if self.yakuhai_call_count >= 2:
            b_list.append("役牌速攻")
        if self.early_game_tendency == "速攻字牌整理型":
            b_list.append("字牌即切り")
        if self.deal_in_rate <= 0.08:
            b_list.append("低放銃率")
        elif self.deal_in_rate >= 0.18:
            b_list.append("放銃注意")

        if not b_list:
            b_list.append("バランス型")
        self.badges = b_list[:4]



class OpponentProfiler:
    """
    Parses mjai logs to profile opponents and compute exploit modifiers
    for StrategyModifier to optimize winrate against specific friends.
    Supports both offline historical paifu parsing and in-match real-time event streaming.
    """

    def __init__(self, profiles_path: Optional[str] = None):
        self.profiles: Dict[str, PlayerProfile] = {}
        self.stats: Dict[int, Dict[str, int]] = {
            i: {
                "agari": 0,
                "houju": 0,
                "reach": 0,
                "call": 0,
                "naki": 0,
                "rounds": 0,
                "tanyao": 0,
                "yakuhai": 0,
            }
            for i in range(4)
        }
        self.profiles_path = profiles_path
        if profiles_path and os.path.exists(profiles_path):
            self.load_profiles(profiles_path)

        # Real-time in-game state
        self.current_game_uuid: Optional[str] = None
        self.match_mode: str = "hanchan"       # "tonpuu" (東風戦) or "hanchan" (東南戦/半荘)
        self.current_bakaze: str = "E"
        self.current_kyoku: int = 1
        self.current_honba: int = 0
        self.live_round_count: int = 0
        self.player_names: List[str] = ["", "", "", ""]
        self._live_riichis: Dict[int, List[str]] = {}  # actor -> list of discards
        self._live_round_calls: set[int] = set()
        self._live_round_riichis: set[int] = set()
        self._live_turn_counter: int = 0
        self._live_round_committed: bool = True

    def get_profile(self, name: str) -> Optional[PlayerProfile]:
        clean_name = clean_player_name(name)
        return self.profiles.get(clean_name)

    def set_match_mode(self, mode: str) -> None:
        """Sets match mode ('tonpuu' or 'hanchan')."""
        if mode in ("tonpuu", "hanchan"):
            self.match_mode = mode

    def set_game_uuid(self, uuid: str) -> None:
        """Sets current game UUID and resets live session statistics for a new match."""
        if self.current_game_uuid != uuid:
            self.current_game_uuid = uuid
            self.live_round_count = 0
            for prof in self.profiles.values():
                prof.reset_live_stats()

    def set_player_name(self, seat: int, name: str) -> None:
        """Registers clean player name for a specific seat, maintaining historical profile."""
        if 0 <= seat < 4 and name:
            SEAT_NAMES = ["東家", "南家", "西家", "北家"]
            clean_name = clean_player_name(name)
            old_name = self.player_names[seat]
            self.player_names[seat] = clean_name
            if clean_name not in self.profiles:
                if old_name and old_name.startswith("Player_Seat") and old_name in self.profiles:
                    p = self.profiles.pop(old_name)
                    p.name = clean_name
                    p.seat_name = SEAT_NAMES[seat]
                    self.profiles[clean_name] = p
                else:
                    self.profiles[clean_name] = PlayerProfile(name=clean_name, seat_name=SEAT_NAMES[seat])
            else:
                self.profiles[clean_name].seat_name = SEAT_NAMES[seat]

    def _ensure_seat_name(self, seat: int) -> str:
        """Ensures a valid name exists for the given seat, assigning placeholder if absent."""
        if 0 <= seat < 4:
            if not self.player_names[seat]:
                SEAT_NAMES = ["東家", "南家", "西家", "北家"]
                placeholder = f"Player_Seat{seat}"
                self.player_names[seat] = placeholder
                if placeholder not in self.profiles:
                    self.profiles[placeholder] = PlayerProfile(name=placeholder, seat_name=SEAT_NAMES[seat])
            return self.player_names[seat]
        return ""

    def is_game_uuid_recorded(self, name: str, game_uuid: str) -> bool:
        """Checks if game_uuid is already recorded for the given player."""
        clean_name = clean_player_name(name)
        prof = self.get_profile(clean_name)
        return bool(prof and game_uuid and game_uuid in prof.recorded_game_uuids)

    def record_game_uuid_for_all(self, game_uuid: str) -> None:
        """Marks game_uuid as processed across all active seated players to prevent duplicates."""
        if not game_uuid:
            return
        for name in self.player_names:
            clean_name = clean_player_name(name)
            if clean_name and clean_name in self.profiles:
                p = self.profiles[clean_name]
                if game_uuid not in p.recorded_game_uuids:
                    p.recorded_game_uuids.append(game_uuid)

    def consume_live_event(self, event: Dict[str, Any]) -> None:
        """
        Consumes a single real-time MJAI event during live play and immediately
        accumulates opponent habits, adjusting Bayesian smoothing & live stats on the fly.
        """
        etype = event.get("type", "")

        def _commit_live_round():
            if not self._live_round_committed:
                visited_call = set()
                for actor in self._live_round_calls:
                    if 0 <= actor < len(self.player_names):
                        name = self.player_names[actor]
                        if name in self.profiles and name not in visited_call:
                            self.profiles[name].rounds_with_call += 1
                            self.profiles[name].live_rounds_with_call += 1
                            visited_call.add(name)
                visited_riichi = set()
                for actor in self._live_round_riichis:
                    if 0 <= actor < len(self.player_names):
                        name = self.player_names[actor]
                        if name in self.profiles and name not in visited_riichi:
                            self.profiles[name].rounds_with_riichi += 1
                            self.profiles[name].live_rounds_with_riichi += 1
                            visited_riichi.add(name)
                self._live_round_committed = True

        if etype == "start_game":
            names = event.get("names", [])
            for i in range(min(4, len(names))):
                name = str(names[i]).strip()
                if name:
                    self.set_player_name(i, name)
            uuid = event.get("game_uuid") or event.get("uuid")
            if uuid:
                self.set_game_uuid(str(uuid))

            if "match_mode" in event:
                self.set_match_mode(event["match_mode"])

        elif etype == "start_kyoku":
            _commit_live_round()
            self._live_riichis.clear()
            self._live_round_calls.clear()
            self._live_round_riichis.clear()
            self._live_turn_counter = 0
            self._live_round_committed = False

            self.current_bakaze = event.get("bakaze", "E")
            self.current_kyoku = event.get("kyoku", 1)
            self.current_honba = event.get("honba", 0)
            self.live_round_count += 1

            if "match_mode" in event:
                self.set_match_mode(event["match_mode"])
            elif self.current_bakaze in ("S", "W", "N"):
                self.set_match_mode("hanchan")

            visited_round_names = set()
            for i in range(4):
                name = self._ensure_seat_name(i)
                if i in self.stats:
                    self.stats[i]["rounds"] += 1
                if name and name in self.profiles and name not in visited_round_names:
                    self.profiles[name].total_rounds += 1
                    self.profiles[name].live_rounds += 1
                    visited_round_names.add(name)

        elif etype in ("chi", "pon", "daiminkan", "kankan", "ankan"):
            actor = event.get("actor", 0)
            if 0 <= actor < 4 and actor in self.stats:
                self.stats[actor]["call"] += 1
                self.stats[actor]["naki"] += 1

            name = self._ensure_seat_name(actor)
            if name in self.profiles:
                p = self.profiles[name]
                p.call_count += 1
                p.live_call_count += 1
                self._live_round_calls.add(actor)

                consumed = event.get("consumed", [])
                pai = event.get("pai", "")
                meld_tiles = [t for t in (consumed + ([pai] if pai else [])) if t and t != "?"]
                if meld_tiles:
                    has_honor = any(normalize_tile(t) in HONORS for t in meld_tiles)
                    if has_honor:
                        p.yakuhai_call_count += 1
                        if 0 <= actor < 4 and actor in self.stats:
                            self.stats[actor]["yakuhai"] += 1
                    else:
                        is_all_simples = all(
                            len(normalize_tile(t)) == 2 and normalize_tile(t)[0] in "2345678"
                            for t in meld_tiles
                        )
                        if is_all_simples:
                            p.tanyao_call_count += 1
                            if 0 <= actor < 4 and actor in self.stats:
                                self.stats[actor]["tanyao"] += 1

                if self._live_turn_counter <= 6:
                    p.early_call_count += 1

        elif etype == "reach":
            actor = event.get("actor", 0)
            if 0 <= actor < 4 and actor in self.stats:
                self.stats[actor]["reach"] += 1

            name = self._ensure_seat_name(actor)
            self._live_round_riichis.add(actor)
            self._live_riichis[actor] = []
            if name in self.profiles:
                p = self.profiles[name]
                p.riichi_count += 1
                p.live_riichi_count += 1

        elif etype == "dahai":
            actor = event.get("actor", 0)
            name = self._ensure_seat_name(actor)
            pai = event.get("pai", "")
            norm_pai = normalize_tile(pai)
            self._live_turn_counter += 1

            if name in self.profiles:
                p = self.profiles[name]
                if self._live_turn_counter <= 6 * 4:
                    p.early_total_discards += 1
                    if norm_pai in HONORS:
                        p.early_honor_discards += 1

            if actor in self._live_riichis:
                self._live_riichis[actor].append(norm_pai)

            other_riichis = [r for r in self._live_riichis.keys() if r != actor]
            if other_riichis and actor not in self._live_riichis and name in self.profiles:
                p = self.profiles[name]
                p.aggressiveness_total_turns += 1
                p.live_aggressiveness_turns += 1

                is_genbutsu = all(norm_pai in self._live_riichis[r] for r in other_riichis)

                def _is_suji(t: str, enemy_discards: List[str]) -> bool:
                    if t in HONORS or len(t) != 2 or not t[0].isdigit():
                        return False
                    val, suit = int(t[0]), t[1]
                    if val in (1, 9):
                        return f"{4 if val == 1 else 6}{suit}" in enemy_discards
                    elif val in (2, 8):
                        return f"5{suit}" in enemy_discards
                    elif val in (3, 7):
                        return f"{6 if val == 3 else 4}{suit}" in enemy_discards
                    elif val == 4:
                        return f"1{suit}" in enemy_discards and f"7{suit}" in enemy_discards
                    elif val == 5:
                        return f"2{suit}" in enemy_discards and f"8{suit}" in enemy_discards
                    elif val == 6:
                        return f"3{suit}" in enemy_discards and f"9{suit}" in enemy_discards
                    return False

                is_suji_safe = all(_is_suji(norm_pai, self._live_riichis[r]) for r in other_riichis)
                if is_genbutsu or is_suji_safe:
                    p.fold_turns += 1
                    p.live_fold_turns += 1
                else:
                    p.aggressiveness_pushes += 1
                    p.live_aggressiveness_pushes += 1

        elif etype == "hora":
            actor = event.get("winner") if "winner" in event else event.get("actor", 0)
            deltas = event.get("deltas", [])
            neg_scorers = [i for i, d in enumerate(deltas) if d < 0]

            raw_zimo = event.get("zimo")
            raw_target = event.get("target")
            raw_loser = event.get("loser")

            if raw_zimo is not None:
                is_zimo = bool(raw_zimo)
            elif raw_loser is not None and raw_loser != actor:
                is_zimo = False
            elif raw_target is not None:
                is_zimo = (raw_target == actor)
            elif len(neg_scorers) >= 2:
                is_zimo = True
            else:
                is_zimo = False
            
            # Target / Loser determination
            target = actor
            loser = None
            if not is_zimo:
                if raw_loser is not None and 0 <= raw_loser < 4 and raw_loser != actor:
                    target = raw_loser
                    loser = raw_loser
                elif raw_target is not None and 0 <= raw_target < 4 and raw_target != actor:
                    target = raw_target
                    loser = raw_target
                elif neg_scorers and len(neg_scorers) == 1 and neg_scorers[0] != actor:
                    target = neg_scorers[0]
                    loser = neg_scorers[0]
                elif neg_scorers:
                    valid_negs = [s for s in neg_scorers if s != actor]
                    if valid_negs:
                        target = valid_negs[0]
                        loser = valid_negs[0]

            if 0 <= actor < 4 and actor in self.stats:
                self.stats[actor]["agari"] += 1

            winner_name = clean_player_name(self._ensure_seat_name(actor))
            if winner_name in self.profiles:
                p = self.profiles[winner_name]
                p.hora_count += 1
                p.live_hora_count += 1
                win_score = 0
                ev_sc = event.get("dadian") or event.get("score")
                if ev_sc is not None and 0 < int(ev_sc) <= 48000:
                    win_score = int(ev_sc)
                else:
                    win_score = 5200
                p.hora_scores_sum += win_score

            if loser is not None and loser != actor:
                if 0 <= loser < 4 and loser in self.stats:
                    self.stats[loser]["houju"] += 1
                victim_name = clean_player_name(self._ensure_seat_name(loser))
                if victim_name in self.profiles:
                    p = self.profiles[victim_name]
                    p.deal_in_count += 1
                    p.live_deal_in_count += 1

            _commit_live_round()
            for p in self.profiles.values():
                p.compute_metrics()

            if self.current_game_uuid:
                self.record_game_uuid_for_all(self.current_game_uuid)

            if self.profiles_path:
                try:
                    self.save_profiles(self.profiles_path)
                except Exception:
                    pass

        elif etype in ("end_kyoku", "ryukyoku"):
            _commit_live_round()
            for p in self.profiles.values():
                p.compute_metrics()

            if self.current_game_uuid:
                self.record_game_uuid_for_all(self.current_game_uuid)

            if self.profiles_path:
                try:
                    self.save_profiles(self.profiles_path)
                except Exception:
                    pass


    def analyze_game_events(self, events: List[Dict[str, Any]], game_uuid: Optional[str] = None) -> None:
        """
        Processes a full game's mjai events and accumulates player habits.
        Strictly prevents double-counting if game_uuid has already been processed.
        """
        SEAT_NAMES = ["東家", "南家", "西家", "北家"]
        player_names: List[str] = ["", "", "", ""]
        current_riichis: Dict[int, List[str]] = {}  # actor -> list of discards for genbutsu
        current_round_calls: set[int] = set()
        current_round_riichis: set[int] = set()
        turn_counter: int = 0
        round_committed: bool = True

        # Extract game_uuid from start_game or argument (including MJAI 'id')
        target_uuid = game_uuid
        if not target_uuid:
            for ev in events:
                if ev.get("type") == "start_game":
                    target_uuid = ev.get("game_uuid") or ev.get("uuid") or ev.get("id")
                    break

        # Deduplication Guard: Check if game_uuid was already processed
        if target_uuid:
            for ev in events:
                if ev.get("type") == "start_game":
                    for n in ev.get("names", []):
                        str_n = clean_player_name(n)
                        if str_n and self.is_game_uuid_recorded(str_n, target_uuid):
                            print(f"[Profiler] Game UUID '{target_uuid}' already recorded for player '{str_n}'. Skipping duplicate analysis.")
                            return

        def _commit_round():
            nonlocal round_committed
            if not round_committed:
                visited_calls = set()
                for actor in current_round_calls:
                    if 0 <= actor < len(player_names):
                        name = player_names[actor]
                        if name and name in self.profiles and name not in visited_calls:
                            self.profiles[name].rounds_with_call += 1
                            visited_calls.add(name)
                visited_riichis = set()
                for actor in current_round_riichis:
                    if 0 <= actor < len(player_names):
                        name = player_names[actor]
                        if name and name in self.profiles and name not in visited_riichis:
                            self.profiles[name].rounds_with_riichi += 1
                            visited_riichis.add(name)
                round_committed = True

        current_last_discard_seat: Optional[int] = None

        for ev in events:
            raw_type = ev.get("type") or ev.get("action_name") or ev.get("name") or ""
            etype = raw_type.split(".")[-1]
            data = ev.get("data") if isinstance(ev.get("data"), dict) else ev

            if etype == "start_game":
                names = ev.get("names", [])
                if not names and "accounts" in ev:
                    names = [a.get("nickname") or a.get("name") for a in ev.get("accounts", []) if isinstance(a, dict)]
                for i in range(min(4, len(names))):
                    name = clean_player_name(names[i])
                    player_names[i] = name
                    if name:
                        if name not in self.profiles:
                            self.profiles[name] = PlayerProfile(name=name, seat_name=SEAT_NAMES[i])
                        else:
                            self.profiles[name].seat_name = SEAT_NAMES[i]
                        self.profiles[name].total_games += 1

            elif etype in ("start_kyoku", "RecordNewRound", "ActionNewRound"):
                # Ensure previous round was committed
                _commit_round()
                current_riichis.clear()
                current_round_calls.clear()
                current_round_riichis.clear()
                turn_counter = 0
                round_committed = False
                current_last_discard_seat = None

                # Extract names if present and player_names not fully populated
                cand_names = ev.get("names") or data.get("names") or []
                if cand_names:
                    for i in range(min(4, len(cand_names))):
                        n = clean_player_name(cand_names[i])
                        if n and not player_names[i]:
                            player_names[i] = n
                            if n not in self.profiles:
                                self.profiles[n] = PlayerProfile(name=n, seat_name=SEAT_NAMES[i])
                            else:
                                self.profiles[n].seat_name = SEAT_NAMES[i]

                # Ensure player profiles exist for seated players
                for i in range(4):
                    if i in self.stats:
                        self.stats[i]["rounds"] += 1
                    if i < len(player_names) and not player_names[i]:
                        placeholder = f"Player_Seat{i}"
                        player_names[i] = placeholder
                        if placeholder not in self.profiles:
                            self.profiles[placeholder] = PlayerProfile(name=placeholder, seat_name=SEAT_NAMES[i])

                # Deduplicated per-round increment: each unique player profile gets exactly 1 total_round
                visited_round_names = set()
                for i in range(4):
                    if 0 <= i < len(player_names):
                        name = player_names[i]
                        if name and name in self.profiles and name not in visited_round_names:
                            self.profiles[name].total_rounds += 1
                            visited_round_names.add(name)

            elif etype in ("chi", "pon", "daiminkan", "RecordChiPengGang", "ActionChiPengGang"):
                actor = ev.get("actor") if "actor" in ev else data.get("seat", 0)
                if 0 <= actor < 4 and actor in self.stats:
                    self.stats[actor]["call"] += 1
                    self.stats[actor]["naki"] += 1

                if 0 <= actor < len(player_names):
                    name = player_names[actor]
                    consumed = ev.get("consumed", []) or data.get("tiles", [])
                    pai = ev.get("pai", "")
                    meld_tiles = [t for t in (consumed + ([pai] if pai else [])) if t and t != "?"]

                    if name in self.profiles:
                        p = self.profiles[name]
                        p.call_count += 1
                        current_round_calls.add(actor)

                        # Classify meld type: strictly requires non-empty meld_tiles to avoid all([]) == True bug
                        if meld_tiles:
                            has_honor = any(normalize_tile(t) in HONORS for t in meld_tiles)
                            if has_honor:
                                p.yakuhai_call_count += 1
                                if 0 <= actor < 4 and actor in self.stats:
                                    self.stats[actor]["yakuhai"] += 1
                            else:
                                is_all_simples = all(
                                    len(normalize_tile(t)) == 2 and normalize_tile(t)[0] in "2345678"
                                    for t in meld_tiles
                                )
                                if is_all_simples:
                                    p.tanyao_call_count += 1
                                    if 0 <= actor < 4 and actor in self.stats:
                                        self.stats[actor]["tanyao"] += 1

                        if turn_counter <= 6:
                            p.early_call_count += 1

            elif etype in ("reach", "RecordDiscardTile", "ActionDiscardTile") and (etype == "reach" or data.get("is_liqi") or data.get("is_wliqi")):
                actor = ev.get("actor") if "actor" in ev else data.get("seat", 0)
                if 0 <= actor < 4 and actor in self.stats:
                    self.stats[actor]["reach"] += 1

                if 0 <= actor < len(player_names):
                    name = player_names[actor]
                    current_round_riichis.add(actor)
                    current_riichis[actor] = []
                    if name in self.profiles:
                        self.profiles[name].riichi_count += 1

                # If this was a discard with is_liqi, record the discard as well
                if etype != "reach":
                    current_last_discard_seat = actor

            elif etype in ("dahai", "RecordDiscardTile", "ActionDiscardTile"):
                actor = ev.get("actor") if "actor" in ev else data.get("seat", 0)
                current_last_discard_seat = actor
                if 0 <= actor < len(player_names):
                    pai = ev.get("pai", "") or data.get("tile", "")
                    norm_pai = normalize_tile(pai)
                    name = player_names[actor]
                    turn_counter += 1

                    if name in self.profiles:
                        p = self.profiles[name]
                        if turn_counter <= 6 * 4:
                            p.early_total_discards += 1
                            if norm_pai in HONORS:
                                p.early_honor_discards += 1

                    if actor in current_riichis:
                        current_riichis[actor].append(norm_pai)

                    other_riichis = [r_actor for r_actor in current_riichis.keys() if r_actor != actor]
                    # Only evaluate push/fold if:
                    # 1. Other players have declared Riichi
                    # 2. Actor is NOT in Riichi oneself (when in Riichi, tsumogiri is forced and not a choice to fold)
                    if other_riichis and actor not in current_riichis and name in self.profiles:
                        self.profiles[name].aggressiveness_total_turns += 1

                        # Determine if discarded tile is safe (genbutsu to all other riichis)
                        is_genbutsu = all(norm_pai in current_riichis[r] for r in other_riichis)

                        # Check if tile is suji (one-chance / suji) against other riichis
                        def _is_suji(t: str, enemy_discards: List[str]) -> bool:
                            if t in HONORS or len(t) != 2 or not t[0].isdigit():
                                return False
                            val, suit = int(t[0]), t[1]
                            if val in (1, 9):
                                return f"{4 if val == 1 else 6}{suit}" in enemy_discards
                            elif val in (2, 8):
                                return f"5{suit}" in enemy_discards
                            elif val in (3, 7):
                                return f"{6 if val == 3 else 4}{suit}" in enemy_discards
                            elif val == 4:
                                return f"1{suit}" in enemy_discards and f"7{suit}" in enemy_discards
                            elif val == 5:
                                return f"2{suit}" in enemy_discards and f"8{suit}" in enemy_discards
                            elif val == 6:
                                return f"3{suit}" in enemy_discards and f"9{suit}" in enemy_discards
                            return False

                        is_suji_safe = all(_is_suji(norm_pai, current_riichis[r]) for r in other_riichis)

                        # A tile is considered a safe/fold choice if it is genbutsu or suji against enemies
                        if is_genbutsu or is_suji_safe:
                            self.profiles[name].fold_turns += 1
                        else:
                            # Genuine dangerous musuji tile pushed
                            self.profiles[name].aggressiveness_pushes += 1

            elif etype in ("hora", "RecordHule", "ActionHule"):
                # Handle hules list (single or multiple/daburon)
                raw_hules = data.get("hules") if isinstance(data.get("hules"), list) else ev.get("hules")
                if not raw_hules:
                    raw_hules = [data if isinstance(data, dict) else ev]

                deltas = data.get("delta_scores") or ev.get("deltas") or []
                neg_scorers = [i for i, d in enumerate(deltas) if d < 0]

                for h in raw_hules:
                    raw_winner = h.get("winner") if "winner" in h else (h.get("actor") if "actor" in h else h.get("seat"))
                    if raw_winner is None:
                        raw_winner = ev.get("winner") if "winner" in ev else ev.get("actor", 0)
                    actor = int(raw_winner) if raw_winner is not None else 0

                    raw_zimo = h.get("zimo") if "zimo" in h else ev.get("zimo")
                    raw_target = h.get("target") if "target" in h else ev.get("target")
                    raw_loser = h.get("loser") if "loser" in h else (data.get("loser") if "loser" in data else ev.get("loser"))

                    if raw_zimo is not None:
                        is_zimo = bool(raw_zimo)
                    elif raw_loser is not None and int(raw_loser) != actor:
                        is_zimo = False
                    elif raw_target is not None:
                        is_zimo = (int(raw_target) == actor)
                    elif len(neg_scorers) >= 2:
                        is_zimo = True
                    else:
                        is_zimo = False

                    # Target / Loser determination:

                    loser = None
                    if not is_zimo:
                        if raw_loser is not None and 0 <= int(raw_loser) < 4 and int(raw_loser) != actor:
                            loser = int(raw_loser)
                        elif raw_target is not None and 0 <= int(raw_target) < 4 and int(raw_target) != actor:
                            loser = int(raw_target)
                        elif neg_scorers and len(neg_scorers) == 1 and neg_scorers[0] != actor:
                            loser = neg_scorers[0]
                        elif current_last_discard_seat is not None and current_last_discard_seat != actor:
                            loser = current_last_discard_seat
                        elif neg_scorers:
                            valid_negs = [s for s in neg_scorers if s != actor]
                            if valid_negs:
                                loser = valid_negs[0]

                    if 0 <= actor < 4 and actor in self.stats:
                        self.stats[actor]["agari"] += 1

                    if 0 <= actor < len(player_names):
                        winner_name = clean_player_name(player_names[actor])
                        if winner_name in self.profiles:
                            p = self.profiles[winner_name]
                            p.hora_count += 1
                            win_score = 0
                            # 1. Prioritize authentic base win score (dadian or score)
                            ev_sc = h.get("dadian") or h.get("score") or ev.get("score")
                            if ev_sc is not None and 0 < int(ev_sc) <= 48000:
                                win_score = int(ev_sc)
                            elif h.get("point_sum") is not None and 0 < int(h["point_sum"]) <= 48000:
                                win_score = int(h["point_sum"])
                            else:
                                win_score = 5200
                            p.hora_scores_sum += win_score

                    # Ron deal-in check: strictly increment deal_in_count for loser only
                    if loser is not None and loser != actor:
                        if 0 <= loser < 4 and loser in self.stats:
                            self.stats[loser]["houju"] += 1
                        if 0 <= loser < len(player_names):
                            victim_name = clean_player_name(player_names[loser])
                            if victim_name in self.profiles:
                                self.profiles[victim_name].deal_in_count += 1

                _commit_round()

            elif etype in ("end_kyoku", "ryukyoku", "RecordLiuJu", "ActionLiuJu", "RecordNoTile", "ActionNoTile"):
                _commit_round()

        # Final commit in case end_game happened without end_kyoku
        _commit_round()

        if target_uuid:
            for n in player_names:
                clean_n = clean_player_name(n)
                if clean_n and clean_n in self.profiles:
                    p = self.profiles[clean_n]
                    if target_uuid not in p.recorded_game_uuids:
                        p.recorded_game_uuids.append(target_uuid)

        # Recompute all metrics after game
        for p in self.profiles.values():
            p.compute_metrics()

    def analyze_files(self, file_paths: List[str]) -> Dict[str, PlayerProfile]:
        """Loads and processes multiple mjai JSON or JSONL paifu logs."""
        for fp in file_paths:
            if not os.path.exists(fp):
                continue
            with open(fp, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    continue
                try:
                    # Try full JSON array
                    if content.startswith("["):
                        events = json.loads(content)
                    else:
                        # JSONL
                        events = [json.loads(line) for line in content.splitlines() if line.strip()]
                    self.analyze_game_events(events)
                except Exception as e:
                    print(f"[Profiler] Error parsing {fp}: {e}")

        return self.profiles

    def save_profiles(self, filepath: str) -> None:
        """Saves compiled player profiles to JSON."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        data = {name: prof.model_dump() for name, prof in self.profiles.items()}
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[Profiler] Saved {len(self.profiles)} player profiles to: {filepath}")

    def load_profiles(self, filepath: str) -> Dict[str, PlayerProfile]:
        """Loads player profiles from JSON."""
        if not os.path.exists(filepath):
            return {}
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.profiles = {name: PlayerProfile(**pdata) for name, pdata in data.items()}
        return self.profiles

    def get_exploit_modifiers(
        self,
        enemy_name: str,
        current_bakaze: str = "E",
        is_enemy_calling: bool = False
    ) -> Dict[str, float]:
        """
        Calculates dynamic tactical adjustment scales against this specific enemy:
        - Incorporates Bayesian smoothed base stats and current live session trends.
        - Tonpuu (East-only) mode: Heightened reaction speed and 1.3x danger penalty against calls.
        - Hanchan (East-South) mode: Fully applies push-fold exploit once entering South wind.
        """
        prof = self.get_profile(enemy_name)
        if not prof:
            return {"defense_scale_mult": 1.0, "danger_penalty_mult": 1.0, "attack_scale_mult": 1.0}

        # Select live session blending weight: 0.40 for Tonpuu, 0.35 for Hanchan
        live_w = 0.40 if self.match_mode == "tonpuu" else 0.35
        eff = prof.get_effective_metrics(live_weight=live_w)

        eff_aggr = eff.get("aggressiveness", 0.50)
        eff_open = eff.get("open_rate", 0.30)

        defense_mult = 1.0
        danger_mult = 1.0
        attack_mult = 1.0
        note = ""

        # Exploit phase scaling:
        # In Hanchan games, East round (Tonba) is data accumulation; South round (Nanba) applies full exploit.
        phase_scale = 1.0
        if self.match_mode == "hanchan":
            phase_scale = 1.0 if current_bakaze in ("S", "W", "N") else 0.70
        else:
            phase_scale = 1.15

        # 1. Push-Fold Tendency Exploitation
        is_pusher = (
            eff_aggr >= 0.55
            or prof.archetype == "Berserker Pusher"
            or (prof.aggressiveness >= 0.65 and prof.aggressiveness_total_turns >= 2)
        )
        is_turtle = (
            (eff_aggr <= 0.32 and prof.aggressiveness_total_turns >= 5)
            or prof.archetype == "Iron Wall Turtle"
            or (prof.aggressiveness <= 0.30 and prof.aggressiveness_total_turns >= 2)
        )

        if is_pusher:
            defense_delta = (0.82 - 1.0) * phase_scale
            danger_delta = (0.75 - 1.0) * phase_scale
            attack_delta = (1.25 - 1.0) * phase_scale

            defense_mult = 1.0 + defense_delta
            danger_mult = 1.0 + danger_delta
            attack_mult = 1.0 + attack_delta
            note = f"Exploit {prof.name} (Aggressive Pusher: Aggr {eff_aggr:.2f}): counter-attack promoted"

        elif is_turtle:
            defense_delta = (1.40 - 1.0) * phase_scale
            danger_delta = (1.35 - 1.0) * phase_scale
            attack_delta = (0.80 - 1.0) * phase_scale

            defense_mult = 1.0 + defense_delta
            danger_mult = 1.0 + danger_delta
            attack_mult = 1.0 + attack_delta
            note = f"Exploit {prof.name} (Iron Wall: Aggr {eff_aggr:.2f}): heavy fold against dangerous riichi"

        # 2. Tonpuu Mode Speed / Call Alert (1.3x weight on calls)
        if self.match_mode == "tonpuu" and (is_enemy_calling or eff_open >= 0.35):
            danger_mult *= 1.30
            defense_mult *= 1.15
            tonpuu_tag = "[東風速攻警戒 1.3x]"
            note = f"{tonpuu_tag} {note}" if note else tonpuu_tag

        return {
            "defense_scale_mult": round(defense_mult, 3),
            "danger_penalty_mult": round(danger_mult, 3),
            "attack_scale_mult": round(attack_mult, 3),
            "exploit_note": note.strip()
        }

