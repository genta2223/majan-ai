"""Opponent Profiling & Tactical Adaptation module for analyzing player habits from paifu logs."""

from __future__ import annotations
import json
import os
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

from core.state_manager import normalize_tile, HONORS


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

    # Detailed habit counts
    yakuhai_call_count: int = 0     # Honor tile melds
    tanyao_call_count: int = 0      # All-simples melds (numbers 2-8)
    early_call_count: int = 0       # Melds made within turns 1-6
    early_honor_discards: int = 0   # Honor discards within turns 1-6
    early_total_discards: int = 0   # Total discards within turns 1-6

    # Computed metrics (0.0 to 1.0 or points)
    hora_rate: float = 0.0          # Hora rounds / total rounds
    deal_in_rate: float = 0.0       # Deal in rounds / total rounds
    riichi_rate: float = 0.0        # Riichi rounds / total rounds
    open_rate: float = 0.0          # Melded rounds / total rounds
    avg_hora_score: float = 0.0     # Average score per win
    aggressiveness: float = 0.50    # Un-safe push rate against riichi
    fold_rate: float = 0.50         # Safe tile fold rate against riichi
    safety_hoarding: float = 0.50   # Hoards honors / genbutsu until late game

    # Tendency profile & badges
    archetype: str = "Balanced"
    call_tendency: str = "バランス型仕掛け"       # 役牌速攻型 / 喰いタン多用 / 遠い鳴き多め / 門前重視
    push_fold_tendency: str = "バランス型押し引き"   # 鉄壁ベタオリ / 超攻撃的プッシャー / やや押し気味 / 安全重視
    early_game_tendency: str = "標準的進行"      # 速攻字牌整理型 / 字牌抱え守備型 / 標準的進行
    badges: List[str] = Field(default_factory=list)

    @property
    def display_name(self) -> str:
        """Returns name formatted with seat info if present, e.g. 'genta2223 (東家)'."""
        if self.seat_name and f"({self.seat_name})" not in self.name:
            return f"{self.name} ({self.seat_name})"
        return self.name

    def compute_metrics(self) -> None:
        """Calculates normalized percentages, behavioral archetype, and badges."""
        if self.total_rounds > 0:
            self.hora_rate = round(self.hora_count / self.total_rounds, 3)
            self.open_rate = round(self.rounds_with_call / self.total_rounds, 3)
            self.riichi_rate = round(self.rounds_with_riichi / self.total_rounds, 3)
            self.deal_in_rate = round(self.deal_in_count / self.total_rounds, 3)

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
    """

    def __init__(self, profiles_path: Optional[str] = None):
        self.profiles: Dict[str, PlayerProfile] = {}
        self.stats: Dict[int, Dict[str, int]] = {
            i: {
                "agari": 0,
                "houju": 0,
                "reach": 0,
                "call": 0,
                "rounds": 0,
                "tanyao": 0,
                "yakuhai": 0,
            }
            for i in range(4)
        }
        self.profiles_path = profiles_path
        if profiles_path and os.path.exists(profiles_path):
            self.load_profiles(profiles_path)

    def get_profile(self, name: str) -> Optional[PlayerProfile]:
        return self.profiles.get(name)

    def analyze_game_events(self, events: List[Dict[str, Any]]) -> None:
        """
        Processes a full game's mjai events and accumulates player habits.
        Strictly maps accounts to seating, and tracks hora, deal_in, calls, and riichi.
        """
        SEAT_NAMES = ["東家", "南家", "西家", "北家"]
        player_names: List[str] = ["", "", "", ""]
        current_riichis: Dict[int, List[str]] = {}  # actor -> list of discards for genbutsu
        current_round_calls: set[int] = set()
        current_round_riichis: set[int] = set()
        turn_counter: int = 0
        round_committed: bool = True

        def _commit_round():
            nonlocal round_committed
            if not round_committed:
                for actor in current_round_calls:
                    if 0 <= actor < len(player_names):
                        name = player_names[actor]
                        if name in self.profiles:
                            self.profiles[name].rounds_with_call += 1
                for actor in current_round_riichis:
                    if 0 <= actor < len(player_names):
                        name = player_names[actor]
                        if name in self.profiles:
                            self.profiles[name].rounds_with_riichi += 1
                round_committed = True

        for ev in events:
            etype = ev.get("type", "")

            if etype == "start_game":
                names = ev.get("names", [])
                for i in range(min(4, len(names))):
                    name = str(names[i]).strip()
                    player_names[i] = name
                    if name:
                        if name not in self.profiles:
                            self.profiles[name] = PlayerProfile(name=name, seat_name=SEAT_NAMES[i])
                        else:
                            if not self.profiles[name].seat_name:
                                self.profiles[name].seat_name = SEAT_NAMES[i]
                        self.profiles[name].total_games += 1

            elif etype == "start_kyoku":
                # Ensure previous round was committed
                _commit_round()
                current_riichis.clear()
                current_round_calls.clear()
                current_round_riichis.clear()
                turn_counter = 0
                round_committed = False
                for i in range(4):
                    if i in self.stats:
                        self.stats[i]["rounds"] += 1
                for name in player_names:
                    if name and name in self.profiles:
                        self.profiles[name].total_rounds += 1

            elif etype in ("chi", "pon", "daiminkan"):
                actor = ev.get("actor", 0)
                if 0 <= actor < 4 and actor in self.stats:
                    self.stats[actor]["call"] += 1

                if 0 <= actor < len(player_names):
                    name = player_names[actor]
                    consumed = ev.get("consumed", [])
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

            elif etype == "reach":
                actor = ev.get("actor", 0)
                if 0 <= actor < 4 and actor in self.stats:
                    self.stats[actor]["reach"] += 1

                if 0 <= actor < len(player_names):
                    name = player_names[actor]
                    current_round_riichis.add(actor)
                    current_riichis[actor] = []
                    if name in self.profiles:
                        self.profiles[name].riichi_count += 1

            elif etype == "dahai":
                actor = ev.get("actor", 0)
                if 0 <= actor < len(player_names):
                    pai = ev.get("pai", "")
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
                    if other_riichis and name in self.profiles:
                        self.profiles[name].aggressiveness_total_turns += 1
                        is_safe = all(norm_pai in current_riichis[r] for r in other_riichis)
                        if not is_safe:
                            self.profiles[name].aggressiveness_pushes += 1
                        else:
                            self.profiles[name].fold_turns += 1

            elif etype == "hora":
                actor = ev.get("actor", 0)
                target = ev.get("target", actor)
                deltas = ev.get("deltas", [])

                if 0 <= actor < 4 and actor in self.stats:
                    self.stats[actor]["agari"] += 1

                if 0 <= actor < len(player_names):
                    winner_name = player_names[actor]
                    if winner_name in self.profiles:
                        p = self.profiles[winner_name]
                        p.hora_count += 1
                        win_score = 0
                        if deltas and len(deltas) > actor and deltas[actor] > 0:
                            win_score = deltas[actor]
                        elif "score" in ev:
                            win_score = ev["score"]
                        else:
                            win_score = 5200
                        p.hora_scores_sum += win_score

                # Ron deal-in check (actor != target means Ron)
                if actor != target:
                    if 0 <= target < 4 and target in self.stats:
                        self.stats[target]["houju"] += 1
                    if 0 <= target < len(player_names):
                        victim_name = player_names[target]
                        if victim_name in self.profiles:
                            self.profiles[victim_name].deal_in_count += 1

                _commit_round()

            elif etype in ("end_kyoku", "ryukyoku"):
                _commit_round()

        # Final commit in case end_game happened without end_kyoku
        _commit_round()

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

    def get_exploit_modifiers(self, enemy_name: str) -> Dict[str, float]:
        """
        Calculates dynamic tactical adjustment scales against this specific enemy:
        - defense_scale_mult: If enemy is Berserker, don't over-fold (reduce defense scale).
                              If enemy is Iron Wall, respect their riichi heavily (increase defense scale).
        - danger_penalty_mult: Multiplier on danger cost.
        - attack_scale_mult: Multiplier on attack value.
        """
        prof = self.get_profile(enemy_name)
        if not prof or (prof.total_rounds < 1 and prof.aggressiveness_total_turns < 1):
            return {"defense_scale_mult": 1.0, "danger_penalty_mult": 1.0, "attack_scale_mult": 1.0}

        # Baseline: aggressiveness = 0.50
        # If enemy pushes aggressively (e.g. 0.80), their riichi is frequently cheap or foolish
        if prof.aggressiveness >= 0.65:
            # Exploit Berserker Pusher: Push back more boldly, penalize danger less
            return {
                "defense_scale_mult": 0.82,
                "danger_penalty_mult": 0.78,
                "attack_scale_mult": 1.25,
                "exploit_note": f"Exploit {prof.name} (Berserker Pusher: Aggr {prof.aggressiveness}): counter-attack encouraged"
            }
        elif prof.aggressiveness <= 0.35:
            # Exploit Iron Wall: When they riichi, they hold monster hands with high tenpai shape!
            return {
                "defense_scale_mult": 1.40,
                "danger_penalty_mult": 1.35,
                "attack_scale_mult": 0.80,
                "exploit_note": f"Exploit {prof.name} (Iron Wall: Aggr {prof.aggressiveness}): heavy fold against high value riichi"
            }

        return {"defense_scale_mult": 1.0, "danger_penalty_mult": 1.0, "attack_scale_mult": 1.0}
