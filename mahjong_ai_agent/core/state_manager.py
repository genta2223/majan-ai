from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple, Any
from pydantic import BaseModel, Field
from utils.debug_tracer import trace_log, mark_trace_event


# Tile utilities
SUITS = ["m", "p", "s"]
HONORS = ["1z", "2z", "3z", "4z", "5z", "6z", "7z"] # ton, nan, sha, pei, haku, hatsu, chun
ALL_CANONICAL_TILES = {f"{i}{s}" for s in SUITS for i in range(1, 10)} | set(HONORS)


def normalize_tile(tile: str) -> str:
    """Normalize tile notation into standard canonical notation (e.g., '5mr' -> '5m', '0m' -> '5m', 'E' -> '1z')."""
    if not tile:
        return ""
    t = str(tile).strip()
    # Map wind/dragon names if present
    wind_map = {
        "E": "1z", "ton": "1z", "higashi": "1z",
        "S": "2z", "nan": "2z", "minami": "2z",
        "W": "3z", "sha": "3z", "nishi": "3z",
        "N": "4z", "pei": "4z", "kita": "4z",
        "P": "5z", "haku": "5z", "shiro": "5z",
        "F": "6z", "hatsu": "6z",
        "C": "7z", "chun": "7z", "naka": "7z",
    }
    if t in wind_map:
        return wind_map[t]
    # Aka-dora normalization to standard five
    if t in ("0m", "5mr"):
        return "5m"
    if t in ("0p", "5pr"):
        return "5p"
    if t in ("0s", "5sr"):
        return "5s"
    # Remove red dora indicator 'r' for standard matching
    return t.replace("r", "")


def is_valid_mahjong_tile(tile: str) -> bool:
    """Validate whether the given string represents a valid mahjong tile."""
    if not tile or not isinstance(tile, str):
        return False
    norm = normalize_tile(tile)
    return norm in ALL_CANONICAL_TILES


def is_aka_dora(tile: str) -> bool:
    """Check if tile is a red five."""
    if not tile:
        return False
    t = str(tile).strip()
    return "r" in t or t in ("0m", "0p", "0s")


def tiles_match(t1: str, t2: str, ignore_red: bool = True) -> bool:
    """Check if two tile representations refer to the same tile."""
    if not t1 or not t2:
        return False
    if ignore_red:
        return normalize_tile(t1) == normalize_tile(t2)
    return (is_aka_dora(t1) == is_aka_dora(t2)) and (normalize_tile(t1) == normalize_tile(t2))


def get_dora_next(indicator: str) -> str:
    """Given a dora indicator, return the dora tile."""
    if not indicator:
        return ""
    norm = normalize_tile(str(indicator))
    if len(norm) != 2:
        return norm
    try:
        val, suit = int(norm[0]), norm[1]
    except Exception:
        return norm

    if suit in SUITS:
        next_val = 1 if val == 9 else val + 1
        return f"{next_val}{suit}"
    elif suit == "z":
        # 1z(E)->2z(S)->3z(W)->4z(N)->1z(E)
        if 1 <= val <= 4:
            next_val = 1 if val == 4 else val + 1
            return f"{next_val}z"
        # 5z(Haku)->6z(Hatsu)->7z(Chun)->5z(Haku)
        if 5 <= val <= 7:
            next_val = 5 if val == 7 else val + 1
            return f"{next_val}z"
    return norm


class Meld(BaseModel):
    """Represents a meld (chi, pon, daiminkan, kankan, ankan)."""
    type: str
    tiles: List[str]
    target: Optional[int] = None
    opened: bool = True


class PlayerState(BaseModel):
    """Tracks state of an individual player."""
    actor_id: int
    name: str = ""
    score: int = 25000
    rank: int = 1
    tehai: List[str] = Field(default_factory=list) # Self hand tiles (others may be empty or known)
    tsumo_tile: Optional[str] = None
    discards: List[str] = Field(default_factory=list)
    tsumogiris: List[bool] = Field(default_factory=list)
    melds: List[Meld] = Field(default_factory=list)
    is_riichi: bool = False
    riichi_turn: Optional[int] = None
    riichi_accepted: bool = False
    is_dealer: bool = False

    @property
    def discards_normalized(self) -> List[str]:
        return [normalize_tile(t) for t in self.discards]

    def has_tile(self, tile: str, ignore_red: bool = True) -> bool:
        """Check if tile physically exists in self hand."""
        return any(tiles_match(t, tile, ignore_red=ignore_red) for t in self.tehai)

    def count_tile(self, tile: str, ignore_red: bool = True) -> int:
        """Count occurrences of tile in self hand."""
        return sum(1 for t in self.tehai if tiles_match(t, tile, ignore_red=ignore_red))

    def get_matching_tile_in_hand(self, tile: str) -> Optional[str]:
        """
        Return the physical tile representation from self.tehai that matches the given tile.
        Prioritizes exact match, then falls back to normalized match.
        """
        if not tile:
            return None
        # 1. Exact match
        for t in self.tehai:
            if t == tile:
                return t
        # 2. Normalized match
        norm = normalize_tile(tile)
        for t in self.tehai:
            if normalize_tile(t) == norm:
                return t
        return None

    def can_consume(self, tiles: List[str], ignore_red: bool = True) -> bool:
        """
        Check if self hand has sufficient tiles to consume the given list of tiles.
        Validates tile pool counting to prevent overdraft.
        """
        if not tiles:
            return True
        pool: Dict[str, int] = {}
        for t in self.tehai:
            if not is_valid_mahjong_tile(t):
                continue
            norm = normalize_tile(t) if ignore_red else t
            pool[norm] = pool.get(norm, 0) + 1
        for req in tiles:
            if not is_valid_mahjong_tile(req):
                return False
            req_key = normalize_tile(req) if ignore_red else req
            if pool.get(req_key, 0) <= 0:
                return False
            pool[req_key] -= 1
        return True


class KyokuState(BaseModel):
    """Represents the state of the current round (Kyoku)."""
    bakaze: str = "E"       # 'E' (East), 'S' (South), 'W' (West)
    kyoku: int = 1          # 1 ~ 4
    honba: int = 0
    kyotaku: int = 0
    oya: int = 0
    turn_count: int = 0
    dora_indicators: List[str] = Field(default_factory=list)
    dora_tiles: List[str] = Field(default_factory=list)
    players: List[PlayerState] = Field(default_factory=list)
    active_actor: int = 0
    last_discard_tile: Optional[str] = None
    last_discard_actor: Optional[int] = None
    is_round_over: bool = False

    def is_ooras(self) -> bool:
        """Return True if this round is the final round (South 4 or West 4 for standard hanchan)."""
        return (self.bakaze == "S" and self.kyoku >= 4) or (self.bakaze == "W" and self.kyoku >= 4)

    def get_dora_count(self, tile: str) -> int:
        """Calculate how many doras (normal + aka) the given tile represents."""
        if not tile:
            return 0
        count = 0
        if is_aka_dora(tile):
            count += 1
        norm = normalize_tile(tile)
        for dt in self.dora_tiles:
            if normalize_tile(dt) == norm:
                count += 1
        return count

    def is_dora_tile(self, tile: str) -> bool:
        """Return True if tile is either a normal dora or an aka dora."""
        return self.get_dora_count(tile) > 0


class MahjongStateManager:
    """Manages full game state by consuming mjai protocol events."""

    def __init__(self, my_actor_id: int = 0):
        self.my_actor_id = my_actor_id
        self.game_active: bool = False
        self.round_state: Optional[KyokuState] = None
        self.initial_scores: List[int] = [25000, 25000, 25000, 25000]

    def reset_round(self, bakaze: str, kyoku: int, honba: int, kyotaku: int, oya: int, scores: List[int], tehais: List[List[str]], dora_marker: Any):
        """Initialize state for a new round."""
        if isinstance(dora_marker, (list, tuple)):
            dora_indicators = [m for m in dora_marker if m and is_valid_mahjong_tile(m)]
        elif isinstance(dora_marker, str) and dora_marker:
            dora_indicators = [dora_marker] if is_valid_mahjong_tile(dora_marker) else []
        else:
            dora_indicators = []
        dora_tiles = [get_dora_next(m) for m in dora_indicators]

        # Defensive coercion for all numeric inputs: guarantees PlayerState.score is always an int
        def _to_int(val: Any, default: int = 25000) -> int:
            try:
                if isinstance(val, int):
                    return val
                if isinstance(val, (bytes, bytearray)):
                    from io_adapter.liqi_proto_decoder import coerce_int
                    return coerce_int(val, default)
                if isinstance(val, str) and val.strip().lstrip("-").isdigit():
                    return int(val.strip())
            except Exception:
                pass
            return default

        players = []
        for i in range(4):
            raw_player_tehai = tehais[i] if i < len(tehais) and tehais[i] else []
            # Filter strictly valid mahjong tiles for self hand
            if i == self.my_actor_id:
                clean_tehai = [t for t in raw_player_tehai if t and t not in ("?", "None") and is_valid_mahjong_tile(t)]
            else:
                clean_tehai = list(raw_player_tehai)

            raw_score = scores[i] if (scores and i < len(scores)) else 25000
            p = PlayerState(
                actor_id=i,
                score=_to_int(raw_score, 25000),
                tehai=clean_tehai,
                is_dealer=(i == oya)
            )
            players.append(p)

        self.round_state = KyokuState(
            bakaze=bakaze,
            kyoku=kyoku,
            honba=honba,
            kyotaku=kyotaku,
            oya=oya,
            dora_indicators=dora_indicators,
            dora_tiles=dora_tiles,
            players=players,
            active_actor=oya
        )
        self._update_ranks()
        self.verify_hand_consistency(context="reset_round")

    def verify_hand_consistency(self, context: str = "") -> bool:
        """
        Verify that player hand state matches mahjong rules and protocol expectations.
        Checks hand size according to melds, eliminates invalid/dummy tiles, and logs warnings.
        """
        my_p = self.my_player
        if not my_p:
            return True

        # 1. Clean invalid entries from tehai (non-tiles, '?', 'None', empty, unrecognized strings)
        clean_tehai = []
        invalid_tiles = []
        for t in my_p.tehai:
            if t and t not in ("?", "None") and is_valid_mahjong_tile(t):
                clean_tehai.append(t)
            else:
                invalid_tiles.append(t)

        if invalid_tiles:
            trace_log("TEHAI_INVALID_ENTRIES", f"Removed {len(invalid_tiles)} invalid tiles in {context}: {invalid_tiles}")
            print(f"[StateManager WARNING] Removed invalid entries from hand in {context}: {invalid_tiles}")
            my_p.tehai = clean_tehai

        # 2. Prevent impossible duplicates: no canonical tile can appear > 4 times in hand
        tile_counts: Dict[str, int] = {}
        deduped_tehai = []
        for t in my_p.tehai:
            norm = normalize_tile(t)
            cnt = tile_counts.get(norm, 0)
            if cnt < 4:
                tile_counts[norm] = cnt + 1
                deduped_tehai.append(t)
            else:
                print(f"[StateManager ALERT] Impossible tile count (>4) for '{t}' (norm: '{norm}') in {context}! Dropped excess tile.")
        my_p.tehai = deduped_tehai

        # 3. Check hand size
        meld_count = len(my_p.melds)
        base_size = 13 - (3 * meld_count)
        draw_size = 14 - (3 * meld_count)
        current_len = len(my_p.tehai)

        if current_len not in (base_size, draw_size):
            msg = (
                f"[StateManager SYNC ALERT] Hand size anomaly in {context}: current={current_len}, "
                f"expected base={base_size} or draw={draw_size} (melds={meld_count}). Hand: {my_p.tehai}"
            )
            print(msg)
            trace_log("HAND_SIZE_ANOMALY", msg, {"tehai": list(my_p.tehai), "melds": meld_count, "context": context})

            # Safe self-healing:
            # 1. If hand size has overflowed beyond draw_size, trim excess
            if current_len > draw_size:
                trim_count = current_len - draw_size
                print(f"[StateManager RECOVERY] Hand size {current_len} > {draw_size}. Trimming {trim_count} overflow tile(s) from hand end: {my_p.tehai[-trim_count:]}")
                my_p.tehai = my_p.tehai[:draw_size]
            # 2. If hand size has collapsed below base_size, resurrect missing tiles from discards to prevent crash
            elif current_len < base_size:
                needed = base_size - current_len
                print(f"[StateManager RECOVERY] Hand size {current_len} < {base_size} in {context}. Reconstructing {needed} missing tile(s) to protect 13-tile integrity...")
                for d in reversed(my_p.discards):
                    if len(my_p.tehai) >= base_size:
                        break
                    if is_valid_mahjong_tile(d) and my_p.tehai.count(d) < 4:
                        my_p.tehai.append(d)
                        print(f"[StateManager RECOVERY] Restored '{d}' from discards into hand: {my_p.tehai}")
            return False

        return True

    def dump_state(self) -> str:
        """Dump concise summary of current round state for debugging and watchdog inspection."""
        if not self.round_state:
            return f"[StateDump] No active round_state (game_active={self.game_active}, my_actor_id={self.my_actor_id})"
        rs = self.round_state
        my_p = self.my_player
        tehai_str = " ".join(my_p.tehai) if my_p else "None"
        tsumo_str = my_p.tsumo_tile if (my_p and my_p.tsumo_tile) else "None"
        scores_str = [p.score for p in rs.players]
        return (
            f"[StateDump] {rs.bakaze}{rs.kyoku}局 {rs.honba}本場 | Oya={rs.oya} | ActiveActor=P{rs.active_actor} | "
            f"MyActor=P{self.my_actor_id} | Tehai=({len(my_p.tehai) if my_p else 0}): [{tehai_str}] | "
            f"Tsumo={tsumo_str} | Discards={len(my_p.discards) if my_p else 0} | Turn={rs.turn_count} | Scores={scores_str}"
        )

    def update_event(self, event: Dict[str, Any]) -> None:
        """Consume a single mjai-formatted event with top-level robust exception guard."""
        try:
            self._update_event_impl(event)
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            ev_type = event.get("type", "unknown") if isinstance(event, dict) else "unknown"
            print(f"[StateManager GUARD ALERT] Suppressed exception updating event '{ev_type}': {e}\n{tb}")
            trace_log("STATE_UPDATE_EXCEPTION", f"Event '{ev_type}' error: {e}\n{tb}")

    def _update_event_impl(self, event: Dict[str, Any]) -> None:
        """Internal consumption logic for a single mjai-formatted event."""
        event_type = event.get("type", "")
        trace_log("STATE_CONSUME_EVENT", f"Type: {event_type}", event)

        # Emergency Rescue Guard: If an in-game turn event arrives before start_kyoku
        # (e.g. agent attached mid-game/mid-round), dynamically synthesize board state immediately.
        if event_type in ("tsumo", "dahai", "chi", "pon", "kan", "daiminkan", "kankan", "ankan", "reach") and not self.round_state:
            print(f"[StateManager RESCUE] In-game event '{event_type}' arrived without active round_state. Initializing emergency board state...")
            bakaze = event.get("bakaze", "E")
            kyoku = event.get("kyoku", 1)
            scores = event.get("scores", self.initial_scores)
            tehais = event.get("tehais", [["?"] * 13 for _ in range(4)])
            dora_marker = event.get("dora_marker", "")
            self.reset_round(
                bakaze=bakaze,
                kyoku=kyoku,
                honba=event.get("honba", 0),
                kyotaku=event.get("kyotaku", 0),
                oya=event.get("oya", 0),
                scores=scores,
                tehais=tehais,
                dora_marker=dora_marker
            )

        if event_type == "start_game":
            self.game_active = True
            names = event.get("names", ["P0", "P1", "P2", "P3"])
            if self.round_state:
                for idx, name in enumerate(names):
                    if idx < len(self.round_state.players):
                        self.round_state.players[idx].name = name
            mark_trace_event("START_GAME", self.dump_state())

        elif event_type == "start_kyoku":
            raw_dora = (
                event.get("dora_marker")
                or event.get("dora")
                or event.get("dora_indicators")
                or event.get("doras")
                or ""
            )
            if isinstance(raw_dora, (list, tuple)):
                dora_marker = raw_dora[0] if raw_dora else ""
            else:
                dora_marker = str(raw_dora) if raw_dora else ""

            self.reset_round(
                bakaze=event.get("bakaze", "E"),
                kyoku=event.get("kyoku", 1),
                honba=event.get("honba", 0),
                kyotaku=event.get("kyotaku", 0),
                oya=event.get("oya", 0),
                scores=event.get("scores", self.initial_scores),
                tehais=event.get("tehais", [[], [], [], []]),
                dora_marker=dora_marker
            )
            mark_trace_event("START_KYOKU", self.dump_state())
            trace_log("ROUND_STATE_INITIALIZED", self.dump_state())

        elif event_type == "tsumo":
            if not self.round_state:
                return
            actor = event["actor"]
            pai = event.get("pai", "")
            self.round_state.active_actor = actor
            p = self.round_state.players[actor]
            p.tsumo_tile = pai

            if actor == self.my_actor_id:
                # Mid-game rescue: If self hand is empty or filled with unknown '?' tiles, recover from event
                if not p.tehai or all(t == "?" for t in p.tehai):
                    raw_hand = event.get("tiles") or event.get("hand") or []
                    if raw_hand:
                        p.tehai = [t for t in raw_hand if is_valid_mahjong_tile(t)]
                        print(f"[StateManager RESCUE] Recovered self hand from event payload: {p.tehai}")

                if pai and pai not in ("?", "None") and is_valid_mahjong_tile(pai):
                    expected_draw_len = 14 - (3 * len(p.melds))
                    if len(p.tehai) >= expected_draw_len:
                        # Replace provisional draw tile (e.g. from dealer ActionNewRound) with verified DealTile draw
                        old_tile = p.tehai[-1]
                        p.tehai[-1] = pai
                        print(f"[StateManager UPDATE] Replaced provisional draw tile '{old_tile}' with verified draw tile '{pai}'.")
                        trace_log("TSUMO_REPLACE_PROVISIONAL", f"Replaced '{old_tile}' with '{pai}' in hand {p.tehai}")
                    else:
                        p.tehai.append(pai)
                self.verify_hand_consistency(context="tsumo")

        elif event_type == "dahai":
            if not self.round_state:
                return
            actor = event["actor"]
            pai = event["pai"]
            tsumogiri = event.get("tsumogiri", False)

            p = self.round_state.players[actor]
            expected_base_len = 13 - (3 * len(p.melds))

            # Guard against duplicate dahai event for self (e.g. sent request followed by recv notification)
            if actor == self.my_actor_id and len(p.tehai) == expected_base_len and p.tsumo_tile is None:
                return

            p.discards.append(pai)
            p.tsumogiris.append(tsumogiri)
            self.round_state.last_discard_tile = pai
            self.round_state.last_discard_actor = actor
            p.tsumo_tile = None
            self.round_state.turn_count += 1

            # Remove from tehai if self
            if actor == self.my_actor_id:
                norm_discard = normalize_tile(pai)
                found_idx = -1
                # 1. Exact match priority
                for idx, t in enumerate(p.tehai):
                    if t == pai:
                        found_idx = idx
                        break
                # 2. Normalized match fallback (ignoring red)
                if found_idx == -1:
                    for idx, t in enumerate(p.tehai):
                        if normalize_tile(t) == norm_discard:
                            found_idx = idx
                            break

                # Strict hand size guard:
                # Discarding must NEVER reduce concealed hand below expected_base_len (13 tiles without melds).
                if len(p.tehai) > expected_base_len:
                    if found_idx != -1:
                        p.tehai.pop(found_idx)
                    else:
                        # Tedashi not found in hand or draw tile was missed: pop last tile to restore valid 13-tile base
                        popped_tile = p.tehai.pop()
                        print(f"[StateManager RECOVERY] Hand was {len(p.tehai)+1} > {expected_base_len}. Popped '{popped_tile}' to restore base hand size.")
                else:
                    # Hand is ALREADY <= expected_base_len! A pop would illegally reduce hand to 12 (or fewer) tiles.
                    # This happens when the draw tile was missed in packet (e.g. ActionDealTile '?').
                    if tsumogiri:
                        msg = f"[StateManager Tsumogiri Draw Missed] Discard '{pai}' was tsumogiri (moqie) but missing from hand. Preserved {len(p.tehai)} hand tiles."
                        print(msg)
                        trace_log("TSUMOGIRI_MISSED_DRAW_PRESERVED", msg)
                    else:
                        # Tedashi: tile was discarded from hand, but draw tile was missing from packet.
                        # Preserve hand size: if tile was found in hand, swap with the discard or preserve size
                        msg = f"[StateManager CRITICAL GUARD] Hand already at {len(p.tehai)} <= {expected_base_len}. Discard '{pai}' prohibited from reducing hand size."
                        print(msg)
                        trace_log("DAHAI_HAND_GUARD", msg)

                # Final fallback safety net: if hand ever drops below expected_base_len, reconstruct by resurrecting discards
                while len(p.tehai) < expected_base_len and p.discards:
                    for d_tile in reversed(p.discards):
                        if is_valid_mahjong_tile(d_tile) and p.tehai.count(d_tile) < 4:
                            p.tehai.append(d_tile)
                            print(f"[StateManager RECOVERY] Restored '{d_tile}' from discards into hand to maintain {len(p.tehai)} tiles.")
                            break
                    if len(p.tehai) < expected_base_len:
                        break

                self.verify_hand_consistency(context="dahai")

        elif event_type == "reach":
            if not self.round_state:
                return
            actor = event["actor"]
            p = self.round_state.players[actor]
            p.is_riichi = True
            p.riichi_turn = len(p.discards)

        elif event_type == "reach_accepted":
            if not self.round_state:
                return
            actor = event["actor"]
            p = self.round_state.players[actor]
            p.riichi_accepted = True
            p.score -= 1000
            self.round_state.kyotaku += 1
            self._update_ranks()

        elif event_type in ("chi", "pon", "daiminkan"):
            if not self.round_state:
                return
            actor = event["actor"]
            target = event.get("target")
            pai = event.get("pai", "")
            consumed = event.get("consumed", [])
            meld_tiles = consumed + ([pai] if pai else [])

            p = self.round_state.players[actor]
            p.melds.append(Meld(type=event_type, tiles=meld_tiles, target=target, opened=True))
            if actor == self.my_actor_id:
                for c in consumed:
                    norm_c = normalize_tile(c)
                    found_idx = -1
                    for idx, t in enumerate(p.tehai):
                        if t == c:
                            found_idx = idx
                            break
                    if found_idx == -1:
                        for idx, t in enumerate(p.tehai):
                            if normalize_tile(t) == norm_c:
                                found_idx = idx
                                break
                    if found_idx != -1:
                        p.tehai.pop(found_idx)
                    else:
                        print(f"[StateManager WARNING] Meld consumed tile '{c}' not found in self hand {p.tehai}!")
                self.verify_hand_consistency(context=f"meld_{event_type}")

        elif event_type in ("kankan", "ankan"):
            if not self.round_state:
                return
            actor = event["actor"]
            consumed = event.get("consumed", [])
            p = self.round_state.players[actor]
            p.melds.append(Meld(type=event_type, tiles=consumed, opened=(event_type == "kankan")))
            if actor == self.my_actor_id:
                for c in consumed:
                    norm_c = normalize_tile(c)
                    found_idx = -1
                    for idx, t in enumerate(p.tehai):
                        if t == c:
                            found_idx = idx
                            break
                    if found_idx == -1:
                        for idx, t in enumerate(p.tehai):
                            if normalize_tile(t) == norm_c:
                                found_idx = idx
                                break
                    if found_idx != -1:
                        p.tehai.pop(found_idx)
                    else:
                        print(f"[StateManager WARNING] Kan consumed tile '{c}' not found in self hand {p.tehai}!")
                self.verify_hand_consistency(context=f"kan_{event_type}")

        elif event_type == "dora":
            if not self.round_state:
                return
            raw_marker = (
                event.get("dora_marker")
                or event.get("dora")
                or event.get("dora_indicators")
                or event.get("doras")
                or ""
            )
            if isinstance(raw_marker, (list, tuple)):
                markers = [m for m in raw_marker if m]
            elif raw_marker:
                markers = [str(raw_marker)]
            else:
                markers = []

            for m in markers:
                if m and is_valid_mahjong_tile(m):
                    self.round_state.dora_indicators.append(m)
                    self.round_state.dora_tiles.append(get_dora_next(m))

        elif event_type in ("hora", "ryukyoku"):
            if not self.round_state:
                return
            self.round_state.is_round_over = True
            deltas = event.get("deltas", None)
            if deltas and len(deltas) == 4:
                for idx, delta in enumerate(deltas):
                    self.round_state.players[idx].score += delta
                self._update_ranks()
            mark_trace_event(event_type.upper(), self.dump_state())
            trace_log("ROUND_OVER", f"{event_type.upper()} | {self.dump_state()}")

        elif event_type == "end_kyoku":
            if self.round_state:
                self.round_state.is_round_over = True
            mark_trace_event("END_KYOKU", self.dump_state())
            trace_log("ROUND_OVER", f"END_KYOKU | {self.dump_state()}")

        elif event_type == "end_game":
            self.game_active = False
            mark_trace_event("END_GAME", "Game finished")
            trace_log("GAME_OVER", "Game finished")

    def _update_ranks(self) -> None:
        """Update ranks for all players based on current scores and tie-breaking by seating."""
        if not self.round_state:
            return
        indexed_scores = [(p.score, -i, i) for i, p in enumerate(self.round_state.players)]
        indexed_scores.sort(reverse=True)
        for rank_zero, (_, _, p_idx) in enumerate(indexed_scores):
            self.round_state.players[p_idx].rank = rank_zero + 1

    @property
    def my_player(self) -> Optional[PlayerState]:
        if not self.round_state or self.my_actor_id >= len(self.round_state.players):
            return None
        return self.round_state.players[self.my_actor_id]

    def get_genbutsu_for_actor(self, riichi_actor: int) -> Set[str]:
        """Return the set of normalized tiles that are 100% safe (Genbutsu) against a riichi actor."""
        if not self.round_state or riichi_actor >= len(self.round_state.players):
            return set()
        actor = self.round_state.players[riichi_actor]
        return set(actor.discards_normalized)

    def get_active_riichi_actors(self) -> List[int]:
        """Return list of other actors who have declared riichi."""
        if not self.round_state:
            return []
        return [
            p.actor_id for p in self.round_state.players
            if p.actor_id != self.my_actor_id and p.is_riichi
        ]

    def get_visible_tiles(self) -> Dict[str, int]:
        """Count all tiles that are publicly visible or in self hand."""
        visible: Dict[str, int] = {}
        if not self.round_state:
            return visible

        def _add(t: str):
            norm = normalize_tile(t)
            visible[norm] = visible.get(norm, 0) + 1

        # Self hand
        my_p = self.my_player
        if my_p:
            for t in my_p.tehai:
                _add(t)

        # Discards & melds of all players
        for p in self.round_state.players:
            for d in p.discards:
                _add(d)
            for m in p.melds:
                for mt in m.tiles:
                    _add(mt)

        # Dora indicators
        for ind in self.round_state.dora_indicators:
            _add(ind)

        return visible

    def estimate_shanten(self, tiles: Optional[List[str]] = None) -> int:
        """
        Precise shanten estimation for the given hand (or self hand).
        Accurately evaluates standard hands (4 mentsu + 1 head), chiitoitsu (7 pairs),
        and kokushi musou without greedy extraction errors.
        Returns integer (-1: agari, 0: tenpai, 1: 1-shanten, etc.).
        """
        if tiles is None:
            my_p = self.my_player
            tiles = my_p.tehai if my_p else []

        if not tiles:
            return 8

        norm_tiles = [normalize_tile(t) for t in tiles]
        counts: Dict[str, int] = {}
        for t in norm_tiles:
            counts[t] = counts.get(t, 0) + 1

        def _decompose_suit(counts_list: List[int]) -> List[tuple[int, int]]:
            """Returns Pareto-optimal (mentsu, taatsu) tuples for a 9-tile suit."""
            results: List[tuple[int, int]] = []

            def _dfs(idx: int, cur: List[int], m: int, t: int):
                while idx < 9 and cur[idx] == 0:
                    idx += 1
                if idx >= 9:
                    results.append((m, t))
                    return

                # Option 1: Triplet (koutsu)
                if cur[idx] >= 3:
                    cur[idx] -= 3
                    _dfs(idx, cur, m + 1, t)
                    cur[idx] += 3

                # Option 2: Sequence (shuntsu)
                if idx <= 6 and cur[idx] > 0 and cur[idx+1] > 0 and cur[idx+2] > 0:
                    cur[idx] -= 1
                    cur[idx+1] -= 1
                    cur[idx+2] -= 1
                    _dfs(idx, cur, m + 1, t)
                    cur[idx] += 1
                    cur[idx+1] += 1
                    cur[idx+2] += 1

                # Option 3: Taatsu - pair (toitsu)
                if cur[idx] >= 2:
                    cur[idx] -= 2
                    _dfs(idx, cur, m, t + 1)
                    cur[idx] += 2

                # Option 4: Taatsu - ryanmen / penchan (i, i+1)
                if idx <= 7 and cur[idx] > 0 and cur[idx+1] > 0:
                    cur[idx] -= 1
                    cur[idx+1] -= 1
                    _dfs(idx, cur, m, t + 1)
                    cur[idx] += 1
                    cur[idx+1] += 1

                # Option 5: Taatsu - kanchan (i, i+2)
                if idx <= 6 and cur[idx] > 0 and cur[idx+2] > 0:
                    cur[idx] -= 1
                    cur[idx+2] -= 1
                    _dfs(idx, cur, m, t + 1)
                    cur[idx] += 1
                    cur[idx+2] += 1

                # Option 6: Skip isolated tile
                cnt = cur[idx]
                cur[idx] = 0
                _dfs(idx + 1, cur, m, t)
                cur[idx] = cnt

            _dfs(0, list(counts_list), 0, 0)
            best: Dict[int, int] = {}
            for m, t in results:
                best[m] = max(best.get(m, -1), t)
            return [(m, t) for m, t in best.items()]

        # Find all pair candidates (janto candidates)
        all_possible_pairs = [tile for tile, cnt in counts.items() if cnt >= 2]
        candidate_heads = [None] + all_possible_pairs
        min_shanten = 8

        for head in candidate_heads:
            work_counts = dict(counts)
            has_pair = 0
            if head:
                work_counts[head] -= 2
                has_pair = 1

            # Honors
            h_mentsu = 0
            h_taatsu = 0
            for h in HONORS:
                c = work_counts.get(h, 0)
                if c >= 3:
                    h_mentsu += 1
                elif c == 2:
                    h_taatsu += 1

            # Number suits
            suit_decomps: List[List[tuple[int, int]]] = []
            for s in SUITS:
                s_counts = [work_counts.get(f"{i}{s}", 0) for i in range(1, 10)]
                if sum(s_counts) == 0:
                    suit_decomps.append([(0, 0)])
                else:
                    suit_decomps.append(_decompose_suit(s_counts))

            # Combine suit decompositions
            for m1, t1 in suit_decomps[0]:
                for m2, t2 in suit_decomps[1]:
                    for m3, t3 in suit_decomps[2]:
                        tot_m = h_mentsu + m1 + m2 + m3
                        tot_t = h_taatsu + t1 + t2 + t3
                        eff_t = min(tot_t, max(0, 4 - tot_m))
                        sh = 8 - 2 * tot_m - eff_t - has_pair
                        if sh < min_shanten:
                            min_shanten = sh

        # Chiitoitsu check
        pair_count = sum(1 for c in counts.values() if c >= 2)
        chiitoi_shanten = 6 - pair_count

        # Kokushi musou check
        terminals = ["1m", "9m", "1p", "9p", "1s", "9s"] + HONORS
        term_count = sum(1 for t in terminals if counts.get(t, 0) > 0)
        has_term_pair = 1 if any(counts.get(t, 0) >= 2 for t in terminals) else 0
        kokushi_shanten = 13 - term_count - has_term_pair

        return max(-1, min(min_shanten, chiitoi_shanten, kokushi_shanten))
