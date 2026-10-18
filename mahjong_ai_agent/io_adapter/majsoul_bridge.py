"""Majsoul (Liqi) protocol to mjai format bridge and tile converter."""

from __future__ import annotations
from typing import List, Dict, Any, Optional
from io_adapter.liqi_proto_decoder import is_valid_tile_code, calculate_pure_hora_score


# Tile mapping tables
LIQI_TO_MJAI_HONORS = {
    "1z": "E",
    "2z": "S",
    "3z": "W",
    "4z": "N",
    "5z": "P",
    "6z": "F",
    "7z": "C",
}

MJAI_TO_LIQI_HONORS = {v: k for k, v in LIQI_TO_MJAI_HONORS.items()}

LIQI_HONOR_ALIASES = {
    "ton": "E", "higashi": "E",
    "nan": "S", "minami": "S",
    "sha": "W", "nishi": "W",
    "pei": "N", "kita": "N",
    "haku": "P", "shiro": "P",
    "hatsu": "F",
    "chun": "C", "naka": "C",
}

LIQI_TO_MJAI_AKA = {
    "0m": "5mr",
    "0p": "5pr",
    "0s": "5sr",
}

MJAI_TO_LIQI_AKA = {v: k for k, v in LIQI_TO_MJAI_AKA.items()}

WIND_CHARS = ["E", "S", "W", "N"]


def safe_coerce_int(value: Any, default: int = 0) -> int:
    """Safely coerces any value (int, bytes, varint, str) to int, never raising exceptions."""
    try:
        from io_adapter.liqi_proto_decoder import coerce_int
        return coerce_int(value, default=default)
    except Exception:
        pass
    try:
        if isinstance(value, int):
            return value
        if isinstance(value, (bytes, bytearray)):
            b = bytes(value)
            return int.from_bytes(b[:8], "little", signed=False) if b else default
        if isinstance(value, str) and value.strip().lstrip("-").isdigit():
            return int(value.strip())
    except Exception:
        pass
    return default


def safe_coerce_int_list(values: Any, default: Optional[List[int]] = None, expected_len: int = 4) -> List[int]:
    """Safely coerces scores/deltas (including proto3 packed varint bytes) to List[int], never raising."""
    fallback = list(default) if default is not None else [25000] * expected_len
    try:
        from io_adapter.liqi_proto_decoder import coerce_int_list
        res = coerce_int_list(values, default=fallback)
        if res:
            while len(res) < expected_len:
                res.append(fallback[len(res)] if len(fallback) > len(res) else 25000)
            return [safe_coerce_int(x, 25000) for x in res[:expected_len]]
    except Exception:
        pass
    return fallback


def liqi_tile_to_mjai(tile: Any) -> str:
    """
    Convert Majsoul (Liqi) tile notation to standard mjai tile notation.
    e.g. '0m' -> '5mr', '1z' -> 'E', '7z' -> 'C', '1m' -> '1m'
    """
    if not tile:
        return ""
    if isinstance(tile, bytes):
        tile = tile.decode("utf-8", errors="replace")
    elif not isinstance(tile, str):
        tile = str(tile)
    if tile in LIQI_TO_MJAI_AKA:
        return LIQI_TO_MJAI_AKA[tile]
    if tile in LIQI_TO_MJAI_HONORS:
        return LIQI_TO_MJAI_HONORS[tile]
    if tile in LIQI_HONOR_ALIASES:
        return LIQI_HONOR_ALIASES[tile]
    return tile


def mjai_tile_to_liqi(tile: str) -> str:
    """
    Convert standard mjai tile notation to Majsoul (Liqi) tile notation.
    e.g. '5mr' -> '0m', 'E' -> '1z', 'C' -> '7z', '1m' -> '1m'
    """
    if not tile:
        return ""
    if tile in MJAI_TO_LIQI_AKA:
        return MJAI_TO_LIQI_AKA[tile]
    if tile in MJAI_TO_LIQI_HONORS:
        return MJAI_TO_LIQI_HONORS[tile]
    # In case input was already '0m' or '1z'
    return tile


class MajsoulBridge:
    """Translates Majsoul (Liqi) action JSON data into one or more mjai events."""

    def __init__(self, my_seat: int = 0, vision_fallback_enabled: bool = True):
        self.my_seat = my_seat
        self.my_account_id: Optional[int] = None
        self.current_game_uuid: Optional[str] = None
        self.match_mode: str = "hanchan"       # "tonpuu" or "hanchan"
        self.player_nicknames: List[str] = ["", "", "", ""]
        self.on_seat_changed: Optional[Callable[[int], None]] = None
        self._current_doras: List[str] = []
        self._last_discard_seat: Optional[int] = None
        self._in_round: bool = False
        self._last_round_params: Dict[str, Any] = {}
        self._initial_round_hand: List[str] = []
        # Vision is a last-resort fallback only. It must never override packet-decoded tiles.
        self.vision_fallback_enabled = vision_fallback_enabled

    def set_my_account_id(self, account_id: int) -> None:
        """Dynamically bind self account_id (e.g. from FastTest.authGame request packet)."""
        if account_id and isinstance(account_id, int) and account_id > 0:
            self.my_account_id = account_id
            print(f"[MajsoulBridge AutoSeat] Successfully bound login account_id: {self.my_account_id}")

    def set_my_seat(self, seat: int) -> None:
        """Update my_seat and notify listeners if changed."""
        if 0 <= seat <= 3 and self.my_seat != seat:
            old = self.my_seat
            self.my_seat = seat
            print(f"[MajsoulBridge AutoSeat] Switched self seat: {old} -> {seat}")
            if self.on_seat_changed:
                try:
                    self.on_seat_changed(seat)
                except Exception as e:
                    print(f"[MajsoulBridge] on_seat_changed callback error: {e}")

    def resolve_seat_from_player_list(self, players_or_seats: Any) -> Optional[int]:
        """
        Matches my_account_id against player list to dynamically resolve self seat (0..3).
        Supports list of dicts, int account_ids, raw_fields, or protobuf submessages.
        """
        if not self.my_account_id or not players_or_seats:
            return None

        # If passed a dict containing raw_fields or nested player containers
        if isinstance(players_or_seats, dict):
            # Check known sub-keys
            for sub_key in ("players", "player_list", "seat_list", "seats", "game_details"):
                if sub_key in players_or_seats and players_or_seats[sub_key]:
                    res = self.resolve_seat_from_player_list(players_or_seats[sub_key])
                    if res is not None:
                        return res
            # Check raw_fields
            if "raw_fields" in players_or_seats:
                rf = players_or_seats["raw_fields"]
                for f_num, f_vals in rf.items():
                    if isinstance(f_vals, (list, tuple)):
                        res = self.resolve_seat_from_player_list(f_vals)
                        if res is not None:
                            return res

        if isinstance(players_or_seats, (list, tuple)):
            for idx, item in enumerate(players_or_seats):
                acc_id = None
                seat_val = idx
                if isinstance(item, dict):
                    acc_id = item.get("account_id") or item.get("accountId") or item.get("id")
                    if not acc_id and "raw_fields" in item:
                        # Scan raw_fields of the player message for account_id
                        p_rf = item["raw_fields"]
                        if 1 in p_rf and p_rf[1] and isinstance(p_rf[1][0], int):
                            acc_id = p_rf[1][0]
                    if "seat" in item and 0 <= safe_coerce_int(item["seat"], -1) <= 3:
                        seat_val = safe_coerce_int(item["seat"], idx)
                    nick = item.get("nickname") or item.get("name")
                    if nick and 0 <= seat_val < 4:
                        self.player_nicknames[seat_val] = str(nick)
                elif isinstance(item, int):
                    acc_id = item
                    seat_val = idx
                elif isinstance(item, (bytes, bytearray)):
                    from io_adapter.liqi_proto_decoder import decode_raw_protobuf
                    p_rf = decode_raw_protobuf(item)
                    if 1 in p_rf and p_rf[1] and isinstance(p_rf[1][0], int):
                        acc_id = p_rf[1][0]

                if acc_id == self.my_account_id:
                    print(f"[MajsoulBridge AutoSeat] Dynamically resolved my_seat={seat_val} for account_id={self.my_account_id}")
                    self.set_my_seat(seat_val)
                    return seat_val
        return None

    def _generate_rescue_start_kyoku(self, data: Dict[str, Any], trigger_action: str) -> Dict[str, Any]:
        """
        Dynamically synthesizes a start_kyoku event when sniffing starts in the middle of a game/round.
        Recovers self seat, wind/round indicators, and hand tiles if available.
        """
        raw_chang = safe_coerce_int(data.get("chang", self._last_round_params.get("chang", 0)), 0)
        chang_idx = raw_chang if 0 <= raw_chang < len(WIND_CHARS) else 0
        bakaze = WIND_CHARS[chang_idx]

        ju = safe_coerce_int(data.get("ju", self._last_round_params.get("ju", 0)), 0)
        kyoku = ju + 1
        honba = safe_coerce_int(data.get("ben", self._last_round_params.get("ben", 0)), 0)
        kyotaku = safe_coerce_int(data.get("liqibang", self._last_round_params.get("liqibang", 0)), 0)
        oya = safe_coerce_int(data.get("oya", ju % 4), ju % 4)

        scores = safe_coerce_int_list(
            data.get("scores", self._last_round_params.get("scores")),
            default=[25000, 25000, 25000, 25000],
            expected_len=4
        )

        # Dora extraction
        raw_doras = data.get("doras") or data.get("dora") or data.get("dora_marker") or []
        if isinstance(raw_doras, str):
            raw_doras = [raw_doras]
        doras = [d for d in raw_doras if is_valid_tile_code(d)]
        dora_marker = liqi_tile_to_mjai(doras[0]) if doras else ""
        if dora_marker:
            self._current_doras = [dora_marker]

        # Dynamic seat resolution (only on deal actions where self tile is unmasked):
        deal_seat = data.get("seat")
        raw_tile = data.get("tile", "")
        if trigger_action in ("ActionDealTile", "RecordDealTile", "NotifyDealTile"):
            if deal_seat is not None and 0 <= safe_coerce_int(deal_seat, -1) <= 3:
                s_int = safe_coerce_int(deal_seat)
                if raw_tile and is_valid_tile_code(raw_tile):
                    self.set_my_seat(s_int)
                elif "operation_list" in data or "operations" in data:
                    self.set_my_seat(s_int)

        # Hand setup
        tehais: List[List[str]] = [["?"] * 13 for _ in range(4)]
        raw_tiles = data.get("tiles") or data.get("hand") or []
        if isinstance(raw_tiles, (list, tuple)):
            valid_tiles = [liqi_tile_to_mjai(t) for t in raw_tiles if is_valid_tile_code(t)]
            if valid_tiles:
                tehais[self.my_seat] = valid_tiles

        print(
            f"[MajsoulBridge RESCUE] Mid-game join detected on '{trigger_action}'. "
            f"Synthesized start_kyoku: {bakaze}{kyoku}局 {honba}本場, MySeat=P{self.my_seat}, Hand={len(tehais[self.my_seat])} tiles."
        )

        rescue_event: Dict[str, Any] = {
            "type": "start_kyoku",
            "action_name": "RescueNewRound",
            "bakaze": bakaze,
            "kyoku": kyoku,
            "honba": honba,
            "kyotaku": kyotaku,
            "oya": oya,
            "scores": scores,
            "tehais": tehais,
            "dora_marker": dora_marker,
            "is_rescue": True,
        }
        if any(self.player_nicknames):
            rescue_event["names"] = list(self.player_nicknames)
        if self.current_game_uuid:
            rescue_event["game_uuid"] = self.current_game_uuid
        if self.match_mode:
            rescue_event["match_mode"] = self.match_mode

        return rescue_event

    def translate_action(self, action_name: str, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Translates a single Liqi action and its payload data into a list of mjai events.
        """
        events: List[Dict[str, Any]] = []
        clean_name = action_name.split(".")[-1]

        # 1. Login / Game Auth Request (.lq.FastTest.authGame)
        if clean_name in ("authGame", "ReqAuthGame"):
            acc_id = safe_coerce_int(data.get("account_id", 0), 0)
            if acc_id > 0:
                self.set_my_account_id(acc_id)
            uuid = data.get("game_uuid") or data.get("uuid")
            if uuid:
                self.current_game_uuid = str(uuid)
            return []

        # 2. Player info updates, Auth responses, table animations, or mid-game sync
        if clean_name in (
            "ResAuthGame", "NotifyRoomPlayerUpdate", "NotifyPlayerLoadGameReady",
            "NotifyRoomGameStart", "ActionMJStart",
            "syncGame", "ResSyncGame", "fetchGamePlayerState"
        ):
            plist = (
                data.get("players")
                or data.get("player_list")
                or data.get("seat_list")
                or data.get("seats")
                or []
            )
            if plist:
                self.resolve_seat_from_player_list(plist)
            uuid = data.get("game_uuid") or data.get("uuid")
            if uuid:
                self.current_game_uuid = str(uuid)

            # ActionMJStart / NotifyRoomGameStart are pre-game table opening animations:
            # They MUST NOT reset hand tiles or emit empty start_kyoku! Keep state clean until ActionNewRound.
            if clean_name in ("NotifyRoomGameStart", "ActionMJStart"):
                print(f"[MajsoulBridge] Pre-game animation '{clean_name}' acknowledged. Ready for authentic ActionNewRound.")
                return []

            # If mid-game state sync has game details and we are not in round, initialize via rescue
            if clean_name in ("syncGame", "ResSyncGame", "fetchGamePlayerState") and not self._in_round:
                rescue_start = self._generate_rescue_start_kyoku(data, clean_name)
                events.append(rescue_start)
                self._in_round = True
            return events

        # 3. Authentic Round Start: ONLY ActionNewRound and RecordNewRound carry verified 13-tile hands
        if clean_name in ("ActionNewRound", "RecordNewRound"):
            # New Round (start_kyoku)
            self._last_discard_seat = None

            # Dynamic Seat Fallback: Check if explicit seat tag is in NewRound data
            if "seat" in data and isinstance(data["seat"], int) and 0 <= data["seat"] <= 3 and data.get("tiles"):
                if self.my_seat != data["seat"]:
                    print(f"[MajsoulBridge AutoSeat] Dynamic fallback: Aligned my_seat {self.my_seat} -> {data['seat']} from ActionNewRound seat tag.")
                    self.set_my_seat(data["seat"])

            raw_chang = safe_coerce_int(data.get("chang", 0), 0)
            chang_idx = raw_chang if 0 <= raw_chang < len(WIND_CHARS) else 0
            bakaze = WIND_CHARS[chang_idx]

            ju = safe_coerce_int(data.get("ju", 0), 0)
            kyoku = ju + 1

            honba = safe_coerce_int(data.get("ben", 0), 0)
            kyotaku = safe_coerce_int(data.get("liqibang", 0), 0)

            oya = ju % 4       # Standard dealer rotation
            scores = safe_coerce_int_list(data.get("scores"), default=[25000, 25000, 25000, 25000], expected_len=4)

            # Dora indicators
            raw_doras = (
                data.get("doras")
                or data.get("dora")
                or data.get("dora_marker")
                or data.get("dora_indicators")
                or []
            )
            if isinstance(raw_doras, str):
                raw_doras = [raw_doras]
            elif not isinstance(raw_doras, (list, tuple)):
                raw_doras = []

            doras = [d for d in raw_doras if is_valid_tile_code(d)]
            dora_marker = liqi_tile_to_mjai(doras[0]) if doras else ""
            self._current_doras = [dora_marker] if dora_marker else []

            # Hand tiles for players
            tehais: List[List[str]] = [[], [], [], []]
            # 1. Direct self-dealt tiles (live client repeated string tiles = 4)
            # Strictly filter with is_valid_tile_code to prevent metadata ('69', etc.) pollution
            raw_my_tiles = data.get("tiles") or []
            my_tiles = [t for t in raw_my_tiles if is_valid_tile_code(t)] if isinstance(raw_my_tiles, (list, tuple)) else []
            if my_tiles:
                self._initial_round_hand = [liqi_tile_to_mjai(t) for t in my_tiles]
                tehais[self.my_seat] = list(self._initial_round_hand)

            # 2. Replay/spectator logs passing tiles0..tiles3
            for i in range(4):
                key = f"tiles{i}"
                if key in data and data[key]:
                    raw_seat_tiles = data[key]
                    valid_seat_tiles = [t for t in raw_seat_tiles if is_valid_tile_code(t)] if isinstance(raw_seat_tiles, (list, tuple)) else []
                    if valid_seat_tiles:
                        tehais[i] = [liqi_tile_to_mjai(t) for t in valid_seat_tiles]

            # 3. If self hand is placed in tiles0 but self.my_seat != 0, move or copy
            if not tehais[self.my_seat] and tehais[0]:
                tehais[self.my_seat] = list(tehais[0])

            # 4. Vision Fallback: If packet has 0 hand tiles for self, capture screen
            # ONLY allow vision fallback for genuine round start (ActionNewRound / RecordNewRound),
            # NEVER for ActionMJStart / NotifyRoomGameStart (which arrive during table start animation).
            my_hand = tehais[self.my_seat]
            is_empty_hand = (not my_hand) or (all(t == "?" for t in my_hand))
            is_genuine_round = clean_name in ("ActionNewRound", "RecordNewRound")
            if is_empty_hand and is_genuine_round and self.vision_fallback_enabled:
                try:
                    from io_adapter.vision_hand_reader import VisionHandReader
                    reader = VisionHandReader()
                    vision_tiles = reader.read_hand()
                    if vision_tiles and len(vision_tiles) >= 13:
                        norm_vision = [liqi_tile_to_mjai(t) for t in vision_tiles]
                        if len(set(norm_vision)) >= 3:
                            tehais[self.my_seat] = norm_vision
                except Exception:
                    pass

            # Fill unknown opponent hands with hidden tile markers
            for i in range(4):
                if not tehais[i]:
                    tehais[i] = ["?"] * 13

            # Dealer initial draw handling:
            # MJAI standard: start_kyoku has 13-tile hands for all players.
            # If dealer starts with >= 14 tiles, MJAI standard splits into 13-tile hand + 14th tile tsumo.
            # If dealer is self (oya == self.my_seat) and hand has 13 tiles, emit initial tsumo immediately
            # so the AI engine is brought into 14-tile state and triggered for initial discard.
            dealer_tsumo_tile = None
            if len(tehais[oya]) >= 14:
                dealer_tsumo_tile = tehais[oya][-1]
                tehais[oya] = list(tehais[oya][:13])
            elif clean_name in ("ActionNewRound", "RecordNewRound") and oya == self.my_seat and len(tehais[oya]) >= 13 and any(t != "?" for t in tehais[oya]):
                cand_tile = data.get("tile") or data.get("first_tile") or data.get("tsumo")
                if cand_tile and is_valid_tile_code(cand_tile):
                    dealer_tsumo_tile = liqi_tile_to_mjai(cand_tile)
                else:
                    dealer_tsumo_tile = tehais[oya][-1]
                # Keep tehais[oya] at 13 tiles; the following tsumo event will transition hand to 14 tiles
                tehais[oya] = list(tehais[oya][:13])

            start_event: Dict[str, Any] = {
                "type": "start_kyoku",
                "action_name": clean_name,
                "bakaze": bakaze,
                "kyoku": kyoku,
                "honba": honba,
                "kyotaku": kyotaku,
                "oya": oya,
                "scores": scores,
                "tehais": tehais,
                "dora_marker": dora_marker,
                "is_rescue": False,
            }
            if any(self.player_nicknames):
                start_event["names"] = list(self.player_nicknames)
            if self.current_game_uuid:
                start_event["game_uuid"] = self.current_game_uuid
            if self.match_mode:
                start_event["match_mode"] = self.match_mode

            self._in_round = True
            self._last_round_params = {
                "chang": raw_chang,
                "ju": ju,
                "ben": honba,
                "liqibang": kyotaku,
                "scores": list(scores),
                "oya": oya,
            }
            events.append(start_event)

            # If dealer has initial draw tile, immediately trigger initial draw event
            if dealer_tsumo_tile:
                events.append({
                    "type": "tsumo",
                    "actor": oya,
                    "pai": dealer_tsumo_tile
                })

        elif clean_name in ("ActionDealTile", "RecordDealTile", "NotifyDealTile"):
            seat = data.get("seat", 0)
            raw_tile = data.get("tile", "")
            
            # Dynamic Seat resolution:
            # If live packet contains operations or multi-tile self hand, this seat IS self!
            has_explicit_operations = bool("operation_list" in data or "operations" in data)
            has_multi_tiles = (
                (isinstance(data.get("tiles"), (list, tuple)) and len(data.get("tiles")) >= 2)
                or (isinstance(data.get("hand"), (list, tuple)) and len(data.get("hand")) >= 2)
            )
            has_self_indicators = has_explicit_operations or has_multi_tiles
            if not clean_name.startswith("Record") and has_self_indicators:
                if self.my_seat != seat and 0 <= safe_coerce_int(seat, -1) <= 3:
                    print(f"[MajsoulBridge AutoSeat] Identified self seat {seat} from deal operations/hand. Switched self seat {self.my_seat} -> {seat}.")
                    self.set_my_seat(safe_coerce_int(seat))

            # Opponent draw isolation:
            # If seat != self.my_seat and not a replay Record, non-self draws are always hidden ('?')
            # to strictly prevent accidental seat switching or phantom tile leaks.
            if seat != self.my_seat and not clean_name.startswith("Record"):
                tile = "?"
            elif raw_tile and is_valid_tile_code(raw_tile):
                tile = liqi_tile_to_mjai(raw_tile)
            elif clean_name.startswith("Record") and raw_tile:
                tile = liqi_tile_to_mjai(raw_tile)
            else:
                tile = "?"

            # Check if new dora indicator was revealed
            raw_doras = data.get("doras") or data.get("dora") or data.get("dora_marker") or []
            if isinstance(raw_doras, str):
                raw_doras = [raw_doras]
            elif not isinstance(raw_doras, (list, tuple)):
                raw_doras = []
            for d in raw_doras:
                if is_valid_tile_code(d):
                    mjai_d = liqi_tile_to_mjai(d)
                    if mjai_d and mjai_d not in self._current_doras:
                        self._current_doras.append(mjai_d)
                        events.append({"type": "dora", "dora_marker": mjai_d})


            tsumo_event: Dict[str, Any] = {
                "type": "tsumo",
                "actor": seat,
                "pai": tile if tile else "?",
            }
            raw_hand = data.get("tiles") or data.get("hand")
            if raw_hand and isinstance(raw_hand, (list, tuple)):
                valid_hand = [liqi_tile_to_mjai(t) for t in raw_hand if is_valid_tile_code(t)]
                if valid_hand:
                    tsumo_event["tiles"] = valid_hand

            raw_ops = data.get("operation_list") or data.get("operations") or data.get("operation") or []
            if isinstance(raw_ops, dict):
                raw_ops = [raw_ops]
            if raw_ops:
                tsumo_event["operation_list"] = self._normalize_operations(raw_ops)
            events.append(tsumo_event)

        elif clean_name in ("ActionDiscardTile", "RecordDiscardTile", "NotifyDiscardTile"):
            seat = safe_coerce_int(data.get("seat", 0), 0)
            self._last_discard_seat = seat
            tile = liqi_tile_to_mjai(data.get("tile", ""))
            moqie = bool(data.get("moqie", False))
            is_liqi = bool(data.get("is_liqi", False) or data.get("is_wliqi", False))

            # Extract operations (calls available on this discard)
            raw_ops = data.get("operation_list") or data.get("operations") or data.get("operation") or []
            if isinstance(raw_ops, dict):
                raw_ops = [raw_ops]
            parsed_ops = self._normalize_operations(raw_ops) if raw_ops else []

            dahai_event: Dict[str, Any] = {
                "type": "dahai",
                "actor": seat,
                "pai": tile,
                "tsumogiri": moqie,
            }
            if parsed_ops:
                dahai_event["operation_list"] = parsed_ops

            # If riichi declaration tile
            if is_liqi:
                events.append({"type": "reach", "actor": seat})
                events.append(dahai_event)
                events.append({"type": "reach_accepted", "actor": seat})
            else:
                events.append(dahai_event)

            # Check doras revealed after kan discards
            raw_doras = data.get("doras") or data.get("dora") or data.get("dora_marker") or []
            if isinstance(raw_doras, str):
                raw_doras = [raw_doras]
            elif not isinstance(raw_doras, (list, tuple)):
                raw_doras = []
            for d in raw_doras:
                if is_valid_tile_code(d):
                    mjai_d = liqi_tile_to_mjai(d)
                    if mjai_d and mjai_d not in self._current_doras:
                        self._current_doras.append(mjai_d)
                        events.append({"type": "dora", "dora_marker": mjai_d})

        elif clean_name in ("ActionChiPengGang", "RecordChiPengGang"):
            # Strictly identify actor (seat who called) vs target (seat who discarded)
            seat = safe_coerce_int(data.get("seat", 0), 0)
            meld_type = safe_coerce_int(data.get("type", 0), 0)  # 0=Chi, 1=Pon, 2=Daiminkan
            raw_tiles = data.get("tiles", [])
            froms = data.get("froms", [])
            mjai_tiles = [liqi_tile_to_mjai(t) for t in raw_tiles if t]

            # Determine who discarded the called tile (target != seat)
            target = None
            if froms:
                for f_seat in froms:
                    f_int = safe_coerce_int(f_seat, -1)
                    if f_int != seat and 0 <= f_int < 4:
                        target = f_int
                        break
            if target is None:
                if self._last_discard_seat is not None and self._last_discard_seat != seat:
                    target = self._last_discard_seat
                else:
                    target = (seat - 1) % 4

            # Segregate called_tile vs consumed
            called_tile = ""
            consumed: List[str] = []
            if len(froms) == len(mjai_tiles) and len(mjai_tiles) > 0:
                for f_seat, t in zip(froms, mjai_tiles):
                    f_int = safe_coerce_int(f_seat, -1)
                    if f_int != seat and not called_tile:
                        called_tile = t
                    else:
                        consumed.append(t)
            if not called_tile and mjai_tiles:
                called_tile = mjai_tiles[-1]
                consumed = mjai_tiles[:-1]

            if meld_type == 0:
                events.append({
                    "type": "chi",
                    "actor": seat,
                    "target": target,
                    "pai": called_tile,
                    "consumed": consumed,
                })
            elif meld_type == 1:
                events.append({
                    "type": "pon",
                    "actor": seat,
                    "target": target,
                    "pai": called_tile,
                    "consumed": consumed,
                })
            else:
                events.append({
                    "type": "daiminkan",
                    "actor": seat,
                    "target": target,
                    "pai": called_tile,
                    "consumed": consumed,
                })

        elif clean_name in ("ActionAnGangAddGang", "RecordAnGangAddGang"):
            seat = safe_coerce_int(data.get("seat", 0), 0)
            gang_type = safe_coerce_int(data.get("type", 3), 3)  # 2=AddGang(Kankan), 3=AnGang
            raw_tiles = data.get("tiles", [])
            mjai_tiles = [liqi_tile_to_mjai(t) for t in raw_tiles if t]

            if gang_type == 2:
                events.append({
                    "type": "kankan",
                    "actor": seat,
                    "consumed": mjai_tiles,
                })
            else:
                events.append({
                    "type": "ankan",
                    "actor": seat,
                    "consumed": mjai_tiles,
                })

        elif clean_name in ("ActionBaBei", "RecordBaBei"):
            seat = safe_coerce_int(data.get("seat", 0), 0)
            events.append({
                "type": "nuki",
                "actor": seat,
                "pai": "4z",
            })

        elif clean_name in ("ActionHule", "RecordHule"):
            hules = data.get("hules", [])
            if not hules and ("seat" in data or "target" in data or "winner" in data):
                hules = [data]
            deltas = safe_coerce_int_list(data.get("delta_scores"), default=[0, 0, 0, 0], expected_len=4)
            neg_scorers = [i for i, d in enumerate(deltas) if d < 0]

            for h in hules:
                raw_winner = h.get("winner") if "winner" in h else (data.get("winner") if "winner" in data else None)
                seat = safe_coerce_int(raw_winner if raw_winner is not None else h.get("seat", 0), 0)
                hu_tile = liqi_tile_to_mjai(h.get("hu_tile", ""))
                is_zimo = bool(h.get("zimo", False))
                raw_loser = h.get("loser") if "loser" in h else (data.get("loser") if "loser" in data else None)
                raw_target = h.get("target")

                if is_zimo:
                    target = seat
                    loser = None
                else:
                    target = None
                    # 1. Direct negative scorer from delta_scores on Ron (financial deal-in is gold truth)
                    if neg_scorers:
                        deal_in_candidates = [s for s in neg_scorers if s != seat]
                        if deal_in_candidates:
                            target = deal_in_candidates[0]
                    # 2. Explicit loser field
                    if target is None and raw_loser is not None and safe_coerce_int(raw_loser, -1) != seat and 0 <= safe_coerce_int(raw_loser, -1) < 4:
                        target = safe_coerce_int(raw_loser, seat)
                    # 3. Explicit target field
                    elif target is None and raw_target is not None and safe_coerce_int(raw_target, -1) != seat and 0 <= safe_coerce_int(raw_target, -1) < 4:
                        target = safe_coerce_int(raw_target, seat)
                    # 4. Last discard seat
                    elif target is None and self._last_discard_seat is not None and self._last_discard_seat != seat:
                        target = self._last_discard_seat

                    if target is None or target == seat:
                        target = seat if is_zimo else (self._last_discard_seat if self._last_discard_seat is not None and self._last_discard_seat != seat else seat)
                    loser = target if (target != seat and not is_zimo) else None

                # Authentic base score calculation (pure score)
                fu_val = safe_coerce_int(h.get("fu"), 0)
                fan_val = safe_coerce_int(h.get("fan"), 0)
                is_dealer_val = bool(h.get("is_dealer", False))
                dadian_val = safe_coerce_int(h.get("dadian"), 0)

                if 0 < dadian_val <= 48000:
                    score_val = dadian_val
                elif fu_val > 0 and fan_val > 0:
                    score_val = calculate_pure_hora_score(fu_val, fan_val, is_dealer_val)
                else:
                    score_val = safe_coerce_int(h.get("score"), 0)
                    if not (0 < score_val <= 48000):
                        score_val = safe_coerce_int(h.get("point_sum"), 0)
                        if not (0 < score_val <= 48000):
                            score_val = 5200

                hora_ev = {
                    "type": "hora",
                    "actor": seat,
                    "target": target,
                    "winner": seat,
                    "loser": loser,
                    "zimo": is_zimo,
                    "pai": hu_tile,
                    "score": score_val,
                    "dadian": score_val,
                    "deltas": deltas,
                }
                if fu_val > 0:
                    hora_ev["fu"] = fu_val
                if fan_val > 0:
                    hora_ev["fan"] = fan_val
                if is_dealer_val:
                    hora_ev["is_dealer"] = is_dealer_val

                events.append(hora_ev)
            self._in_round = False
            events.append({"type": "end_kyoku"})

        elif clean_name in ("ActionNoTile", "RecordNoTile", "ActionLiuJu", "RecordLiuJu"):
            self._in_round = False
            events.append({
                "type": "ryukyoku",
                "deltas": safe_coerce_int_list(data.get("delta_scores"), default=[0, 0, 0, 0], expected_len=4),
            })
            events.append({"type": "end_kyoku"})

        return events

    def _normalize_operations(self, raw_ops: List[Any]) -> List[Dict[str, Any]]:
        """Normalizes and translates Liqi operations and combination tiles to standard mjai format."""
        parsed_ops: List[Dict[str, Any]] = []
        op_names = {
            1: "none", 2: "chi", 3: "pon", 4: "ankan", 5: "daiminkan",
            6: "kankan", 7: "reach", 8: "hora", 9: "hora", 10: "ryukyoku", 11: "nuki"
        }
        for op in raw_ops:
            if not isinstance(op, dict):
                continue
            op_type = op.get("type", 0)
            t_name = op.get("type_name") or op_names.get(op_type, "none")
            raw_combs = op.get("combinations", op.get("combination", []))
            converted_combs = []
            for c in raw_combs:
                if isinstance(c, str):
                    parts = [liqi_tile_to_mjai(p) for p in c.split("|")]
                    converted_combs.append(parts)
                elif isinstance(c, list):
                    converted_combs.append([liqi_tile_to_mjai(p) for p in c])

            parsed_ops.append({
                "type": op_type,
                "type_name": t_name,
                "combinations": converted_combs,
                "seat": op.get("seat", self.my_seat)
            })
        return parsed_ops
