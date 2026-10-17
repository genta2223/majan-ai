"""Majsoul (Liqi) protocol to mjai format bridge and tile converter."""

from __future__ import annotations
from typing import List, Dict, Any, Optional
from io_adapter.liqi_proto_decoder import is_valid_tile_code


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
        self._current_doras: List[str] = []
        self._last_discard_seat: Optional[int] = None
        # Vision is a last-resort fallback only. It must never override packet-decoded tiles.
        self.vision_fallback_enabled = vision_fallback_enabled

    def translate_action(self, action_name: str, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Translates a single Liqi action and its payload data into a list of mjai events.
        """
        events: List[Dict[str, Any]] = []
        clean_name = action_name.split(".")[-1]

        if clean_name in ("ActionNewRound", "RecordNewRound", "ActionMJStart", "NotifyRoomGameStart"):
            # New Round (start_kyoku)
            self._last_discard_seat = None
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
                tehais[self.my_seat] = [liqi_tile_to_mjai(t) for t in my_tiles]

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
            # NEVER for ActionMJStart (which arrives during table start animation and reads all tiles as blank/P).
            my_hand = tehais[self.my_seat]
            is_empty_hand = (not my_hand) or (all(t == "?" for t in my_hand))
            is_genuine_round = clean_name in ("ActionNewRound", "RecordNewRound")
            if is_empty_hand and is_genuine_round and self.vision_fallback_enabled:
                try:
                    from io_adapter.vision_hand_reader import VisionHandReader
                    reader = VisionHandReader()
                    vision_tiles = reader.read_hand()
                    if vision_tiles and len(vision_tiles) >= 13:
                        # Sanity check: must not be 13+ identical tiles (e.g. all 'P' from blank animation)
                        norm_vision = [liqi_tile_to_mjai(t) for t in vision_tiles]
                        if len(set(norm_vision)) >= 3:
                            tehais[self.my_seat] = norm_vision
                            print(f"[Vision Fallback] Overrode self hand with {len(tehais[self.my_seat])} vision tiles: {tehais[self.my_seat]}")
                        else:
                            print(f"[Vision Fallback] Rejected vision tiles due to low diversity (likely animation blank screen): {norm_vision}")
                except Exception as ve:
                    print(f"[Vision Fallback] Warning: Hand capture fallback failed: {ve}")

            # Fill unknown opponent hands with hidden tile markers
            for i in range(4):
                if not tehais[i]:
                    tehais[i] = ["?"] * 13

            # Dealer initial 14-tile handling:
            # If dealer starts with 14 tiles, MJAI standard splits into 13-tile hand + 14th tile tsumo
            dealer_tsumo_tile = None
            if len(tehais[oya]) >= 14:
                dealer_tsumo_tile = tehais[oya][-1]
                tehais[oya] = tehais[oya][:-1]

            events.append({
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
            })

            # If dealer has 14th tile, immediately trigger initial draw event
            if dealer_tsumo_tile:
                events.append({
                    "type": "tsumo",
                    "actor": oya,
                    "pai": dealer_tsumo_tile
                })

        elif clean_name in ("ActionDealTile", "RecordDealTile", "NotifyDealTile"):
            seat = data.get("seat", 0)
            tile = liqi_tile_to_mjai(data.get("tile", ""))
            # In live play, non-self draws are always hidden ('?') to prevent accidental seat switching or phantom tile leaks
            if seat != self.my_seat and not clean_name.startswith("Record"):
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

            # Vision only when the packet itself was undecodable (wire_ok is False).
            # A structurally valid DealTile without a tile is an opponent's hidden draw:
            # vision must NOT run there (it caused phantom tiles / wrong seat switching).
            packet_corrupt = data.get("wire_ok") is False
            if (self.vision_fallback_enabled and packet_corrupt
                    and seat == self.my_seat and (not tile or tile == "?")):
                try:
                    from io_adapter.vision_hand_reader import VisionHandReader
                    reader = VisionHandReader()
                    vision_tiles = reader.read_hand()
                    if vision_tiles and len(vision_tiles) >= 14:
                        if vision_tiles[-1] and vision_tiles[-1] != "?":
                            tile = vision_tiles[-1]
                            print(f"[Bridge Vision Sync] Identified self tsumo tile: {tile}")
                except Exception as ve:
                    pass

            tsumo_event: Dict[str, Any] = {
                "type": "tsumo",
                "actor": seat,
                "pai": tile if tile else "?",
            }
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

        elif clean_name in ("ActionHule", "RecordHule"):
            hules = data.get("hules", [])
            if not hules and ("seat" in data or "target" in data):
                hules = [data]
            deltas = safe_coerce_int_list(data.get("delta_scores"), default=[0, 0, 0, 0], expected_len=4)
            for h in hules:
                seat = safe_coerce_int(h.get("seat", 0), 0)
                hu_tile = liqi_tile_to_mjai(h.get("hu_tile", ""))
                is_zimo = bool(h.get("zimo", False))
                raw_target = h.get("target")

                if is_zimo:
                    target = seat
                elif raw_target is not None and safe_coerce_int(raw_target, -1) != seat and 0 <= safe_coerce_int(raw_target, -1) < 4:
                    target = safe_coerce_int(raw_target, seat)
                elif self._last_discard_seat is not None and self._last_discard_seat != seat:
                    target = self._last_discard_seat
                else:
                    target = seat if is_zimo else (self._last_discard_seat if self._last_discard_seat is not None else seat)

                score_val = safe_coerce_int(h.get("score") or h.get("point_sum"), 0)
                if not (0 < score_val <= 48000):
                    if 0 <= seat < len(deltas) and 0 < deltas[seat] <= 48000:
                        score_val = deltas[seat]
                    else:
                        score_val = 5200

                events.append({
                    "type": "hora",
                    "actor": seat,
                    "target": target,
                    "pai": hu_tile,
                    "score": score_val,
                    "deltas": deltas,
                })
            events.append({"type": "end_kyoku"})

        elif clean_name in ("ActionNoTile", "RecordNoTile", "ActionLiuJu", "RecordLiuJu"):
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
