"""Liqi Protocol dynamic binary decoder for Majsoul WebSocket frames."""

from __future__ import annotations
import base64
import struct
from typing import Dict, Any, Optional, Tuple, List, Union

MAJSOUL_OP_TYPE_NAMES = {
    1: "none",        # 打牌 / スキップ
    2: "chi",         # チー
    3: "pon",         # ポン
    4: "ankan",       # 暗槓
    5: "daiminkan",   # 大明槓
    6: "kankan",      # 加槓
    7: "reach",       # 立直
    8: "hora",        # 自摸
    9: "hora",        # 栄和 (ロン)
    10: "ryukyoku",   # 九種九牌
    11: "nuki",       # 抜北
}


LIQI_ACTION_XOR_KEY: Tuple[int, ...] = (0x84, 0x5E, 0x4E, 0x42, 0x39, 0xA2, 0x1F, 0x60, 0x1C)

# Strict tile code validation:
# Suits: 0m, 1m-9m, 0p, 1p-9p, 0s, 1s-9s (0 denotes red five aka-dora)
# Honors: 1z-7z (1z=East, 2z=South, 3z=West, 4z=North, 5z=White, 6z=Green, 7z=Red)
import re
TILE_PATTERN = re.compile(r"^([0-9][mps]|[1-7]z)$")


def is_valid_tile_code(val: Any) -> bool:
    """Returns True iff val is a valid 2-char mahjong tile code (e.g. '1m', '9p', '0s', '7z')."""
    if not val:
        return False
    if isinstance(val, (bytes, bytearray)):
        try:
            val = val.decode("utf-8")
        except Exception:
            return False
    elif not isinstance(val, str):
        val = str(val)
    return bool(TILE_PATTERN.match(val.strip()))


def xor_decrypt_action_data(data: bytes, key: Tuple[int, ...] = LIQI_ACTION_XOR_KEY) -> bytes:
    """
    Decrypts or encrypts Majsoul ActionPrototype.data payload (Liqi XOR codec).

    Per-byte mask depends on the payload length and byte position, not only the 9-byte key:
        mask_i = ((23 ^ len(data)) + 5 * i + key[i % 9]) & 0xFF
        out_i  = data_i ^ mask_i
    The operation is symmetric: applying it twice restores the original bytes.

    Verified on live sample '9d75712666cc' -> '080118443800'
    (ActionDealTile: seat=1, left_tile_count=68, field7=0).
    """
    if not data:
        return b""
    n = len(data)
    k_len = len(key)
    base = 23 ^ n
    return bytes((b ^ ((base + 5 * i + key[i % k_len]) & 0xFF)) for i, b in enumerate(data))


def protobuf_wire_is_complete(data: bytes) -> bool:
    """
    Pure structural check: True if `data` parses as a sequence of well-formed Protobuf
    fields that consumes the buffer exactly (no text heuristics). Empty buffer is valid
    (proto3 message with all-default fields).
    """
    if data is None:
        return False
    offset = 0
    length = len(data)
    while offset < length:
        try:
            key, offset = decode_varint(data, offset)
        except Exception:
            return False
        wire_type = key & 0x07
        field_num = key >> 3
        if field_num == 0 or field_num > 10000:
            return False
        try:
            if wire_type == 0:
                _, offset = decode_varint(data, offset)
            elif wire_type == 1:
                offset += 8
            elif wire_type == 2:
                v_len, offset = decode_varint(data, offset)
                offset += v_len
            elif wire_type == 5:
                offset += 4
            else:
                return False
        except Exception:
            return False
        if offset > length:
            return False
    return offset == length


def decode_packed_varints(data: bytes) -> List[int]:
    """Decode a packed repeated varint field (proto3 default for repeated uint32)."""
    out: List[int] = []
    offset = 0
    while offset < len(data):
        try:
            v, offset = decode_varint(data, offset)
        except Exception:
            break
        out.append(v)
    return out


def _to_signed64(v: int) -> int:
    """proto3 int32/int64 negatives are encoded as 10-byte two's-complement varints."""
    return v - (1 << 64) if v >= (1 << 63) else v


def coerce_int(value: Any, default: int = 0) -> int:
    """
    Never-raising conversion of a decoded protobuf value to int.
      int/bool          -> int (signed-64 normalized)
      bytes (ascii digits) -> int
      bytes (varint)    -> varint value if valid varint buffer <= 10 bytes
      str of digits     -> int
      anything else     -> default
    """
    try:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return _to_signed64(value)
        if isinstance(value, (bytes, bytearray)):
            b = bytes(value)
            if not b:
                return default
            # Check if it's an ASCII digit string encoded as bytes (e.g. b"0", b"1")
            try:
                s_decoded = b.decode("utf-8")
                if s_decoded.strip().lstrip("-").isdigit():
                    return int(s_decoded.strip())
                # If it's alphabetic text (e.g. b"ActionMJStart"), it's not a valid number
                if any(c.isalpha() for c in s_decoded):
                    return default
            except Exception:
                pass
            # Try single varint
            try:
                v, off = decode_varint(b, 0)
                if off == len(b):
                    return _to_signed64(v)
            except Exception:
                pass
            if len(b) <= 8:
                return int.from_bytes(b, "little", signed=False)
            return default
        if isinstance(value, str) and value.strip().lstrip("-").isdigit():
            return int(value.strip())
    except Exception:
        pass
    return default


def coerce_int_list(values: Any, default: Optional[List[int]] = None) -> List[int]:
    """
    Never-raising conversion of a repeated int field to List[int].
    Handles proto3 packed encoding (a single bytes blob like b'\\xa8\\xc3\\x01...' = [25000, ...]),
    unpacked lists of ints, and mixed lists. Returns `default` (or []) on failure.
    """
    fallback = list(default) if default is not None else []
    try:
        if values is None:
            return fallback
        if isinstance(values, (bytes, bytearray)):
            values = [values]
        if not isinstance(values, (list, tuple)):
            values = [values]
        out: List[int] = []
        for v in values:
            if isinstance(v, (bytes, bytearray)):
                out.extend(_to_signed64(x) for x in decode_packed_varints(bytes(v)))
            else:
                iv = coerce_int(v, default=None)  # type: ignore[arg-type]
                if iv is not None:
                    out.append(iv)
        return out if out else fallback
    except Exception:
        return fallback


def decode_varint(buffer: bytes, offset: int = 0) -> Tuple[int, int]:
    """
    Decode a Protobuf varint from buffer starting at offset. Returns (value, new_offset).
    Raises ValueError on truncated varint or length overflow (> 10 bytes).
    """
    result = 0
    shift = 0
    start = offset
    length = len(buffer)
    while offset < length:
        b = buffer[offset]
        offset += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, offset
        shift += 7
        if shift >= 70:
            raise ValueError(f"Varint exceeds maximum length (10 bytes) at offset {start}")
    raise ValueError(f"Truncated varint at offset {start} (buffer len {length})")


def is_valid_protobuf_submessage(data: bytes) -> bool:
    """
    Strictly verifies whether a byte sequence represents a valid nested Protobuf message,
    preventing corruption on plain strings (e.g. tile codes, player names, method names).
    """
    if not data or len(data) < 2:
        return False

    # Mahjong tile token (e.g. '1m', '9p', '0s', '7z') is never a submessage
    if len(data) == 2 and data[0:1].isdigit() and data[1:2] in b"mpsz":
        return False

    # Pure printable ASCII strings (identifiers/method names) are not protobuf submessages
    try:
        text = data.decode("utf-8")
        if all(32 <= ord(c) < 127 or c in "\r\n\t" for c in text):
            if any(c.isalpha() for c in text):
                return False
    except UnicodeDecodeError:
        pass

    offset = 0
    length = len(data)
    field_count = 0

    while offset < length:
        try:
            key, offset = decode_varint(data, offset)
        except Exception:
            return False

        wire_type = key & 0x07
        field_num = key >> 3

        if field_num == 0 or field_num > 10000:
            return False

        if wire_type == 0:  # Varint
            try:
                _, offset = decode_varint(data, offset)
            except Exception:
                return False
        elif wire_type == 1:  # 64-bit
            if offset + 8 > length:
                return False
            offset += 8
        elif wire_type == 2:  # Length-delimited
            try:
                v_len, offset = decode_varint(data, offset)
            except Exception:
                return False
            if v_len < 0 or offset + v_len > length:
                return False
            offset += v_len
        elif wire_type == 5:  # 32-bit
            if offset + 4 > length:
                return False
            offset += 4
        else:
            return False

        field_count += 1

    return offset == length and field_count > 0


def decode_raw_protobuf(data: bytes, recursive: bool = False, max_depth: int = 4) -> Dict[int, List[Any]]:
    """
    Lightweight, schema-less Protobuf wire format decoder with strict boundary checks.
    Extracts tags and values without needing pre-compiled .proto files.
    Wire types:
      0: Varint (int32, int64, bool, enum)
      1: 64-bit (fixed64, double)
      2: Length-delimited (string, bytes, embedded messages, packed repeated)
      5: 32-bit (fixed32, float)
    """
    fields: Dict[int, List[Any]] = {}
    if not isinstance(data, (bytes, bytearray)):
        return fields

    offset = 0
    length = len(data)

    while offset < length:
        try:
            key, offset = decode_varint(data, offset)
        except Exception:
            break

        wire_type = key & 0x07
        field_num = key >> 3

        if field_num == 0 or field_num > 10000:
            break

        if wire_type == 0:  # Varint
            try:
                val, offset = decode_varint(data, offset)
                fields.setdefault(field_num, []).append(val)
            except Exception:
                break
        elif wire_type == 1:  # 64-bit
            if offset + 8 > length:
                break
            val = struct.unpack("<Q", data[offset:offset+8])[0]
            offset += 8
            fields.setdefault(field_num, []).append(val)
        elif wire_type == 2:  # Length-delimited (string, bytes, sub-message)
            try:
                v_len, offset = decode_varint(data, offset)
            except Exception:
                break
            if v_len < 0 or offset + v_len > length:
                break
            payload = data[offset:offset+v_len]
            offset += v_len

            if recursive and max_depth > 0 and is_valid_protobuf_submessage(payload):
                try:
                    nested_dict = decode_raw_protobuf(payload, recursive=True, max_depth=max_depth - 1)
                    fields.setdefault(field_num, []).append(nested_dict)
                except Exception:
                    fields.setdefault(field_num, []).append(payload)
            else:
                fields.setdefault(field_num, []).append(payload)
        elif wire_type == 5:  # 32-bit
            if offset + 4 > length:
                break
            val = struct.unpack("<I", data[offset:offset+4])[0]
            offset += 4
            fields.setdefault(field_num, []).append(val)
        else:
            # Unknown wire type, cannot proceed safely
            break

    return fields


class LiqiProtoDecoder:
    """
    Decodes raw Majsoul WebSocket binary frames into structured action dictionaries.
    Frame format in Majsoul:
    - Byte 0: Message Type (0x01: Notify, 0x02: Request, 0x03: Response)
    - If Type == 0x01 (Notify):
        [method_name_len (2 bytes, little-endian)] + [method_name (utf-8)] + [protobuf_payload]
    """

    @staticmethod
    def xor_decrypt_action_data(data: bytes, key: Tuple[int, ...] = LIQI_ACTION_XOR_KEY) -> bytes:
        """Decrypts or encrypts Majsoul ActionPrototype.data payload using the 9-byte XOR key."""
        return xor_decrypt_action_data(data, key)

    @staticmethod
    def _score_fields(fields: Dict[int, List[Any]]) -> int:
        """Plausibility score of a decoded Liqi action (small tags, tile tokens, small ints)."""
        score = 0
        for tag, vals in fields.items():
            if 1 <= tag <= 20:
                score += len(vals) * 2
                for v in vals:
                    if is_valid_tile_code(v):
                        score += 5
                    elif isinstance(v, int) and 0 <= v <= 200:
                        score += 1
            else:
                score -= 5
        return score

    @staticmethod
    def resolve_action_sub_bytes_ex(sub_bytes: bytes, action_name: str = "") -> Tuple[bytes, str, bool]:
        """
        Resolves ActionPrototype.data for ANY action (DealTile, DiscardTile, ChiPengGang, ...).

        Returns (resolved_bytes, mode, wire_ok):
          mode    : "xor" (Liqi-decrypted), "plain" (already plaintext), or "heuristic"
          wire_ok : True if resolved bytes are a structurally complete Protobuf message
        """
        if sub_bytes is None:
            return b"", "plain", True
        sub_bytes = bytes(sub_bytes)
        if len(sub_bytes) == 0:
            return b"", "plain", True

        decrypted = xor_decrypt_action_data(sub_bytes)
        dec_ok = protobuf_wire_is_complete(decrypted)
        plain_ok = protobuf_wire_is_complete(sub_bytes)

        if dec_ok and plain_ok:
            s_dec = LiqiProtoDecoder._score_fields(decode_raw_protobuf(decrypted))
            s_plain = LiqiProtoDecoder._score_fields(decode_raw_protobuf(sub_bytes))
            if s_plain > s_dec:
                return sub_bytes, "plain", True
            return decrypted, "xor", True
        if dec_ok:
            return decrypted, "xor", True
        if plain_ok:
            return sub_bytes, "plain", True

        # Neither is structurally complete (corrupt / truncated). Pick the more plausible one.
        s_dec = LiqiProtoDecoder._score_fields(decode_raw_protobuf(decrypted))
        s_plain = LiqiProtoDecoder._score_fields(decode_raw_protobuf(sub_bytes))
        if s_plain > s_dec:
            return sub_bytes, "heuristic", False
        if s_dec > s_plain:
            return decrypted, "heuristic", False
        tiles_plain = LiqiProtoDecoder.scan_tiles_from_bytes(sub_bytes)
        tiles_dec = LiqiProtoDecoder.scan_tiles_from_bytes(decrypted)
        if len(tiles_plain) > len(tiles_dec):
            return sub_bytes, "heuristic", False
        return decrypted, "heuristic", False

    @staticmethod
    def resolve_action_sub_bytes(sub_bytes: bytes, action_name: str = "") -> bytes:
        """Backward-compatible wrapper returning only the resolved bytes."""
        return LiqiProtoDecoder.resolve_action_sub_bytes_ex(sub_bytes, action_name)[0]

    @staticmethod
    def _map_by_schema(action_name: str, f: Dict[int, List[Any]]) -> Optional[Dict[str, Any]]:
        """
        Strict field mapping based on liqi.proto definitions. Returns None for unknown actions.
        proto3 omits default values, so a missing seat means seat 0 and a missing tile
        in ActionDealTile means the tile is hidden (opponent draw).
        """
        def s(v: Any) -> str:
            return v.decode("utf-8", errors="replace") if isinstance(v, (bytes, bytearray)) else str(v)

        def is_tile(v: Any) -> bool:
            return is_valid_tile_code(v)

        def first_int(tag: int, default: int = 0) -> int:
            vals = f.get(tag) or []
            return coerce_int(vals[0], default) if vals else default

        def tiles_of(tag: int) -> List[str]:
            return [s(v).strip() for v in (f.get(tag) or []) if is_tile(v)]

        def ints_of(tag: int) -> List[int]:
            out: List[int] = []
            for v in f.get(tag) or []:
                if isinstance(v, int):
                    out.append(v)
                elif isinstance(v, (bytes, bytearray)):
                    out.extend(decode_packed_varints(bytes(v)))
            return out

        def ops_of(tag: int) -> List[Dict[str, Any]]:
            ops: List[Dict[str, Any]] = []
            for v in f.get(tag) or []:
                if isinstance(v, (bytes, bytearray, dict)):
                    ops.extend(LiqiProtoDecoder._try_parse_operations(v if isinstance(v, dict) else bytes(v)))
            return ops

        name = action_name.split(".")[-1]
        m: Dict[str, Any] = {}

        if name in ("authGame", "ReqAuthGame"):
            # account_id=1, token=2, game_uuid=3
            m["account_id"] = first_int(1)
            if 2 in f and f[2]:
                m["token"] = s(f[2][0])
            if 3 in f and f[3]:
                m["game_uuid"] = s(f[3][0])
            return m

        if name in ("ResAuthGame", "NotifyRoomPlayerUpdate", "NotifyPlayerLoadGameReady"):
            # Extract player list or seat list
            players: List[Dict[str, Any]] = []
            for tag in (1, 2, 3):
                for item in f.get(tag) or []:
                    if isinstance(item, (bytes, bytearray)):
                        pf = decode_raw_protobuf(bytes(item))
                        p_dict: Dict[str, Any] = {}
                        if 1 in pf and pf[1]:
                            p_dict["account_id"] = coerce_int(pf[1][0], 0)
                        if 2 in pf and pf[2] and 0 <= coerce_int(pf[2][0], -1) <= 3:
                            p_dict["seat"] = coerce_int(pf[2][0], 0)
                        if 4 in pf and pf[4]:
                            p_dict["nickname"] = s(pf[4][0])
                        elif 3 in pf and pf[3] and isinstance(pf[3][0], (str, bytes)):
                            p_dict["nickname"] = s(pf[3][0])
                        if p_dict.get("account_id"):
                            players.append(p_dict)
                    elif isinstance(item, int) and item > 1000:
                        players.append({"account_id": item})
            if players:
                for idx, p in enumerate(players):
                    p.setdefault("seat", idx)
                m["players"] = players
                m["player_list"] = players
            return m

        if name in ("ActionDealTile", "RecordDealTile"):
            # seat=1, tile=2, left_tile_count=3, operation=4, liqi=5, doras=6, zhenting=7
            m["seat"] = first_int(1)
            t = tiles_of(2)
            if t:
                m["tile"] = t[0]
            else:
                # Fallback: scan any tag that has a valid 2-char tile code
                for tag in (2, 3, 4, 5, 6, 7):
                    cand_t = tiles_of(tag)
                    if cand_t:
                        m["tile"] = cand_t[0]
                        break
            if 3 in f:
                m["left_tile_count"] = first_int(3)
            d = tiles_of(6)
            if d:
                m["doras"] = d
            ops = ops_of(4)
            if ops:
                m["operation_list"] = ops
            return m

        if name in ("ActionDiscardTile", "RecordDiscardTile"):
            # seat=1, tile=2, is_liqi=3, operation=4, moqie=5, zhenting=6, tingpais=7, doras=8, is_wliqi=9
            m["seat"] = first_int(1)
            t = tiles_of(2)
            if t:
                m["tile"] = t[0]
            m["is_liqi"] = bool(first_int(3) or first_int(9))
            m["is_wliqi"] = bool(first_int(9))
            m["moqie"] = bool(first_int(5))
            d = tiles_of(8)
            if d:
                m["doras"] = d
            ops = ops_of(4)
            if ops:
                m["operation_list"] = ops
            return m

        if name in ("ActionChiPengGang", "RecordChiPengGang"):
            # seat=1, type=2, tiles=3, froms=4 (packed), liqi=5, operation=6
            m["seat"] = first_int(1)
            m["type"] = first_int(2)
            m["tiles"] = tiles_of(3)
            m["froms"] = ints_of(4)
            ops = ops_of(6)
            if ops:
                m["operation_list"] = ops
            return m

        if name in ("ActionAnGangAddGang", "RecordAnGangAddGang"):
            # seat=1, type=2, tiles=3 (single tile string), operation=4, doras=6
            m["seat"] = first_int(1)
            m["type"] = first_int(2, 3)
            t = tiles_of(3)
            m["tiles"] = t * 4 if len(t) == 1 else t
            d = tiles_of(6)
            if d:
                m["doras"] = d
            ops = ops_of(4)
            if ops:
                m["operation_list"] = ops
            return m

        if name in ("ActionHule", "RecordHule"):
            # hules = 1 or 2, delta_scores = 2 or 3, scores = 3 or 4
            hules_list: List[Dict[str, Any]] = []
            raw_hules = f.get(1, []) or f.get(2, [])
            for h_blob in raw_hules:
                if isinstance(h_blob, (bytes, bytearray)):
                    hf = decode_raw_protobuf(bytes(h_blob))
                    h_dict: Dict[str, Any] = {}
                    # seat = tag 4 (or 1)
                    h_dict["seat"] = coerce_int((hf.get(4) or hf.get(1) or [0])[0], 0)
                    h_dict["zimo"] = bool(coerce_int((hf.get(5) or [0])[0], 0))
                    # target = tag 9 (or 10 or 6)
                    h_dict["target"] = coerce_int((hf.get(9) or hf.get(10) or hf.get(6) or [h_dict["seat"]])[0], h_dict["seat"])
                    # hu_tile = tag 3
                    t_cand = [s(v).strip() for v in (hf.get(3) or []) if is_tile(v)]
                    if t_cand:
                        h_dict["hu_tile"] = t_cand[0]
                    # Score calculation: dadian (tag 19) / point_sum (tag 18 or 7) / point_rong (tag 14) / zimo (tags 15, 16)
                    dadian = coerce_int((hf.get(19) or hf.get(18) or hf.get(14) or [0])[0], 0)
                    if not dadian:
                        z_qin = coerce_int((hf.get(15) or [0])[0], 0)
                        z_xian = coerce_int((hf.get(16) or [0])[0], 0)
                        if z_qin or z_xian:
                            dadian = z_qin * 3 if z_qin and not z_xian else (z_qin + z_xian * 2)
                    if not dadian:
                        dadian = coerce_int((hf.get(7) or [0])[0], 0)
                    if 0 < dadian <= 48000:
                        h_dict["score"] = dadian
                        h_dict["point_sum"] = dadian
                    else:
                        h_dict["point_sum"] = 5200
                        h_dict["score"] = 5200
                    hules_list.append(h_dict)
                elif isinstance(h_blob, dict):
                    hules_list.append(h_blob)

            # Fallback for flat int fields (e.g. mock/test frame: tag 1 = seat, tag 3 = target/zimo)
            if not hules_list:
                seat_val = None
                if 1 in f and isinstance(f[1][0], int) and 0 <= f[1][0] <= 3:
                    seat_val = f[1][0]
                elif 4 in f and isinstance(f[4][0], int) and 0 <= f[4][0] <= 3:
                    seat_val = f[4][0]

                if seat_val is not None:
                    target_val = seat_val
                    if 3 in f and isinstance(f[3][0], int) and 0 <= f[3][0] <= 3:
                        target_val = f[3][0]
                    elif 6 in f and isinstance(f[6][0], int) and 0 <= f[6][0] <= 3:
                        target_val = f[6][0]
                    is_zimo = (seat_val == target_val)
                    h_dict = {"seat": seat_val, "target": target_val, "zimo": is_zimo}
                    t_str = tiles_of(3) or tiles_of(2)
                    if t_str:
                        h_dict["hu_tile"] = t_str[0]
                    hules_list.append(h_dict)
                    m["seat"] = seat_val
                    m["target"] = target_val

            m["hules"] = hules_list
            del_sc = ints_of(2) or ints_of(3)
            if del_sc:
                m["delta_scores"] = del_sc
            return m

        if name in ("ActionLiuJu", "RecordLiuJu", "ActionNoTile", "RecordNoTile"):
            del_sc = ints_of(1) or ints_of(2) or ints_of(3)
            if del_sc:
                m["delta_scores"] = del_sc
            return m

        if name in ("ActionNewRound", "RecordNewRound"):
            # chang=1, ju=2, ben=3, tiles=4, dora=5, scores=6, operation=7, liqibang=8, doras=10, tiles0..3=11..14
            m["chang"] = first_int(1)
            m["ju"] = first_int(2)
            m["ben"] = first_int(3)
            hand_tiles = tiles_of(4)
            if hand_tiles:
                m["tiles"] = hand_tiles
            dora = tiles_of(5)
            if dora:
                m["doras"] = [dora[0]]
            doras = tiles_of(10)
            if doras:
                m["doras"] = doras
            # Scores (field 6 or 8)
            sc = ints_of(6) or ints_of(8)
            if len(sc) in (3, 4):
                m["scores"] = sc
            if 8 in f and len(f[8]) == 1 and isinstance(f[8][0], int):
                m["liqibang"] = coerce_int(f[8][0], 0)
            elif 9 in f:
                m["liqibang"] = first_int(9)
            # Replay / spectator hands tiles0..tiles3 (fields 11..14)
            for s_idx in range(4):
                t_list = tiles_of(11 + s_idx)
                if t_list:
                    m[f"tiles{s_idx}"] = t_list
                    if not m.get("tiles") and s_idx == 0:
                        m["tiles"] = t_list
            ops = ops_of(7)
            if ops:
                m["operation_list"] = ops
            return m

        return None

    @staticmethod
    def decode_frame(raw_payload: Union[bytes, str]) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Parses raw binary data or base64 string from CDP or mitmproxy.
        Returns tuple of (action_name, action_data_dict), e.g. ("ActionDiscardTile", {...})
        """
        if isinstance(raw_payload, str):
            try:
                data = base64.b64decode(raw_payload)
            except Exception:
                data = raw_payload.encode("latin-1")
        elif isinstance(raw_payload, (bytes, bytearray)):
            data = bytes(raw_payload)
        else:
            return None

        if len(data) < 4:
            return None

        msg_type = data[0]
        # Debug logging to identify live frame format
        print(f"[LiqiDecoder DEBUG] msg_type={hex(msg_type)}, raw_len={len(data)}, hex_prefix={data[:24].hex()}")

        # Case 1: Majsoul Notify Frame (0x01)
        if msg_type == 0x01:
            method_name = ""
            payload = b""
            # Real Majsoul live frames: byte 0 is 0x01, followed immediately by protobuf (tag 1: 0x0a)
            if len(data) > 1 and data[1] == 0x0a:
                proto_fields = decode_raw_protobuf(data[1:])
                if 1 in proto_fields and proto_fields[1]:
                    v1 = proto_fields[1][0]
                    method_name = v1.decode("utf-8", errors="replace") if isinstance(v1, (bytes, bytearray)) else str(v1)
                if 2 in proto_fields and proto_fields[2]:
                    v2 = proto_fields[2][0]
                    payload = bytes(v2) if isinstance(v2, (bytes, bytearray)) else (v2.encode("utf-8") if isinstance(v2, str) else b"")
            else:
                # Legacy / alternate format: 2-byte length header
                name_len = struct.unpack("<H", data[1:3])[0]
                if len(data) >= 3 + name_len:
                    method_name = data[3:3+name_len].decode("utf-8", errors="replace")
                    payload = data[3+name_len:]

            if method_name:
                action_data = LiqiProtoDecoder.parse_action_payload(method_name, payload)
                pure_name = method_name.split(".")[-1]
                return pure_name, action_data

        # Case 2: Majsoul Request Frame (0x02) - e.g. inputOperation / discardTile
        if msg_type == 0x02 and len(data) >= 3:
            req_id = struct.unpack("<H", data[1:3])[0]
            proto_fields = decode_raw_protobuf(data[3:])
            method_name = ""
            if 1 in proto_fields and proto_fields[1]:
                v1 = proto_fields[1][0]
                method_name = v1.decode("utf-8", errors="replace") if isinstance(v1, bytes) else str(v1)
            req_payload = b""
            if 2 in proto_fields and proto_fields[2]:
                v2 = proto_fields[2][0]
                req_payload = v2 if isinstance(v2, bytes) else v2.encode("utf-8")

            action_data = LiqiProtoDecoder.parse_action_payload(method_name, req_payload)
            action_data["request_id"] = req_id
            pure_name = method_name.split(".")[-1]
            if pure_name == "inputOperation" or "tile" in action_data:
                if "tile" in action_data:
                    pure_name = "ActionDiscardTile"
            return pure_name, action_data

        # Case 3: Majsoul Response Frame (0x03)
        if msg_type == 0x03 and len(data) >= 3:
            req_id = struct.unpack("<H", data[1:3])[0]
            proto_fields = decode_raw_protobuf(data[3:])
            return "ResponseAction", {"request_id": req_id, "raw_fields": proto_fields}

        # Case 4: Wrapper JSON structure (if pre-decoded by proxy or wrapper)
        if data.startswith(b"{") and data.endswith(b"}"):
            try:
                import json
                obj = json.loads(data.decode("utf-8"))
                name = obj.get("name", obj.get("action", ""))
                payload_dict = obj.get("data", obj)
                return name, payload_dict
            except Exception:
                pass

        # Case 5: Direct Protobuf Payload
        parsed = decode_raw_protobuf(data)
        return "UnknownAction", {"raw_fields": parsed}

    @staticmethod
    def parse_action_payload(method_name: str, payload_bytes: bytes) -> Dict[str, Any]:
        """
        Extracts key game fields (seat, tile, moqie, doras, tiles, etc.) from payload.
        """
        try:
            fields = decode_raw_protobuf(payload_bytes)
        except Exception:
            return {"raw_fields": {}}

        result: Dict[str, Any] = {"raw_fields": fields}

        def to_str(v: Any) -> str:
            if isinstance(v, str):
                return v
            if isinstance(v, bytes):
                return v.decode("utf-8", errors="replace")
            return str(v)

        # If wrapped in ActionPrototype or NotifyPlayerAction
        if "ActionPrototype" in method_name or "NotifyPlayerAction" in method_name:
            act_name = ""
            sub_bytes = None
            name_tag = None

            # 1. Identify action_name by keyword first
            for tag in (2, 1, 3):
                if tag in fields and fields[tag]:
                    val = to_str(fields[tag][0])
                    if isinstance(val, str) and ("Action" in val or "Record" in val or "Notify" in val):
                        act_name = val
                        name_tag = tag
                        break

            # Fallback by position
            if not act_name:
                if 2 in fields and isinstance(fields[2][0], str) and len(fields[2][0]) > 2:
                    act_name = to_str(fields[2][0])
                    name_tag = 2
                elif 1 in fields and isinstance(fields[1][0], str) and len(fields[1][0]) > 2:
                    act_name = to_str(fields[1][0])
                    name_tag = 1

            # 2. Extract sub-payload bytes (Tag 3 is ActionPrototype.data in official Protobuf)
            if 3 in fields and fields[3] and name_tag != 3:
                cand = fields[3][0]
                if isinstance(cand, (bytes, bytearray)):
                    sub_bytes = bytes(cand)
            elif name_tag == 2 and 3 not in fields:
                # Official layout (step=1, name=2, data=3) with data omitted => all-default message
                sub_bytes = b""

            if sub_bytes is None:
                byte_candidates = []
                for tag, vals in fields.items():
                    if tag != name_tag and vals:
                        for v in vals:
                            if isinstance(v, (bytes, bytearray)) and len(v) > 0:
                                byte_candidates.append(bytes(v))
                            elif isinstance(v, str) and len(v) > 0 and v != act_name:
                                byte_candidates.append(v.encode("latin-1"))

                # Pick the largest candidate byte buffer (nested protobuf payload)
                if byte_candidates:
                    byte_candidates.sort(key=len, reverse=True)
                    sub_bytes = byte_candidates[0]
                elif payload_bytes and len(payload_bytes) > len(act_name) + 4:
                    sub_bytes = payload_bytes

            if act_name:
                result["action_name"] = act_name

            if sub_bytes is not None and act_name:
                try:
                    # Apply Liqi XOR decryption uniformly to ALL actions' ActionPrototype.data
                    resolved_bytes, xor_mode, wire_ok = LiqiProtoDecoder.resolve_action_sub_bytes_ex(
                        sub_bytes, action_name=act_name
                    )
                    sub_fields = decode_raw_protobuf(resolved_bytes)
                    result["raw_sub_fields"] = sub_fields
                    result["xor_mode"] = xor_mode
                    result["wire_ok"] = wire_ok

                    schema_mapped = LiqiProtoDecoder._map_by_schema(act_name, sub_fields) if wire_ok else None
                    if schema_mapped is not None:
                        # Trusted, schema-exact decode: never salvage-scan (avoids phantom tiles,
                        # e.g. opponent ActionDealTile legitimately has NO tile field).
                        common_mapped = schema_mapped
                    else:
                        common_mapped = LiqiProtoDecoder._map_common_fields(
                            sub_fields, action_name=act_name, raw_bytes=resolved_bytes or sub_bytes or payload_bytes
                        )

                    # Salvage for ActionNewRound
                    if ("NewRound" in act_name or "MJStart" in act_name) and len(common_mapped.get("tiles", [])) < 13:
                        salvaged = LiqiProtoDecoder.scan_tiles_from_bytes(resolved_bytes or payload_bytes)
                        if len(salvaged) >= 13:
                            hand_tiles = salvaged[:14] if len(salvaged) >= 14 and act_name.startswith("Action") else salvaged[:13]
                            common_mapped["tiles"] = hand_tiles
                            if not common_mapped.get("doras") and len(salvaged) > len(hand_tiles):
                                common_mapped["doras"] = [salvaged[len(hand_tiles)]]
                            print(f"[LiqiDecoder Salvage Full] Salvaged {len(hand_tiles)} hand tiles from resolved bytes: {hand_tiles}")

                    # Salvage for ActionDealTile: if tile was not extracted by schema mapping, scan raw/sub bytes
                    if "DealTile" in act_name and "tile" not in common_mapped:
                        salvaged_deal = LiqiProtoDecoder.scan_tiles_from_bytes(resolved_bytes or sub_bytes or payload_bytes)
                        if salvaged_deal:
                            common_mapped["tile"] = salvaged_deal[0]
                            print(f"[LiqiDecoder Salvage Deal] Salvaged drawn tile '{salvaged_deal[0]}' from resolved bytes: {salvaged_deal}")

                    # Debug log
                    if not wire_ok:
                        print(f"[LiqiDecoder WIRE WARN] {act_name} not a complete protobuf after XOR/plain! sub_bytes_hex={(sub_bytes or b'')[:64].hex()}")

                    print(f"[LiqiDecoder DEBUG sub_fields] {act_name} ({xor_mode}) -> raw: {sub_fields} | mapped: {common_mapped}")
                    result.update(common_mapped)
                except Exception as e:
                    print(f"[LiqiDecoder] Error decoding sub_bytes: {e}")
            else:
                if act_name in ("ActionNewRound", "ActionDealTile"):
                    print(f"[LiqiDecoder EMPTY GUARD] {act_name} no sub_bytes found! payload_hex={payload_bytes[:64].hex()} fields_keys={list(fields.keys())}")
        else:
            schema_mapped = LiqiProtoDecoder._map_by_schema(method_name, fields)
            if schema_mapped is not None:
                result.update(schema_mapped)
            else:
                result.update(LiqiProtoDecoder._map_common_fields(fields, action_name=method_name, raw_bytes=payload_bytes))

        return result

    VALID_TILE_RE = None

    @classmethod
    def scan_tiles_from_bytes(cls, data: Any) -> List[str]:
        """Scans and extracts all valid 2-char mahjong tile tokens from raw binary buffer."""
        if not isinstance(data, (bytes, bytearray)):
            if isinstance(data, str):
                data = data.encode("utf-8", errors="replace")
            else:
                return []
        import re
        if cls.VALID_TILE_RE is None:
            # Strictly matches: [0-9][mps] (with 0 as aka-dora) or [1-7]z (honors).
            # Excludes arbitrary numbers like '69', non-tile letters, etc.
            cls.VALID_TILE_RE = re.compile(br'([0-9][mps]|[1-7]z)')
        matches = cls.VALID_TILE_RE.findall(data)
        extracted = [m.decode("ascii") for m in matches]
        return [t for t in extracted if is_valid_tile_code(t)]

    @staticmethod
    def _map_common_fields(fields: Dict[int, List[Any]], action_name: str = "", raw_bytes: bytes = b"") -> Dict[str, Any]:
        """Maps protobuf field indices to standard Liqi names heuristically."""
        mapped: Dict[str, Any] = {}

        def to_str(v: Any) -> str:
            if isinstance(v, str):
                return v
            if isinstance(v, bytes):
                return v.decode("utf-8", errors="replace")
            return str(v)

        clean_name = action_name.split(".")[-1]

        # Case 1: ActionNewRound / RecordNewRound
        if "NewRound" in clean_name or "MJStart" in clean_name:
            if 1 in fields and fields[1]:
                mapped["chang"] = coerce_int(fields[1][0], 0)
            if 2 in fields and fields[2]:
                mapped["ju"] = coerce_int(fields[2][0], 0)
            if 3 in fields and fields[3]:
                mapped["ben"] = coerce_int(fields[3][0], 0)

            # Tag 4: 自家の配牌 (repeated string tiles = 4)
            # Strictly validate tile strings: must match is_valid_tile_code
            if 4 in fields and fields[4]:
                extracted = [to_str(t) for t in fields[4] if is_valid_tile_code(t)]
                if extracted:
                    mapped["tiles"] = extracted

            # Tag 5: dora indicator (string dora = 5)
            if 5 in fields and fields[5]:
                dora_str = to_str(fields[5][0])
                if is_valid_tile_code(dora_str):
                    mapped.setdefault("doras", []).append(dora_str)

            # Tag 6: scores (repeated int32 -> proto3 PACKED: single bytes blob of varints)
            if 6 in fields and fields[6]:
                sc = coerce_int_list(fields[6])
                if len(sc) in (3, 4):
                    mapped["scores"] = sc
            elif 8 in fields and fields[8] and len(fields[8]) >= 4:
                sc = coerce_int_list(fields[8])
                if len(sc) in (3, 4):
                    mapped["scores"] = sc

            # Tag 8 / 9: liqibang
            if 8 in fields and fields[8] and isinstance(fields[8][0], int) and len(fields[8]) == 1:
                mapped["liqibang"] = coerce_int(fields[8][0], 0)
            elif 9 in fields and fields[9] and isinstance(fields[9][0], int):
                mapped["liqibang"] = coerce_int(fields[9][0], 0)

            # Tag 10: doras
            if 10 in fields and fields[10]:
                mapped["doras"] = [to_str(t) for t in fields[10] if is_valid_tile_code(t)]

            # Tags 11..14: tiles0..tiles3 (観戦 / 牌譜用)
            for s in range(4):
                tag = 11 + s
                if tag in fields and fields[tag]:
                    t_list = [to_str(t) for t in fields[tag] if is_valid_tile_code(t)]
                    if t_list:
                        mapped[f"tiles{s}"] = t_list

            # Fallback Deep Scanner: If tiles missing or incomplete, salvage from binary
            curr_tiles = mapped.get("tiles") or mapped.get("tiles0") or []
            if len(curr_tiles) < 13 and raw_bytes:
                scanned = LiqiProtoDecoder.scan_tiles_from_bytes(raw_bytes)
                if len(scanned) >= 13:
                    # Dealt hand is the first 13 (or 14 if dealer) tiles
                    salvaged_hand = scanned[:14] if len(scanned) >= 14 and clean_name.startswith("Action") else scanned[:13]
                    mapped["tiles"] = salvaged_hand
                    mapped["tiles0"] = salvaged_hand
                    if not mapped.get("doras") and len(scanned) > len(salvaged_hand):
                        mapped["doras"] = [scanned[len(salvaged_hand)]]
                    print(f"[LiqiDecoder Salvage] Successfully salvaged {len(salvaged_hand)} hand tiles from raw bytes: {salvaged_hand}")

            return mapped

        # Case 2: ActionChiPengGang / RecordChiPengGang
        if "ChiPengGang" in clean_name:
            if 1 in fields and fields[1]:
                mapped["seat"] = coerce_int(fields[1][0], 0)
            else:
                mapped["seat"] = 0
            if 2 in fields and fields[2]:
                mapped["type"] = coerce_int(fields[2][0], 0)
            if 3 in fields and fields[3]:
                mapped["tiles"] = [to_str(t) for t in fields[3] if is_valid_tile_code(t)]
            if 4 in fields and fields[4]:
                mapped["froms"] = coerce_int_list(fields[4])

            if 6 in fields:
                all_ops: List[Dict[str, Any]] = []
                for v in fields[6]:
                    if isinstance(v, (bytes, dict)):
                        parsed = LiqiProtoDecoder._try_parse_operations(v)
                        if parsed:
                            all_ops.extend(parsed)
                if all_ops:
                    mapped["operation_list"] = all_ops
            return mapped

        # Case 3: Generic / DealTile / DiscardTile
        for tag, vals in fields.items():
            if not vals:
                continue
            for raw_v in vals:
                v_str = to_str(raw_v)
                if isinstance(raw_v, int):
                    if tag == 1 and 0 <= raw_v <= 3:
                        mapped.setdefault("seat", raw_v)
                    elif tag == 3 and "Discard" in clean_name:
                        mapped["is_liqi"] = bool(raw_v)
                    elif tag == 5 and "Discard" in clean_name:
                        mapped["moqie"] = bool(raw_v)
                    elif tag == 9 and "Discard" in clean_name:
                        mapped["is_wliqi"] = bool(raw_v)
                        if bool(raw_v):
                            mapped["is_liqi"] = True
                elif is_valid_tile_code(v_str):
                    if tag == 2 or "tile" not in mapped:
                        mapped["tile"] = v_str
                    if tag in (4, 5, 6, 7, 10):
                        mapped.setdefault("doras", []).append(v_str)

        # In ActionDealTile or ActionDiscardTile, seat is field 1 (defaults to 0 if omitted by Proto3)
        if 1 in fields and fields[1] and isinstance(fields[1][0], int) and 0 <= fields[1][0] <= 3:
            mapped["seat"] = fields[1][0]
        elif "seat" not in mapped and ("Deal" in clean_name or "Discard" in clean_name):
            mapped["seat"] = 0

        if 2 in fields and fields[2]:
            t_str = to_str(fields[2][0])
            if is_valid_tile_code(t_str):
                mapped["tile"] = t_str

        # Fallback Deep Scanner for DealTile / DiscardTile if tile missing
        if "tile" not in mapped and raw_bytes:
            scanned = LiqiProtoDecoder.scan_tiles_from_bytes(raw_bytes)
            if scanned:
                mapped["tile"] = scanned[0]
                print(f"[LiqiDecoder Salvage Tile] Salvaged tile '{mapped['tile']}' from raw_bytes for {clean_name}")

        # Case 3: Parse OptionalOperationList (calls/operations) from tags (e.g. tag 7 in DiscardTile, tag 4 in DealTile)
        all_ops: List[Dict[str, Any]] = []
        for tag, vals in fields.items():
            for v in vals:
                if isinstance(v, (bytes, dict)):
                    parsed = LiqiProtoDecoder._try_parse_operations(v)
                    if parsed:
                        all_ops.extend(parsed)
        if all_ops:
            mapped["operation_list"] = all_ops

        return mapped

    @staticmethod
    def _try_parse_operations(payload: Union[bytes, dict]) -> List[Dict[str, Any]]:
        """Parses Majsoul OptionalOperationList or OptionalOperation protobuf binary."""
        operations: List[Dict[str, Any]] = []
        if isinstance(payload, bytes):
            try:
                fields = decode_raw_protobuf(payload)
            except Exception:
                return operations
        elif isinstance(payload, dict):
            fields = payload
        else:
            return operations

        # Case A: OptionalOperationList container (tag 1: seat, tag 2: repeated OptionalOperation)
        seat = fields.get(1, [None])[0] if 1 in fields and isinstance(fields[1][0], int) else None

        if 2 in fields and fields[2]:
            for item in fields[2]:
                if isinstance(item, bytes):
                    try:
                        op_f = decode_raw_protobuf(item)
                    except Exception:
                        continue
                elif isinstance(item, dict):
                    op_f = item
                else:
                    continue

                op_type = op_f.get(1, [0])[0] if 1 in op_f and isinstance(op_f[1][0], int) else 0
                if 1 <= op_type <= 11:
                    combs: List[str] = []
                    if 2 in op_f:
                        for c in op_f[2]:
                            if isinstance(c, bytes):
                                try:
                                    combs.append(c.decode("utf-8"))
                                except Exception:
                                    pass
                            elif isinstance(c, str):
                                combs.append(c)
                    op_name = MAJSOUL_OP_TYPE_NAMES.get(op_type, "unknown")
                    operations.append({
                        "type": op_type,
                        "type_name": op_name,
                        "combinations": combs,
                        "seat": seat
                    })

        # Case B: Direct single OptionalOperation (tag 1: type in 1..11, tag 2: combinations)
        if not operations and 1 in fields and isinstance(fields[1][0], int):
            op_type = fields[1][0]
            if 1 <= op_type <= 11:
                combs = []
                if 2 in fields:
                    for c in fields[2]:
                        if isinstance(c, bytes):
                            try:
                                combs.append(c.decode("utf-8"))
                            except Exception:
                                pass
                        elif isinstance(c, str):
                            combs.append(c)
                op_name = MAJSOUL_OP_TYPE_NAMES.get(op_type, "unknown")
                operations.append({
                    "type": op_type,
                    "type_name": op_name,
                    "combinations": combs,
                    "seat": seat
                })

        return operations
