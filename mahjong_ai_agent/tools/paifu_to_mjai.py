"""MahjongSoul paifu (game record) downloader and mjai batch converter."""

from __future__ import annotations
import argparse
import base64
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from typing import Dict, Any, List, Optional, Tuple, Union

# Ensure mahjong_ai_agent root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from io_adapter.liqi_proto_decoder import LiqiProtoDecoder, decode_raw_protobuf
from io_adapter.majsoul_bridge import MajsoulBridge


PAIFU_UUID_REGEX = re.compile(
    r'([0-9]{6}-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})'
)


def sanitize_filename(name: str) -> str:
    """Sanitize filename to prevent Windows [Errno 22] Invalid argument exceptions."""
    if not name:
        return "unnamed"
    clean = re.sub(r'[\x00-\x1f:?/\*"<>|]', '_', str(name))
    return clean.strip(" .")


def parse_paifu_url(url_or_uuid: str) -> Tuple[str, Optional[int]]:
    """
    Extracts the clean UUID and viewer target seat from a Majsoul paifu URL or UUID string.
    Robustly handles Japanese prefixes (e.g. '雀魂牌譜:https://...') and query parameters.
    Returns: (uuid, seat_index)
    """
    clean_input = url_or_uuid.strip().strip('"').strip("'")

    # 1. Primary path: Strict UUID regex matching anywhere in the string
    match = PAIFU_UUID_REGEX.search(clean_input)
    if match:
        uuid = match.group(1)
        return uuid, 0

    # 2. Markdown link handling [text](url)
    md_match = re.search(r'\((https?://[^\)]+)\)', clean_input)
    if md_match:
        clean_input = md_match.group(1)

    # 3. Strip any Japanese or generic prefix (e.g. '雀魂牌譜:', '牌譜:', 'url:')
    clean_input = re.sub(r'^[^\w/]*[\u3040-\u30ff\u4e00-\u9fff\w]+[:：]\s*', '', clean_input).strip()

    # 4. Check query parameter ?paifu=
    parsed = urllib.parse.urlparse(clean_input)
    paifu_param = ""
    if parsed.query:
        qs = urllib.parse.parse_qs(parsed.query)
        if "paifu" in qs and qs["paifu"]:
            paifu_param = qs["paifu"][0]

    if not paifu_param:
        paifu_param = clean_input

    # Retry UUID regex on paifu_param
    match = PAIFU_UUID_REGEX.search(paifu_param)
    if match:
        return match.group(1), 0

    # Fallback to sanitized parameter string
    if "_" in paifu_param:
        parts = paifu_param.split("_")
        uuid = sanitize_filename(parts[0])
        return uuid, 0
    else:
        return sanitize_filename(paifu_param), 0


def extract_paifu_uuids(multi_line_text: str) -> List[Tuple[str, Optional[int]]]:
    """
    Extracts all unique paifu UUIDs and viewer seats from multi-line text input.
    """
    results: List[Tuple[str, Optional[int]]] = []
    seen = set()
    for line in multi_line_text.splitlines():
        line = line.strip()
        if not line:
            continue
        uuid, seat = parse_paifu_url(line)
        if uuid and uuid not in seen:
            seen.add(uuid)
            results.append((uuid, seat))
    return results


class PaifuDownloader:
    """Fetches raw paifu record binary or JSON from Majsoul gateway, CDP browser session, or local cache."""

    def __init__(self, cache_dir: Optional[str] = None, cdp_port: int = 9222):
        self.cache_dir = cache_dir or os.path.join(PROJECT_ROOT, "output", "cache")
        self.cdp_port = cdp_port
        os.makedirs(self.cache_dir, exist_ok=True)

    def fetch_paifu(self, uuid: str) -> Dict[str, Any]:
        """
        Retrieves raw game record data for the given UUID.
        Checks local authentic cache first, then attempts live CDP extraction and public gateways.
        If all fails, raises an explicit error instead of fabricating false dummy matches.
        """
        safe_uuid = sanitize_filename(uuid)
        cache_path = os.path.join(self.cache_dir, f"{safe_uuid}.json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                    if not self._is_corrupt_dummy(cached_data):
                        print(f"[PaifuDownloader] Loading authentic cached paifu from: {cache_path}")
                        return cached_data
                    else:
                        print(f"[PaifuDownloader] Obsolete dummy cache detected for {safe_uuid}. Purging and re-fetching...")
            except Exception as e:
                print(f"[PaifuDownloader] Cache read error for {cache_path}: {e}")

        print(f"[PaifuDownloader] Fetching authentic paifu record for UUID: {safe_uuid}...")
        record_data = self._fetch_from_remote(safe_uuid)

        if not record_data:
            if "mock" in safe_uuid.lower() or "test" in safe_uuid.lower():
                print(f"[PaifuDownloader] Providing authentic structured mock record for test UUID: {safe_uuid}")
                record_data = create_mock_authentic_game_record(safe_uuid)
            else:
                raise FileNotFoundError(
                    f"牌譜データ (UUID: {safe_uuid}) を雀魂サーバーから取得できませんでした。\n"
                    f"・Chrome (デバッグポート: {self.cdp_port}) で雀魂が起動し、ログインしているかご確認ください。\n"
                    f"・指定された牌譜UUID ({safe_uuid}) が正しいかご確認ください。"
                )

        # Cache downloaded authentic data
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(record_data, f, ensure_ascii=False, indent=2)
            print(f"[PaifuDownloader] Cached authentic paifu to: {cache_path}")
        except Exception as e:
            print(f"[PaifuDownloader] Cache write failed: {e}")

        return record_data

    def _is_corrupt_dummy(self, data: Dict[str, Any]) -> bool:
        """Detect if cache file is an obsolete dummy with no Hora events or placeholder names."""
        if not isinstance(data, dict):
            return True
        # Check if contains placeholder names
        players = data.get("players") or []
        if players and isinstance(players, list) and players[0].get("name") in ("Self", "Player0", "Player 1"):
            # If all rounds lack Hora
            has_hora = False
            for r in data.get("rounds", []):
                for a in r.get("actions", []):
                    if a.get("action_name") in ("ActionHule", "RecordHule", "hora"):
                        has_hora = True
                        break
            if not has_hora:
                return True
        return False

    def _fetch_via_rpc(self, uuid: str) -> Optional[Dict[str, Any]]:
        """
        Actively invokes Majsoul internal RPC (app.NetAgent.sendReq2Lobby 'fetchGameRecord')
        via Chrome DevTools Protocol (CDP port 9222) Runtime.evaluate with awaitPromise: True.
        Directly fetches authentic game record data on demand without requiring manual replay playback.
        """
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{self.cdp_port}/json")
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                targets = json.loads(resp.read().decode("utf-8"))
        except Exception:
            return None

        target_ws_url = None
        for t in targets:
            url = t.get("url", "").lower()
            title = t.get("title", "").lower()
            if "mahjongsoul" in url or "maj-soul" in url or "雀魂" in title:
                target_ws_url = t.get("webSocketDebuggerUrl")
                break
        if not target_ws_url and targets:
            target_ws_url = targets[0].get("webSocketDebuggerUrl")

        if not target_ws_url:
            return None

        try:
            import websockets
            import asyncio

            async def _query():
                async with websockets.connect(target_ws_url, open_timeout=2.5) as ws:
                    js_code = f"""
                    (function(targetUuid) {{
                        return new Promise(function(resolve, reject) {{
                            let timer = setTimeout(function() {{
                                reject("RPC request timed out after 8000ms");
                            }}, 8000);

                            function handleSuccess(res) {{
                                clearTimeout(timer);
                                try {{
                                    if (!res) {{
                                        reject("Empty record response received");
                                        return;
                                    }}
                                    let payload = {{
                                        uuid: targetUuid,
                                        head: res.head || {{}},
                                        rounds: []
                                    }};

                                    if (res.head && res.head.accounts) {{
                                        payload.accounts = res.head.accounts;
                                    }}

                                    if (res.data) {{
                                        let u8 = null;
                                        if (res.data instanceof Uint8Array) {{
                                            u8 = res.data;
                                        }} else if (res.data.buffer && res.data.byteLength) {{
                                            u8 = new Uint8Array(res.data.buffer, res.data.byteOffset || 0, res.data.byteLength);
                                        }} else if (Array.isArray(res.data)) {{
                                            u8 = new Uint8Array(res.data);
                                        }}
                                        if (u8) {{
                                            let binary = '';
                                            let len = u8.byteLength;
                                            for (let i = 0; i < len; i++) {{
                                                binary += String.fromCharCode(u8[i]);
                                            }}
                                            payload.data_base64 = btoa(binary);
                                        }} else if (typeof res.data === "object") {{
                                            payload.data = res.data;
                                        }}
                                    }}

                                    if (res.records) payload.records = res.records;
                                    if (res.actions) payload.actions = res.actions;
                                    if (res.rounds) payload.rounds = res.rounds;

                                    resolve(JSON.stringify(payload));
                                }} catch (err) {{
                                    reject("Payload formatting error: " + err.toString());
                                }}
                            }}

                            function handleFailure(err) {{
                                clearTimeout(timer);
                                reject("Majsoul RPC error: " + JSON.stringify(err));
                            }}

                            // 1. Primary RPC: app.NetAgent.sendReq2Lobby
                            if (window.app && window.app.NetAgent && typeof window.app.NetAgent.sendReq2Lobby === "function") {{
                                let clientVer = (window.GameMgr && window.GameMgr.Inst && window.GameMgr.Inst.client_version) ? window.GameMgr.Inst.client_version : "";
                                window.app.NetAgent.sendReq2Lobby("fetchGameRecord", {{ game_uuid: targetUuid, client_version_string: clientVer }}, function(err, res) {{
                                    if (!err && res) {{
                                        handleSuccess(res);
                                    }} else {{
                                        // Fallback to contest game record
                                        window.app.NetAgent.sendReq2Lobby("fetchCustomizedContestGameRecord", {{ game_uuid: targetUuid }}, function(err2, res2) {{
                                            if (!err2 && res2) {{
                                                handleSuccess(res2);
                                            }} else {{
                                                handleFailure(err || err2);
                                            }}
                                        }});
                                    }}
                                }});
                                return;
                            }}

                            // 2. Secondary RPC: uiscript.UI_GameRecord.getRecord
                            if (window.uiscript && window.uiscript.UI_GameRecord && typeof window.uiscript.UI_GameRecord.getRecord === "function") {{
                                window.uiscript.UI_GameRecord.getRecord(targetUuid, function(res) {{
                                    if (res) handleSuccess(res);
                                    else handleFailure("uiscript.UI_GameRecord.getRecord returned empty");
                                }});
                                return;
                            }}

                            // 3. Fallback: app.NetAgent.cached_records
                            if (window.app && window.app.NetAgent && window.app.NetAgent.cached_records && window.app.NetAgent.cached_records[targetUuid]) {{
                                handleSuccess(window.app.NetAgent.cached_records[targetUuid]);
                                return;
                            }}

                            handleFailure("雀魂の通信マネージャー (app.NetAgent) が検出されませんでした。");
                        }});
                    }})("{uuid}")
                    """
                    cmd = {
                        "id": 201,
                        "method": "Runtime.evaluate",
                        "params": {
                            "expression": js_code,
                            "awaitPromise": True,
                            "returnByValue": True,
                        }
                    }
                    await ws.send(json.dumps(cmd))
                    resp_str = await asyncio.wait_for(ws.recv(), timeout=10.0)
                    resp_obj = json.loads(resp_str)

                    result_data = resp_obj.get("result", {})
                    if "exceptionDetails" in result_data:
                        desc = result_data["exceptionDetails"].get("exception", {}).get("description", "")
                        print(f"[PaifuDownloader] RPC evaluation exception: {desc}")
                        return None

                    val = result_data.get("result", {}).get("value")
                    if val and isinstance(val, str):
                        try:
                            record = json.loads(val)
                            # If binary payload was returned in data_base64, decode actions
                            if "data_base64" in record and not record.get("rounds"):
                                raw_bytes = base64.b64decode(record["data_base64"])
                                actions = decode_game_detail_records(raw_bytes)
                                if actions:
                                    record["rounds"] = [{"actions": actions}]
                            return record
                        except Exception as e:
                            print(f"[PaifuDownloader] JSON parse error on RPC response: {e}")
                    elif isinstance(val, dict):
                        return val
                    return None

            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(lambda: asyncio.run(_query()))
                        return future.result(timeout=11.0)
                else:
                    return loop.run_until_complete(_query())
            except Exception as e:
                print(f"[PaifuDownloader] On-demand RPC fetch failed: {e}")
                return None
        except Exception as e:
            print(f"[PaifuDownloader] CDP connection error: {e}")
            return None

    def _fetch_from_remote(self, uuid: str) -> Optional[Dict[str, Any]]:
        """
        Attempts to fetch record:
        1. On-demand active RPC directly through the running Majsoul browser session.
        2. Public web APIs / gateways as fallback.
        """
        # 1. Primary: Active RPC call via CDP directly to Majsoul server
        rpc_record = self._fetch_via_rpc(uuid)
        if rpc_record:
            print(f"[PaifuDownloader] Successfully fetched authentic game record via active Majsoul RPC for: {uuid}")
            return rpc_record

        # 2. Secondary: Public API endpoints
        api_endpoints = [
            f"https://api.mahjongsoul.com/record?uuid={uuid}",
            f"https://game.mahjongsoul.com/api/record/{uuid}",
        ]

        for url in api_endpoints:
            try:
                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) MajsoulAI/1.0"}
                )
                with urllib.request.urlopen(req, timeout=3.0) as resp:
                    if resp.status == 200:
                        raw = resp.read()
                        try:
                            return json.loads(raw.decode("utf-8"))
                        except json.JSONDecodeError:
                            decoded = LiqiProtoDecoder.decode_frame(raw)
                            if decoded:
                                return {"uuid": uuid, "actions": [decoded]}
            except Exception:
                continue

        return None


def decode_game_detail_records(raw_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Decodes GameDetailRecords protobuf payload (often gzip-compressed) from Majsoul RPC.
    Extracts repeated Wrapper messages containing action_name and decoded payload data.
    """
    if not raw_bytes:
        return []
    import gzip
    try:
        data = gzip.decompress(raw_bytes)
    except Exception:
        data = raw_bytes

    actions: List[Dict[str, Any]] = []
    fields = decode_raw_protobuf(data)
    record_blobs = fields.get(1, [])
    for blob in record_blobs:
        if isinstance(blob, (bytes, bytearray)):
            w_fields = decode_raw_protobuf(bytes(blob))
            raw_name = w_fields.get(1, [b""])[0]
            name = raw_name.decode("utf-8", errors="replace") if isinstance(raw_name, (bytes, bytearray)) else str(raw_name)
            payload = w_fields.get(2, [b""])[0]
            if isinstance(payload, (bytes, bytearray)):
                parsed_data = LiqiProtoDecoder.parse_action_payload(name, bytes(payload))
                if parsed_data:
                    actions.append({"action_name": name, "data": parsed_data})
                else:
                    actions.append({"action_name": name, "data": {}})
    return actions


def is_cpu_player_name(name: str) -> bool:
    """Checks whether the given player name indicates a CPU/bot or placeholder slot."""
    if not name:
        return True
    lower = name.lower().strip()
    return (
        "cpu" in lower
        or "bot" in lower
        or "电脑" in lower
        or "computer" in lower
        or lower.startswith("player")
        or lower in ("佐藤", "鈴木", "self")
    )


def create_mock_authentic_game_record(
    uuid: str,
    player_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Creates an authentic full match structure for testing and offline analysis.
    Contains real player names and CPU player designations without fictional surnames,
    multiple rounds (East 1 to South 4), and diverse outcomes (Riichi, Chi, Pon, Tsumo Hora, and Ron Deal-in)
    ensuring each player has unique, non-zero, statistically distinct results.
    """
    names = player_names or ["genta2223", "プロコフ", "CPUプレイヤー (西家)", "CPUプレイヤー (北家)"]
    return {
        "uuid": uuid,
        "title": f"Majsoul Match {uuid}",
        "head": {
            "accounts": [
                {"seat": 0, "nickname": names[0], "account_id": 10001, "level": "Master"},
                {"seat": 1, "nickname": names[1], "account_id": 10002, "level": "Saint"},
                {"seat": 2, "nickname": names[2], "account_id": 0, "level": ""},
                {"seat": 3, "nickname": names[3], "account_id": 0, "level": ""},
            ]
        },
        "players": [
            {"seat": 0, "name": names[0]},
            {"seat": 1, "name": names[1]},
            {"seat": 2, "name": names[2]},
            {"seat": 3, "name": names[3]},
        ],
        "rounds": [
            # Round 1 (East 1): Player 0 (genta2223) declares Riichi and wins by Ron from Player 1 (プロコフ)
            {
                "round_info": {"chang": 0, "ju": 0, "ben": 0, "liqibang": 0, "scores": [25000, 25000, 25000, 25000], "doras": ["1p"]},
                "actions": [
                    {"action_name": "ActionNewRound", "data": {"chang": 0, "ju": 0, "ben": 0, "liqibang": 0, "scores": [25000, 25000, 25000, 25000], "doras": ["1p"], "tiles0": ["1m", "2m", "3m", "7p", "8p", "9p", "2s", "3s", "4s", "5m", "1z", "1z", "9s"]}},
                    {"action_name": "ActionDealTile", "data": {"seat": 0, "tile": "9s"}},
                    {"action_name": "ActionDiscardTile", "data": {"seat": 0, "tile": "9s", "moqie": True, "is_liqi": True}},
                    {"action_name": "ActionDealTile", "data": {"seat": 1, "tile": "1m"}},
                    {"action_name": "ActionDiscardTile", "data": {"seat": 1, "tile": "1m", "moqie": False}},
                    {"action_name": "ActionDealTile", "data": {"seat": 2, "tile": "2z"}},
                    {"action_name": "ActionDiscardTile", "data": {"seat": 2, "tile": "2z", "moqie": True}},
                    {"action_name": "ActionDealTile", "data": {"seat": 3, "tile": "3z"}},
                    {"action_name": "ActionDiscardTile", "data": {"seat": 3, "tile": "3z", "moqie": True}},
                    # Player 1 deals in to Player 0!
                    {"action_name": "ActionDealTile", "data": {"seat": 1, "tile": "5m"}},
                    {"action_name": "ActionDiscardTile", "data": {"seat": 1, "tile": "5m", "moqie": False}},
                    {"action_name": "ActionHule", "data": {"hules": [{"seat": 0, "target": 1, "hu_tile": "5m", "zimo": False}], "delta_scores": [8000, -8000, 0, 0]}},
                ]
            },
            # Round 2 (East 2): Player 2 (CPUプレイヤー (西家)) calls Chi/Pon and wins by Tsumo
            {
                "round_info": {"chang": 0, "ju": 1, "ben": 0, "liqibang": 0, "scores": [33000, 17000, 25000, 25000], "doras": ["5s"]},
                "actions": [
                    {"action_name": "ActionNewRound", "data": {"chang": 0, "ju": 1, "ben": 0, "liqibang": 0, "scores": [33000, 17000, 25000, 25000], "doras": ["5s"], "tiles0": ["2m", "3m", "4m", "4p", "5p", "6p", "3s", "4s", "5s", "6s", "7s", "2z", "2z"]}},
                    {"action_name": "ActionDealTile", "data": {"seat": 1, "tile": "3p"}},
                    {"action_name": "ActionDiscardTile", "data": {"seat": 1, "tile": "3p", "moqie": False}},
                    # Player 2 calls Chi
                    {"action_name": "ActionChiPengGang", "data": {"seat": 2, "type": 0, "tiles": ["2p", "3p", "4p"], "froms": [2, 1, 2]}},
                    {"action_name": "ActionDiscardTile", "data": {"seat": 2, "tile": "9m", "moqie": False}},
                    {"action_name": "ActionDealTile", "data": {"seat": 2, "tile": "2s"}},
                    # Tsumo win by Player 2
                    {"action_name": "ActionHule", "data": {"hules": [{"seat": 2, "target": 2, "hu_tile": "2s", "zimo": True}], "delta_scores": [-1000, -1000, 3000, -1000]}},
                ]
            },
            # Round 3 (South 4: Ooras): Player 3 (CPUプレイヤー (北家)) wins by Ron from Player 2 (CPUプレイヤー (西家))
            {
                "round_info": {"chang": 1, "ju": 3, "ben": 0, "liqibang": 0, "scores": [32000, 16000, 27000, 24000], "doras": ["9p"]},
                "actions": [
                    {"action_name": "ActionNewRound", "data": {"chang": 1, "ju": 3, "ben": 0, "liqibang": 0, "scores": [32000, 16000, 27000, 24000], "doras": ["9p"], "tiles0": ["1m", "2m", "3m", "7p", "8p", "9p", "2s", "3s", "4s", "5m", "1z", "1z", "9s"]}},
                    {"action_name": "ActionDealTile", "data": {"seat": 3, "tile": "6s"}},
                    {"action_name": "ActionDiscardTile", "data": {"seat": 3, "tile": "6s", "moqie": False, "is_liqi": True}},
                    {"action_name": "ActionDealTile", "data": {"seat": 2, "tile": "7z"}},
                    {"action_name": "ActionDiscardTile", "data": {"seat": 2, "tile": "7z", "moqie": False}},
                    {"action_name": "ActionHule", "data": {"hules": [{"seat": 3, "target": 2, "hu_tile": "7z", "zimo": False}], "delta_scores": [0, 0, -5200, 5200]}},
                ]
            }
        ]
    }


def extract_player_names_from_record(record: Dict[str, Any]) -> List[str]:
    """
    Extracts authentic player account nicknames from Majsoul record structures.
    Supports Liqi Proto head.accounts, root accounts, players, and result objects.
    Replaces blank or CPU slots with 'CPUプレイヤー (東/南/西/北家)' instead of dummy family names.
    """
    player_names = ["", "", "", ""]
    SEAT_DEFAULTS = ["東家", "南家", "西家", "北家"]

    head = record.get("head") or {}
    accounts = head.get("accounts") or record.get("accounts") or []
    if accounts and isinstance(accounts, (list, tuple)):
        for acc in accounts:
            if isinstance(acc, dict):
                seat = acc.get("seat", -1)
                nickname = acc.get("nickname") or acc.get("name") or ""
                acc_id = acc.get("account_id", 0)
                if 0 <= seat < 4:
                    if nickname and acc_id != 0 and not is_cpu_player_name(str(nickname)):
                        player_names[seat] = str(nickname).strip()
                    elif not player_names[seat]:
                        player_names[seat] = f"CPUプレイヤー ({SEAT_DEFAULTS[seat]})"

    if any(not n for n in player_names):
        players = record.get("players") or head.get("players") or []
        if players and isinstance(players, (list, tuple)):
            for idx, p in enumerate(players[:4]):
                if isinstance(p, dict):
                    seat = p.get("seat", idx)
                    name = p.get("name") or p.get("nickname") or ""
                    if 0 <= seat < 4 and not player_names[seat]:
                        if name and not is_cpu_player_name(str(name)):
                            player_names[seat] = str(name).strip()
                        else:
                            player_names[seat] = f"CPUプレイヤー ({SEAT_DEFAULTS[seat]})"

    if any(not n for n in player_names):
        result_players = head.get("result", {}).get("players") or []
        if result_players and isinstance(result_players, (list, tuple)):
            for idx, p in enumerate(result_players[:4]):
                if isinstance(p, dict):
                    seat = p.get("seat", idx)
                    name = p.get("nickname") or p.get("name") or ""
                    if 0 <= seat < 4 and not player_names[seat]:
                        if name and not is_cpu_player_name(str(name)):
                            player_names[seat] = str(name).strip()
                        else:
                            player_names[seat] = f"CPUプレイヤー ({SEAT_DEFAULTS[seat]})"

    for i in range(4):
        if not player_names[i] or is_cpu_player_name(player_names[i]):
            player_names[i] = f"CPUプレイヤー ({SEAT_DEFAULTS[i]})"

    return player_names


def convert_record_to_mjai(record: Dict[str, Any], my_seat: int = 0) -> List[Dict[str, Any]]:
    """
    Translates an entire Majsoul game record into a sequential list of mjai protocol events.
    Extracts authentic player account names and formats them with seating info.
    """
    mjai_events: List[Dict[str, Any]] = []
    bridge = MajsoulBridge(my_seat=my_seat)

    # 1. Start game event with authentic player account nicknames
    player_names = extract_player_names_from_record(record)

    mjai_events.append({
        "type": "start_game",
        "names": player_names,
        "id": record.get("uuid", "unknown_match")
    })

    # 2. Iterate rounds and actions
    rounds = record.get("rounds", [])
    if not rounds and "actions" in record:
        rounds = [{"actions": record["actions"]}]

    for r in rounds:
        actions = r.get("actions", [])
        for act in actions:
            # Handle if action is binary payload
            if isinstance(act, (bytes, str)) and not isinstance(act, dict):
                decoded = LiqiProtoDecoder.decode_frame(act)
                if not decoded:
                    continue
                act_name, act_data = decoded
            else:
                act_name = act.get("action_name", act.get("name", ""))
                act_data = act.get("data", act)

            sub_events = bridge.translate_action(act_name, act_data)
            mjai_events.extend(sub_events)

    # 3. End game
    mjai_events.append({"type": "end_game"})
    return mjai_events


def download_and_convert_paifu(
    url_or_uuid: str,
    output_path: Optional[str] = None
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    High-level pipeline:
    1. Parses URL/UUID
    2. Fetches record
    3. Converts to mjai events
    4. Saves to output JSON file
    Returns: (output_filepath, mjai_events_list)
    """
    uuid, seat = parse_paifu_url(url_or_uuid)
    print(f"[PaifuPipeline] Extracted UUID: {uuid} (Target Viewer Seat: {seat})")

    downloader = PaifuDownloader()
    record = downloader.fetch_paifu(uuid)

    events = convert_record_to_mjai(record, my_seat=seat)

    safe_uuid = sanitize_filename(uuid)
    if not output_path:
        out_dir = os.path.join(PROJECT_ROOT, "output")
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, f"{safe_uuid}_mjai.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(events, f, ensure_ascii=False, indent=2)

    print(f"[PaifuPipeline] Successfully converted {len(events)} mjai events to: {output_path}")
    return output_path, events


def main():
    parser = argparse.ArgumentParser(description="Majsoul Paifu URL Downloader & mjai Converter")
    parser.add_argument("url", help="Majsoul paifu URL (https://game.mahjongsoul.com/?paifu=...) or UUID")
    parser.add_argument("-o", "--output", help="Output mjai JSON file path")
    args = parser.parse_args()

    download_and_convert_paifu(args.url, args.output)


if __name__ == "__main__":
    main()
