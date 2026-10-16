"""Live traffic sniffer for Windows Desktop Majsoul client via CDP or mitmproxy."""

from __future__ import annotations
import asyncio
import json
import sys
import urllib.request
from typing import AsyncIterator, Dict, Any, Optional, Callable, Awaitable

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass

from io_adapter.liqi_proto_decoder import LiqiProtoDecoder
from io_adapter.majsoul_bridge import MajsoulBridge
from utils.debug_tracer import trace_log


class LiveSniffer:
    """
    Captures live Majsoul WebSocket traffic from:
    1. CDP (Chrome DevTools Protocol) attached to the Windows app running with --remote-debugging-port
    2. Local mitmproxy addon WebSocket bridge
    """

    def __init__(
        self,
        mode: str = "cdp",
        cdp_port: int = 9222,
        mitm_port: int = 8765,
        my_seat: int = 0
    ):
        self.mode = mode
        self.cdp_port = cdp_port
        self.mitm_port = mitm_port
        self.bridge = MajsoulBridge(my_seat=my_seat)
        self.decoder = LiqiProtoDecoder()
        self.event_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        self.on_connected_callback: Optional[Callable[[], None]] = None
        self._is_running = False

    async def start(self) -> None:
        """Start sniffing according to chosen mode."""
        self._is_running = True
        try:
            if self.mode == "cdp":
                await self._run_cdp_listener()
            elif self.mode == "mitm":
                await self._run_mitm_ws_server()
            else:
                raise ValueError(f"Unknown sniffer mode: {self.mode}")
        finally:
            self._is_running = False

    async def stream_events(self) -> AsyncIterator[Dict[str, Any]]:
        """Asynchronously stream parsed mjai events in real-time."""
        # Wait up to 5 seconds for start() to initialize
        for _ in range(50):
            if self._is_running:
                break
            await asyncio.sleep(0.1)

        while self._is_running:
            try:
                event = await asyncio.wait_for(self.event_queue.get(), timeout=0.5)
                yield event
                self.event_queue.task_done()
            except asyncio.TimeoutError:
                continue

    def stop(self) -> None:
        self._is_running = False

    async def _run_cdp_listener(self) -> None:
        """Connects to Chromium DevTools Protocol via WebSocket."""
        try:
            import websockets
        except ImportError:
            raise ImportError("Please install websockets: pip install websockets")

        # 1. Discover target WebSocket debugger URL
        target_ws_url = self._get_cdp_target_ws_url()
        if not target_ws_url:
            raise ConnectionError(
                f"Could not find valid Majsoul CDP target at http://127.0.0.1:{self.cdp_port}/json. "
                "Ensure Chrome / Windows app is running with --remote-debugging-port=9222"
            )

        print(f"[CDPSniffer] Opening WebSocket to {target_ws_url}...")
        try:
            async with websockets.connect(
                target_ws_url,
                open_timeout=5.0,
                ping_interval=None,
                max_size=10 * 1024 * 1024
            ) as ws:
                print("[CDPSniffer] Connected! Sending Network.enable...")
                enable_cmd = {"id": 1, "method": "Network.enable", "params": {}}
                await ws.send(json.dumps(enable_cmd))
                resp = await asyncio.wait_for(ws.recv(), timeout=5.0)
                print(f"[CDPSniffer] Network domain enabled: {resp}")
                print("[CDPSniffer] Listening for WebSocket frames...")
                if self.on_connected_callback:
                    try:
                        self.on_connected_callback()
                    except Exception as cb_err:
                        print(f"[CDPSniffer] Callback error: {cb_err}")

                async for message in ws:
                    if not self._is_running:
                        break
                    try:
                        cdp_msg = json.loads(message)
                        method = cdp_msg.get("method", "")
                        params = cdp_msg.get("params", {})

                        if method in ("Network.webSocketFrameReceived", "Network.webSocketFrameSent"):
                            response = params.get("response", {})
                            payload_data = response.get("payloadData", "")
                            direction = "RECV" if "Received" in method else "SENT"
                            trace_log("CDP_WS_FRAME", f"{direction} frame len={len(payload_data)} method={method}")
                            print(f"[CDP FRAME] {direction} | Method: {method} | Payload Len: {len(payload_data)}")
                            if payload_data:
                                await self._process_raw_frame(payload_data)
                    except Exception as e:
                        trace_log("CDP_FRAME_ERR", f"Error processing frame: {e}")
                        print(f"[CDPSniffer] Error processing message: {e}")
                        continue
        except asyncio.TimeoutError:
            trace_log("CDP_TIMEOUT", f"Timeout connecting to {target_ws_url}")
            raise TimeoutError(f"[CDPSniffer] Timeout while connecting or enabling Network domain on {target_ws_url}")
        except Exception as e:
            trace_log("CDP_ERR", f"WebSocket error: {e}")
            print(f"[CDPSniffer] WebSocket error: {e}")
            raise

    def _get_cdp_target_ws_url(self) -> Optional[str]:
        """Fetch list of open targets and select Majsoul page tab with strict priority."""
        url = f"http://127.0.0.1:{self.cdp_port}/json"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "MahjongAI"})
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                raw_bytes = resp.read()
                targets = json.loads(raw_bytes.decode("utf-8", errors="replace"))
        except Exception as e:
            print(f"[CDPSniffer] Target discovery failed at {url}: {e}")
            raise ConnectionError(f"Failed to connect to CDP endpoint at {url}: {e}") from e

        if not targets:
            raise ConnectionError(f"[CDPSniffer] No targets found at {url}. Is Chrome/client running with remote debugging?")

        # Filter and rank candidates
        # Priority 1: type == "page" AND title contains "雀魂"
        # Priority 2: type == "page" AND url contains "mahjongsoul" or "maj-soul"
        # Priority 3: title contains "雀魂" or "majsoul" (any type)
        # Priority 4: any type == "page" (excluding chrome-extension or internal)
        selected_target = None
        match_reason = ""

        for target in targets:
            title = target.get("title", "")
            t_url = target.get("url", "")
            t_type = target.get("type", "")

            if t_type == "page" and "雀魂" in title:
                selected_target = target
                match_reason = "Priority 1 (type=='page' and title contains '雀魂')"
                break

        if not selected_target:
            for target in targets:
                title = target.get("title", "")
                t_url = target.get("url", "").lower()
                t_type = target.get("type", "")
                if t_type == "page" and ("mahjongsoul" in t_url or "maj-soul" in t_url or "majsoul" in title.lower()):
                    selected_target = target
                    match_reason = "Priority 2 (type=='page' and URL/title contains 'mahjongsoul')"
                    break

        if not selected_target:
            for target in targets:
                title = target.get("title", "")
                if "雀魂" in title:
                    selected_target = target
                    match_reason = "Priority 3 (title contains '雀魂')"
                    break

        if not selected_target:
            for target in targets:
                t_type = target.get("type", "")
                t_url = target.get("url", "")
                if t_type == "page" and not t_url.startswith("chrome://") and not t_url.startswith("chrome-extension://"):
                    selected_target = target
                    match_reason = "Priority 4 (general web page)"
                    break

        if selected_target:
            ws_url = selected_target.get("webSocketDebuggerUrl")
            title = selected_target.get("title", "")
            t_url = selected_target.get("url", "")
            t_id = selected_target.get("id", "")
            print("==================================================================")
            print(f"  [CDPSniffer] Majsoul Target Detected ({match_reason})")
            print(f"  ├─ Title  : {title}")
            print(f"  ├─ URL    : {t_url}")
            print(f"  ├─ Target : {t_id}")
            print(f"  └─ WS URL : {ws_url}")
            print("==================================================================")
            if ws_url:
                return ws_url

        # Not found - dump detailed target list
        print(f"\n[CDPSniffer] ERROR: Could not find Majsoul game tab in {len(targets)} targets at {url}:")
        for idx, t in enumerate(targets, 1):
            print(f"  [{idx}] Type: {t.get('type')} | Title: {t.get('title')} | URL: {t.get('url')}")
            print(f"      WS: {t.get('webSocketDebuggerUrl')}")

        raise ConnectionError(f"No suitable Majsoul game tab detected at {url}. See target list above.")

    async def _run_mitm_ws_server(self) -> None:
        """Run WebSocket server to accept frames pushed from tools/mitm_addon.py."""
        try:
            import websockets
        except ImportError:
            raise ImportError("Please install websockets: pip install websockets")

        async def handler(websocket):
            print(f"[MitmServer] Client connected from {websocket.remote_address}")
            async for message in websocket:
                await self._process_raw_frame(message)

        print(f"[MitmServer] Listening for mitmproxy bridge on ws://127.0.0.1:{self.mitm_port}")
        async with websockets.serve(handler, "127.0.0.1", self.mitm_port):
            while self._is_running:
                await asyncio.sleep(0.5)

    async def _process_raw_frame(self, raw_data: Any) -> None:
        """Decode raw frame to Liqi action and push mjai events to queue."""
        if not raw_data:
            return
        try:
            decoded = self.decoder.decode_frame(raw_data)
            if not decoded:
                return

            action_name, action_data = decoded
            if not action_name:
                trace_log("FRAME_NO_ACTION", "Frame decoded without action_name")
                return

            # If wrapped in ActionPrototype or NotifyPlayerAction
            if "action_name" in action_data and action_data["action_name"]:
                action_name = action_data["action_name"]

            trace_log("LIQI_ACTION_DECODED", f"Action: {action_name}", action_data)

            # Translate into mjai events via bridge
            mjai_events = self.bridge.translate_action(action_name, action_data)
            if mjai_events:
                types = [e.get("type") for e in mjai_events]
                trace_log("MJAI_TRANSLATED", f"Generated {len(mjai_events)} events: {types}")
                print(f"[CDPSniffer] Parsed {len(mjai_events)} mjai event(s) from {action_name}: {types}")
            for ev in mjai_events:
                await self.event_queue.put(ev)
                trace_log("EVENT_QUEUED", f"Queued {ev.get('type')}", ev)
        except Exception as e:
            trace_log("FRAME_PROCESS_ERR", f"Failed to parse frame: {e}")
            # Fallback - never crash or stop listener on unexpected packets
            print(f"[CDPSniffer] Warning: Failed to parse frame ({e}). Skipping frame safely.")
            import traceback
            traceback.print_exc()
