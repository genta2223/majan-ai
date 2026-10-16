"""Mitmproxy addon script to forward Majsoul WebSocket binary frames to Mahjong AI Agent.

Usage with mitmproxy:
    mitmweb -s tools/mitm_addon.py --listen-port 8080
"""

import asyncio
import threading
import websockets


AGENT_WS_URL = "ws://127.0.0.1:8765"


class MajsoulMitmBridge:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._start_loop, daemon=True)
        self.thread.start()
        self.client_ws = None
        asyncio.run_coroutine_threadsafe(self._connect_to_agent(), self.loop)

    def _start_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    async def _connect_to_agent(self):
        while True:
            try:
                self.client_ws = await websockets.connect(AGENT_WS_URL)
                print(f"[MitmAddon] Connected to Mahjong AI bridge at {AGENT_WS_URL}")
                await self.client_ws.wait_closed()
            except Exception:
                await asyncio.sleep(2.0)

    def websocket_message(self, flow):
        """Intercept WebSocket messages and forward game frames."""
        url = flow.request.pretty_url.lower()
        # Filter Majsoul game server domains
        if any(d in url for d in ("maj-soul", "mahjongsoul", "game-gateway", "gateway")):
            msg = flow.websocket.messages[-1]
            if msg.from_client:
                # Client outgoing
                pass
            else:
                # Server incoming message (game state, discards, etc.)
                if self.client_ws and not self.client_ws.closed:
                    asyncio.run_coroutine_threadsafe(
                        self.client_ws.send(msg.content),
                        self.loop
                    )


addons = [MajsoulMitmBridge()]
