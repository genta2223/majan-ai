"""Real-time mjai stream communication client supporting stdio and WebSocket with instant pass filtering."""

from __future__ import annotations
import asyncio
import json
import sys
import time
from typing import Optional, Dict, Any, AsyncIterator, Callable, Awaitable, Tuple

from core.state_manager import MahjongStateManager
from core.engine import DecisionEngine
from core.strategy_modifier import StrategyModifier
from io_adapter.action_controller import ActionController, OutputAction


def is_my_decision_required(
    event: Dict[str, Any],
    state_mgr: MahjongStateManager,
    engine: DecisionEngine
) -> bool:
    """
    Strict guard checking whether this event genuinely demands a thoughtful decision from self.
    - Self draw (tsumo with actor == my_actor_id): True
    - Enemy discard (dahai with actor != my_actor_id):
        True ONLY if engine identifies at least one valid call (Ron, Pon, Chi, Kan).
        If no calls are possible, returns False immediately.
    - All other events (enemy tsumo, start_kyoku, dora, etc.): False
    """
    event_type = event.get("type", "")
    actor = event.get("actor")

    if event_type == "tsumo":
        return (actor == state_mgr.my_actor_id)

    if event_type == "start_kyoku":
        # If self is dealer and hand already has 14 tiles, self initial discard is required
        oya = event.get("oya", 0)
        if oya == state_mgr.my_actor_id:
            my_hand = state_mgr.my_player.tehai if state_mgr.my_player else []
            if len(my_hand) >= 14:
                return True
        return False

    if event_type == "dahai":
        if actor == state_mgr.my_actor_id:
            return False

        # 1. Check if event contains an explicit operation_list for self
        op_list = event.get("operation_list") or []
        for op in op_list:
            op_seat = op.get("seat")
            if op_seat is None or op_seat == state_mgr.my_actor_id:
                if op.get("type_name") in ("chi", "pon", "kan", "daiminkan", "hora", "ron") or op.get("type") in (2, 3, 5, 9):
                    return True

        # 2. Fast candidate evaluation
        candidates = engine.evaluate(state_mgr, valid_actions=op_list)
        has_valid_call = any(c.action_type in ("hora", "pon", "chi", "kan", "daiminkan") for c in candidates)
        return has_valid_call

    return False


class MjaiStreamHandler:
    """
    Coordinates state updates, instant pass filtering, and strategic action emission.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        my_actor_id: int = 0,
        apply_sleep: bool = True
    ):
        self.config = config
        self.my_actor_id = my_actor_id
        self.apply_sleep = apply_sleep

        engine_cfg = self.config.get("engine", {})
        self.state_mgr = MahjongStateManager(my_actor_id=self.my_actor_id)
        self.engine = DecisionEngine(
            model_path=engine_cfg.get("model_path"),
            device=engine_cfg.get("device", "cpu"),
            use_fallback=engine_cfg.get("use_fallback_if_missing", True),
            max_inference_time_ms=engine_cfg.get("max_inference_time_ms", 200)
        )
        self.modifier = StrategyModifier(self.config)
        self.action_controller = ActionController(self.config)

    async def process_event(self, event: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], float]:
        """
        Processes incoming mjai event.
        Returns tuple of (response_event_dict, elapsed_wait_seconds).

        Guaranteed:
        - If decision is NOT required: returns ({"type": "none"} or None) in < 0.001s (0 wait).
        - If decision IS required: computes best action and applies Gaussian human delay.
        """
        t0 = time.perf_counter()

        # 1. Update state first
        self.state_mgr.update_event(event)

        # 2. Strict guard condition for instant pass / skip
        needs_decision = is_my_decision_required(event, self.state_mgr, self.engine)

        if not needs_decision:
            # If enemy discard and requires mjai response, return instant none without delay
            if event.get("type") == "dahai" and event.get("actor") != self.my_actor_id:
                return {"type": "none"}, (time.perf_counter() - t0)
            return None, (time.perf_counter() - t0)

        # 3. Decision genuinely required: Run inference
        t_inf_start = time.perf_counter()
        raw_candidates = self.engine.evaluate(self.state_mgr)
        comp_time_ms = (time.perf_counter() - t_inf_start) * 1000

        if not raw_candidates:
            return {"type": "none"}, (time.perf_counter() - t0)

        # 4. Modify Q-values tactically
        modified = self.modifier.modify_candidates(raw_candidates, self.state_mgr)
        best = modified[0]

        # 5. Apply human-like delay and calculate output action
        output: OutputAction = await self.action_controller.execute_action(
            candidate=best,
            state_mgr=self.state_mgr,
            comp_time_ms=comp_time_ms,
            apply_sleep=self.apply_sleep
        )

        # Format mjai response JSON
        resp_event: Dict[str, Any]
        if output.action_type == "dahai":
            resp_event = {
                "type": "dahai",
                "actor": self.my_actor_id,
                "pai": output.tile,
                "tsumogiri": output.metadata.get("is_tsumogiri", False),
            }
        elif output.action_type == "reach":
            resp_event = {
                "type": "reach",
                "actor": self.my_actor_id,
            }
        elif output.action_type in ("pon", "chi", "kan"):
            resp_event = {
                "type": output.action_type,
                "actor": self.my_actor_id,
                "target": self.state_mgr.round_state.last_discard_actor if self.state_mgr.round_state else None,
                "pai": output.tile,
                "consumed": best.consumed or [],
            }
        elif output.action_type == "hora":
            resp_event = {
                "type": "hora",
                "actor": self.my_actor_id,
                "target": self.state_mgr.round_state.last_discard_actor if self.state_mgr.round_state else self.my_actor_id,
                "pai": output.tile,
            }
        else:
            resp_event = {"type": "none"}

        total_elapsed = time.perf_counter() - t0
        return resp_event, total_elapsed


class StdioStreamClient:
    """Manages mjai communication over standard I/O (stdin/stdout lines)."""

    def __init__(self, handler: MjaiStreamHandler):
        self.handler = handler

    async def run(self) -> None:
        """Read lines from stdin asynchronously, process, and write to stdout."""
        loop = asyncio.get_running_loop()
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await loop.connect_read_pipe(lambda: protocol, sys.stdin)

        while True:
            line_bytes = await reader.readline()
            if not line_bytes:
                break
            line_str = line_bytes.decode("utf-8").strip()
            if not line_str:
                continue
            try:
                event = json.loads(line_str)
            except json.JSONDecodeError:
                continue

            resp, _ = await self.handler.process_event(event)
            if resp is not None:
                out_str = json.dumps(resp, ensure_ascii=False) + "\n"
                sys.stdout.write(out_str)
                sys.stdout.flush()


class WebSocketStreamClient:
    """
    Connects to a WebSocket server (e.g. ws://127.0.0.1:8765) to receive and send mjai events.
    """

    def __init__(self, handler: MjaiStreamHandler, uri: str = "ws://127.0.0.1:8765"):
        self.handler = handler
        self.uri = uri

    async def run(self) -> None:
        """Run WebSocket connection loop."""
        try:
            import websockets
        except ImportError:
            print("[WebSocketClient] Error: 'websockets' package is not installed. Please pip install websockets.")
            return

        async with websockets.connect(self.uri) as ws:
            print(f"[WebSocketClient] Connected to {self.uri}")
            async for message in ws:
                try:
                    event = json.loads(message)
                except json.JSONDecodeError:
                    continue

                resp, _ = await self.handler.process_event(event)
                if resp is not None:
                    await ws.send(json.dumps(resp, ensure_ascii=False))
