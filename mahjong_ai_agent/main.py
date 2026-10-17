"""Main entry point for Mahjong AI Agent asynchronous event loop."""

import asyncio
import argparse
import json
import os
import sys
import time
import yaml
from typing import Optional, Dict, Any

from core.state_manager import MahjongStateManager, is_valid_mahjong_tile
from core.engine import DecisionEngine
from core.strategy_modifier import StrategyModifier
from io_adapter.input_simulator import InputSimulator, load_mjai_log
from io_adapter.action_controller import ActionController, OutputAction
from utils.debug_tracer import trace_log, mark_trace_event

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass


class MahjongAIAgent:
    """Orchestrates state management, neural engine, strategy modifiers, and action output."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        my_actor_id: int = 0,
        apply_sleep: bool = True,
        output_mode: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        if config is not None:
            self.config = config
        elif config_path:
            if not os.path.exists(config_path):
                from utils.path_helper import get_config_path
                config_path = get_config_path(os.path.basename(config_path))
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {}

        if output_mode:
            self.config["output_mode"] = output_mode

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
        from core.opponent_profiler import OpponentProfiler
        prof_path = self.config.get("opponent_profiles_path")
        self.profiler = self.modifier.profiler or OpponentProfiler(prof_path)
        self.modifier.profiler = self.profiler

        self.action_controller = ActionController(self.config)
        self.status_callback: Optional[Callable[[Dict[str, Any]], None]] = None
        # Vision is a strict last-resort fallback. Once packet decoding delivered a real hand
        # for the current round, vision is locked out until the next start_kyoku.
        self.vision_fallback_enabled: bool = bool(self.config.get("vision_fallback_enabled", True))
        self.packet_hand_trusted: bool = False

    def set_my_actor_id(self, seat: int) -> None:
        """Dynamically updates self actor ID and state manager."""
        self.my_actor_id = seat
        self.state_mgr.my_actor_id = seat

    async def handle_event(self, event: Dict[str, Any]) -> Optional[OutputAction]:
        """Consumes a single mjai event with guaranteed top-level exception immunity."""
        try:
            return await self._handle_event_impl(event)
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            ev_type = event.get("type", "unknown") if isinstance(event, dict) else "unknown"
            print(f"[Main ERROR] Unhandled exception in handle_event for '{ev_type}': {e}\n{tb}")
            trace_log("HANDLE_EVENT_EXCEPTION", f"Event '{ev_type}' error: {e}\n{tb}")
            return None

    async def _handle_event_impl(self, event: Dict[str, Any]) -> Optional[OutputAction]:
        event_type = event.get("type")

        # 0. Real-time opponent profiling stream feed
        if self.profiler:
            try:
                self.profiler.consume_live_event(event)
            except Exception as pe:
                print(f"[Main Profiler] Warning processing live event: {pe}")

        # 0. Automatic Seat Detection from server messages
        if event_type == "start_kyoku":
            tehais = event.get("tehais", [])
            has_visible = False
            for s_idx, hand in enumerate(tehais):
                visible_tiles = [t for t in hand if t and t != "?"]
                if len(visible_tiles) >= 13:
                    has_visible = True
                    if self.my_actor_id != s_idx:
                        print(f"\n[Main AutoSeat] Auto-detected self seat: Player {s_idx} ({len(visible_tiles)} tiles). Switched actor ID {self.my_actor_id} -> {s_idx}")
                        self.my_actor_id = s_idx
                        self.state_mgr.my_actor_id = s_idx
                    break
            self.packet_hand_trusted = has_visible
            if has_visible:
                print(f"[Main Guard] Packet hand trusted for this round -> Vision fallback LOCKED")

            # Vision Fallback safety net (non-blocking thread with 0.8s timeout)
            is_genuine_round = event.get("action_name") in ("ActionNewRound", "RecordNewRound") if "action_name" in event else True
            if self.vision_fallback_enabled and is_genuine_round and not has_visible and len(tehais) > self.my_actor_id:
                try:
                    def _capture():
                        from io_adapter.vision_hand_reader import VisionHandReader
                        reader = VisionHandReader()
                        return reader.read_hand()

                    raw_vision_hand = await asyncio.wait_for(asyncio.to_thread(_capture), timeout=0.8)
                    vision_hand = [t for t in raw_vision_hand if is_valid_mahjong_tile(t)]
                    if vision_hand and len(vision_hand) >= 13 and len(set(vision_hand)) >= 3:
                        tehais[self.my_actor_id] = vision_hand[:13]
                        print(f"[Main Vision Fallback] Injected {len(vision_hand)} vision tiles into Player {self.my_actor_id}: {tehais[self.my_actor_id]}")
                        if event.get("oya", 0) == self.my_actor_id and len(vision_hand) >= 14:
                            tehais[self.my_actor_id] = vision_hand[:14]
                except (asyncio.TimeoutError, Exception) as ve:
                    pass

        elif event_type == "tsumo":
            actor = event.get("actor", 0)
            pai = event.get("pai", "")
            if pai and pai != "?" and self.my_actor_id != actor:
                print(f"\n[Main AutoSeat] Auto-detected self drawn tile ({pai}) on Player {actor}. Switched actor ID {self.my_actor_id} -> {actor}")
                self.my_actor_id = actor
                self.state_mgr.my_actor_id = actor

            # Live Vision Resynchronization on self turn
            if actor == self.my_actor_id:
                if not self.state_mgr.round_state:
                    self.state_mgr.update_event({"type": "tsumo", "actor": actor, "pai": pai, "tiles": event.get("tiles")})
                elif event.get("tiles") and self.state_mgr.round_state:
                    p = self.state_mgr.round_state.players[self.my_actor_id]
                    if len(p.tehai) < 13 or all(t == "?" or not is_valid_mahjong_tile(t) for t in p.tehai):
                        p.tehai = [t for t in event["tiles"] if is_valid_mahjong_tile(t)]
                        print(f"[Main RESCUE] Synchronized self hand from tsumo event: {p.tehai}")

                needs_sync = (not pai or pai == "?")
                if not needs_sync and self.state_mgr.round_state:
                    p = self.state_mgr.round_state.players[self.my_actor_id]
                    if len(p.tehai) < 13 or all(t == "?" or not is_valid_mahjong_tile(t) for t in p.tehai):
                        needs_sync = True

                if needs_sync and (self.packet_hand_trusted or not self.vision_fallback_enabled):
                    if not pai or pai == "?":
                        print(f"[Main Guard] Self tsumo tile missing from packet; vision LOCKED (packet hand trusted). Keeping state.")
                    needs_sync = False

                if needs_sync:
                    try:
                        def _capture_turn():
                            from io_adapter.vision_hand_reader import VisionHandReader
                            reader = VisionHandReader()
                            return reader.read_hand()

                        raw_vision_hand = await asyncio.wait_for(asyncio.to_thread(_capture_turn), timeout=0.8)
                        vision_hand = [t for t in raw_vision_hand if is_valid_mahjong_tile(t)]
                        if vision_hand and len(vision_hand) >= 13:
                            if len(vision_hand) >= 14:
                                tsumo_tile = vision_hand[-1]
                                hand_tiles = [t for t in vision_hand[:13] if is_valid_mahjong_tile(t)]
                            else:
                                tsumo_tile = vision_hand[-1]
                                hand_tiles = [t for t in vision_hand[:-1] if is_valid_mahjong_tile(t)]

                            if tsumo_tile and is_valid_mahjong_tile(tsumo_tile):
                                event["pai"] = tsumo_tile

                            if self.state_mgr.round_state and hand_tiles:
                                p = self.state_mgr.round_state.players[self.my_actor_id]
                                p.tehai = list(hand_tiles)
                                p.tsumo_tile = tsumo_tile
                            print(f"[Live Vision Sync] Resynchronized player {self.my_actor_id} hand ({len(vision_hand)} tiles): {vision_hand} | Tsumo: {tsumo_tile}")
                    except (asyncio.TimeoutError, Exception) as ve:
                        pass

        # 1. Update board state
        self.state_mgr.update_event(event)

        # Clear previous recommendation cache on new turn draw or self discard to prevent stale recommendations
        if event_type == "tsumo":
            if event.get("actor") == self.my_actor_id:
                self.action_controller.clear_recommendation(notify=True)
            else:
                self.action_controller.clear_recommendation(notify=False)
        elif event_type == "dahai" and event.get("actor") == self.my_actor_id:
            self.action_controller.clear_recommendation(notify=True)

        if self.status_callback and event_type in ("start_kyoku", "tsumo", "dahai", "chi", "pon", "kan", "hora", "ryukyoku"):
            try:
                round_st = self.state_mgr.round_state
                round_str = f"{round_st.bakaze}{round_st.kyoku}局 {round_st.honba}本場" if round_st else "対局中"
                seat_str = ["東家", "南家", "西家", "北家"][self.my_actor_id] if 0 <= self.my_actor_id <= 3 else f"P{self.my_actor_id}"
                
                profiler_status = ""
                if self.profiler:
                    mode_name = "東風戦" if self.profiler.match_mode == "tonpuu" else "半荘戦"
                    live_rds = self.profiler.live_round_count
                    profiler_status = f"[Live学習] {mode_name}モード: {live_rds}局蓄積済 (牌譜データとハイブリッド統合中)"

                self.status_callback({
                    "status": "in_game",
                    "round_info": round_str,
                    "seat_name": seat_str,
                    "event_type": event_type,
                    "profiler_status": profiler_status,
                })
            except Exception:
                pass

        # Strict guard condition checking if self decision is genuinely required
        from io_adapter.stream_client import is_my_decision_required
        if not is_my_decision_required(event, self.state_mgr, self.engine):
            return None

        # 2. Run inference engine (Target: < 200ms)
        t0 = time.perf_counter()
        raw_candidates = self.engine.evaluate(self.state_mgr)
        comp_time_ms = (time.perf_counter() - t0) * 1000

        if not raw_candidates:
            return None

        # 3. Dynamic tactical modification
        modified_candidates = self.modifier.modify_candidates(raw_candidates, self.state_mgr)
        best_candidate = modified_candidates[0]

        # 4. Action controller execution (applies 1.0 - 3.0s thinking delay & screen coordinates)
        output_action = await self.action_controller.execute_action(
            candidate=best_candidate,
            state_mgr=self.state_mgr,
            comp_time_ms=comp_time_ms,
            apply_sleep=self.apply_sleep,
            all_candidates=modified_candidates
        )

        return output_action

    async def run_stream(self, event_stream) -> None:
        """Asynchronous loop consuming events from an async generator with permanent liveness."""
        print("==================================================================")
        print("  [Mahjong AI Agent] Initialized (Tactical Push-Fold Engine)      ")
        print(f"  Actor ID: {self.my_actor_id} | Sleep Delay: {self.apply_sleep}")
        print("==================================================================")

        import traceback
        try:
            async for event in event_stream:
                try:
                    if not event or not isinstance(event, dict):
                        continue
                    ev_type = event.get("type", "")
                    trace_log("AGENT_STREAM_RECV", f"Event: {ev_type}", event)
                    if ev_type in ("start_kyoku", "reach", "hora", "ryukyoku", "tsumo", "dahai", "chi", "pon", "kan"):
                        actor_str = f"P{event.get('actor')}" if "actor" in event else ""
                        pai_str = event.get("pai", "")
                        print(f"[STREAM EVENT] {ev_type.upper():<12} {actor_str:<4} {pai_str:<4}")
                    action = await self.handle_event(event)
                    if action:
                        trace_log("DECISION_OUTPUT", f"Action: {action.action_type} Tile: {action.tile}")
                        self._print_decision(action)
                except Exception as e:
                    tb = traceback.format_exc()
                    trace_log("AGENT_EVENT_ERR", f"Error in event {event}: {e}\n{tb}")
                    print(f"[Main Stream] Error processing event {event}: {e}\n{tb}")
                    continue
        except Exception as outer_e:
            tb = traceback.format_exc()
            trace_log("AGENT_STREAM_FATAL", f"Fatal event_stream error: {outer_e}\n{tb}")
            print(f"[Main Stream FATAL] Event stream loop caught fatal error: {outer_e}\n{tb}")

    def _print_decision(self, action: OutputAction) -> None:
        coords_str = f"({action.screen_coordinates.x}, {action.screen_coordinates.y})" if action.screen_coordinates else "N/A"
        bd = action.metadata.get("breakdown", {})
        bd_str = " | ".join([f"{k}: {v:+}" if isinstance(v, (int, float)) else f"{k}: {v}" for k, v in bd.items()]) if bd else "None"
        print(f"\n>>> [DECISION] Output Action: {action.action_type.upper()} {action.tile or ''}")
        print(f"    ├─ Computation Time : {action.computation_time_ms:.2f} ms (Target: <= 200 ms)")
        print(f"    ├─ Human Wait Time  : {action.thinking_time_sec:.2f} s")
        print(f"    ├─ Slot Index       : {action.target_slot_index} | Screen Coords: {coords_str}")
        print(f"    ├─ Q Breakdown      : {bd_str}")
        print(f"    └─ Tactical Summary : {action.reasoning_summary}")


async def main():
    parser = argparse.ArgumentParser(description="Mahjong AI Agent Runner")
    parser.add_argument("--config", default="config/strategy_config.yaml", help="Path to config YAML")
    parser.add_argument("--my-id", type=int, default=0, help="Actor ID of the AI (0-3)")
    parser.add_argument("--scenario", choices=["ooras_top", "ooras_last", "file"], default="ooras_top",
                        help="Select test scenario or supply a file")
    parser.add_argument("--log-file", type=str, default="", help="Path to mjai JSON/JSONL log file")
    parser.add_argument("--paifu-url", type=str, default="", help="Majsoul paifu URL (https://game.mahjongsoul.com/?paifu=...) or UUID")
    parser.add_argument("--output-mode", choices=["dashboard", "auto_click", "semi_auto", "text_only"], default="dashboard",
                        help="Choose HUD dashboard assistant, auto_click mouse simulation, semi_auto assist, or text_only HUD")
    parser.add_argument("--no-sleep", action="store_true", help="Disable human-like delay for fast testing")
    parser.add_argument("--speed", type=float, default=0.0, help="Stream replay speed between events (seconds)")

    # Phase 3 Live Sniffer Options
    parser.add_argument("--live-cdp", action="store_true", help="Attach to live Windows Desktop app via Chrome DevTools Protocol")
    parser.add_argument("--live-mitm", action="store_true", help="Listen for live traffic from mitmproxy bridge")
    parser.add_argument("--port", type=int, default=9222, help="CDP port (default: 9222) or mitm ws port (default: 8765)")
    parser.add_argument("--enable-mouse", action="store_true", help="Enable actual OS mouse clicks (pyautogui)")
    args = parser.parse_args()

    agent = MahjongAIAgent(
        config_path=args.config,
        my_actor_id=args.my_id,
        apply_sleep=not args.no_sleep,
        output_mode=args.output_mode
    )

    if args.enable_mouse:
        agent.config.setdefault("mouse_control", {})["enable_pyautogui"] = True
        agent.action_controller.mouse_cfg["enable_pyautogui"] = True
        print("[Security/Notice] Actual OS mouse control ENABLED.")

    # Live Mode (CDP or Mitmproxy)
    if args.live_cdp or args.live_mitm:
        from io_adapter.live_sniffer import LiveSniffer
        mode = "cdp" if args.live_cdp else "mitm"
        sniffer = LiveSniffer(
            mode=mode,
            cdp_port=args.port if args.live_cdp else 9222,
            mitm_port=args.port if args.live_mitm else 8765,
            my_seat=args.my_id
        )
        print(f"==================================================================")
        print(f"  [LIVE MODE ACTIVATED] Mode: {mode.upper()} | Port: {args.port}")
        print(f"==================================================================")
        sniffer_task = asyncio.create_task(sniffer.start())
        stream_task = asyncio.create_task(agent.run_stream(sniffer.stream_events()))

        try:
            done, pending = await asyncio.wait(
                [sniffer_task, stream_task],
                return_when=asyncio.FIRST_COMPLETED
            )
            for t in done:
                if t.exception():
                    raise t.exception()
        except (KeyboardInterrupt, asyncio.CancelledError):
            print("\n[Main] Shutdown signal received. Exiting live mode cleanly...")
        finally:
            sniffer.stop()
            sniffer_task.cancel()
            stream_task.cancel()
            await asyncio.gather(sniffer_task, stream_task, return_exceptions=True)
        return

    # Paifu URL Download & Conversion Mode
    if args.paifu_url:
        from tools.paifu_to_mjai import download_and_convert_paifu, parse_paifu_url
        _, seat = parse_paifu_url(args.paifu_url)
        agent.my_actor_id = seat
        agent.state_mgr.my_actor_id = seat
        out_json_path, raw_events = download_and_convert_paifu(args.paifu_url)
        print(f"[Main] Replaying downloaded paifu from {out_json_path} (Perspective: Player {seat})")
        simulator = InputSimulator(events=raw_events, replay_speed=args.speed)
        await agent.run_stream(simulator.stream_events())
        return

    # Simulation / File Replay Mode
    if args.log_file and os.path.exists(args.log_file):
        raw_events = load_mjai_log(args.log_file)
    elif args.scenario == "ooras_top":
        raw_events = InputSimulator.create_mock_ooras_scenario(my_id=args.my_id, my_score=38000, enemy_riichi=True)
    elif args.scenario == "ooras_last":
        raw_events = InputSimulator.create_mock_ooras_scenario(my_id=args.my_id, my_score=8000, enemy_riichi=True)
    else:
        raw_events = InputSimulator.create_mock_ooras_scenario(my_id=args.my_id)

    simulator = InputSimulator(events=raw_events, replay_speed=args.speed)
    await agent.run_stream(simulator.stream_events())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[Main] Terminated by user.")
        sys.exit(0)
