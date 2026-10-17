"""Test scenarios verifying push-fold tactics, inference speed, timing delay,
Majsoul bridge, instant pass filtering, and PC Majsoul mouse control mechanics.
"""

import pytest
import time
import yaml
import os
import sys

# Ensure mahjong_ai_agent root is in sys.path
AGENT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if AGENT_ROOT not in sys.path:
    sys.path.insert(0, AGENT_ROOT)

from core.state_manager import MahjongStateManager, normalize_tile
from core.engine import DecisionEngine
from core.strategy_modifier import StrategyModifier
from io_adapter.input_simulator import InputSimulator
from io_adapter.action_controller import ActionController, ScreenPoint
from io_adapter.majsoul_bridge import MajsoulBridge, liqi_tile_to_mjai, mjai_tile_to_liqi
from io_adapter.stream_client import MjaiStreamHandler, is_my_decision_required


@pytest.fixture
def test_config():
    cfg_path = os.path.join(AGENT_ROOT, "config", "strategy_config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_inference_latency_under_200ms(test_config):
    """Verify that inference completes within 200ms target."""
    engine = DecisionEngine(use_fallback=True, max_inference_time_ms=200)
    state_mgr = MahjongStateManager(my_actor_id=0)

    events = InputSimulator.create_mock_ooras_scenario(my_id=0, my_score=38000, enemy_riichi=True)
    for ev in events:
        state_mgr.update_event(ev)

    t0 = time.perf_counter()
    candidates = engine.evaluate(state_mgr)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    print(f"\nInference completed in {elapsed_ms:.3f}ms")
    assert elapsed_ms <= 200.0, f"Inference took {elapsed_ms:.2f}ms, exceeding 200ms limit!"
    assert len(candidates) > 0


def test_ooras_1st_place_folds_against_riichi(test_config):
    """
    Scenario: Ooras (South 4), Self is 1st place (38,000 pts).
    Player 1 (2nd place) declares Riichi.
    Self draws a dangerous tile (5s).
    Player 1 has previously discarded 1m (Genbutsu safe tile).
    Result expected: Self MUST choose the safe Genbutsu tile (1m) instead of dangerous tile (5s).
    """
    engine = DecisionEngine(use_fallback=True)
    modifier = StrategyModifier(test_config)
    state_mgr = MahjongStateManager(my_actor_id=0)

    events = InputSimulator.create_mock_ooras_scenario(my_id=0, my_score=38000, enemy_riichi=True)
    for ev in events:
        state_mgr.update_event(ev)

    assert state_mgr.my_player.rank == 1, "Self should be 1st place"
    assert state_mgr.round_state.is_ooras(), "Should be Ooras"
    assert len(state_mgr.get_active_riichi_actors()) == 1, "Player 1 should be riichi"

    # Evaluate raw candidates
    candidates = engine.evaluate(state_mgr)

    # Apply tactical strategy modification
    modified = modifier.modify_candidates(candidates, state_mgr)
    best_action = modified[0]

    print("\n--- Ooras 1st Place Candidates ---")
    for c in modified[:5]:
        print(f"Tile: {c.tile} | Mod Q: {c.modified_q_value} | Raw Q: {c.raw_q_value} | Danger: {c.danger_score}")

    assert normalize_tile(best_action.tile) == "1m", f"Expected safe tile 1m, got {best_action.tile}"
    assert best_action.danger_score == 0.0, "Chosen tile must be safe (Genbutsu)"


def test_ooras_4th_place_pushes_aggressively(test_config):
    """
    Scenario: Ooras (South 4), Self is 4th place (8,000 pts).
    Enemy declares Riichi, but 4th place needs points to avoid loss.
    Result expected: Attack scale increases, defense penalty is attenuated.
    """
    engine = DecisionEngine(use_fallback=True)
    modifier = StrategyModifier(test_config)
    state_mgr = MahjongStateManager(my_actor_id=0)

    # In this scenario, my score is 8,000 pts (clearly 4th place)
    events = InputSimulator.create_mock_ooras_scenario(my_id=0, my_score=8000, enemy_riichi=True)
    for ev in events:
        state_mgr.update_event(ev)

    assert state_mgr.my_player.rank == 4, "Self should be 4th place"

    candidates = engine.evaluate(state_mgr)
    modified_4th = modifier.modify_candidates(candidates, state_mgr)

    # Compare with 1st place modifier output
    state_mgr.my_player.score = 45000
    state_mgr._update_ranks()
    modified_1st = modifier.modify_candidates(candidates, state_mgr)

    atk_scale_4th = modified_4th[0].metadata["strategy_context"]["attack_scale"]
    atk_scale_1st = modified_1st[0].metadata["strategy_context"]["attack_scale"]
    assert atk_scale_4th > atk_scale_1st, f"4th place attack ({atk_scale_4th}) should exceed 1st place ({atk_scale_1st})"


def test_action_controller_timing_and_coordinates(test_config):
    """Verify realistic timing distribution and screen coordinate generation."""
    controller = ActionController(test_config)
    state_mgr = MahjongStateManager(my_actor_id=0)
    events = InputSimulator.create_mock_ooras_scenario(my_id=0, my_score=35000, enemy_riichi=False)
    for ev in events:
        state_mgr.update_event(ev)

    engine = DecisionEngine(use_fallback=True)
    candidates = engine.evaluate(state_mgr)
    c = candidates[0]

    comp_time_ms = 15.0
    delay = controller.calculate_thinking_delay(c, state_mgr, comp_time_ms)
    print(f"\nCalculated realistic human delay: {delay:.2f}s")
    assert 0.8 <= delay <= 3.2, f"Thinking delay {delay}s out of expected bounds"

    slot_idx, coords = controller.calculate_screen_coordinates(c, state_mgr)
    assert coords is not None
    assert 0 <= coords.x <= 1920
    assert 0 <= coords.y <= 1080


# ==============================================================================
# Phase 2 New Tests: Majsoul Bridge, Instant Pass, and PC Mouse Control
# ==============================================================================

def test_majsoul_tile_conversion_and_bridge():
    """
    Test 1: Verify Majsoul (Liqi) tile notation and action conversion to standard mjai format.
    - Akadora: 0m <-> 5mr, 0p <-> 5pr, 0s <-> 5sr
    - Honors: 1z <-> E, 2z <-> S, 3z <-> W, 4z <-> N, 5z <-> P, 6z <-> F, 7z <-> C
    - Standard suits: 1m <-> 1m, 9s <-> 9s
    """
    # 1. Tile conversions
    assert liqi_tile_to_mjai("0m") == "5mr"
    assert liqi_tile_to_mjai("0p") == "5pr"
    assert liqi_tile_to_mjai("0s") == "5sr"
    assert liqi_tile_to_mjai("1z") == "E"
    assert liqi_tile_to_mjai("2z") == "S"
    assert liqi_tile_to_mjai("5z") == "P"
    assert liqi_tile_to_mjai("7z") == "C"
    assert liqi_tile_to_mjai("3p") == "3p"

    # Reverse conversions
    assert mjai_tile_to_liqi("5mr") == "0m"
    assert mjai_tile_to_liqi("5pr") == "0p"
    assert mjai_tile_to_liqi("5sr") == "0s"
    assert mjai_tile_to_liqi("E") == "1z"
    assert mjai_tile_to_liqi("P") == "5z"
    assert mjai_tile_to_liqi("C") == "7z"
    assert mjai_tile_to_liqi("9s") == "9s"

    # 2. Bridge Action translation test
    bridge = MajsoulBridge(my_seat=0)
    new_round_events = bridge.translate_action("ActionNewRound", {
        "chang": 1,  # South
        "ju": 3,     # Round 4 (S4)
        "ben": 1,
        "liqibang": 1000,
        "scores": [35000, 25000, 22000, 18000],
        "doras": ["9p"],
        "tiles0": ["0m", "1m", "2m", "1z", "7z"]
    })

    assert len(new_round_events) == 1
    ev = new_round_events[0]
    assert ev["type"] == "start_kyoku"
    assert ev["bakaze"] == "S"
    assert ev["kyoku"] == 4
    assert ev["honba"] == 1
    assert ev["tehais"][0] == ["5mr", "1m", "2m", "E", "C"]


@pytest.mark.asyncio
async def test_stream_client_instant_pass_on_enemy_turns(test_config):
    """
    Test 2: Stream client instant pass verification across a full 1-turn cycle (4 players).
    For non-self actions that self cannot call, thinking latency MUST be 0s (completed in < 0.005s).
    Only on self turn (Player 0 tsumo) should realistic human latency be applied.
    """
    handler = MjaiStreamHandler(test_config, my_actor_id=0, apply_sleep=True)

    # Initialize a clean round
    start_event = {
        "type": "start_kyoku",
        "bakaze": "E",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 0,
        "oya": 0,
        "scores": [25000, 25000, 25000, 25000],
        "tehais": [
            ["1m", "2m", "3m", "7p", "8p", "9p", "2s", "3s", "4s", "5m", "1z", "1z", "9s"],
            [], [], []
        ],
        "dora_marker": "9p"
    }
    await handler.process_event(start_event)

    # Player 1 turn: Tsumo & Discard (Self cannot call 9m)
    p1_tsumo = {"type": "tsumo", "actor": 1, "pai": "3p"}
    resp, elapsed = await handler.process_event(p1_tsumo)
    assert resp is None
    assert elapsed < 0.05, f"Enemy tsumo took {elapsed:.4f}s, must be instant (< 50ms)!"

    p1_dahai = {"type": "dahai", "actor": 1, "pai": "9m", "tsumogiri": True}
    resp, elapsed = await handler.process_event(p1_dahai)
    # Self has no 9m pair/run, so it's an uncallable discard -> instant pass
    assert resp == {"type": "none"}
    assert elapsed < 0.05, f"Uncallable enemy discard took {elapsed:.4f}s, must be instant (< 50ms)!"

    # Player 2 turn: Tsumo & Discard
    p2_tsumo = {"type": "tsumo", "actor": 2, "pai": "4p"}
    _, elapsed = await handler.process_event(p2_tsumo)
    assert elapsed < 0.05

    p2_dahai = {"type": "dahai", "actor": 2, "pai": "8m", "tsumogiri": True}
    resp, elapsed = await handler.process_event(p2_dahai)
    assert resp == {"type": "none"}
    assert elapsed < 0.05

    # Player 3 turn: Tsumo & Discard
    p3_tsumo = {"type": "tsumo", "actor": 3, "pai": "5p"}
    _, elapsed = await handler.process_event(p3_tsumo)
    assert elapsed < 0.05

    p3_dahai = {"type": "dahai", "actor": 3, "pai": "7m", "tsumogiri": True}
    resp, elapsed = await handler.process_event(p3_dahai)
    assert resp == {"type": "none"}
    assert elapsed < 0.05

    # Player 0 turn: SELF TSUMO! Must trigger human thinking delay (>= 1.0s)
    # Using handler with apply_sleep=False just to test calculation or True for sleep
    p0_tsumo = {"type": "tsumo", "actor": 0, "pai": "5s"}
    # Check guard function directly first
    assert is_my_decision_required(p0_tsumo, handler.state_mgr, handler.engine) is True

    # Now execute with sleep enabled
    t_start = time.perf_counter()
    resp, elapsed = await handler.process_event(p0_tsumo)
    assert resp is not None
    assert resp["type"] == "dahai"
    assert resp["actor"] == 0
    # Verified human latency was applied
    print(f"\nSelf turn total thinking latency elapsed: {elapsed:.2f}s")
    assert elapsed >= 0.95, f"Self turn should take >= 1.0s thinking delay, took {elapsed:.2f}s"


def test_pc_majsoul_mouse_sequence_generation(test_config):
    """
    Test 3: Verify the 4-step PC Majsoul mouse interaction sequence:
    1. Bezier Approach + Jitter (+-5~8px)
    2. Hover Delay (0.08s - 0.18s) for tile slide-up popup
    3. Single Click (MouseDown -> Hold 0.04-0.09s -> MouseUp)
    4. Safe Neutral Rest Retreat (0.2s - 0.5s)
    """
    controller = ActionController(test_config)
    target_point = ScreenPoint(x=850, y=965)  # Slot 5 hand tile coordinate

    steps = controller.plan_mouse_interaction_sequence(
        target_point=target_point,
        is_riichi_sequence=False
    )

    print(f"\nGenerated {len(steps)} PC Majsoul mouse steps:")
    for s in steps:
        print(f"  [{s.step_type:<12}] target: ({s.target_point.x:>4}, {s.target_point.y:>4}) dur: {s.duration_sec:.3f}s | {s.note}")

    assert len(steps) == 5, f"Expected 5 mouse steps (approach, hover, down, up, retreat), got {len(steps)}"

    # 1. Step 1: Bezier Approach
    step_approach = steps[0]
    assert step_approach.step_type == "approach"
    assert step_approach.trajectory_sample_count > 0
    # Verify coordinate jitter (+- 8px)
    assert abs(step_approach.target_point.x - target_point.x) <= 8
    assert abs(step_approach.target_point.y - target_point.y) <= 8

    # 2. Step 2: Hover Delay (0.08s - 0.18s)
    step_hover = steps[1]
    assert step_hover.step_type == "hover"
    assert 0.075 <= step_hover.duration_sec <= 0.185, f"Hover duration {step_hover.duration_sec}s outside [0.08, 0.18]"

    # 3. Step 3: Single Click (MouseDown -> MouseUp with 0.04-0.09s hold)
    step_down = steps[2]
    step_up = steps[3]
    assert step_down.step_type == "mouse_down"
    assert step_up.step_type == "mouse_up"
    assert 0.035 <= step_up.duration_sec <= 0.095, f"Click hold duration {step_up.duration_sec}s outside [0.04, 0.09]"

    # 4. Step 4: Neutral Rest Retreat (0.2s - 0.5s)
    step_retreat = steps[4]
    assert step_retreat.step_type == "rest_retreat"
    assert 0.19 <= step_retreat.duration_sec <= 0.51, f"Retreat duration {step_retreat.duration_sec}s outside [0.2, 0.5]"

    # Rest position must be in safe area (x: 1400..1750, y: 650..800) away from hand area
    assert 1400 <= step_retreat.target_point.x <= 1750
    assert 650 <= step_retreat.target_point.y <= 800


# ==============================================================================
# Phase 3 New Tests: Liqi Binary Protocol Decoder & Window Client Calibration
# ==============================================================================

def test_liqi_proto_decoder_frame_parsing():
    """
    Test Phase 3: Verify decoding of raw Majsoul binary frames (0x01 Notify + ActionPrototype).
    """
    import struct
    from io_adapter.liqi_proto_decoder import LiqiProtoDecoder
    from io_adapter.majsoul_bridge import MajsoulBridge

    # Build a synthetic ActionDiscardTile frame:
    # 0x01 (Type Notify) + len(method) + method + protobuf(tag 1: seat=0, tag 2: tile="1m", tag 3: moqie=False)
    method_name = ".lq.ActionPrototype"
    method_bytes = method_name.encode("utf-8")

    # Protobuf for ActionDiscardTile:
    # field 1 (seat=0): key = (1 << 3) | 0 = 0x08, val = 0x00
    # field 2 (tile="1m"): key = (2 << 3) | 2 = 0x12, len = 2, val = b"1m"
    # field 3 (moqie=0): key = (3 << 3) | 0 = 0x18, val = 0x00
    sub_proto = bytes([0x08, 0x00, 0x12, 0x02, ord('1'), ord('m'), 0x18, 0x00])

    # Outer ActionPrototype:
    # field 1 (action_name="ActionDiscardTile"): key = 0x0a, len=17, "ActionDiscardTile"
    # field 2 (data): key = 0x12, len=len(sub_proto), sub_proto
    act_name = "ActionDiscardTile".encode("utf-8")
    outer_proto = bytes([0x0A, len(act_name)]) + act_name + bytes([0x12, len(sub_proto)]) + sub_proto

    # Final frame: 0x01 + 2-byte len + method_bytes + outer_proto
    frame = bytes([0x01]) + struct.pack("<H", len(method_bytes)) + method_bytes + outer_proto

    decoded = LiqiProtoDecoder.decode_frame(frame)
    assert decoded is not None
    action_name, action_data = decoded
    assert action_name == "ActionPrototype"
    assert action_data.get("action_name") == "ActionDiscardTile"
    assert action_data.get("tile") == "1m"
    assert action_data.get("seat") == 0

    # Test bridging to mjai
    bridge = MajsoulBridge(my_seat=0)
    mjai_events = bridge.translate_action("ActionDiscardTile", action_data)
    assert len(mjai_events) == 1
    assert mjai_events[0]["type"] == "dahai"
    assert mjai_events[0]["pai"] == "1m"
    assert mjai_events[0]["actor"] == 0


def test_client_area_calibration_scaling():
    """
    Test Phase 3: Verify client rectangle scaling and 16:9 aspect fit calculation.
    """
    from tools.calibrate_ui import calculate_screen_mapping_from_rect

    # Test 1: Exact 1920x1080 at offset (100, 50)
    map_1080 = calculate_screen_mapping_from_rect(x0=100, y0=50, w=1920, h=1080)
    viewport = map_1080["client_viewport"]
    assert viewport["x"] == 100
    assert viewport["y"] == 50
    assert viewport["width"] == 1920
    assert viewport["height"] == 1080
    assert viewport["scale_x"] == 1.0
    assert viewport["scale_y"] == 1.0

    # Hand slot start_x should be 100 + 430 = 530
    assert map_1080["hand_area"]["start_x"] == 530
    assert map_1080["hand_area"]["y"] == 1030

    # Test 2: Scaled down 1280x720 window
    map_720 = calculate_screen_mapping_from_rect(x0=0, y0=0, w=1280, h=720)
    scale_x = 1280.0 / 1920.0  # ~0.6667
    assert map_720["client_viewport"]["scale_x"] == round(scale_x, 4)
    expected_start_x = int(430 * scale_x)
    assert map_720["hand_area"]["start_x"] == expected_start_x

    # Test 3: Letterbox / Pillarbox aspect fit (e.g. wide 2000x1000)
    map_wide = calculate_screen_mapping_from_rect(x0=50, y0=50, w=2000, h=1000)
    # Render viewport must maintain 16:9 -> height=1000, width=int(1000*16/9) = 1777
    vp_wide = map_wide["client_viewport"]
    assert vp_wide["height"] == 1000
    assert vp_wide["width"] == 1777
    assert vp_wide["x"] > 50  # Offset should be centered with pillarbox padding


def test_paifu_url_parsing_and_batch_conversion():
    """
    Test Paifu Tool: Verify URL parsing, download/conversion pipeline to mjai JSON.
    """
    import json
    from tools.paifu_to_mjai import parse_paifu_url, download_and_convert_paifu, create_mock_authentic_game_record

    # Test 1: Parse standard URL and Markdown formatted URL
    sample_url = "https://game.mahjongsoul.com/?paifu=261016-64fee530-a705-4f80-b5ab-23d9eb7c6b36_a804191042"
    uuid, seat = parse_paifu_url(sample_url)
    assert uuid == "261016-64fee530-a705-4f80-b5ab-23d9eb7c6b36"
    assert seat == 0

    # Ensure authentic test cache exists for offline testing
    cache_dir = os.path.join(AGENT_ROOT, "output", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{uuid}.json")
    if not os.path.exists(cache_path) or os.path.getsize(cache_path) == 0:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(create_mock_authentic_game_record(uuid), f, ensure_ascii=False, indent=2)

    md_url = "[https://game.mahjongsoul.com/?paifu=261016-64fee530-a705-4f80-b5ab-23d9eb7c6b36_a804191042](https://game.mahjongsoul.com/?paifu=261016-64fee530-a705-4f80-b5ab-23d9eb7c6b36_a804191042)"
    uuid_md, seat_md = parse_paifu_url(md_url)
    assert uuid_md == "261016-64fee530-a705-4f80-b5ab-23d9eb7c6b36"
    assert seat_md == 0

    # Test 2: Full download and conversion to mjai events
    out_file, events = download_and_convert_paifu(sample_url)
    assert os.path.exists(out_file)
    assert len(events) > 5

    # Check that events start with start_game and contain start_kyoku
    assert events[0]["type"] == "start_game"
    assert any(ev.get("type") == "start_kyoku" for ev in events)
    assert any(ev.get("type") == "dahai" for ev in events)
    assert events[-1]["type"] == "end_game"


# ==============================================================================
# Phase 4 New Tests: Semi-Auto Mode & Opponent Profiling / Exploit Adaptation
# ==============================================================================

@pytest.mark.asyncio
async def test_semi_auto_mode_hover_and_wait_sequence(test_config):
    """
    Test Phase 4: Verify semi_auto assist mode.
    Sequence MUST move cursor to tile, trigger hover popup, and pause at semi_auto_wait
    WITHOUT executing auto-clicks or retreating to rest position.
    """
    cfg = dict(test_config)
    cfg["output_mode"] = "semi_auto"
    cfg["semi_auto"] = {"hover_timeout_sec": 0.1, "auto_click_on_timeout": False}

    controller = ActionController(cfg)
    assert controller.output_mode == "semi_auto"

    target_pt = ScreenPoint(x=700, y=965)
    steps = controller.plan_mouse_interaction_sequence(target_pt, is_semi_auto=True)

    print(f"\nSemi-Auto Planned Steps ({len(steps)} steps):")
    for s in steps:
        print(f"  [{s.step_type:<15}] -> ({s.target_point.x:>4}, {s.target_point.y:>4}) {s.note}")

    assert len(steps) == 3, f"Expected exactly 3 steps in semi_auto, got {len(steps)}"
    assert steps[0].step_type == "approach"
    assert steps[1].step_type == "hover"
    assert steps[2].step_type == "semi_auto_wait"

    # Verify no auto-click or retreat is planned
    assert not any(s.step_type in ("mouse_down", "mouse_up", "rest_retreat") for s in steps)

    # Test via execute_action
    state_mgr = MahjongStateManager(my_actor_id=0)
    events = InputSimulator.create_mock_ooras_scenario(my_id=0, my_score=35000, enemy_riichi=False)
    for ev in events:
        state_mgr.update_event(ev)

    engine = DecisionEngine(use_fallback=True)
    cands = engine.evaluate(state_mgr)

    output = await controller.execute_action(
        candidate=cands[0],
        state_mgr=state_mgr,
        comp_time_ms=5.0,
        apply_sleep=False
    )

    assert any(s.step_type == "semi_auto_wait" for s in output.mouse_steps)


def test_opponent_profiling_and_exploit_adaptation(test_config):
    """
    Test Phase 4: Verify opponent profiling from game logs and dynamic exploit adjustments.
    - Aggressive Berserker: pushes non-safe tiles against enemy riichi -> triggers counter-attack exploit
    - Turtle Iron Wall: folds completely with 0 pushes -> triggers extreme respect & defense exploit
    """
    from core.opponent_profiler import OpponentProfiler

    # Synthetic game: Player 1 (Aggro) vs Player 2 (Turtle) vs Player 0 (Self)
    events = [
        {"type": "start_game", "names": ["Self", "AggroPlayer", "TurtlePlayer", "Player3"]},
        {
            "type": "start_kyoku",
            "bakaze": "E", "kyoku": 1, "honba": 0, "kyotaku": 0, "oya": 0,
            "scores": [25000, 25000, 25000, 25000], "tehais": [[], [], [], []], "dora_marker": "1p"
        },
        # Self declares Riichi discarding 1m
        {"type": "tsumo", "actor": 0, "pai": "1m"},
        {"type": "reach", "actor": 0},
        {"type": "dahai", "actor": 0, "pai": "1m", "tsumogiri": False},
        {"type": "reach_accepted", "actor": 0},

        # Turn 1:
        # AggroPlayer pushes dangerous un-safe 5s!
        {"type": "tsumo", "actor": 1, "pai": "5s"},
        {"type": "dahai", "actor": 1, "pai": "5s", "tsumogiri": True},
        # TurtlePlayer safely discards 1m (genbutsu)
        {"type": "tsumo", "actor": 2, "pai": "1m"},
        {"type": "dahai", "actor": 2, "pai": "1m", "tsumogiri": False},
        # P3 discards genbutsu 1m
        {"type": "tsumo", "actor": 3, "pai": "1m"},
        {"type": "dahai", "actor": 3, "pai": "1m", "tsumogiri": False},

        # Turn 2:
        # AggroPlayer pushes dangerous un-safe 4p!
        {"type": "tsumo", "actor": 1, "pai": "4p"},
        {"type": "dahai", "actor": 1, "pai": "4p", "tsumogiri": True},
        # TurtlePlayer discards another genbutsu 1m
        {"type": "tsumo", "actor": 2, "pai": "1m"},
        {"type": "dahai", "actor": 2, "pai": "1m", "tsumogiri": False},

        # Round ends with AggroPlayer dealing in to Self Ron
        {"type": "hora", "actor": 0, "target": 1, "pai": "4p"},
        {"type": "end_kyoku"},
        {"type": "end_game"}
    ]

    profiler = OpponentProfiler()
    profiler.analyze_game_events(events)

    prof_aggro = profiler.get_profile("AggroPlayer")
    prof_turtle = profiler.get_profile("TurtlePlayer")

    assert prof_aggro is not None
    assert prof_turtle is not None

    # Aggro pushed 2/2 times (100% aggressiveness)
    assert prof_aggro.aggressiveness == 1.0
    assert prof_aggro.archetype == "Berserker Pusher"
    assert prof_aggro.deal_in_count == 1

    # Turtle folded 2/2 times (0% aggressiveness)
    assert prof_turtle.aggressiveness == 0.0

    # Test Exploit Modifiers in StrategyModifier
    mod_aggro = profiler.get_exploit_modifiers("AggroPlayer")
    mod_turtle = profiler.get_exploit_modifiers("TurtlePlayer")

    # Berserker should encourage counter-attack (reduced danger penalty, boosted attack)
    assert mod_aggro["danger_penalty_mult"] < 1.0
    assert mod_aggro["attack_scale_mult"] > 1.0

    # Iron wall should demand heavy fold (increased danger penalty and defense scale)
    assert mod_turtle["defense_scale_mult"] > 1.0
    assert mod_turtle["danger_penalty_mult"] > 1.0


def test_cdp_target_selection_priorities_and_error_handling(monkeypatch):
    """
    Test LiveSniffer CDP target discovery logic:
    1. Prioritize page with '雀魂' over Stripe iframes and Omnibox popups.
    2. Fallback to URL matching 'mahjongsoul'.
    3. Raise ConnectionError when no suitable tab is available.
    """
    from io_adapter.live_sniffer import LiveSniffer
    import urllib.request
    import json
    import io

    sniffer = LiveSniffer(mode="cdp", cdp_port=9222)

    # Mock targets with mixed iframes, popups, and 雀魂 tab
    mock_targets = [
        {
            "id": "1",
            "type": "iframe",
            "title": "https://js.stripe.com/v3/m-outer.html",
            "url": "https://js.stripe.com/v3/m-outer.html",
            "webSocketDebuggerUrl": "ws://127.0.0.1:9222/devtools/page/1"
        },
        {
            "id": "2",
            "type": "browser_ui",
            "title": "Omnibox Popup",
            "url": "chrome://omnibox-popup.top-chrome/",
            "webSocketDebuggerUrl": "ws://127.0.0.1:9222/devtools/page/2"
        },
        {
            "id": "3",
            "type": "page",
            "title": "雀魂-じゃんたま-",
            "url": "https://game.mahjongsoul.com/index.html",
            "webSocketDebuggerUrl": "ws://127.0.0.1:9222/devtools/page/3"
        }
    ]

    class MockResponse:
        def __init__(self, data):
            self.data = data
        def read(self):
            return self.data
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    # Case 1: Priority 1 (雀魂 page tab selected despite iframes/popups)
    raw_json = json.dumps(mock_targets, ensure_ascii=False).encode("utf-8")
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=3.0: MockResponse(raw_json))

    ws_url = sniffer._get_cdp_target_ws_url()
    assert ws_url == "ws://127.0.0.1:9222/devtools/page/3"

    # Case 2: Priority 2 (No '雀魂' in title, but URL contains mahjongsoul)
    mock_targets_en = [
        mock_targets[0],
        mock_targets[1],
        {
            "id": "4",
            "type": "page",
            "title": "Anime Mahjong Game",
            "url": "https://game.mahjongsoul.com/index.html",
            "webSocketDebuggerUrl": "ws://127.0.0.1:9222/devtools/page/4"
        }
    ]
    raw_json_en = json.dumps(mock_targets_en, ensure_ascii=False).encode("utf-8")
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=3.0: MockResponse(raw_json_en))

    ws_url_en = sniffer._get_cdp_target_ws_url()
    assert ws_url_en == "ws://127.0.0.1:9222/devtools/page/4"

    # Case 3: No valid tab -> must raise ConnectionError with diagnostics
    mock_targets_none = [mock_targets[0], mock_targets[1]]
    raw_json_none = json.dumps(mock_targets_none, ensure_ascii=False).encode("utf-8")
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=3.0: MockResponse(raw_json_none))

    with pytest.raises(ConnectionError) as exc_info:
        sniffer._get_cdp_target_ws_url()
    assert "No suitable Majsoul game tab detected" in str(exc_info.value)


def test_furo_operation_extraction_and_mouse_hover(test_config):
    """
    Test: Verify full Furo (Call) operation pipeline:
    1. majsoul_bridge parses operation_list (Chi, Pon, Daiminkan, Ron) from ActionDiscardTile.
    2. engine evaluates Chi/Pon/Kan vs Pass on enemy discard.
    3. action_controller correctly maps buttons (chi->1200,850, pon->1300,850, kan->1400,850, ron->1500,850, pass->1600,850)
       and plans semi-auto hover guidance.
    """
    from io_adapter.majsoul_bridge import MajsoulBridge
    from io_adapter.stream_client import is_my_decision_required
    from core.state_manager import MahjongStateManager
    from core.engine import DecisionEngine, ActionCandidate
    from core.strategy_modifier import StrategyModifier
    from io_adapter.action_controller import ActionController

    # 1. Test majsoul_bridge operation_list extraction
    bridge = MajsoulBridge(my_seat=1)
    discard_data = {
        "seat": 0,
        "tile": "3m",
        "operation_list": [
            {"type": 2, "type_name": "chi", "combinations": ["1m|2m", "2m|4m"], "seat": 1},
            {"type": 3, "type_name": "pon", "combinations": ["3m|3m"], "seat": 1},
        ]
    }
    events = bridge.translate_action("ActionDiscardTile", discard_data)
    assert len(events) == 1
    assert events[0]["type"] == "dahai"
    assert "operation_list" in events[0]
    assert len(events[0]["operation_list"]) == 2
    assert events[0]["operation_list"][0]["type_name"] == "chi"
    assert events[0]["operation_list"][0]["combinations"] == [["1m", "2m"], ["2m", "4m"]]

    # 2. Test Engine Chi/Pon/Pass Evaluation
    state_mgr = MahjongStateManager(my_actor_id=1)
    state_mgr.update_event({
        "type": "start_kyoku", "bakaze": "E", "kyoku": 1, "honba": 0, "kyotaku": 0,
        "scores": [25000, 25000, 25000, 25000], "dora_marker": "9p",
        "tehais": [["?"]*13, ["1m", "2m", "4m", "5m", "7p", "8p", "9p", "1s", "2s", "3s", "E", "E", "E"], ["?"]*13, ["?"]*13]
    })
    # Kamicha (Player 0) discards 3m
    state_mgr.update_event(events[0])

    engine = DecisionEngine(use_fallback=True)
    assert is_my_decision_required(events[0], state_mgr, engine) is True

    candidates = engine.evaluate(state_mgr)
    cand_types = [c.action_type for c in candidates]
    assert "chi" in cand_types
    assert "none" in cand_types

    # 3. Test ActionController Coordinate Mapping for all furo buttons
    controller = ActionController(test_config)
    for act, expected_x, expected_y in [
        ("chi", 1200, 850),
        ("pon", 1300, 850),
        ("kan", 1400, 850),
        ("hora", 1500, 850),
        ("none", 1600, 850),
    ]:
        cand = ActionCandidate(action_type=act, tile="3m", raw_q_value=5.0, metadata={"target": 0})
        slot, coords = controller.calculate_screen_coordinates(cand, state_mgr)
        assert slot is None
        assert coords is not None
        assert coords.x == expected_x
        assert coords.y == expected_y

    # 4. Verify semi_auto hover sequence on button
    chi_cand = [c for c in candidates if c.action_type == "chi"][0]
    slot, coords = controller.calculate_screen_coordinates(chi_cand, state_mgr)
    steps = controller.plan_mouse_interaction_sequence(coords, is_semi_auto=True)
    assert steps[-1].step_type == "semi_auto_wait"
    assert abs(steps[-1].target_point.x - 1200) <= 8
    assert abs(steps[-1].target_point.y - 850) <= 8


def test_nested_action_prototype_sub_bytes_decode_offset():
    """
    Test regression: ActionPrototype sub_bytes must preserve raw bytes without premature
    UTF-8/latin-1 string conversion causing offset shifts (e.g. tag 1859 or 2034).
    """
    from io_adapter.liqi_proto_decoder import LiqiProtoDecoder
    # Construct ActionDealTile wrapped in ActionPrototype
    deal_tile_bytes = bytes([0x12, 0x02, ord('2'), ord('p'), 0x18, 69])
    ap_bytes = bytes([0x08, 12, 0x12, 14]) + b'ActionDealTile' + bytes([0x1a, len(deal_tile_bytes)]) + deal_tile_bytes
    frame = bytes([0x01, 0x0a, 19]) + b'.lq.ActionPrototype' + bytes([0x12, len(ap_bytes)]) + ap_bytes

    action_name, action_data = LiqiProtoDecoder.decode_frame(frame)
    assert action_name == "ActionPrototype"
    assert action_data.get("action_name") == "ActionDealTile"
    assert action_data.get("tile") == "2p"
    assert action_data.get("seat") == 0
    raw_sub = action_data.get("raw_sub_fields", {})
    # Verify no corrupted gigantic tags like 1859 or 2034 exist
    assert 1859 not in raw_sub
    assert 2034 not in raw_sub
    assert 2 in raw_sub  # tile field


@pytest.mark.asyncio
async def test_text_only_hud_output_mode_and_callbacks(test_config):
    """
    Test Phase 6: Verify text_only output mode.
    - Completely bypasses mouse controls and coordinate planning.
    - Fires on_recommendation_callback with formatted Japanese tiles and metrics.
    """
    from core.engine import ActionCandidate

    cfg = dict(test_config)
    cfg["output_mode"] = "text_only"
    controller = ActionController(cfg)
    assert controller.output_mode == "text_only"

    captured_recs = []
    controller.on_recommendation_callback = lambda rec: captured_recs.append(rec)

    # 1. Dahai recommendation
    cand_dahai = ActionCandidate(
        action_type="dahai",
        tile="8p",
        raw_q_value=2.45,
        modified_q_value=2.45,
        shanten_after=1,
        metadata={"shanten": 1, "top_candidates": [{"tile": "8p", "expected_value": 2.45}]}
    )
    state_mgr = MahjongStateManager(my_actor_id=0)
    state_mgr.update_event({"type": "tsumo", "actor": 0, "pai": "8p"})

    result = await controller.execute_action(cand_dahai, state_mgr, comp_time_ms=50.0)
    assert result.output_mode == "text_only"
    assert len(result.mouse_steps) == 0
    assert result.screen_coordinates is None
    assert len(captured_recs) == 1
    assert captured_recs[0]["type"] == "dahai"
    assert captured_recs[0]["target"] == "8p"
    assert captured_recs[0]["shanten"] == 1

    # 2. Furo (pon) recommendation
    cand_pon = ActionCandidate(
        action_type="pon",
        tile="7m",
        raw_q_value=3.10,
        metadata={"target": 1}
    )
    result_pon = await controller.execute_action(cand_pon, state_mgr, comp_time_ms=30.0)
    assert result_pon.output_mode == "text_only"
    assert len(result_pon.mouse_steps) == 0
    assert len(captured_recs) == 2
    assert captured_recs[1]["type"] == "pon"
    assert captured_recs[1]["target"] == "7m"


def test_gui_chrome_discovery_and_app_initialization():
    """
    Test Phase 6: Verify gui.py Chrome discovery and MahjongHUDApp initialization.
    """
    import gui
    import tkinter as tk

    # Verify Chrome path lookup returns a valid string or None without crashing
    chrome_path = gui.find_chrome_executable()
    if chrome_path:
        assert os.path.isfile(chrome_path)

    # Verify Tkinter HUD app initializes properly
    root = tk.Tk()
    try:
        app = gui.MahjongHUDApp(root)
        assert app.is_topmost.get() is True
        assert app.output_mode_var.get() == "text_only"
        assert app.lbl_rec_main["text"] != ""

        # 1. Check recommendation update
        app._on_recommendation({
            "type": "dahai",
            "target": "8p",
            "score": 2.4,
            "shanten": 1,
            "top_candidates": [{"tile": "8p", "expected_value": 2.4}]
        })
        root.update()
        assert "8筒" in app.lbl_rec_main["text"]

        # 2. Check game status sync
        app._on_agent_status({
            "round_info": "東1局 0本場",
            "seat_name": "南家",
            "event_type": "start_kyoku"
        })
        root.update()
        assert "東1局 0本場" in app.lbl_round_info["text"]
        assert "南家" in app.lbl_round_info["text"]

        # 3. Check stdout redirect to txt_log
        print("[TEST_LOG] Live CDP frame received")
        app._poll_log_queue()
        root.update()
        log_text = app.txt_log.get("1.0", tk.END)
        assert "[TEST_LOG] Live CDP frame received" in log_text

        # 4. Check exploit mode toggle and modifier integration
        app.active_custom_exploit = {
            "defense_scale_mult": 0.80,
            "danger_penalty_mult": 0.75,
            "attack_scale_mult": 1.30,
            "exploit_note": "Berserker Pusher counter-attack"
        }
        app.is_exploit_mode.set(True)
        assert app.is_exploit_mode.get() is True
    finally:
        app.close()


def test_action_new_round_binary_decoding_and_dealer_first_turn_recommendation():
    """
    Test Phase 7: Verify ActionNewRound Protobuf & binary salvage for dealer initial deal.
    1. Extracts 13-14 tiles from ActionNewRound payload.
    2. Deep Tile Scanner salvages hand even if Protobuf offsets are corrupted.
    3. MajsoulBridge correctly populates self hand and generates initial tsumo for dealer.
    """
    from io_adapter.liqi_proto_decoder import LiqiProtoDecoder
    from io_adapter.majsoul_bridge import MajsoulBridge

    tiles_14 = ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p", "5p"]
    # 1. Standard ActionNewRound protobuf construction
    tiles_bytes = b"".join(bytes([0x22, len(t)]) + t.encode('ascii') for t in tiles_14)
    # tag 1: chang=0, tag 2: ju=0, tag 3: ben=0, tag 4: tiles, tag 5: dora="1s"
    nr_payload = bytes([0x08, 0, 0x10, 0, 0x18, 0]) + tiles_bytes + bytes([0x2a, 2]) + b"1s"
    ap_payload = bytes([0x08, 1, 0x12, len("ActionNewRound")]) + b"ActionNewRound" + bytes([0x1a, len(nr_payload)]) + nr_payload
    frame = bytes([0x01, 0x0a, len(".lq.ActionPrototype")]) + b".lq.ActionPrototype" + bytes([0x12, len(ap_payload)]) + ap_payload

    action_name, action_data = LiqiProtoDecoder.decode_frame(frame)
    assert action_name == "ActionPrototype"
    assert action_data.get("action_name") == "ActionNewRound"
    assert len(action_data.get("tiles", [])) == 14
    assert action_data["tiles"][0] == "1m"
    assert action_data.get("doras") == ["1s"]

    # 2. Corrupted Protobuf wire format (with offset shift like 608558417870476624)
    # Must be rescued by Deep Tile Scanner
    corrupted_nr_payload = bytes([0x79, 0x50, 0x55, 0x3e, 0x22, 0xa8, 0x08, 0x72, 0x08]) + tiles_bytes
    corrupted_ap = bytes([0x08, 1, 0x12, len("ActionNewRound")]) + b"ActionNewRound" + bytes([0x1a, len(corrupted_nr_payload)]) + corrupted_nr_payload
    corrupted_frame = bytes([0x01, 0x0a, len(".lq.ActionPrototype")]) + b".lq.ActionPrototype" + bytes([0x12, len(corrupted_ap)]) + corrupted_ap

    _, salvaged_data = LiqiProtoDecoder.decode_frame(corrupted_frame)
    assert salvaged_data.get("action_name") == "ActionNewRound"
    assert len(salvaged_data.get("tiles", [])) >= 13
    assert salvaged_data["tiles"][:3] == ["1m", "2m", "3m"]

    # 3. MajsoulBridge translation for Dealer (oya=0, my_seat=0)
    bridge = MajsoulBridge(my_seat=0)
    events = bridge.translate_action("ActionNewRound", action_data)
    # Must produce start_kyoku followed by dealer's 14th tile tsumo
    assert len(events) == 2
    assert events[0]["type"] == "start_kyoku"
    assert events[0]["oya"] == 0
    assert len(events[0]["tehais"][0]) == 13
    assert events[1]["type"] == "tsumo"
    assert events[1]["actor"] == 0
    assert events[1]["pai"] == "5p"


def test_action_deal_tile_salvage_and_tsumo_resolution():
    """
    Test Phase 7: Verify ActionDealTile drawn tile recovery when Protobuf wire format is corrupted.
    Ensures '8p' is salvaged and correctly emitted as {'type': 'tsumo', 'actor': 0, 'pai': '8p'}.
    """
    from io_adapter.liqi_proto_decoder import LiqiProtoDecoder
    from io_adapter.majsoul_bridge import MajsoulBridge

    # Corrupted DealTile frame where sub_fields has offset issue but raw binary has '8p'
    deal_payload = bytes([0x79, 0x01, 0x02]) + b'8p' + bytes([0x18, 0])
    ap_payload = bytes([0x08, 1, 0x12, len("ActionDealTile")]) + b"ActionDealTile" + bytes([0x1a, len(deal_payload)]) + deal_payload
    frame = bytes([0x01, 0x0a, len(".lq.ActionPrototype")]) + b".lq.ActionPrototype" + bytes([0x12, len(ap_payload)]) + ap_payload

    _, action_data = LiqiProtoDecoder.decode_frame(frame)
    assert action_data.get("action_name") == "ActionDealTile"
    assert action_data.get("tile") == "8p"

    bridge = MajsoulBridge(my_seat=0)
    events = bridge.translate_action("ActionDealTile", action_data)
    assert len(events) == 1
    assert events[0]["type"] == "tsumo"
    assert events[0]["actor"] == 0
    assert events[0]["pai"] == "8p"
    assert events[0]["pai"] != "?"


def test_packet_parse_exception_safety():
    """
    Test Phase 8: Verify that unexpected or non-bytes packet inputs never raise
    'can't concat int to bytes' or cause uncaught exceptions.
    """
    from io_adapter.liqi_proto_decoder import LiqiProtoDecoder
    from io_adapter.majsoul_bridge import liqi_tile_to_mjai, MajsoulBridge

    # 1. Non-bytes/int payload to decode_frame
    assert LiqiProtoDecoder.decode_frame(12345) is None
    assert LiqiProtoDecoder.decode_frame(None) is None
    assert LiqiProtoDecoder.decode_frame(b"") is None
    assert LiqiProtoDecoder.decode_frame("not base64 string") is not None

    # 2. Non-bytes/int to scan_tiles_from_bytes
    assert LiqiProtoDecoder.scan_tiles_from_bytes(12345) == []
    assert LiqiProtoDecoder.scan_tiles_from_bytes(None) == []

    # 3. Non-string/bytes/int to liqi_tile_to_mjai
    assert liqi_tile_to_mjai(b"1m") == "1m"
    assert liqi_tile_to_mjai(123) == "123"
    assert liqi_tile_to_mjai(None) == ""

    # 4. ActionNewRound with non-list/int attributes (e.g. ju as bytes or string)
    bridge = MajsoulBridge(my_seat=0)
    events1 = bridge.translate_action("ActionNewRound", {"tiles": None, "doras": None, "chang": 0, "ju": 0})
    assert len(events1) >= 1
    assert events1[0]["type"] == "start_kyoku"

    # Specifically test ju as bytes like b'ActionMJStart' which caused "can't concat int to bytes"
    events2 = bridge.translate_action("ActionNewRound", {
        "chang": b"0",
        "ju": b"ActionMJStart",
        "ben": "1",
        "liqibang": None,
        "doras": [],
        "tiles": ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p"]
    })
    assert len(events2) >= 1
    assert events2[0]["kyoku"] == 1
    assert events2[0]["oya"] == 0


def test_vision_hand_fallback_on_empty_new_round(monkeypatch):
    """
    Test Phase 8: When ActionNewRound provides 0 hand tiles (doras: [] only),
    VisionHandReader should trigger and populate the player's 13-14 tiles.
    """
    from io_adapter.majsoul_bridge import MajsoulBridge
    from io_adapter.vision_hand_reader import VisionHandReader

    # Mock VisionHandReader.read_hand to return 14 recognized tiles for dealer
    mock_tiles = ["1m", "8m", "8p", "nan", "1s", "9s", "2p", "3p", "4p", "ton", "sha", "pei", "chun", "8p"]
    monkeypatch.setattr(VisionHandReader, "read_hand", lambda self: mock_tiles)

    bridge = MajsoulBridge(my_seat=0)
    empty_new_round_data = {
        "chang": 0,
        "ju": 0,
        "ben": 0,
        "liqibang": 0,
        "doras": [],
        "tiles": [],  # Empty packet
    }

    events = bridge.translate_action("ActionNewRound", empty_new_round_data)
    assert len(events) == 2
    assert events[0]["type"] == "start_kyoku"

    # Self hand (seat 0) should be populated with 13 vision tiles normalized to mjai
    self_hand = events[0]["tehais"][0]
    assert len(self_hand) == 13
    assert "1m" in self_hand
    assert "8p" in self_hand
    assert "S" in self_hand   # 'nan' normalized to 'S'
    assert "E" in self_hand   # 'ton' normalized to 'E'

    # Dealer 14th tile emitted as initial draw
    assert events[1]["type"] == "tsumo"
    assert events[1]["actor"] == 0
    assert events[1]["pai"] == "8p"


@pytest.mark.asyncio
async def test_agent_recommendation_with_vision_fallback_hand(monkeypatch):
    """
    Test Phase 8: Ensure MahjongAIAgent immediately outputs a concrete tile discard
    (never defaulting to '? (?)') when starting hand is populated via Vision Fallback.
    """
    import os
    import yaml
    from main import MahjongAIAgent
    from io_adapter.vision_hand_reader import VisionHandReader

    mock_tiles = ["1m", "8m", "8p", "S", "1s", "9s", "2p", "3p", "4p", "E", "W", "N", "C", "8p"]
    monkeypatch.setattr(VisionHandReader, "read_hand", lambda self: mock_tiles)

    cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "strategy_config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    agent = MahjongAIAgent(config=cfg, my_actor_id=0, apply_sleep=False)

    # 1. Feed start_kyoku with 13 tiles
    start_event = {
        "type": "start_kyoku",
        "bakaze": "E",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 0,
        "oya": 0,
        "scores": [25000, 25000, 25000, 25000],
        "tehais": [mock_tiles[:13], ["?"]*13, ["?"]*13, ["?"]*13],
        "dora_marker": "1p"
    }
    await agent.handle_event(start_event)

    # 2. Feed dealer's 14th tile tsumo
    tsumo_event = {
        "type": "tsumo",
        "actor": 0,
        "pai": "8p"
    }
    action = await agent.handle_event(tsumo_event)

    # 3. Action must be a dahai and tile must NOT be None or '?'
    assert action is not None
    assert action.action_type == "dahai"
    assert action.tile is not None
    assert action.tile != "?"
    # Recommended tile must be one of the player's real hand tiles
    assert action.tile in mock_tiles


def test_vision_hand_reader_slot_cropping_and_debug_images(monkeypatch, tmp_path):
    """
    Test Phase 8: Verify that VisionHandReader crops 14 tile slots and generates:
    - output/debug_window.png
    - output/debug_tile_0.png .. debug_tile_13.png
    from an arbitrary sized window (e.g. 960x1080 half-screen on the right).
    """
    from PIL import Image, ImageDraw
    from io_adapter.vision_hand_reader import VisionHandReader

    # Create synthetic 960x1080 mock game window (simulating browser on right half)
    w, h = 960, 1080
    mock_canvas = Image.new("RGB", (w, h), color=(30, 45, 60))
    draw = ImageDraw.Draw(mock_canvas)

    # Draw white tile faces in the hand slot area (Y: H*0.88 .. H*0.98)
    tile_y0 = int(h * 0.88)
    tile_y1 = int(h * 0.98)
    slot_pitch = 0.03906 * w
    tile_w = int(slot_pitch * 0.92)
    start_x0 = int(0.224 * w)

    for i in range(13):
        sx0 = int(start_x0 + i * slot_pitch)
        sx1 = sx0 + tile_w
        # Draw ivory tile
        draw.rectangle((sx0, tile_y0, sx1, tile_y1), fill=(245, 245, 235), outline=(180, 180, 170))
        # Draw red dot for pinzu/man
        draw.ellipse((sx0 + 5, tile_y0 + 10, sx1 - 5, tile_y0 + 30), fill=(220, 40, 40))

    # Draw 14th tile (tsumo)
    tsumo_x0 = int(0.7786 * w)
    draw.rectangle((tsumo_x0, tile_y0, tsumo_x0 + tile_w, tile_y1), fill=(245, 245, 235), outline=(180, 180, 170))

    # Mock capture_screen_or_window to return our synthetic right-side window
    reader = VisionHandReader(debug_save=True)
    reader.output_dir = str(tmp_path)
    reader.debug_save_dir = str(tmp_path / "captured_tiles")
    monkeypatch.setattr(reader, "capture_screen_or_window", lambda: (mock_canvas, {"x": 960, "y": 0, "width": w, "height": h}))

    tiles = reader.read_hand()

    # Verify tile count
    assert len(tiles) in (13, 14)

    # Verify required debug files were generated
    output_dir = reader.output_dir
    assert os.path.exists(os.path.join(output_dir, "debug_window.png"))
    for i in range(14):
        slot_file = os.path.join(output_dir, f"debug_tile_{i}.png")
        assert os.path.exists(slot_file), f"Missing {slot_file}"

    # Verify cropped tile image has valid dimensions
    tile_0_img = Image.open(os.path.join(output_dir, "debug_tile_0.png"))
    assert tile_0_img.width > 10
    assert tile_0_img.height > 20


def test_real_tile_template_discrimination():
    """
    Test Phase 8: Verify that real tiles from game capture (South, West, Hatsu, etc.)
    are precisely identified and never misclassified as North (Pei/N).
    """
    from PIL import Image
    from io_adapter.vision_hand_reader import VisionHandReader

    reader = VisionHandReader()

    # Verify that S (South), W (West), and F (Hatsu) match their exact names and NEVER 'N' (North)
    test_cases = [
        ("9m.png", "9m"),
        ("1p.png", "1p"),
        ("8p.png", "8p"),
        ("1s.png", "1s"),
        ("2s.png", "2s"),
        ("3s.png", "3s"),
        ("4s.png", "4s"),
        ("5s.png", "5s"),
        ("6s.png", "6s"),
        ("S.png", "S"),
        ("W.png", "W"),
        ("F.png", "F"),
    ]

    for fn, expected in test_cases:
        p = os.path.join(reader.template_dir, fn)
        if os.path.exists(p):
            img = Image.open(p)
            recognized = reader.recognize_tile(img)
            assert recognized == expected, f"{fn} was recognized as {recognized}, expected {expected}"
            assert recognized != "N", f"{fn} must never be misclassified as North (N)!"


def test_debug_tile_pngs_hand_recognition_and_discard_recommendation():
    """
    Test Phase 8: Verify that actual cropped debug tile files (output/debug_tile_0.png .. 13.png)
    are recognized with 100% precision without any North (N) false positives,
    and the DecisionEngine recommends a valid tile present in the player's hand.
    """
    from PIL import Image
    from io_adapter.vision_hand_reader import VisionHandReader
    from core.state_manager import MahjongStateManager
    from core.engine import DecisionEngine

    reader = VisionHandReader()
    expected_hand = ['9m', '1p', '8p', '8p', '1s', '2s', '3s', '4s', '5s', '6s', 'S', 'W', 'F', '1p']

    recognized_hand = []
    for i in range(14):
        p = os.path.join(reader.project_root, "assets", "reference_hand", f"debug_tile_{i}.png")
        if not os.path.exists(p):
            p = os.path.join(reader.output_dir, f"debug_tile_{i}.png")
        assert os.path.exists(p), f"Missing debug tile file: {p}"
        tile_img = Image.open(p)
        name = reader.recognize_tile(tile_img, slot_index=i)
        recognized_hand.append(name)

    # 1. Verify complete hand matches expectation
    assert recognized_hand == expected_hand, f"Recognized hand {recognized_hand} does not match {expected_hand}"

    # 2. Strict guarantee: North (N) must NEVER be in this hand
    assert "N" not in recognized_hand, "North (N) falsely detected in hand!"

    # 3. DecisionEngine must recommend a tile that genuinely exists in this hand
    state_mgr = MahjongStateManager(my_actor_id=0)
    state_mgr.update_event({
        'type': 'start_kyoku', 'bakaze': 'E', 'kyoku': 1, 'honba': 0, 'kyotaku': 0, 'oya': 0,
        'dora_marker': '1p', 'tehais': [recognized_hand[:13], ['?']*13, ['?']*13, ['?']*13]
    })
    state_mgr.update_event({'type': 'tsumo', 'actor': 0, 'pai': recognized_hand[13]})

    engine = DecisionEngine(use_fallback=True)
    cands = engine.evaluate(state_mgr)

    assert len(cands) > 0, "No decision candidates produced!"
    best_action = cands[0]
    assert best_action.tile in recognized_hand, f"Recommended tile {best_action.tile} not in player's real hand!"
    assert best_action.tile != "N", "Decision engine recommended North (N) which is not in hand!"
    assert best_action.tile != "?", "Decision engine produced unknown tile '?'!"


@pytest.mark.asyncio
async def test_live_vision_per_turn_sync_and_exclusion_of_2s(monkeypatch):
    """
    Test Phase 9: Verify that when hand is [5m, 6m, 7m, 8m, 2p, 6p, 6p, 8p, 3s, 3s, S, S, W]
    and packet tsumo is '?', VisionHandReader is invoked on each tsumo turn,
    2s is strictly excluded, and only real tiles in hand are recommended.
    """
    from PIL import Image
    from io_adapter.vision_hand_reader import VisionHandReader
    from main import MahjongAIAgent

    real_hand = ['5m', '6m', '7m', '8m', '2p', '6p', '6p', '8p', '3s', '3s', 'S', 'S', 'W', 'W']
    reader = VisionHandReader()

    # 1. Verify tile recognition of each tile in the real hand
    recognized = []
    for t in real_hand:
        p = os.path.join(reader.template_dir, f"{t}.png")
        assert os.path.exists(p), f"Missing template for {t}: {p}"
        tile_img = Image.open(p)
        rec = reader.recognize_tile(tile_img)
        recognized.append(rec)

    assert recognized == real_hand, f"Recognized {recognized} != {real_hand}"
    assert "2s" not in recognized, "2s falsely detected in hand!"

    # 2. Mock read_hand to return the recognized hand on live capture
    monkeypatch.setattr(VisionHandReader, "read_hand", lambda self: list(real_hand))

    agent = MahjongAIAgent(my_actor_id=0, apply_sleep=False, output_mode="text_only")

    # Start kyoku with empty/hidden hand
    await agent.handle_event({
        "type": "start_kyoku", "bakaze": "E", "kyoku": 1, "honba": 0, "kyotaku": 0, "oya": 1,
        "dora_marker": "1p", "tehais": [["?"]*13, ["?"]*13, ["?"]*13, ["?"]*13]
    })

    # P1 discards
    await agent.handle_event({
        "type": "dahai", "actor": 1, "pai": "1m", "tsumogiri": True
    })

    # P0 (self) tsumo with unknown tile '?'
    # Must trigger live vision sync, populate hand with real_hand, and recommend discard
    action = await agent.handle_event({
        "type": "tsumo", "actor": 0, "pai": "?"
    })

    assert action is not None, "Agent produced no action on self tsumo!"
    assert action.action_type == "dahai"
    assert action.tile in real_hand, f"Recommended tile {action.tile} not in real hand!"
    assert action.tile != "2s", "Agent recommended 2s which is NOT in hand!"
    assert action.tile != "?", "Agent produced unknown '?' tile!"


def test_liqi_xor_decryption_action_new_round_real_payload():
    """
    Test XOR Decryption and Protobuf parsing accuracy for ActionNewRound:
    1. Encodes ActionNewRound with 13 dealt tiles and dora.
    2. Encrypts payload with Majsoul XOR key [0x84, 0x5E, 0x4E, 0x42, 0x39, 0xA2, 0x1F, 0x60, 0x1C].
    3. Verifies raw encrypted bytes fail standard Protobuf wire decoding (returns {}).
    4. Passes through LiqiProtoDecoder.decode_frame:
       - XOR decryption unlocks the wire format.
       - Hand of 13 tiles is extracted 100% cleanly in one shot without falling back to salvage.
       - Action name, chang, ju, ben, and doras are fully expanded.
    """
    from io_adapter.liqi_proto_decoder import (
        LiqiProtoDecoder,
        decode_raw_protobuf,
        xor_decrypt_action_data,
        LIQI_ACTION_XOR_KEY,
    )
    from io_adapter.majsoul_bridge import MajsoulBridge

    expected_tiles = ["1m", "9m", "1p", "9p", "1s", "9s", "1z", "2z", "3z", "4z", "5z", "6z", "7z"]
    expected_dora = "5m"

    # 1. Build standard ActionNewRound protobuf payload
    # tag 1: chang=0, tag 2: ju=0, tag 3: ben=0
    # tag 4: tiles (repeated string)
    # tag 5: dora (string)
    tiles_bytes = b"".join(bytes([0x22, len(t)]) + t.encode("ascii") for t in expected_tiles)
    dora_bytes = bytes([0x2A, len(expected_dora)]) + expected_dora.encode("ascii")
    plain_new_round = bytes([0x08, 0, 0x10, 0, 0x18, 0]) + tiles_bytes + dora_bytes

    # 2. XOR Encrypt with Majsoul live key
    encrypted_sub_bytes = xor_decrypt_action_data(plain_new_round, LIQI_ACTION_XOR_KEY)
    sub_bytes_hex = encrypted_sub_bytes.hex()
    assert len(sub_bytes_hex) > 0

    # 3. Prove raw encrypted bytes cannot be decoded by schema-less protobuf directly
    raw_before_xor = decode_raw_protobuf(encrypted_sub_bytes)
    assert raw_before_xor == {}, f"Encrypted bytes should not decode cleanly without XOR: {raw_before_xor}"

    # 4. Wrap into ActionPrototype (tag 1: step=1, tag 2: "ActionNewRound", tag 3: sub_bytes)
    # and Notify Frame (0x01 + tag 1: ".lq.ActionPrototype" + tag 2: ap_payload)
    act_name = "ActionNewRound"
    ap_payload = (
        bytes([0x08, 1])
        + bytes([0x12, len(act_name)]) + act_name.encode("ascii")
        + bytes([0x1A, len(encrypted_sub_bytes)]) + encrypted_sub_bytes
    )
    method_name = ".lq.ActionPrototype"
    frame = (
        bytes([0x01, 0x0A, len(method_name)]) + method_name.encode("ascii")
        + bytes([0x12, len(ap_payload)]) + ap_payload
    )

    # 5. Decode frame through LiqiProtoDecoder
    action_name, action_data = LiqiProtoDecoder.decode_frame(frame)

    assert action_name == "ActionPrototype"
    assert action_data.get("action_name") == "ActionNewRound"

    # Verify raw_sub_fields is NOT empty {}
    assert "raw_sub_fields" in action_data
    assert len(action_data["raw_sub_fields"]) > 0
    assert 4 in action_data["raw_sub_fields"]  # tag 4 has tiles

    # Verify 13 tiles extracted cleanly
    extracted_tiles = action_data.get("tiles", [])
    assert extracted_tiles == expected_tiles, f"Extracted {extracted_tiles} != {expected_tiles}"
    assert action_data.get("doras") == [expected_dora]
    assert action_data.get("chang") == 0
    assert action_data.get("ju") == 0
    assert action_data.get("ben") == 0

    # 6. Verify MajsoulBridge translation into start_kyoku event
    from io_adapter.majsoul_bridge import liqi_tile_to_mjai
    bridge = MajsoulBridge(my_seat=0)
    events = bridge.translate_action("ActionNewRound", action_data)
    assert len(events) >= 1
    assert events[0]["type"] == "start_kyoku"
    expected_mjai_tiles = [liqi_tile_to_mjai(t) for t in expected_tiles]
    assert events[0]["tehais"][0] == expected_mjai_tiles
    assert events[0]["dora_marker"] == expected_dora


def test_liqi_xor_decryption_action_deal_tile_real_payload():
    """
    Test XOR Decryption and Protobuf parsing accuracy for ActionDealTile (tsumo):
    1. Encodes ActionDealTile with drawn tile '8p' and seat 0.
    2. Encrypts with Majsoul XOR key.
    3. Decodes frame and verifies tile '8p' and seat 0 are extracted in one shot.
    """
    from io_adapter.liqi_proto_decoder import LiqiProtoDecoder, xor_decrypt_action_data, LIQI_ACTION_XOR_KEY
    from io_adapter.majsoul_bridge import MajsoulBridge

    # Build ActionDealTile protobuf: tag 1: seat=0, tag 2: tile="8p", tag 3: moqie=0
    plain_deal = bytes([0x08, 0, 0x12, 2]) + b"8p" + bytes([0x18, 0])
    encrypted_deal = xor_decrypt_action_data(plain_deal, LIQI_ACTION_XOR_KEY)

    # ActionPrototype wrapper
    act_name = "ActionDealTile"
    ap_payload = (
        bytes([0x08, 2])
        + bytes([0x12, len(act_name)]) + act_name.encode("ascii")
        + bytes([0x1A, len(encrypted_deal)]) + encrypted_deal
    )
    method_name = ".lq.ActionPrototype"
    frame = (
        bytes([0x01, 0x0A, len(method_name)]) + method_name.encode("ascii")
        + bytes([0x12, len(ap_payload)]) + ap_payload
    )

    action_name, action_data = LiqiProtoDecoder.decode_frame(frame)
    assert action_name == "ActionPrototype"
    assert action_data.get("action_name") == "ActionDealTile"
    assert action_data.get("tile") == "8p"
    assert action_data.get("seat") == 0

    bridge = MajsoulBridge(my_seat=0)
    events = bridge.translate_action("ActionDealTile", action_data)
    assert len(events) == 1
    assert events[0]["type"] == "tsumo"
    assert events[0]["actor"] == 0
    assert events[0]["pai"] == "8p"


def test_nested_protobuf_strict_boundary_verification():
    """
    Test robust schema-less boundary decoding:
    1. Validates that partial/truncated varints raise or cleanly break without corrupting offsets.
    2. Validates nested protobuf messages (e.g. OptionalOperationList) are decoded accurately.
    3. Validates plain text strings are not falsely unpacked as nested messages.
    """
    from io_adapter.liqi_proto_decoder import (
        decode_raw_protobuf,
        is_valid_protobuf_submessage,
        decode_varint,
    )

    # 1. Truncated varint at buffer end must raise ValueError
    truncated_buf = bytes([0x80, 0x80])  # MSB set but buffer ends
    with pytest.raises(ValueError):
        decode_varint(truncated_buf, 0)

    # 2. String tokens must not be treated as nested protobuf messages
    assert not is_valid_protobuf_submessage(b"1m")
    assert not is_valid_protobuf_submessage(b"ActionNewRound")
    assert not is_valid_protobuf_submessage(b"ActionDiscardTile")
    assert not is_valid_protobuf_submessage(b".lq.ActionPrototype")

    # 3. Valid nested protobuf structure (tag 1: varint 2, tag 2: string "2m")
    nested_msg = bytes([0x08, 2, 0x12, 2]) + b"2m"
    assert is_valid_protobuf_submessage(nested_msg)

    # 4. Container message with nested payload (recursive=True)
    container = bytes([0x0A, len(nested_msg)]) + nested_msg
    decoded_plain = decode_raw_protobuf(container, recursive=False)
    assert 1 in decoded_plain
    assert decoded_plain[1][0] == nested_msg  # raw bytes preserved by default

    decoded_recursive = decode_raw_protobuf(container, recursive=True)
    assert 1 in decoded_recursive
    assert isinstance(decoded_recursive[1][0], dict)
    assert decoded_recursive[1][0].get(1) == [2]
    assert decoded_recursive[1][0].get(2) == [b"2m"]


def _build_action_prototype_frame(act_name: str, data: bytes, step: int = 5) -> bytes:
    """Helper: wrap data as live Notify(.lq.ActionPrototype{step=1,name=2,data=3})."""
    name_b = act_name.encode("ascii")
    ap = bytes([0x08, step, 0x12, len(name_b)]) + name_b + bytes([0x1A, len(data)]) + data
    method = b".lq.ActionPrototype"
    return bytes([0x01, 0x0A, len(method)]) + method + bytes([0x12, len(ap)]) + ap


def test_liqi_xor_real_live_deal_tile_sample_opponent_draw(monkeypatch):
    """
    Real live sample sub_bytes_hex='9d75712666cc' (previously decoded as raw: {}).
    Correct Liqi XOR -> 08 01 18 44 38 00 => seat=1, left_tile_count=68, no tile (opponent draw).
    Must NOT salvage a phantom tile and must NOT invoke vision.
    """
    from io_adapter.liqi_proto_decoder import LiqiProtoDecoder, xor_decrypt_action_data
    from io_adapter.majsoul_bridge import MajsoulBridge
    from io_adapter.vision_hand_reader import VisionHandReader

    live = bytes.fromhex("9d75712666cc")
    assert xor_decrypt_action_data(live).hex() == "080118443800"

    _, data = LiqiProtoDecoder.decode_frame(_build_action_prototype_frame("ActionDealTile", live))
    assert data["action_name"] == "ActionDealTile"
    assert data["xor_mode"] == "xor" and data["wire_ok"] is True
    assert data["raw_sub_fields"] == {1: [1], 3: [68], 7: [0]}
    assert data["seat"] == 1
    assert data["left_tile_count"] == 68
    assert "tile" not in data

    def _boom(self):
        raise AssertionError("Vision must not run for a valid opponent DealTile")
    monkeypatch.setattr(VisionHandReader, "read_hand", _boom)

    bridge = MajsoulBridge(my_seat=1)  # even if seat matches, valid packet => no vision
    events = bridge.translate_action("ActionDealTile", data)
    assert events == [{"type": "tsumo", "actor": 1, "pai": "?"}]


@pytest.mark.parametrize("seat,tile", [(0, "3s"), (2, "9s"), (3, "0p"), (1, "7z")])
def test_liqi_xor_self_deal_tile_extracts_real_tile(seat, tile):
    """Encrypted self ActionDealTile must yield the exact drawn tile in mapped['tile']."""
    from io_adapter.liqi_proto_decoder import LiqiProtoDecoder, xor_decrypt_action_data

    plain = (bytes([0x08, seat]) if seat else b"") + bytes([0x12, 2]) + tile.encode() + bytes([0x18, 55])
    enc = xor_decrypt_action_data(plain)
    _, data = LiqiProtoDecoder.decode_frame(_build_action_prototype_frame("ActionDealTile", enc))
    assert data["xor_mode"] == "xor"
    assert data["seat"] == seat
    assert data["tile"] == tile
    assert data["left_tile_count"] == 55


def test_liqi_xor_applies_to_discard_and_chipenggang():
    """XOR decryption is applied uniformly to every action type, with schema-exact mapping."""
    from io_adapter.liqi_proto_decoder import LiqiProtoDecoder, xor_decrypt_action_data
    from io_adapter.majsoul_bridge import MajsoulBridge

    # ActionDiscardTile: seat=2, tile='5m', is_liqi=1, moqie=1
    discard = bytes([0x08, 2, 0x12, 2]) + b"5m" + bytes([0x18, 1, 0x28, 1])
    _, d = LiqiProtoDecoder.decode_frame(
        _build_action_prototype_frame("ActionDiscardTile", xor_decrypt_action_data(discard)))
    assert d["xor_mode"] == "xor"
    assert (d["seat"], d["tile"], d["is_liqi"], d["moqie"]) == (2, "5m", True, True)

    # ActionChiPengGang: seat=3, type=1 (pon), tiles=['6p','6p','6p'], froms packed [3,3,1]
    tiles = b"".join(bytes([0x1A, 2]) + t for t in (b"6p", b"6p", b"6p"))
    cpg = bytes([0x08, 3, 0x10, 1]) + tiles + bytes([0x22, 3, 3, 3, 1])
    _, c = LiqiProtoDecoder.decode_frame(
        _build_action_prototype_frame("ActionChiPengGang", xor_decrypt_action_data(cpg)))
    assert c["xor_mode"] == "xor"
    assert c["seat"] == 3 and c["type"] == 1
    assert c["tiles"] == ["6p", "6p", "6p"]
    assert c["froms"] == [3, 3, 1]

    events = MajsoulBridge(my_seat=0).translate_action("ActionChiPengGang", c)
    assert events[0]["type"] == "pon" and events[0]["actor"] == 3 and events[0]["target"] == 1


@pytest.mark.asyncio
async def test_vision_locked_out_when_packet_hand_trusted(monkeypatch):
    """Once start_kyoku carries a real packet hand, vision must never overwrite it."""
    from main import MahjongAIAgent
    from io_adapter.vision_hand_reader import VisionHandReader

    calls = []
    monkeypatch.setattr(VisionHandReader, "read_hand", lambda self: calls.append(1) or ["C"] * 14)

    agent = MahjongAIAgent(my_actor_id=0, apply_sleep=False, output_mode="text_only")
    hand = ["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "E", "E", "P", "P"]
    await agent.handle_event({
        "type": "start_kyoku", "bakaze": "E", "kyoku": 1, "honba": 0, "kyotaku": 0, "oya": 1,
        "dora_marker": "1p", "tehais": [list(hand), ["?"] * 13, ["?"] * 13, ["?"] * 13],
    })
    assert agent.packet_hand_trusted is True

    await agent.handle_event({"type": "dahai", "actor": 1, "pai": "1m", "tsumogiri": True})
    # Self tsumo with a missing tile must NOT trigger vision when packet hand is trusted
    await agent.handle_event({"type": "tsumo", "actor": 0, "pai": "?"})
    assert calls == [], "Vision was invoked despite trusted packet hand"
    p = agent.state_mgr.round_state.players[0]
    assert "C" not in p.tehai, "Vision hallucination overwrote packet hand"


def test_action_new_round_packed_bytes_scores_conversion():
    """
    Test Phase 9: Verify ActionNewRound with packed bytes scores (e.g. b'\\xa8\\xc3\\x01...')
    safely decodes to List[int] without Pydantic int_parsing crash.
    """
    from io_adapter.liqi_proto_decoder import LiqiProtoDecoder
    from io_adapter.majsoul_bridge import MajsoulBridge
    from core.state_manager import MahjongStateManager

    # 25000 in Protobuf varint is: 25000 = 0x61a8 -> 0xa8, 0xc3, 0x01
    score_varint_bytes = bytes([0xA8, 0xC3, 0x01])  # 25000
    scores_packed_bytes = score_varint_bytes * 4      # 4 players with 25000

    # Build NewRound payload with tag 6 as packed bytes
    tiles = ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p"]
    tiles_bytes = b"".join(bytes([0x22, len(t)]) + t.encode("ascii") for t in tiles)
    nr_payload = (
        bytes([0x08, 0, 0x10, 0, 0x18, 0])
        + tiles_bytes
        + bytes([0x2A, 2]) + b"1s"
        + bytes([0x32, len(scores_packed_bytes)]) + scores_packed_bytes
    )

    ap_payload = bytes([0x08, 1, 0x12, len("ActionNewRound")]) + b"ActionNewRound" + bytes([0x1A, len(nr_payload)]) + nr_payload
    frame = bytes([0x01, 0x0A, len(".lq.ActionPrototype")]) + b".lq.ActionPrototype" + bytes([0x12, len(ap_payload)]) + ap_payload

    _, action_data = LiqiProtoDecoder.decode_frame(frame)
    assert action_data.get("action_name") == "ActionNewRound"
    scores = action_data.get("scores")
    assert scores == [25000, 25000, 25000, 25000]

    # Pass through bridge to generate mjai event
    bridge = MajsoulBridge(my_seat=0)
    events = bridge.translate_action("ActionNewRound", action_data)
    assert len(events) >= 1
    start_kyoku = events[0]
    assert start_kyoku["type"] == "start_kyoku"
    assert start_kyoku["scores"] == [25000, 25000, 25000, 25000]
    assert all(isinstance(s, int) for s in start_kyoku["scores"])

    # Update state_manager to verify PlayerState validation succeeds without int_parsing error
    state_mgr = MahjongStateManager(my_actor_id=0)
    state_mgr.update_event(start_kyoku)
    assert state_mgr.round_state is not None
    assert all(p.score == 25000 for p in state_mgr.round_state.players)
    assert all(isinstance(p.score, int) for p in state_mgr.round_state.players)


def test_action_new_round_malformed_bytes_scores_graceful_fallback():
    """
    Test Phase 9: If scores contain arbitrary malformed bytes or garbage,
    it gracefully falls back to [25000, 25000, 25000, 25000] instead of crashing.
    """
    from io_adapter.majsoul_bridge import MajsoulBridge
    from core.state_manager import MahjongStateManager

    raw_data = {
        "chang": b"\x00",
        "ju": b"\x00",
        "ben": b"\x00",
        "scores": [b"\xff\xff\xff", b"invalid", 30000, "20000"],
        "tiles": ["1m"] * 13,
        "doras": ["1p"]
    }

    bridge = MajsoulBridge(my_seat=0)
    events = bridge.translate_action("ActionNewRound", raw_data)
    start_kyoku = events[0]
    assert all(isinstance(s, int) for s in start_kyoku["scores"])

    state_mgr = MahjongStateManager(my_actor_id=0)
    # Must NOT raise pydantic ValidationError
    state_mgr.update_event(start_kyoku)
    assert state_mgr.round_state is not None
    assert all(isinstance(p.score, int) for p in state_mgr.round_state.players)


def test_action_new_round_metadata_numeric_strings_filtered_out():
    """
    Test Phase 10: Verify metadata like '69', '123', or arbitrary non-tile strings
    present in raw binary payloads or proto fields are strictly filtered out and never
    pollute tiles, doras, or player tehais.
    """
    from io_adapter.liqi_proto_decoder import LiqiProtoDecoder, is_valid_tile_code
    from io_adapter.majsoul_bridge import MajsoulBridge
    from core.state_manager import MahjongStateManager

    # 1. Direct validation check
    assert is_valid_tile_code("1m") is True
    assert is_valid_tile_code("0m") is True
    assert is_valid_tile_code("9s") is True
    assert is_valid_tile_code("7z") is True
    assert is_valid_tile_code("69") is False
    assert is_valid_tile_code("123") is False
    assert is_valid_tile_code("00") is False
    assert is_valid_tile_code("8z") is False
    assert is_valid_tile_code("") is False
    assert is_valid_tile_code(None) is False
    assert is_valid_tile_code(69) is False

    # 2. Build ActionNewRound proto payload where tag 4 (tiles) contains '69' alongside valid tiles
    # Tag 4: string tiles = repeated string
    valid_tiles = ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p"]
    dirty_tiles = ["1m", "69", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p"]
    
    nr_body = b""
    # tag 1: chang = 0
    nr_body += bytes([0x08, 0])
    # tag 2: ju = 0
    nr_body += bytes([0x10, 0])
    # tag 3: ben = 0
    nr_body += bytes([0x18, 0])
    # tag 4: repeated tiles including '69'
    for t in dirty_tiles:
        b_t = t.encode("utf-8")
        nr_body += bytes([0x22, len(b_t)]) + b_t
    # tag 5: dora = '1z'
    nr_body += bytes([0x2A, 2]) + b"1z"
    # tag 6: scores = packed [25000, 25000, 25000, 25000]
    packed_scores = bytes([0xA8, 0xC3, 0x01, 0xA8, 0xC3, 0x01, 0xA8, 0xC3, 0x01, 0xA8, 0xC3, 0x01])
    nr_body += bytes([0x32, len(packed_scores)]) + packed_scores

    # Wrap in ActionPrototype: step=1, name='ActionNewRound', data=nr_body
    ap_payload = bytes([0x08, 1, 0x12, len("ActionNewRound")]) + b"ActionNewRound" + bytes([0x1A, len(nr_body)]) + nr_body
    # Wrap in frame 0x01 .lq.ActionPrototype
    frame = bytes([0x01, 0x0A, len(".lq.ActionPrototype")]) + b".lq.ActionPrototype" + bytes([0x12, len(ap_payload)]) + ap_payload

    _, action_data = LiqiProtoDecoder.decode_frame(frame)
    assert action_data.get("action_name") == "ActionNewRound"
    
    # Assert '69' was filtered out of tiles and tiles0
    extracted_tiles = action_data.get("tiles", [])
    assert "69" not in extracted_tiles
    assert len(extracted_tiles) == 13
    assert extracted_tiles == valid_tiles

    # 3. Pass through MajsoulBridge
    bridge = MajsoulBridge(my_seat=0)
    events = bridge.translate_action("ActionNewRound", action_data)
    assert len(events) >= 1
    start_kyoku = events[0]
    tehais = start_kyoku["tehais"]
    assert "69" not in tehais[0]
    assert all("69" not in h for h in tehais)
    assert len(tehais[0]) == 13

    # 4. Verify scan_tiles_from_bytes never extracts '69' from binary with '69'
    raw_mixed = b"some prefix 69 1m 2p 69 7z other 99 metadata"
    scanned = LiqiProtoDecoder.scan_tiles_from_bytes(raw_mixed)
    assert "69" not in scanned
    assert "99" not in scanned
    assert scanned == ["1m", "2p", "7z"]

    # 5. Verify state manager accepts it cleanly without unknown tile pollution
    state_mgr = MahjongStateManager(my_actor_id=0)
    state_mgr.update_event(start_kyoku)
    assert state_mgr.round_state is not None
    my_player_tehai = state_mgr.round_state.players[0].tehai
    assert "69" not in my_player_tehai


@pytest.mark.asyncio
async def test_reach_action_flow_and_post_riichi_state_transition():
    """
    Test Phase 11: Verify reach action execution, state update (reach -> dahai -> reach_accepted),
    and smooth transition into subsequent turns (e.g. self tsumo automatically recommends tsumogiri).
    """
    from core.state_manager import MahjongStateManager
    from core.engine import DecisionEngine
    from core.strategy_modifier import StrategyModifier
    from io_adapter.action_controller import ActionController
    from io_adapter.majsoul_bridge import MajsoulBridge
    from io_adapter.stream_client import is_my_decision_required

    # 1. Initialize components
    state_mgr = MahjongStateManager(my_actor_id=0)
    engine = DecisionEngine(use_fallback=True)
    modifier = StrategyModifier({})
    controller = ActionController({"output_mode": "text_only"})

    captured_gui_recs = []
    controller.on_recommendation_callback = lambda rec: captured_gui_recs.append(rec)

    # 2. Setup game state where self is tenpai: 123m 456p 789s 11z + draw 8s (tenpai on 1z)
    # Tenpai with 14 tiles
    start_event = {
        "type": "start_kyoku",
        "bakaze": "E",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 0,
        "oya": 0,
        "scores": [25000, 25000, 25000, 25000],
        # 13 tiles: 123m 456m 789m 123p 5s. Drawn tile: 8s.
        # Discarding 8s leaves 123m 456m 789m 123p + 5s (single tile, waiting for 5s, Tenpai!)
        "tehais": [
            ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "5s"],
            ["?"] * 13,
            ["?"] * 13,
            ["?"] * 13,
        ],
        "dora_marker": "1p"
    }
    state_mgr.update_event(start_event)

    # Self draws 8s
    draw_event = {"type": "tsumo", "actor": 0, "pai": "8s"}
    state_mgr.update_event(draw_event)

    # 3. DecisionEngine should propose reach
    candidates = engine.evaluate(state_mgr)
    reach_candidates = [c for c in candidates if c.action_type == "reach"]
    assert len(reach_candidates) > 0
    best_candidate = modifier.modify_candidates(candidates, state_mgr)[0]
    assert best_candidate.action_type == "reach"

    # 4. ActionController executes REACH action and triggers GUI callback
    output = await controller.execute_action(
        candidate=best_candidate,
        state_mgr=state_mgr,
        comp_time_ms=10.0,
        apply_sleep=False,
        all_candidates=candidates
    )
    assert output.action_type == "reach"
    assert len(captured_gui_recs) == 1
    assert captured_gui_recs[0]["action_type"] == "reach"
    assert "立直" in captured_gui_recs[0]["recommendation_text"]

    # 5. Simulate Majsoul packet flow when Riichi discard is confirmed:
    # ActionDiscardTile with is_liqi=True translates to: ['reach', 'dahai', 'reach_accepted']
    bridge = MajsoulBridge(my_seat=0)
    discard_packet_data = {
        "seat": 0,
        "tile": "7s",
        "is_liqi": True,
        "moqie": True,
        "is_wliqi": False,
    }
    mjai_events = bridge.translate_action("ActionDiscardTile", discard_packet_data)
    event_types = [e["type"] for e in mjai_events]
    assert event_types == ["reach", "dahai", "reach_accepted"]

    # Apply these events to state_mgr
    for ev in mjai_events:
        state_mgr.update_event(ev)

    # Verify self is now recorded as in Riichi
    my_p = state_mgr.my_player
    assert my_p.is_riichi is True
    assert my_p.riichi_accepted is True
    assert my_p.score == 24000
    assert state_mgr.round_state.kyotaku == 1

    # 6. Next turn: Other players discard, then self draws again
    # P1 dahai
    p1_dahai = {"type": "dahai", "actor": 1, "pai": "1p", "tsumogiri": True}
    state_mgr.update_event(p1_dahai)
    assert is_my_decision_required(p1_dahai, state_mgr, engine) is False

    # Self draws next tile: 4m (not agari)
    next_tsumo = {"type": "tsumo", "actor": 0, "pai": "4m"}
    state_mgr.update_event(next_tsumo)
    assert is_my_decision_required(next_tsumo, state_mgr, engine) is True

    # 7. Post-riichi evaluation MUST select 4m auto-tsumogiri as top choice without crashing or getting stuck
    post_riichi_candidates = engine.evaluate(state_mgr)
    best_post_cand = modifier.modify_candidates(post_riichi_candidates, state_mgr)[0]
    assert best_post_cand.action_type == "dahai"
    assert best_post_cand.tile == "4m"
    assert best_post_cand.is_tsumogiri is True

    # Execute action for post-riichi tsumogiri
    post_output = await controller.execute_action(
        candidate=best_post_cand,
        state_mgr=state_mgr,
        comp_time_ms=5.0,
        apply_sleep=False,
    )
    assert post_output.action_type == "dahai"
    assert post_output.tile == "4m"


def test_format_tile_jp_mappings_and_edge_cases():
    """
    Test Phase 12: Verify that format_tile_jp correctly transforms mjai, liqi,
    honor aliases, aka dora, and handles edge cases without outputting '?' for valid tiles.
    """
    from io_adapter.action_controller import format_tile_jp

    # 1. Honors in mjai notation ('E', 'S', 'W', 'N', 'P', 'F', 'C')
    assert "北 (N)" in format_tile_jp("N")
    assert "東 (E)" in format_tile_jp("E")
    assert "南 (S)" in format_tile_jp("S")
    assert "西 (W)" in format_tile_jp("W")
    assert "白 (P)" in format_tile_jp("P")
    assert "發 (F)" in format_tile_jp("F")
    assert "中 (C)" in format_tile_jp("C")

    # 2. Honors in numeric notation ('1z' .. '7z')
    assert "北 (4z)" in format_tile_jp("4z")
    assert "東 (1z)" in format_tile_jp("1z")
    assert "中 (7z)" in format_tile_jp("7z")

    # 3. Honor aliases
    assert "北 (pei)" in format_tile_jp("pei")
    assert "北 (kita)" in format_tile_jp("kita")
    assert "東 (ton)" in format_tile_jp("ton")

    # 4. Number tiles (1m..9m, 1p..9p, 1s..9s)
    assert "8索 (8s)" in format_tile_jp("8s")
    assert "1萬 (1m)" in format_tile_jp("1m")
    assert "9筒 (9p)" in format_tile_jp("9p")

    # 5. Red dora ('0m', '0p', '0s', '5mr', '5pr', '5sr')
    assert "赤5萬" in format_tile_jp("0m")
    assert "赤5筒" in format_tile_jp("0p")
    assert "赤5索" in format_tile_jp("0s")
    assert "赤5萬" in format_tile_jp("5mr")

    # 6. Unicode inclusion
    assert "🀃 北 (N)" == format_tile_jp("N", include_unicode=True)
    assert "🀄 中 (C)" == format_tile_jp("C", include_unicode=True)
    assert "🀗 8索 (8s)" == format_tile_jp("8s", include_unicode=True)

    # 7. Edge cases: None, empty string, '?', invalid
    assert format_tile_jp(None) == "？"
    assert format_tile_jp("") == "？"
    assert format_tile_jp("?") == "？"
    assert format_tile_jp("None") == "？"


@pytest.mark.asyncio
async def test_decision_engine_and_action_controller_fallback_on_corrupt_hand():
    """
    Test Phase 13: Verify that DecisionEngine and ActionController never emit
    tile=None or tile='?' when tehai contains unknown tokens or when candidate.tile is corrupted.
    """
    from core.state_manager import MahjongStateManager
    from core.engine import DecisionEngine, ActionCandidate
    from io_adapter.action_controller import ActionController

    state_mgr = MahjongStateManager(my_actor_id=0)
    # Start round with hand containing unknown '?' tokens and valid '1m'
    start_event = {
        "type": "start_kyoku",
        "bakaze": "E",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 0,
        "oya": 0,
        "scores": [25000, 25000, 25000, 25000],
        "tehais": [
            ["?", "?", "?", "?", "?", "?", "?", "?", "?", "?", "?", "?", "1m"],
            ["?"] * 13,
            ["?"] * 13,
            ["?"] * 13,
        ],
        "dora_marker": "1p"
    }
    state_mgr.update_event(start_event)

    # 1. Self turn draws '9s'
    draw_event = {"type": "tsumo", "actor": 0, "pai": "9s"}
    state_mgr.update_event(draw_event)

    engine = DecisionEngine(use_fallback=True)
    candidates = engine.evaluate(state_mgr)
    assert len(candidates) > 0
    # Every candidate tile must be a valid non-'?' tile
    for c in candidates:
        assert c.tile not in ("?", None, "")
        assert c.tile in ("1m", "9s")

    # 2. Test edge case: Tehai is completely all '?'
    state_mgr.round_state.players[0].tehai = ["?"] * 13
    state_mgr.round_state.players[0].tsumo_tile = "?"
    fallback_candidates = engine.evaluate(state_mgr)
    assert len(fallback_candidates) > 0
    assert fallback_candidates[0].tile != "?"
    assert fallback_candidates[0].tile is not None

    # 3. Test ActionController fallback guard when given an intentionally corrupted candidate (tile='?')
    controller = ActionController({"output_mode": "text_only"})
    corrupt_candidate = ActionCandidate(
        action_type="dahai",
        tile="?",
        raw_q_value=1.0,
        danger_score=0.0,
        shanten_after=3
    )

    # Set valid tehai so controller can extract fallback
    state_mgr.round_state.players[0].tehai = ["3p", "4p", "5p"]
    state_mgr.round_state.players[0].tsumo_tile = "6p"

    output = await controller.execute_action(
        candidate=corrupt_candidate,
        state_mgr=state_mgr,
        comp_time_ms=5.0,
        apply_sleep=False
    )
    # Must have recovered to a real tile ('6p' tsumo or '5p' tehai), NOT '?' or None
    assert output.tile != "?"
    assert output.tile is not None
    assert output.tile in ("6p", "5p")
    assert "？" not in output.reasoning_summary


def test_tile_efficiency_evaluation_not_bypassed():
    """
    Test Phase 14: Verify that tile efficiency logic evaluates all distinct hand tiles,
    taking into account ukeire and hand structure, without being preemptively short-circuited
    by fallbacks or automatic tsumogiri.
    """
    from core.state_manager import MahjongStateManager
    from core.engine import DecisionEngine
    from core.strategy_modifier import StrategyModifier

    state_mgr = MahjongStateManager(my_actor_id=0)
    # Hand: 1m, 2m, 3m, 4p, 5p, 6p, 7s, 8s, 9s, 2s, 3s, 9m, 9m + draw: 1z (isolated East wind)
    # Hand is 1-shanten. Discarding isolated '1z' maintains 1-shanten and maximizes efficiency.
    start_event = {
        "type": "start_kyoku",
        "bakaze": "E",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 0,
        "oya": 0,
        "scores": [25000, 25000, 25000, 25000],
        "tehais": [
            ["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "2s", "3s", "9m", "9m"],
            ["?"] * 13,
            ["?"] * 13,
            ["?"] * 13,
        ],
        "dora_marker": "1p"
    }
    state_mgr.update_event(start_event)

    # Draws 1z (East)
    draw_event = {"type": "tsumo", "actor": 0, "pai": "1z"}
    state_mgr.update_event(draw_event)

    engine = DecisionEngine(use_fallback=True)
    candidates = engine.evaluate(state_mgr)

    # 1. Verify that all distinct tiles in hand are evaluated (not short-circuited)
    distinct_tiles = set(state_mgr.my_player.tehai)
    evaluated_tiles = {c.tile for c in candidates if c.action_type == "dahai"}
    assert distinct_tiles.issubset(evaluated_tiles)
    assert len(candidates) >= len(distinct_tiles)

    # 2. Verify that ukeire was computed for candidates
    ukeire_values = [c.metadata.get("ukeire", 0) for c in candidates if c.action_type == "dahai"]
    assert any(u > 0 for u in ukeire_values)

    # 3. Discarding isolated 1z should have higher efficiency score than breaking mentsu (e.g. 2m)
    c_1z = next(c for c in candidates if c.tile == "1z")
    c_2m = next(c for c in candidates if c.tile == "2m")
    assert c_1z.raw_q_value > c_2m.raw_q_value

    # 4. StrategyModifier should pick 1z as optimal discard (or reach discarding 1z)
    modifier = StrategyModifier({})
    modified = modifier.modify_candidates(candidates, state_mgr)
    assert modified[0].tile == "1z"
    assert modified[0].action_type in ("reach", "dahai")


def test_toitsu_preservation_over_isolated_tiles():
    """
    Test Phase 15: Verify that DecisionEngine preserves existing pairs (toitsu)
    like 2s-2s and prioritizes discarding truly isolated tiles (e.g. 1m, 7s, or isolated honor).
    """
    from core.state_manager import MahjongStateManager
    from core.engine import DecisionEngine
    from core.strategy_modifier import StrategyModifier

    state_mgr = MahjongStateManager(my_actor_id=0)
    # Hand: 1m (isolated), 7s (isolated), 2s, 2s (toitsu), 3s, 4s, 678p, 345m, 9m, 9m
    # 2s is part of toitsu + ryanmen (2s 2s 3s 4s). 1m and 7s are completely isolated.
    hand = ["1m", "7s", "2s", "2s", "3s", "4s", "6p", "7p", "8p", "3m", "4m", "5m", "9m", "9m"]
    start_event = {
        "type": "start_kyoku",
        "bakaze": "E",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 0,
        "oya": 0,
        "scores": [25000, 25000, 25000, 25000],
        "tehais": [hand[:13], ["?"] * 13, ["?"] * 13, ["?"] * 13],
        "dora_marker": "1z"
    }
    state_mgr.update_event(start_event)
    state_mgr.update_event({"type": "tsumo", "actor": 0, "pai": hand[13]})

    engine = DecisionEngine(use_fallback=True)
    modifier = StrategyModifier({})
    candidates = engine.evaluate(state_mgr)
    best = modifier.modify_candidates(candidates, state_mgr)[0]

    # Best discard must be one of the isolated tiles (7s or 1m), NEVER the 2s pair
    assert best.tile in ("7s", "1m")
    assert best.tile != "2s"

    # Verify score comparison: isolated tiles must score significantly higher than breaking 2s pair
    c_2s = next(c for c in candidates if c.tile == "2s")
    c_iso = next(c for c in candidates if c.tile == best.tile)
    assert c_iso.raw_q_value > c_2s.raw_q_value


def test_debug_tracer_and_watchdog_resilience():
    """
    Test Phase 16: Verify DebugTracer logging, state dumping, and StateManager resilience
    against unexpected packet sequences without silent freezing or crashing.
    """
    from utils.debug_tracer import get_tracer, trace_log, mark_trace_event
    from core.state_manager import MahjongStateManager
    from main import MahjongAIAgent

    tracer = get_tracer()
    tracer.enabled = True
    trace_log("TEST_TRACE", "Testing debug trace logger", {"sample": 123})
    assert tracer.last_event_type == "TEST_TRACE"
    assert tracer.get_idle_seconds() >= 0.0

    state_mgr = MahjongStateManager(my_actor_id=0)
    # 1. State dump before any rounds
    dump1 = state_mgr.dump_state()
    assert "No active round_state" in dump1

    # 2. Out-of-order events: send tsumo and dahai BEFORE start_kyoku
    # Must not crash, should log safely
    state_mgr.update_event({"type": "tsumo", "actor": 0, "pai": "1m"})
    state_mgr.update_event({"type": "dahai", "actor": 0, "pai": "1m"})

    # 3. Normal start_kyoku with malformed/missing fields
    corrupt_start = {
        "type": "start_kyoku",
        "bakaze": "E",
        "kyoku": 1,
        # missing oya, scores, tehais
    }
    state_mgr.update_event(corrupt_start)
    dump2 = state_mgr.dump_state()
    assert "E1局 0本場" in dump2
    assert "MyActor=P0" in dump2

    # 4. End round events trace verification
    state_mgr.update_event({"type": "hora", "actor": 1, "target": 2, "pai": "5m"})
    assert state_mgr.round_state.is_round_over is True
    assert tracer.last_event_type in ("HORA", "ROUND_OVER")


def test_harmful_call_prevention_yaku_less(test_config):
    """
    Verify that harmful calls without yaku (e.g. chi/pon on non-yakuhai tiles with terminal/honors)
    are strictly rejected and pass (none) dominates with a huge margin.
    """
    engine = DecisionEngine(use_fallback=True)
    modifier = StrategyModifier(test_config)
    state_mgr = MahjongStateManager(my_actor_id=0)

    # Initialize a round: East 1, player 0 has a closed hand with terminals and honors, no yakuhai
    # Hand: 1m, 9m, 2p, 3p, 4p, 7s, 8s, 9s, 1z, 2z, 4z, 6p, 6p
    init_event = {
        "type": "start_kyoku",
        "bakaze": "E",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 0,
        "oya": 1,
        "scores": [25000, 25000, 25000, 25000],
        "tehais": [
            ["1m", "9m", "2p", "3p", "4p", "7s", "8s", "9s", "1z", "2z", "4z", "6p", "6p"],
            [], [], []
        ],
        "dora_marker": "5m"
    }
    state_mgr.update_event(init_event)

    # Kamicha (Player 3) discards 6p. Player 0 could Pon 6p.
    # But hand has 1m, 9m, 1z, 2z, 4z! Ponning 6p will open the hand with 0 YAKU (no tanyao, no yakuhai).
    # This must be rejected in favor of PASS (none).
    discard_event = {
        "type": "dahai",
        "actor": 3,
        "pai": "6p",
        "tsumogiri": False
    }
    state_mgr.update_event(discard_event)

    candidates = engine.evaluate(state_mgr)
    modified = modifier.modify_candidates(candidates, state_mgr)

    best_action = modified[0]
    assert best_action.action_type == "none", f"Expected Pass (none), but got {best_action.action_type}"

    # Verify that pon candidate received heavy penalty
    pon_cands = [c for c in modified if c.action_type == "pon"]
    if pon_cands:
        pon_cand = pon_cands[0]
        assert pon_cand.modified_q_value < 0.0, f"No-yaku pon must have negative Q-value, got {pon_cand.modified_q_value}"
        assert pon_cand.metadata["breakdown"]["has_yaku"] is False


def test_taatsu_and_toitsu_protection_over_isolated(test_config):
    """
    Verify that dahai evaluation protects valid taatsu (e.g. 23p, 78s) and toitsu (e.g. 5m pair),
    and prioritizes discarding genuinely isolated tiles (e.g. isolated 1z, 9m) first.
    """
    engine = DecisionEngine(use_fallback=True)
    modifier = StrategyModifier(test_config)
    state_mgr = MahjongStateManager(my_actor_id=0)

    # Hand: 2p, 3p, 4p (completed meld)
    # 6p, 7p, 8p (completed meld)
    # 7s, 8s (ryanmen taatsu)
    # 5m, 5m (essential pair / janto)
    # 4s, 5s (ryanmen taatsu)
    # 1z (isolated honor)
    # Tsumo: 9m (isolated terminal)
    init_event = {
        "type": "start_kyoku",
        "bakaze": "E",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 0,
        "oya": 0,
        "scores": [25000, 25000, 25000, 25000],
        "tehais": [
            ["2p", "3p", "4p", "6p", "7p", "8p", "7s", "8s", "5m", "5m", "4s", "5s", "1z"],
            [], [], []
        ],
        "dora_marker": "3s"
    }
    state_mgr.update_event(init_event)

    tsumo_event = {
        "type": "tsumo",
        "actor": 0,
        "pai": "9m"
    }
    state_mgr.update_event(tsumo_event)

    candidates = engine.evaluate(state_mgr)
    modified = modifier.modify_candidates(candidates, state_mgr)

    best_action = modified[0]
    # Best action must discard isolated tile (1z or 9m), NOT break the pair 5m, nor break ryanmen 2p/3p or 7s/8s!
    assert normalize_tile(best_action.tile) in ("1z", "9m"), f"Expected isolated tile 1z or 9m, got {best_action.tile}"

    # Also verify that 5m (the pair) has a much lower score than 1z
    pair_cands = [c for c in modified if normalize_tile(c.tile) == "5m"]
    iso_cands = [c for c in modified if normalize_tile(c.tile) == "1z"]
    assert len(pair_cands) > 0 and len(iso_cands) > 0
    assert iso_cands[0].modified_q_value > pair_cands[0].modified_q_value, "Isolated honor must score higher than breaking a pair"


def test_evaluation_breakdown_logging(test_config):
    """
    Verify that candidate action metadata includes structured breakdown of Q-value factors.
    """
    engine = DecisionEngine(use_fallback=True)
    state_mgr = MahjongStateManager(my_actor_id=0)

    init_event = {
        "type": "start_kyoku",
        "bakaze": "E",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 0,
        "oya": 0,
        "scores": [25000, 25000, 25000, 25000],
        "tehais": [
            ["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "1z", "2z", "3z", "4z"],
            [], [], []
        ],
        "dora_marker": "5m"
    }
    state_mgr.update_event(init_event)

    tsumo_event = {
        "type": "tsumo",
        "actor": 0,
        "pai": "5s"
    }
    state_mgr.update_event(tsumo_event)

    candidates = engine.evaluate(state_mgr)
    assert len(candidates) > 0

    for c in candidates:
        bd = c.metadata.get("breakdown")
        assert bd is not None, f"Candidate {c.tile} must have breakdown metadata"
        assert "base_shanten" in bd
        assert "ukeire" in bd
        assert "shape" in bd


def test_nonexistent_hand_tile_filtered_out(test_config):
    """
    Verify that DecisionEngine and ActionController strictly reject any tile not in the hand (e.g. 9s),
    and guarantee that any recommended dahai tile exists in player's current tehai.
    """
    engine = DecisionEngine(use_fallback=True)
    modifier = StrategyModifier(test_config)
    state_mgr = MahjongStateManager(my_actor_id=0)

    # Hand without 9s: 1m, 2m, 3m, 4p, 5p, 6p, 1s, 2s, 3s, 1z, 1z, 2z, 3z + draw 4z
    # (9s is definitely NOT in this hand!)
    init_event = {
        "type": "start_kyoku",
        "bakaze": "E",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 0,
        "oya": 0,
        "scores": [25000, 25000, 25000, 25000],
        "tehais": [
            ["1m", "2m", "3m", "4p", "5p", "6p", "1s", "2s", "3s", "1z", "1z", "2z", "3z"],
            [], [], []
        ],
        "dora_marker": "5m"
    }
    state_mgr.update_event(init_event)

    tsumo_event = {
        "type": "tsumo",
        "actor": 0,
        "pai": "4z"
    }
    state_mgr.update_event(tsumo_event)

    # 1. Normal evaluation: every candidate tile must be physically in the hand
    candidates = engine.evaluate(state_mgr)
    my_hand = state_mgr.my_player.tehai
    for c in candidates:
        if c.action_type in ("dahai", "reach"):
            assert c.tile in my_hand or normalize_tile(c.tile) in [normalize_tile(t) for t in my_hand], (
                f"Candidate {c.tile} must exist in hand {my_hand}"
            )
            assert normalize_tile(c.tile) != "9s", "9s must NEVER be evaluated as candidate when not in hand"

    # 2. Inject a fake rogue candidate with 9s to simulate model/heuristic corruption
    from core.engine import ActionCandidate
    rogue_cand = ActionCandidate(
        action_type="dahai",
        tile="9s",
        raw_q_value=999.0, # Attempt to trick AI into recommending it
        danger_score=0.0,
        shanten_after=0
    )

    # If an external caller or model passed rogue candidates, DecisionEngine.evaluate filters it
    class MockModelEngine(DecisionEngine):
        def _evaluate_with_pytorch(self, sm, va):
            return [rogue_cand] + candidates

    mock_engine = MockModelEngine(use_fallback=True)
    mock_engine.model = object() # Trigger pytorch branch

    sanitized_candidates = mock_engine.evaluate(state_mgr)
    assert not any(c.tile == "9s" for c in sanitized_candidates), "Rogue 9s candidate must be strictly dropped"

    # 3. Strategy modification should select a valid hand tile
    best = modifier.select_best_action(sanitized_candidates, state_mgr)
    assert best is not None
    assert best.tile in my_hand or normalize_tile(best.tile) in [normalize_tile(t) for t in my_hand]


def test_meld_candidates_strictly_validated_against_hand_pool(test_config):
    """
    Verify that DecisionEngine strictly validates consumed tiles for all melds (chi, pon, kan),
    and forcibly drops any candidate requiring tiles not in hand or exceeding hand pool quantity.
    """
    from core.engine import DecisionEngine, ActionCandidate
    from core.state_manager import MahjongStateManager

    engine = DecisionEngine(use_fallback=True)
    state_mgr = MahjongStateManager(my_actor_id=0)

    # Hand has only ONE 2m, and NO 9s: ['1m', '2m', '3m', '4p', '5p', '6p', '1s', '2s', '3s', '1z', '1z', '2z', '3z']
    init_event = {
        "type": "start_kyoku",
        "bakaze": "E",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 0,
        "oya": 1, # Enemy is dealer
        "scores": [25000, 25000, 25000, 25000],
        "tehais": [
            ["1m", "2m", "3m", "4p", "5p", "6p", "1s", "2s", "3s", "1z", "1z", "2z", "3z"],
            ["?"] * 13, ["?"] * 13, ["?"] * 13
        ],
        "dora_marker": "1p"
    }
    state_mgr.update_event(init_event)

    # Enemy discards 2m
    enemy_discard = {
        "type": "dahai",
        "actor": 1,
        "pai": "2m",
        "tsumogiri": False
    }
    state_mgr.update_event(enemy_discard)

    # 1. Normal evaluation should NOT offer Pon for 2m because player only has 1 instance of 2m in hand
    candidates = engine.evaluate(state_mgr)
    pon_2m = [c for c in candidates if c.action_type == "pon" and c.tile == "2m"]
    assert len(pon_2m) == 0, "Pon on 2m must NOT be generated when hand contains only 1 2m"

    # 2. Inject rogue meld candidates:
    # a) Rogue Pon requiring two 2m when hand only has one
    # b) Rogue Chi requiring non-existent 9s
    class RogueMeldEngine(DecisionEngine):
        def _evaluate_with_fallback(self, sm, va):
            return [
                ActionCandidate(action_type="pon", tile="2m", consumed=["2m", "2m"], raw_q_value=50.0),
                ActionCandidate(action_type="chi", tile="8s", consumed=["7s", "9s"], raw_q_value=40.0), # 9s is not in hand!
                ActionCandidate(action_type="none", raw_q_value=1.0)
            ]

    rogue_engine = RogueMeldEngine(use_fallback=True)
    filtered = rogue_engine.evaluate(state_mgr)

    # Both rogue melds must be dropped! Only 'none' survives.
    assert not any(c.action_type == "pon" for c in filtered), "Rogue Pon exceeding hand pool must be dropped"
    assert not any(c.action_type == "chi" for c in filtered), "Rogue Chi requiring non-existent 9s must be dropped"
    assert any(c.action_type == "none" for c in filtered), "'none' action should be preserved"


def test_state_manager_dahai_sync_with_aka_dora_and_safety_checks():
    """
    Verify that MahjongStateManager accurately removes aka-dora (0s / 5sr) upon dahai,
    and protects hand integrity from accidental pops when unmatched discards occur.
    """
    from core.state_manager import MahjongStateManager, normalize_tile

    state_mgr = MahjongStateManager(my_actor_id=0)

    # Player starts with red 5s ("5sr" or "0s")
    init_event = {
        "type": "start_kyoku",
        "bakaze": "E",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 0,
        "oya": 0,
        "scores": [25000, 25000, 25000, 25000],
        "tehais": [
            ["1m", "2m", "3m", "4p", "5p", "6p", "1s", "2s", "3s", "5sr", "1z", "2z", "3z"],
            [], [], []
        ],
        "dora_marker": "1p"
    }
    state_mgr.update_event(init_event)
    p = state_mgr.my_player

    # Draw 4z
    state_mgr.update_event({"type": "tsumo", "actor": 0, "pai": "4z"})
    assert len(p.tehai) == 14
    assert "5sr" in p.tehai

    # Discard 0s (Liqi notation for red 5s): state manager must recognize 0s == 5sr and remove 5sr
    state_mgr.update_event({"type": "dahai", "actor": 0, "pai": "0s", "tsumogiri": False})
    assert len(p.tehai) == 13
    assert "5sr" not in p.tehai, "5sr should have been removed when 0s was discarded"
    assert "4z" in p.tehai, "4z must still remain in hand"

    # Now hand is 13 tiles. If an unmatched discard event arrives (e.g. '9s' which is not in hand),
    # the state manager must NOT pop another tile from the valid 13-tile hand!
    state_mgr.update_event({"type": "dahai", "actor": 0, "pai": "9s", "tsumogiri": False})
    assert len(p.tehai) == 13, "Hand must NOT pop tiles below 13 on unmatched discard"


def test_packet_sync_seat_isolation_and_no_phantom_tiles():
    """
    Verify that ActionNewRound and ActionDealTile in LiqiProtoDecoder and MajsoulBridge
    do not contaminate seat 0 when player is seat 1 (South), preventing false AutoSeat
    and preventing phantom tile leaks.
    """
    from io_adapter.liqi_proto_decoder import LiqiProtoDecoder
    from io_adapter.majsoul_bridge import MajsoulBridge
    from core.state_manager import MahjongStateManager

    # Simulated decoded Liqi ActionNewRound for player in seat 1 (South)
    # The packet only has field 4 ('tiles') which represents the player's own hand
    raw_payload_data = {
        "chang": 0,
        "ju": 0,
        "ben": 0,
        "tiles": ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p"],
        "doras": ["1z"],
        "scores": [25000, 25000, 25000, 25000]
    }

    # Bridge configured for seat 1 (South)
    bridge = MajsoulBridge(my_seat=1)
    events = bridge.translate_action("ActionNewRound", raw_payload_data)

    start_event = next(e for e in events if e["type"] == "start_kyoku")
    tehais = start_event["tehais"]

    # Player 1's hand should have the 13 tiles
    assert len(tehais[1]) == 13
    assert "1m" in tehais[1]

    # Player 0's hand must NOT have player 1's tiles; it should be obscured with '?'
    assert all(t == "?" for t in tehais[0]), "Player 0 hand must be hidden with '?' for seat 1 player"

    # StateManager initialized for seat 1
    state_mgr = MahjongStateManager(my_actor_id=1)
    for ev in events:
        state_mgr.update_event(ev)

    assert state_mgr.my_player.actor_id == 1
    assert len(state_mgr.my_player.tehai) == 13
    assert not state_mgr.my_player.has_tile("9s"), "Hand should not contain 9s"

    # Opponent draw on seat 0: ActionDealTile
    deal_events = bridge.translate_action("ActionDealTile", {"seat": 0, "tile": "9s"})
    for ev in deal_events:
        state_mgr.update_event(ev)

    # Opponent draw must be hidden ('?') and must NEVER leak into self hand
    assert deal_events[0]["pai"] == "?"
    assert len(state_mgr.my_player.tehai) == 13
    assert not state_mgr.my_player.has_tile("9s"), "Opponent draw must not affect player's hand"


def test_phantom_tile_candidate_strictly_filtered_out(capsys):
    """
    Test Phase 19: Strict validation guard in DecisionEngine.
    Verify that if a candidate tile (e.g. '9s') is NOT physically in self hand,
    it is forcefully dropped and a warning alert is printed to console.
    """
    from core.state_manager import MahjongStateManager
    from core.engine import DecisionEngine, ActionCandidate

    state_mgr = MahjongStateManager(my_actor_id=0)
    # Hand without 9s: 14 tiles of manzu & pinzu
    hand = ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p", "5p"]
    start_event = {
        "type": "start_kyoku",
        "bakaze": "E",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 0,
        "oya": 0,
        "scores": [25000, 25000, 25000, 25000],
        "tehais": [hand[:13], ["?"] * 13, ["?"] * 13, ["?"] * 13],
        "dora_marker": "1p"
    }
    state_mgr.update_event(start_event)
    state_mgr.update_event({"type": "tsumo", "actor": 0, "pai": hand[13]})

    assert len(state_mgr.my_player.tehai) == 14
    assert not state_mgr.my_player.has_tile("9s")

    engine = DecisionEngine(use_fallback=True)

    # Subclass or monkeypatch _evaluate_with_fallback to intentionally inject a phantom '9s' candidate
    orig_fallback = engine._evaluate_with_fallback
    def mock_eval_with_phantom(sm, valid_actions=None):
        cands = orig_fallback(sm, valid_actions)
        # Maliciously insert a phantom discard candidate for '9s' which is not in hand!
        cands.insert(0, ActionCandidate(
            action_type="dahai",
            tile="9s",
            raw_q_value=999.0,
            shanten_after=0
        ))
        cands.insert(1, ActionCandidate(
            action_type="reach",
            tile="9s",
            raw_q_value=999.0,
            shanten_after=0
        ))
        return cands

    engine._evaluate_with_fallback = mock_eval_with_phantom

    # Run evaluate
    evaluated_cands = engine.evaluate(state_mgr)

    # 1. '9s' must NEVER appear in evaluated_cands
    discard_tiles = [c.tile for c in evaluated_cands if c.action_type in ("dahai", "reach")]
    assert "9s" not in discard_tiles, f"Phantom tile '9s' must be strictly filtered out! Found: {discard_tiles}"

    # 2. All surviving discard candidates must physically exist in hand
    for t in discard_tiles:
        assert state_mgr.my_player.has_tile(t), f"Candidate tile '{t}' does not exist in hand {hand}!"

    # 3. Alert message must be printed
    captured = capsys.readouterr()
    assert "[Engine Guard ALERT]" in captured.out
    assert "9s" in captured.out


def test_phantom_tile_meld_candidate_strictly_filtered_out(capsys):
    """
    Verify that meld candidates with consumed tiles exceeding the hand pool are dropped.
    """
    from core.state_manager import MahjongStateManager
    from core.engine import DecisionEngine, ActionCandidate

    state_mgr = MahjongStateManager(my_actor_id=0)
    # Hand has only one '2s'
    hand = ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "2s"]
    start_event = {
        "type": "start_kyoku",
        "bakaze": "E",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 0,
        "oya": 1,
        "scores": [25000, 25000, 25000, 25000],
        "tehais": [hand, ["?"] * 13, ["?"] * 13, ["?"] * 13],
        "dora_marker": "1p"
    }
    state_mgr.update_event(start_event)

    # Player 3 discards 2s
    state_mgr.update_event({"type": "dahai", "actor": 3, "pai": "2s", "tsumogiri": False})

    engine = DecisionEngine(use_fallback=True)

    # Intentionally inject impossible Pon candidate requiring two '2s' consumed
    orig_fallback = engine._evaluate_with_fallback
    def mock_eval_with_impossible_pon(sm, valid_actions=None):
        cands = orig_fallback(sm, valid_actions)
        cands.insert(0, ActionCandidate(
            action_type="pon",
            tile="2s",
            consumed=["2s", "2s"],
            raw_q_value=999.0
        ))
        return cands

    engine._evaluate_with_fallback = mock_eval_with_impossible_pon
    evaluated_cands = engine.evaluate(state_mgr)

    # Impossible pon must be eliminated
    surviving_pons = [c for c in evaluated_cands if c.action_type == "pon"]
    assert len(surviving_pons) == 0, "Impossible pon exceeding hand count must be dropped!"

    captured = capsys.readouterr()
    assert "[Engine Guard ALERT]" in captured.out


def test_action_packet_hand_state_synchronization():
    """
    Verify state synchronization for ActionNewRound -> ActionDealTile -> ActionDiscardTile.
    Ensures that hand tile list strictly tracks dealt and discarded tiles.
    """
    from core.state_manager import MahjongStateManager
    from io_adapter.majsoul_bridge import MajsoulBridge

    bridge = MajsoulBridge(my_seat=0)
    state_mgr = MahjongStateManager(my_actor_id=0)

    # 1. ActionNewRound: dealt 13 tiles
    nr_data = {
        "chang": 0,
        "ju": 0,
        "ben": 0,
        "liqibang": 0,
        "scores": [25000, 25000, 25000, 25000],
        "doras": ["1p"],
        "tiles": ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p"]
    }
    events = bridge.translate_action("ActionNewRound", nr_data)
    for ev in events:
        state_mgr.update_event(ev)

    # Dealer is brought into 14-tile state immediately (13-tile hand + initial draw) so initial discard is triggered
    assert len(state_mgr.my_player.tehai) == 14
    assert state_mgr.my_player.has_tile("1m")
    assert not state_mgr.my_player.has_tile("9s")

    # 2. ActionDealTile (self draw '5s')
    deal_events = bridge.translate_action("ActionDealTile", {"seat": 0, "tile": "5s"})
    for ev in deal_events:
        state_mgr.update_event(ev)

    assert len(state_mgr.my_player.tehai) == 14
    assert state_mgr.my_player.has_tile("5s")
    assert state_mgr.my_player.tsumo_tile == "5s"

    # 3. ActionDiscardTile (self discards '5s')
    discard_events = bridge.translate_action("ActionDiscardTile", {"seat": 0, "tile": "5s", "moqie": True})
    for ev in discard_events:
        state_mgr.update_event(ev)

    assert len(state_mgr.my_player.tehai) == 13
    assert not state_mgr.my_player.has_tile("5s"), "'5s' was discarded, must not be in tehai!"
    assert state_mgr.my_player.tsumo_tile is None
    assert state_mgr.round_state.last_discard_tile == "5s"


@pytest.mark.asyncio
async def test_action_controller_phantom_tile_replacement_guard(capsys):
    """
    Verify ActionController replaces any phantom candidate tile with an actual physical hand tile.
    """
    from core.state_manager import MahjongStateManager
    from core.engine import ActionCandidate
    from io_adapter.action_controller import ActionController

    controller = ActionController({"output_mode": "text_only"})
    state_mgr = MahjongStateManager(my_actor_id=0)

    # Initialize round
    start_event = {
        "type": "start_kyoku",
        "bakaze": "E",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 0,
        "oya": 0,
        "scores": [25000, 25000, 25000, 25000],
        "tehais": [
            ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p"],
            ["?"] * 13,
            ["?"] * 13,
            ["?"] * 13,
        ],
        "dora_marker": "1p"
    }
    state_mgr.update_event(start_event)
    state_mgr.update_event({"type": "tsumo", "actor": 0, "pai": "5p"})

    # Intentionally craft a phantom candidate tile '9s'
    phantom_cand = ActionCandidate(
        action_type="dahai",
        tile="9s",
        raw_q_value=5.0,
        modified_q_value=5.0,
        shanten_after=1
    )

    output = await controller.execute_action(
        candidate=phantom_cand,
        state_mgr=state_mgr,
        comp_time_ms=10.0,
        apply_sleep=False
    )

    # Output action must NOT be 9s! It must have been replaced with a real tile from hand (e.g. '5p')
    assert output.tile != "9s", "Phantom tile '9s' must never be emitted as output action!"
    assert state_mgr.my_player.has_tile(output.tile), f"Output tile '{output.tile}' must physically exist in hand!"

    captured = capsys.readouterr()
    assert "[ActionController ALERT]" in captured.out
    assert "9s" in captured.out


def test_complex_block_toitsu_taatsu_cut_2p_not_7p():
    """
    Test Phase 20: Verify that in complex multi-block hands (2p 4p 5p 5p 7p 7p 8p),
    the exterior floater '2p' is prioritized for discard, and the connected pair/taatsu
    '7p' is strictly protected and never chosen.

    Hand shape:
    - Manzu: 6m 8m
    - Souzu: 4s 5s (tsumo 4s -> 4s 4s 5s)
    - Pinzu: 2p 4p 5p 5p 7p 7p 8p
    - Honors: 6z 6z (F F / Hatsu pair)
    """
    from core.state_manager import MahjongStateManager
    from core.engine import DecisionEngine

    state_mgr = MahjongStateManager(my_actor_id=0)
    # 13 tiles before draw
    hand_13 = ["6m", "8m", "2p", "4p", "5p", "5p", "7p", "7p", "8p", "4s", "5s", "6z", "6z"]
    start_event = {
        "type": "start_kyoku",
        "bakaze": "E",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 0,
        "oya": 0,
        "scores": [25000, 25000, 25000, 25000],
        "tehais": [hand_13, ["?"] * 13, ["?"] * 13, ["?"] * 13],
        "dora_marker": "1m"
    }
    state_mgr.update_event(start_event)
    # Self draws 4s (making 4s 4s 5s)
    state_mgr.update_event({"type": "tsumo", "actor": 0, "pai": "4s"})

    assert len(state_mgr.my_player.tehai) == 14
    assert state_mgr.my_player.count_tile("7p") == 2
    assert state_mgr.my_player.count_tile("2p") == 1

    engine = DecisionEngine(use_fallback=True)
    candidates = engine.evaluate(state_mgr)

    assert len(candidates) > 0

    best = candidates[0]
    print(f"\n[Test Result] Best Recommended Action: {best.action_type} {best.tile} (Q: {best.raw_q_value})")
    print(f"Top 3 candidates:")
    for idx, c in enumerate(candidates[:5]):
        print(f"  #{idx+1}: {c.action_type} {c.tile} | Q: {c.raw_q_value} | Breakdown: {c.metadata.get('breakdown', {})}")

    # 1. 2p must be the highest evaluated discard action
    assert best.action_type == "dahai"
    assert best.tile == "2p", f"Engine must recommend discarding exterior floater '2p', but recommended '{best.tile}'!"

    # 2. 7p must NEVER be the top choice
    assert best.tile != "7p", "7p must strictly not be chosen!"

    # 3. Find 7p candidate and verify 2p has higher Q-value than 7p
    c_2p = next(c for c in candidates if c.tile == "2p")
    c_7p = next(c for c in candidates if c.tile == "7p")
    assert c_2p.raw_q_value > c_7p.raw_q_value, f"2p Q-value ({c_2p.raw_q_value}) must be higher than 7p ({c_7p.raw_q_value})"
    assert c_7p.metadata.get("breakdown", {}).get("shape", 0) <= -3.5, "7p should receive at least -3.5 shape penalty"


def test_dora_recognition_and_heavy_discard_penalty():
    """
    Emergency Regression Test:
    Verify that when dora indicator is 6p (dora = 7p):
    1. Dora indicator is correctly mapped and recognized (6p -> 7p, 9p -> 1p, N -> 1z, P -> 6z).
    2. Dora counter and tile classification functions work accurately.
    3. In the hand [6m 8m, 4s 5s (tsumo 4s), 2p 4p 5p 5p 7p 7p 8p, F F],
       discarding 7p receives a severe dora penalty (-16.0 or lower) in breakdown['dora'].
    4. 7p is strictly NOT chosen; 2p is recommended as the top discard.
    """
    from core.state_manager import MahjongStateManager, get_dora_next
    from core.engine import DecisionEngine

    # 1. Indicator-to-Dora calculation verification
    assert get_dora_next("6p") == "7p", "6p indicator must yield 7p dora"
    assert get_dora_next("9p") == "1p", "9p indicator must yield 1p dora"
    assert get_dora_next("4z") == "1z", "4z (North) indicator must yield 1z (East) dora"
    assert get_dora_next("N") == "1z", "N indicator must yield 1z dora"
    assert get_dora_next("P") == "6z", "P (Haku) indicator must yield 6z (Hatsu) dora"
    assert get_dora_next("5z") == "6z", "5z indicator must yield 6z dora"

    # 2. State manager round initialization with dora indicator '6p'
    state_mgr = MahjongStateManager(my_actor_id=0)
    hand_13 = ["6m", "8m", "2p", "4p", "5p", "5p", "7p", "7p", "8p", "4s", "5s", "6z", "6z"]
    start_event = {
        "type": "start_kyoku",
        "bakaze": "E",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 0,
        "oya": 0,
        "scores": [25000, 25000, 25000, 25000],
        "tehais": [hand_13, ["?"] * 13, ["?"] * 13, ["?"] * 13],
        "dora_marker": "6p"  # Dora is 7p
    }
    state_mgr.update_event(start_event)
    state_mgr.update_event({"type": "tsumo", "actor": 0, "pai": "4s"})

    assert state_mgr.round_state is not None
    assert "7p" in state_mgr.round_state.dora_tiles or any(t == "7p" for t in state_mgr.round_state.dora_tiles)
    assert state_mgr.round_state.get_dora_count("7p") == 1, "7p must have dora count 1"
    assert state_mgr.round_state.is_dora_tile("7p") is True, "7p must be recognized as dora"
    assert state_mgr.round_state.get_dora_count("2p") == 0, "2p is not dora"

    # 3. DecisionEngine evaluation
    engine = DecisionEngine(use_fallback=True)
    candidates = engine.evaluate(state_mgr)
    assert len(candidates) > 0

    best = candidates[0]
    print(f"\n[Test Result] Top Action: {best.action_type} {best.tile} (Q: {best.raw_q_value})")
    for idx, c in enumerate(candidates):
        print(f"  #{idx+1}: {c.action_type} {c.tile} | Q: {c.raw_q_value:.2f} | Breakdown: {c.metadata.get('breakdown', {})}")

    # Top recommendation must be 2p, definitely not 7p
    assert best.tile == "2p", f"Expected top discard to be '2p', but got '{best.tile}'"
    assert best.tile != "7p", "7p (dora pair) must NEVER be recommended!"

    # Verify breakdown for 7p
    c_7p = next(c for c in candidates if c.tile == "7p")
    breakdown_7p = c_7p.metadata.get("breakdown", {})
    dora_val = breakdown_7p.get("dora", 0.0)
    shape_val = breakdown_7p.get("shape", 0.0)

    print(f"7p Breakdown -> dora: {dora_val}, shape: {shape_val}")
    assert dora_val <= -10.0, f"Dora penalty for 7p must be <= -10.0, but got {dora_val}"
    assert dora_val != 0.0, "Dora penalty must not be 0.0!"
    assert shape_val <= -3.5, f"Shape penalty for 7p pair must be <= -3.5, but got {shape_val}"

    # Verify 7p is near the bottom of the candidate list
    rank_7p = next(i for i, c in enumerate(candidates) if c.tile == "7p")
    assert rank_7p >= len(candidates) - 2, f"7p should rank near the bottom (got rank {rank_7p+1}/{len(candidates)})"


def test_paifu_url_parsing_and_sanitization():
    r"""
    Test verifying paifu URL sanitization, Japanese prefix parsing, and habit profiling:
    1. Robust UUID extraction from '雀魂牌譜:https://...' and multi-line pasted inputs.
    2. Complete elimination of Windows filename invalid characters (: ? / \ * " < > |).
    3. Error-free download_and_convert_paifu execution without [Errno 22].
    4. Accurate OpponentProfiler aggregation of new stats (hora, deal_in, calls, badges).
    """
    import json
    from tools.paifu_to_mjai import (
        parse_paifu_url,
        sanitize_filename,
        extract_paifu_uuids,
        download_and_convert_paifu,
        PaifuDownloader,
    )
    from core.opponent_profiler import OpponentProfiler

    # 1. Japanese prefix parsing
    test_prefixed_url = "雀魂牌譜:https://game.mahjongsoul.com/?paifu=261016-64fee530-a705-4f80-b5ab-23d9eb7c6b36_a804191042"
    uuid, seat = parse_paifu_url(test_prefixed_url)
    assert uuid == "261016-64fee530-a705-4f80-b5ab-23d9eb7c6b36", f"Failed to extract clean UUID, got: {uuid}"
    assert ":" not in uuid
    assert "?" not in uuid
    assert "/" not in uuid

    # 2. Sanitize filename safety check
    dirty_name = "雀魂牌譜:https://game.mahjongsoul.com/?paifu=261016*test?uuid<1>|2"
    clean_name = sanitize_filename(dirty_name)
    for forbidden in [":", "?", "/", "\\", "*", '"', "<", ">", "|"]:
        assert forbidden not in clean_name, f"Forbidden char '{forbidden}' found in sanitized name: {clean_name}"

    # 3. Multi-line pasted input handling
    multi_input = """
    雀魂牌譜:https://game.mahjongsoul.com/?paifu=261016-64fee530-a705-4f80-b5ab-23d9eb7c6b36_a804191042
    https://game.mahjongsoul.com/?paifu=261017-11111111-2222-3333-4444-555555555555
    261018-aaaa1111-bbbb-2222-cccc-333333333333
    雀魂牌譜: 261016-64fee530-a705-4f80-b5ab-23d9eb7c6b36
    """
    extracted = extract_paifu_uuids(multi_input)
    assert len(extracted) == 3, f"Expected 3 unique UUIDs from multi-line input, got {len(extracted)}"
    assert extracted[0][0] == "261016-64fee530-a705-4f80-b5ab-23d9eb7c6b36"
    assert extracted[1][0] == "261017-11111111-2222-3333-4444-555555555555"
    assert extracted[2][0] == "261018-aaaa1111-bbbb-2222-cccc-333333333333"

    # 4. Pipeline execution: Ensure authentic record cache is available and download_and_convert_paifu works without [Errno 22]
    cache_dir = os.path.join(AGENT_ROOT, "output", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{uuid}.json")
    # Write authentic mock cache if not present or corrupt
    from tools.paifu_to_mjai import create_mock_authentic_game_record
    auth_mock = create_mock_authentic_game_record(uuid)
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(auth_mock, f, ensure_ascii=False, indent=2)

    out_file, events = download_and_convert_paifu(test_prefixed_url)
    assert os.path.exists(out_file), f"Output file does not exist: {out_file}"
    assert len(events) > 0, "Converted events list must not be empty"

    # 5. Profiling aggregation verification
    profiler = OpponentProfiler()
    profiler.analyze_game_events(events)
    assert len(profiler.profiles) > 0, "Profiler must have generated profiles"

    # Check that new stats and badges were computed
    for p_name, p in profiler.profiles.items():
        assert hasattr(p, "hora_rate")
        assert hasattr(p, "call_tendency")
        assert hasattr(p, "push_fold_tendency")
        assert hasattr(p, "early_game_tendency")
        assert hasattr(p, "badges")
        assert len(p.badges) > 0, f"Player {p_name} must have at least one badge"
        print(f"\n[Test Profiler] {p.display_name}: Badges={p.badges} | Archetype={p.archetype} | Hora={p.hora_rate*100:.1f}% | Deal-in={p.deal_in_rate*100:.1f}%")


def test_authentic_paifu_player_names_and_independent_stats():
    """
    Emergency Regression Test:
    1. Verify extraction of authentic player account nicknames from head.accounts and players.
    2. Verify CPU player slot fallback replaces blanks or placeholder names with 'CPUプレイヤー (西家/北家)'
       instead of arbitrary fictional Japanese surnames.
    3. Verify that player seating (東家, 南家, 西家, 北家) is properly tracked and reflected in display_name.
    4. Verify that across multiple rounds with distinct actions:
       - Player 0 (genta2223): 1 Hora, 1 Riichi, 0 Deal-in -> hora_rate = 1/3 (0.333), deal_in_rate = 0.0
       - Player 1 (プロコフ): 0 Hora, 0 Riichi, 1 Deal-in -> deal_in_rate = 1/3 (0.333), hora_rate = 0.0
       - Player 2 (CPUプレイヤー (西家)): 1 Hora (Tsumo), 1 Call, 1 Deal-in -> hora_rate = 1/3, open_rate = 1/3, deal_in_rate = 1/3
       - Player 3 (CPUプレイヤー (北家)): 1 Hora (Ron), 1 Riichi, 0 Deal-in -> hora_rate = 1/3, deal_in_rate = 0.0
    5. Confirm that stats are completely independent and NOT duplicated across players.
    """
    from tools.paifu_to_mjai import (
        create_mock_authentic_game_record,
        convert_record_to_mjai,
        extract_player_names_from_record,
    )
    from core.opponent_profiler import OpponentProfiler

    uuid = "261016-64fee530-a705-4f80-b5ab-23d9eb7c6b36"
    record = create_mock_authentic_game_record(uuid)

    # 1. Name extraction check (CPU slots must be CPUプレイヤー without fictional family names)
    extracted_names = extract_player_names_from_record(record)
    assert extracted_names[0] == "genta2223"
    assert extracted_names[1] == "プロコフ"
    assert "CPUプレイヤー" in extracted_names[2]
    assert "CPUプレイヤー" in extracted_names[3]
    assert "佐藤" not in extracted_names[2] and "鈴木" not in extracted_names[3]

    # 2. MJAI conversion check
    events = convert_record_to_mjai(record, my_seat=0)
    assert events[0]["type"] == "start_game"
    assert events[0]["names"] == extracted_names

    # 3. Analyze events in profiler
    profiler = OpponentProfiler()
    profiler.analyze_game_events(events)

    # 4. Check each player's independent stats
    p0 = profiler.get_profile("genta2223")
    p1 = profiler.get_profile("プロコフ")
    p2 = profiler.get_profile(extracted_names[2])
    p3 = profiler.get_profile(extracted_names[3])

    assert p0 is not None and p1 is not None and p2 is not None and p3 is not None

    # Check seating and display name formatting
    assert p0.seat_name == "東家"
    assert p0.display_name == "genta2223 (東家)"
    assert p1.seat_name == "南家"
    assert p1.display_name == "プロコフ (南家)"
    assert p2.seat_name == "西家"
    assert "CPUプレイヤー" in p2.display_name and "西家" in p2.display_name
    assert p3.seat_name == "北家"
    assert "CPUプレイヤー" in p3.display_name and "北家" in p3.display_name

    # Total rounds should be 3
    for p in (p0, p1, p2, p3):
        assert p.total_rounds == 3

    # Player 0: 1 Hora (Ron), 1 Riichi, 0 Deal-in
    assert p0.hora_count == 1
    assert p0.deal_in_count == 0
    assert p0.riichi_count == 1
    assert p0.hora_rate == round(1 / 3, 3)
    assert p0.deal_in_rate == 0.0

    # Player 1: 0 Hora, 1 Deal-in (dealt into p0), 0 Riichi
    assert p1.hora_count == 0
    assert p1.deal_in_count == 1
    assert p1.deal_in_rate == round(1 / 3, 3)

    # Player 2 (CPU West): 1 Hora (Tsumo), 1 Call (Chi), 1 Deal-in (dealt into p3 in South 4)
    assert p2.hora_count == 1
    assert p2.call_count >= 1
    assert p2.rounds_with_call == 1
    assert p2.open_rate == round(1 / 3, 3)
    assert p2.deal_in_count == 1

    # Player 3 (CPU North): 1 Hora (Ron), 1 Riichi, 0 Deal-in
    assert p3.hora_count == 1
    assert p3.deal_in_count == 0
    assert p3.riichi_count == 1

    # Ensure stats are independent (NOT duplicated like Self/Player1 in the bug report)
    assert p0.hora_count != p1.hora_count
    assert p0.deal_in_count != p1.deal_in_count
    assert p1.deal_in_count != p3.deal_in_count


def test_paifu_downloader_subthread_asyncio_loop():
    """
    Test verifying asyncio event loop safety in background worker threads:
    Calls _fetch_via_rpc and fetch_paifu from within a fresh threading.Thread (simulating GUI worker).
    Asserts that NO 'RuntimeError: There is no current event loop in thread' is raised.
    """
    import threading
    from tools.paifu_to_mjai import PaifuDownloader

    downloader = PaifuDownloader(cdp_port=9222)
    thread_exceptions = []

    def _worker():
        try:
            # 1. Test _fetch_via_rpc in subthread without active loop (offline port -> gracefully returns None)
            res = downloader._fetch_via_rpc("261016-subthread-test-uuid")
            assert res is None or isinstance(res, dict)

            # 2. Test fetch_paifu in subthread with mock/test UUID
            rec = downloader.fetch_paifu("261016-mock-subthread-worker")
            assert rec is not None
            assert rec.get("uuid") == "261016-mock-subthread-worker"
        except Exception as e:
            thread_exceptions.append(e)

    t = threading.Thread(target=_worker, name="TestWorkerThread")
    t.start()
    t.join(timeout=5.0)

    assert not thread_exceptions, f"Subthread raised unexpected exception: {thread_exceptions}"


def test_paifu_rpc_cdp_target_selection_and_structured_error_handling(monkeypatch, capsys):
    """
    Test CDP target selection and structured {success, error, payload} evaluation:
    1. Target selection priority:
       - Prefers type == 'page' and URL/Title matching 'mahjongsoul' / '雀魂' over generic pages.
    2. Structured error handling:
       - Catches {success: false, error: ...} and prints descriptive error details without crashing.
    3. Structured success handling:
       - Extracts payload when {success: true, payload: ...} is returned.
    4. Runtime exceptionDetails extraction:
       - Formats exception details properly instead of printing empty string.
    """
    import io
    import json
    import asyncio
    import urllib.request
    from tools.paifu_to_mjai import PaifuDownloader

    mock_targets = [
        {"type": "background_page", "title": "Extension", "url": "chrome-extension://abc", "webSocketDebuggerUrl": "ws://dummy/ext"},
        {"type": "page", "title": "Google", "url": "https://www.google.com/", "webSocketDebuggerUrl": "ws://dummy/google"},
        {"type": "page", "title": "雀魂 -じゃんたま-", "url": "https://game.mahjongsoul.com/", "webSocketDebuggerUrl": "ws://dummy/majsoul"},
    ]

    class MockHTTPResponse:
        def __init__(self, data):
            self._data = data
        def read(self):
            return self._data
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: MockHTTPResponse(json.dumps(mock_targets).encode("utf-8")))

    connected_urls = []

    class MockCDPWebSocket:
        def __init__(self, url, main_eval_result, match_cid=2):
            self.url = url
            self.main_eval_result = main_eval_result
            self.match_cid = match_cid
            self.sent_commands = []
            connected_urls.append(url)
            self._incoming = asyncio.Queue()

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

        async def send(self, msg):
            cmd = json.loads(msg)
            self.sent_commands.append(cmd)
            cid = cmd.get("id")
            method = cmd.get("method")
            params = cmd.get("params", {})

            if method == "Runtime.enable":
                # Simulate emission of executionContextCreated for context 2 (e.g. game iframe)
                await self._incoming.put(json.dumps({
                    "method": "Runtime.executionContextCreated",
                    "params": {"context": {"id": 2, "origin": "https://game.mahjongsoul.com", "name": "game_iframe"}}
                }))
                await self._incoming.put(json.dumps({"id": cid, "result": {}}))
            elif method == "Runtime.evaluate":
                expr = params.get("expression", "")
                eval_cid = params.get("contextId")
                if "Boolean((typeof app" in expr:
                    # Probe check
                    is_match = (eval_cid == self.match_cid)
                    await self._incoming.put(json.dumps({"id": cid, "result": {"result": {"type": "boolean", "value": is_match}}}))
                elif "Boolean((typeof uiscript" in expr:
                    await self._incoming.put(json.dumps({"id": cid, "result": {"result": {"type": "boolean", "value": False}}}))
                else:
                    # Main evaluation script
                    await self._incoming.put(json.dumps({"id": cid, "result": self.main_eval_result}))

        async def recv(self):
            return await self._incoming.get()

    downloader = PaifuDownloader(cdp_port=9222)

    import websockets
    # Test 1: Structured failure response (e.g. game object not found in any context)
    fail_eval_result = {
        "result": {
            "type": "string",
            "value": json.dumps({
                "success": False,
                "error": "雀魂のゲームオブジェクト (app.NetAgent / uiscript) が見つかりません。(検出iframe数: 1, 現在URL: https://game.mahjongsoul.com/, 主なwindow変数: Laya)"
            })
        }
    }
    # match_cid=999 means no context matches the probe
    ws_fail = MockCDPWebSocket("ws://dummy/majsoul", fail_eval_result, match_cid=999)
    monkeypatch.setattr(websockets, "connect", lambda url, open_timeout=None: ws_fail)

    res_fail = downloader._fetch_via_rpc("261017-error-test-uuid")
    assert res_fail is None
    assert connected_urls[-1] == "ws://dummy/majsoul"

    captured = capsys.readouterr().out
    assert "Selected primary target for evaluation" in captured
    assert "雀魂のゲームオブジェクト (app.NetAgent / uiscript) が見つかりません" in captured
    assert "検出iframe数: 1" in captured
    assert "Majsoul RPC Error for 261017-error-test-uuid" in captured

    # Test 2: Structured success response with contextId=2 targeted
    success_eval_result = {
        "result": {
            "type": "string",
            "value": json.dumps({
                "success": True,
                "payload": {
                    "uuid": "261017-success-test-uuid",
                    "head": {"accounts": [{"nickname": "genta2223"}]},
                    "rounds": []
                }
            })
        }
    }
    ws_success = MockCDPWebSocket("ws://dummy/majsoul", success_eval_result, match_cid=2)
    monkeypatch.setattr(websockets, "connect", lambda url, open_timeout=None: ws_success)

    res_success = downloader._fetch_via_rpc("261017-success-test-uuid")
    assert res_success is not None
    assert res_success.get("uuid") == "261017-success-test-uuid"
    assert res_success.get("head", {}).get("accounts", [])[0].get("nickname") == "genta2223"

    # Verify that the main evaluation command was sent with contextId == 2
    main_eval_cmds = [c for c in ws_success.sent_commands if c.get("method") == "Runtime.evaluate" and "targetUuid" in c.get("params", {}).get("expression", "")]
    assert len(main_eval_cmds) == 1
    assert main_eval_cmds[0]["params"].get("contextId") == 2

    captured_success = capsys.readouterr().out
    assert "Identified MahjongSoul game context: contextId=2" in captured_success

    # Test 3: Multi-target scanning (Site Isolation) finds game iframe target directly
    multi_targets = [
        {"type": "page", "title": "雀魂 -じゃんたま-", "url": "https://game.mahjongsoul.com/", "webSocketDebuggerUrl": "ws://dummy/outer_page"},
        {"type": "iframe", "title": "Game Engine", "url": "https://game.mahjongsoul.com/game_canvas", "webSocketDebuggerUrl": "ws://dummy/game_iframe"},
    ]
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: MockHTTPResponse(json.dumps(multi_targets).encode("utf-8")))

    class MultiTargetWebSocket:
        def __init__(self, url):
            self.url = url
            self._incoming = asyncio.Queue()

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

        async def send(self, msg):
            cmd = json.loads(msg)
            cid = cmd.get("id")
            method = cmd.get("method")
            params = cmd.get("params", {})
            if method == "Runtime.enable":
                await self._incoming.put(json.dumps({"id": cid, "result": {}}))
            elif method == "Runtime.evaluate":
                expr = params.get("expression", "")
                if "Boolean((typeof app" in expr:
                    # Outer page returns False, game_iframe returns True
                    is_active = (self.url == "ws://dummy/game_iframe")
                    await self._incoming.put(json.dumps({"id": cid, "result": {"result": {"type": "boolean", "value": is_active}}}))
                elif "Boolean((typeof uiscript" in expr:
                    await self._incoming.put(json.dumps({"id": cid, "result": {"result": {"type": "boolean", "value": False}}}))
                else:
                    await self._incoming.put(json.dumps({
                        "id": cid,
                        "result": {
                            "result": {
                                "type": "string",
                                "value": json.dumps({
                                    "success": True,
                                    "payload": {"uuid": "261017-multi-target-uuid", "head": {}, "rounds": []}
                                })
                            }
                        }
                    }))

        async def recv(self):
            return await self._incoming.get()

    monkeypatch.setattr(websockets, "connect", lambda url, open_timeout=None: MultiTargetWebSocket(url))
    res_multi = downloader._fetch_via_rpc("261017-multi-target-uuid")
    assert res_multi is not None
    assert res_multi.get("uuid") == "261017-multi-target-uuid"

    captured_multi = capsys.readouterr().out
    assert "Discovered game target [iframe] Title='Game Engine': app.NetAgent active!" in captured_multi

    # Test 3: CDP exceptionDetails returned (must print exception details, not empty)
    exc_eval_result = {
        "exceptionDetails": {
            "text": "Uncaught",
            "exception": {
                "type": "object",
                "value": "ReferenceError: NetAgent is not defined"
            }
        }
    }
    ws_exc = MockCDPWebSocket("ws://dummy/majsoul", exc_eval_result, match_cid=2)
    monkeypatch.setattr(websockets, "connect", lambda url, open_timeout=None: ws_exc)

    res_exc = downloader._fetch_via_rpc("261017-exc-test-uuid")
    assert res_exc is None
    captured_exc = capsys.readouterr().out
    assert "CDP Runtime evaluation exception: ReferenceError: NetAgent is not defined" in captured_exc


def test_paifu_downloader_http_direct_fetch_priority(monkeypatch, tmp_path):
    """
    Verifies that HTTP direct fetching is the highest priority strategy in PaifuDownloader,
    correctly normalizing JSON responses and decoding binary Protobuf records.
    """
    from tools.paifu_to_mjai import PaifuDownloader
    import json
    import base64
    import io
    import urllib.request

    downloader = PaifuDownloader(cache_dir=str(tmp_path))
    test_uuid = "261017-http-test-uuid"

    # 1. Test JSON response normalization
    sample_json = {
        "uuid": test_uuid,
        "head": {
            "accounts": [
                {"seat": 0, "nickname": "genta2223"},
                {"seat": 1, "nickname": "プロコフ"},
                {"seat": 2, "nickname": ""},
                {"seat": 3, "nickname": ""}
            ]
        },
        "rounds": [{"actions": [{"action_name": "RecordNewRound", "chang": 0, "ju": 0}]}]
    }

    class MockHTTPResponse:
        def __init__(self, data: bytes, status: int = 200):
            self.data = data
            self.status = status
            self.headers = {"Content-Type": "application/json"}
        def read(self):
            return self.data
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    # Mock urlopen to return sample_json for HTTP direct request
    def mock_urlopen(req, timeout=None):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        if "127.0.0.1" in url:
            raise urllib.error.URLError("No local CDP running")
        return MockHTTPResponse(json.dumps(sample_json).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    # Verify _fetch_from_http parses JSON
    fetched = downloader._fetch_from_http(test_uuid)
    assert fetched is not None
    assert fetched["uuid"] == test_uuid
    assert len(fetched["accounts"]) == 4
    assert fetched["accounts"][0]["nickname"] == "genta2223"
    assert len(fetched["rounds"]) == 1

    # Verify _fetch_from_remote prioritizes HTTP direct fetch over CDP and RPC
    called_order = []
    def spy_http(uuid):
        called_order.append("http")
        return sample_json
    def spy_unity(uuid):
        called_order.append("unity_cdp")
        return None
    def spy_rpc(uuid):
        called_order.append("rpc")
        return None

    monkeypatch.setattr(downloader, "_fetch_from_http", spy_http)
    monkeypatch.setattr(downloader, "_fetch_via_unity_cdp", spy_unity)
    monkeypatch.setattr(downloader, "_fetch_via_rpc", spy_rpc)

    res = downloader._fetch_from_remote(test_uuid)
    assert res == sample_json
    assert called_order == ["http"]


def test_paifu_downloader_unity_webgl_cdp_interception(monkeypatch, tmp_path):
    """
    Verifies that when HTTP direct fetching is unavailable, PaifuDownloader falls back to
    Unity WebGL CDP Network interception, triggering replay navigation and capturing game packets.
    """
    from tools.paifu_to_mjai import PaifuDownloader
    import json
    import asyncio
    import websockets
    import urllib.request

    downloader = PaifuDownloader(cache_dir=str(tmp_path))
    test_uuid = "261017-unity-cdp-uuid"

    # Mock CDP targets list
    targets_data = [
        {
            "id": "target-unity",
            "type": "page",
            "title": "雀魂 - 4.0.12 Unity WebGL",
            "url": "https://game.mahjongsoul.com/",
            "webSocketDebuggerUrl": "ws://dummy/unity_target"
        }
    ]

    class MockCDPTargetResponse:
        def __init__(self, data: bytes):
            self.data = data
            self.status = 200
            self.headers = {}
        def read(self):
            return self.data
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    def mock_cdp_urlopen(req, timeout=None):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        if "/json/new" in url:
            new_tab_info = {
                "id": "temp-tab-123",
                "type": "page",
                "title": "",
                "url": url,
                "webSocketDebuggerUrl": "ws://dummy/unity_target"
            }
            return MockCDPTargetResponse(json.dumps(new_tab_info).encode("utf-8"))
        elif "/json/close" in url:
            return MockCDPTargetResponse(b"Target is closing")
        elif "127.0.0.1" in url and "/json" in url:
            return MockCDPTargetResponse(json.dumps(targets_data).encode("utf-8"))
        raise urllib.error.URLError("Not found")

    monkeypatch.setattr(urllib.request, "urlopen", mock_cdp_urlopen)

    # Mock WebSocket connection that simulates Unity WebGL CDP Network events
    class MockUnityCDPWebSocket:
        def __init__(self, url):
            self.url = url
            self._incoming = asyncio.Queue()
            self._sent_cmds = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

        async def send(self, msg_str):
            msg = json.loads(msg_str)
            self._sent_cmds.append(msg)
            method = msg.get("method")
            cid = msg.get("id")

            if method in ("Page.enable", "Network.enable", "Runtime.enable"):
                await self._incoming.put(json.dumps({"id": cid, "result": {}}))
                # Emit Network.responseReceived with paipu url
                await self._incoming.put(json.dumps({
                    "method": "Network.responseReceived",
                    "params": {
                        "requestId": "req-paifu-123",
                        "response": {"url": f"https://game.mahjongsoul.com/api/record/{test_uuid}", "status": 200}
                    }
                }))
            elif method == "Page.navigate":
                await self._incoming.put(json.dumps({"id": cid, "result": {}}))
            elif method == "Network.getResponseBody":
                # Return JSON game record
                record_payload = {
                    "uuid": test_uuid,
                    "head": {"accounts": [{"seat": 0, "nickname": "genta2223"}]},
                    "rounds": [{"actions": [{"action_name": "RecordNewRound"}]}]
                }
                await self._incoming.put(json.dumps({
                    "id": cid,
                    "result": {
                        "body": json.dumps(record_payload),
                        "base64Encoded": False
                    }
                }))

        async def recv(self):
            return await self._incoming.get()

    monkeypatch.setattr(websockets, "connect", lambda url, open_timeout=None: MockUnityCDPWebSocket(url))

    # Verify fallback order: HTTP fails -> Unity CDP succeeds
    monkeypatch.setattr(downloader, "_fetch_from_http", lambda uuid: None)

    res = downloader._fetch_from_remote(test_uuid)
    assert res is not None
    assert res.get("uuid") == test_uuid
    assert len(res.get("rounds", [])) == 1
    assert res.get("accounts", [])[0]["nickname"] == "genta2223"


def test_paifu_downloader_unity_webgl_websocket_binary_frame(monkeypatch, tmp_path):
    """
    Verifies that PaifuDownloader intercepts binary WebSocket frames (WSS) and
    extracts game actions via decode_game_detail_records.
    """
    from tools.paifu_to_mjai import PaifuDownloader, LiqiProtoDecoder
    import json
    import gzip
    import base64
    import asyncio
    import websockets
    import urllib.request

    downloader = PaifuDownloader(cache_dir=str(tmp_path))
    test_uuid = "261017-ws-frame-uuid"

    # Construct mock protobuf payload wrapped in gzip with prefix and trailing bytes (arbitrary offset)
    name_bytes = b"RecordNewRound"
    wrapper_bytes = b"\x0a" + bytes([len(name_bytes)]) + name_bytes + b"\x12\x02\x08\x00"
    payload = b"\x0a" + bytes([len(wrapper_bytes)]) + wrapper_bytes
    gzipped = gzip.compress(payload)
    # Add Liqi RPC header prefix (offset > 0) and trailing protobuf metadata bytes
    prefix = b"\x02\x00\x05\x12\x34"
    trailing = b"\x22\x06extra!"
    framed_packet = prefix + gzipped + trailing
    b64_payload = base64.b64encode(framed_packet).decode("ascii")

    targets_data = [
        {
            "id": "target-unity",
            "type": "page",
            "title": "雀魂 - Unity WebGL",
            "url": "https://game.mahjongsoul.com/",
            "webSocketDebuggerUrl": "ws://dummy/unity_ws"
        }
    ]

    class MockCDPTargetResponse:
        def __init__(self, data: bytes):
            self.data = data
            self.status = 200
            self.headers = {}
        def read(self):
            return self.data
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    def mock_urlopen(req, timeout=None):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        if "/json/new" in url:
            new_tab = {
                "id": "temp-ws-tab",
                "type": "page",
                "title": "",
                "url": url,
                "webSocketDebuggerUrl": "ws://dummy/temp_ws"
            }
            return MockCDPTargetResponse(json.dumps(new_tab).encode("utf-8"))
        elif "/json/close" in url:
            return MockCDPTargetResponse(b"Closed")
        return MockCDPTargetResponse(json.dumps(targets_data).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    class MockBinaryWSWebSocket:
        def __init__(self, url):
            self.url = url
            self._incoming = asyncio.Queue()

        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

        async def send(self, msg_str):
            msg = json.loads(msg_str)
            cid = msg.get("id")
            method = msg.get("method")
            if method in ("Page.enable", "Network.enable", "Runtime.enable"):
                await self._incoming.put(json.dumps({"id": cid, "result": {}}))
                # Push binary WebSocket frame containing gzipped game detail records at offset 5
                await self._incoming.put(json.dumps({
                    "method": "Network.webSocketFrameReceived",
                    "params": {
                        "response": {
                            "opcode": 2,
                            "payloadData": b64_payload
                        }
                    }
                }))

        async def recv(self):
            return await self._incoming.get()

    monkeypatch.setattr(websockets, "connect", lambda url, open_timeout=None: MockBinaryWSWebSocket(url))
    monkeypatch.setattr(downloader, "_fetch_from_http", lambda uuid: None)

    res = downloader._fetch_from_remote(test_uuid)
    assert res is not None
    assert res.get("uuid") == test_uuid
    assert len(res.get("rounds", [])) == 1
    assert res["rounds"][0]["actions"][0]["action_name"] == "RecordNewRound"


def test_paifu_downloader_decompress_embedded_gzip_offset(tmp_path):
    """
    Direct unit test verifying _decompress_gzip_slice and _try_decode_records
    correctly find and extract gzip stream buried at arbitrary offset with trailing data.
    """
    from tools.paifu_to_mjai import PaifuDownloader
    import gzip

    downloader = PaifuDownloader(cache_dir=str(tmp_path))
    test_uuid = "261017-embedded-gzip-test"

    name_bytes = b"RecordNewRound"
    wrapper_bytes = b"\x0a" + bytes([len(name_bytes)]) + name_bytes + b"\x12\x02\x08\x00"
    payload = b"\x0a" + bytes([len(wrapper_bytes)]) + wrapper_bytes
    gzipped = gzip.compress(payload)

    # Buried at offset 12 with 8 bytes trailing metadata
    garbage_prefix = b"LIQI_PACKET_"
    garbage_suffix = b"_ENDDATA"
    framed_data = garbage_prefix + gzipped + garbage_suffix

    decompressed = downloader._decompress_gzip_slice(framed_data)
    assert decompressed is not None
    assert payload in decompressed

    record = downloader._try_decode_records(decompressed, test_uuid)
    assert record is not None
    assert record["uuid"] == test_uuid
    assert len(record["rounds"][0]["actions"]) == 1
    assert record["rounds"][0]["actions"][0]["action_name"] == "RecordNewRound"



def test_paifu_downloader_skips_legacy_rpc_on_unity(monkeypatch, tmp_path, capsys):
    """
    Verifies that when Unity WebGL is detected, PaifuDownloader skips obsolete LayaAir app.NetAgent probe.
    """
    from tools.paifu_to_mjai import PaifuDownloader
    downloader = PaifuDownloader(cache_dir=str(tmp_path))

    monkeypatch.setattr(downloader, "_fetch_from_http", lambda uuid: None)
    monkeypatch.setattr(downloader, "_fetch_via_unity_cdp", lambda uuid: None)
    monkeypatch.setattr(downloader, "_is_unity_environment", lambda: True)

    rpc_called = False
    def mock_rpc(uuid):
        nonlocal rpc_called
        rpc_called = True
        return None

    monkeypatch.setattr(downloader, "_fetch_via_rpc", mock_rpc)

    res = downloader._fetch_from_remote("261017-skip-rpc-uuid")
    assert res is None
    assert not rpc_called, "Legacy RPC should NOT be called in Unity WebGL environment!"
    captured = capsys.readouterr().out
    assert "Unity WebGL environment detected: Skipping obsolete LayaAir app.NetAgent RPC probe." in captured


def test_paifu_downloader_large_liqi_frame_res_game_record(tmp_path):
    """
    Verifies that a large 70KB-110KB Liqi WebSocket response frame (type 0x03, req_id 2 bytes)
    containing ResGameRecord with field 2 (gzipped GameDetailRecords) and field 4 (head accounts)
    is decoded directly and extracts both authentic player names and action events.
    """
    from tools.paifu_to_mjai import PaifuDownloader
    import gzip
    import base64

    downloader = PaifuDownloader(cache_dir=str(tmp_path))
    test_uuid = "261017-large-liqi-frame-uuid"

    # 1. Construct GameDetailRecords payload (records field 1) with multiple actions
    action_1 = b"\x0a\x0eRecordNewRound\x12\x02\x08\x00"
    action_2 = b"\x0a\x0eActionDealTile\x12\x04\x08\x01\x10\x12"
    action_3 = b"\x0a\x0aRecordHule\x12\x06\x08\x00\x10\xc8\x01\x18\x01"
    
    # Construct authentic records payload
    records_payload = (
        b"\x0a" + bytes([len(action_1)]) + action_1 +
        b"\x0a" + bytes([len(action_2)]) + action_2 +
        b"\x0a" + bytes([len(action_3)]) + action_3
    )
    gzipped_records = gzip.compress(records_payload)

    # 2. Construct RecordGame (field 4: head) with accounts (field 3)
    # Account 0: seat=0, nickname="genta2223"
    acc0_inner = b"\x08\x00\x12\x09genta2223"
    acc0_blob = b"\x1a" + bytes([len(acc0_inner)]) + acc0_inner
    # Account 1: seat=1, nickname="プロコフ"
    nick_utf8 = "プロコフ".encode("utf-8")
    acc1_inner = b"\x08\x01\x12" + bytes([len(nick_utf8)]) + nick_utf8
    acc1_blob = b"\x1a" + bytes([len(acc1_inner)]) + acc1_inner
    head_payload = acc0_blob + acc1_blob

    # 3. Construct ResGameRecord:
    # Field 2: bytes data (length-delimited tag (2<<3)|2 = 0x12)
    def encode_varint(n: int) -> bytes:
        res = bytearray()
        while n >= 0x80:
            res.append((n & 0x7F) | 0x80)
            n >>= 7
        res.append(n)
        return bytes(res)

    res_game_record = (
        b"\x12" + encode_varint(len(gzipped_records)) + gzipped_records +
        b"\x22" + encode_varint(len(head_payload)) + head_payload
    )

    # 4. Construct Liqi RPC frame (type 0x03, req_id 0x0005)
    liqi_header = b"\x03\x05\x00"
    large_frame = liqi_header + res_game_record

    # Pad large_frame to ~75KB to simulate realistic 75KB-110KB match packet
    padding_comment = bytes([(i * 37) % 256 for i in range(75000)])
    large_frame += padding_comment
    assert len(large_frame) >= 75000

    # Execute decoding
    result = downloader._try_decode_records(large_frame, test_uuid)
    assert result is not None, "Failed to decode large Liqi ResGameRecord frame!"
    assert result["uuid"] == test_uuid
    assert len(result["rounds"][0]["actions"]) >= 3
    assert result["rounds"][0]["actions"][0]["action_name"] == "RecordNewRound"
    
    # Check that accounts were extracted from head
    assert "accounts" in result
    assert len(result["accounts"]) == 2
    assert result["accounts"][0]["nickname"] == "genta2223"
    assert result["accounts"][1]["nickname"] == "プロコフ"
def test_paifu_sniffer_process_frame_and_deduplication(tmp_path):
    """
    Verifies that PaifuSniffer.process_frame correctly decodes a large ResGameRecord packet,
    caches it to disk, adds unique UUID to captured_records, and drops duplicate UUID frames.
    """
    from tools.paifu_to_mjai import PaifuSniffer
    import gzip

    sniffer = PaifuSniffer(cdp_port=9222, max_games=3, cache_dir=str(tmp_path))

    def create_mock_frame(uuid_str: str, nick: str) -> bytes:
        action_1 = b"\x0a\x0eRecordNewRound\x12\x02\x08\x00"
        action_2 = b"\x0a\x0eActionDealTile\x12\x04\x08\x01\x10\x12"
        action_3 = b"\x0a\x0aRecordHule\x12\x06\x08\x00\x10\xc8\x01\x18\x01"
        records_payload = (
            b"\x0a" + bytes([len(action_1)]) + action_1 +
            b"\x0a" + bytes([len(action_2)]) + action_2 +
            b"\x0a" + bytes([len(action_3)]) + action_3
        )
        gzipped = gzip.compress(records_payload)

        u_bytes = uuid_str.encode("utf-8")
        uuid_field = b"\x0a" + bytes([len(u_bytes)]) + u_bytes
        nick_bytes = nick.encode("utf-8")
        acc_inner = b"\x08\x01\x12" + bytes([len(nick_bytes)]) + nick_bytes
        acc_field = b"\x1a" + bytes([len(acc_inner)]) + acc_inner
        head = uuid_field + acc_field

        def encode_varint(n: int) -> bytes:
            res = bytearray()
            while n >= 0x80:
                res.append((n & 0x7F) | 0x80)
                n >>= 7
            res.append(n)
            return bytes(res)

        res_game = (
            b"\x12" + encode_varint(len(gzipped)) + gzipped +
            b"\x22" + encode_varint(len(head)) + head
        )
        frame = b"\x03\x01\x00" + res_game
        frame += b"\x00" * 12000
        return frame

    frame1 = create_mock_frame("261017-match-001", "プレイヤーA")
    frame2 = create_mock_frame("261017-match-002", "プレイヤーB")

    # 1. Process frame 1
    rec1 = sniffer.process_frame(frame1)
    assert rec1 is not None
    assert rec1["uuid"] == "261017-match-001"
    assert len(sniffer.captured_records) == 1
    assert "261017-match-001" in sniffer.captured_uuids

    # Verify cache file created
    cache_file = os.path.join(str(tmp_path), "261017-match-001.json")
    assert os.path.exists(cache_file)

    # 2. Duplicate frame 1 should be ignored (None returned, count stays 1)
    rec1_dup = sniffer.process_frame(frame1)
    assert rec1_dup is None
    assert len(sniffer.captured_records) == 1

    # 3. Process frame 2 (unique match)
    rec2 = sniffer.process_frame(frame2)
    assert rec2 is not None
    assert rec2["uuid"] == "261017-match-002"
    assert len(sniffer.captured_records) == 2


@pytest.mark.asyncio
async def test_paifu_sniffer_sequential_capture_and_profiler_integration(tmp_path, monkeypatch):
    """
    Verifies that PaifuSniffer attaches to an existing Chrome WebSocket session,
    sends Network.enable, captures up to 3 sequential paifu packets via Network.webSocketFrameReceived,
    stops when reaching max_games=3, and feeds all game events into OpponentProfiler.
    """
    from tools.paifu_to_mjai import PaifuSniffer
    from core.opponent_profiler import OpponentProfiler
    import gzip
    import base64
    import json
    import asyncio

    sniffer = PaifuSniffer(cdp_port=9222, max_games=3, cache_dir=str(tmp_path))

    def make_frame_base64(uuid_str: str, nick: str) -> str:
        action_1 = b"\x0a\x0eRecordNewRound\x12\x02\x08\x00"
        action_2 = b"\x0a\x0eActionDealTile\x12\x04\x08\x01\x10\x12"
        action_3 = b"\x0a\x0aRecordHule\x12\x06\x08\x00\x10\xc8\x01\x18\x01"
        records_payload = (
            b"\x0a" + bytes([len(action_1)]) + action_1 +
            b"\x0a" + bytes([len(action_2)]) + action_2 +
            b"\x0a" + bytes([len(action_3)]) + action_3
        )
        gzipped = gzip.compress(records_payload)
        u_bytes = uuid_str.encode("utf-8")
        uuid_field = b"\x0a" + bytes([len(u_bytes)]) + u_bytes
        nick_bytes = nick.encode("utf-8")
        acc_inner = b"\x08\x01\x12" + bytes([len(nick_bytes)]) + nick_bytes
        acc_field = b"\x1a" + bytes([len(acc_inner)]) + acc_inner
        head = uuid_field + acc_field

        def encode_varint(n: int) -> bytes:
            res = bytearray()
            while n >= 0x80:
                res.append((n & 0x7F) | 0x80)
                n >>= 7
            res.append(n)
            return bytes(res)

        res_game = (
            b"\x12" + encode_varint(len(gzipped)) + gzipped +
            b"\x22" + encode_varint(len(head)) + head
        )
        raw_b = b"\x03\x01\x00" + res_game + (b"\x00" * 12000)
        return base64.b64encode(raw_b).decode("ascii")

    frames_b64 = [
        make_frame_base64(f"261017-stream-match-00{i}", f"Opponent_{i}")
        for i in range(1, 4)
    ]

    sent_messages = []
    class MockWS:
        def __init__(self):
            self.msg_idx = 0
            self.closed = False

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            self.closed = True

        async def send(self, msg):
            sent_messages.append(json.loads(msg))

        async def recv(self):
            if self.msg_idx < len(frames_b64):
                payload = frames_b64[self.msg_idx]
                self.msg_idx += 1
                msg = {
                    "method": "Network.webSocketFrameReceived",
                    "params": {
                        "response": {
                            "payloadData": payload
                        }
                    }
                }
                return json.dumps(msg)
            await asyncio.sleep(0.1)
            raise asyncio.TimeoutError()

    monkeypatch.setattr(sniffer, "_find_majsoul_target", lambda: "ws://127.0.0.1:9222/devtools/page/TEST")

    import websockets
    monkeypatch.setattr(websockets, "connect", lambda *args, **kwargs: MockWS())

    captured_list = []
    status_list = []
    finished_list = []

    profiler = OpponentProfiler()

    def on_captured(rec, count, max_count, events):
        captured_list.append((rec, count, max_count))
        if events:
            profiler.analyze_game_events(events)

    def on_status(msg):
        status_list.append(msg)

    def on_finished(records):
        finished_list.append(records)

    sniffer.is_running = True
    await sniffer._listen_async(
        on_captured=on_captured,
        on_status=on_status,
        on_finished=on_finished,
        on_error=None
    )

    # 1. Verify Network.enable was sent
    assert any(m.get("method") == "Network.enable" for m in sent_messages)

    # 2. Verify all 3 matches captured sequentially
    assert len(captured_list) == 3
    assert captured_list[0][1] == 1
    assert captured_list[1][1] == 2
    assert captured_list[2][1] == 3
    assert len(sniffer.captured_records) == 3

    # 3. Verify finished callback called with 3 records
    assert len(finished_list) == 1
    assert len(finished_list[0]) == 3

    # 4. Verify profiler updated
    assert len(profiler.profiles) > 0


def test_paifu_authentic_player_names_and_round_action_profiling(tmp_path):
    """
    Verifies that authentic player nicknames (genta2223, ブロコフ, 龍斗たっちゃん, 横丁の隠居)
    are accurately extracted from protobuf head.accounts across seats 0..3,
    passed to mjai events, and correctly populate OpponentProfiler round stats (not 0 rounds / 0.0%).
    """
    from tools.paifu_to_mjai import PaifuDownloader, extract_player_names_from_record, convert_record_to_mjai
    from core.opponent_profiler import OpponentProfiler
    import gzip

    downloader = PaifuDownloader(cache_dir=str(tmp_path))
    test_uuid = "261017-authentic-names-match"

    expected_players = ["genta2223", "ブロコフ", "龍斗たっちゃん", "横丁の隠居"]

    def encode_varint(n: int) -> bytes:
        res = bytearray()
        while n >= 0x80:
            res.append((n & 0x7F) | 0x80)
            n >>= 7
        res.append(n)
        return bytes(res)

    # 1. Build authentic account blobs for all 4 seats
    acc_blobs = []
    for seat, name in enumerate(expected_players):
        nb = name.encode("utf-8")
        # Format: field 1: acc_id, field 2: seat, field 3: nickname
        acc_payload = (
            b"\x08" + encode_varint(20001 + seat) +
            b"\x10" + bytes([seat]) +
            b"\x1a" + bytes([len(nb)]) + nb
        )
        acc_blobs.append(b"\x1a" + bytes([len(acc_payload)]) + acc_payload)

    # 2. Build RecordGame head with UUID and repeated accounts
    uuid_bytes = test_uuid.encode("utf-8")
    head_payload = (
        b"\x0a" + bytes([len(uuid_bytes)]) + uuid_bytes +
        b"".join(acc_blobs)
    )

    # 3. Build multiple authentic round actions
    action_1 = b"\x0a\x0eRecordNewRound\x12\x02\x08\x00"
    action_2 = b"\x0a\x0eActionDealTile\x12\x04\x08\x01\x10\x12"
    action_3 = b"\x0a\x12RecordDiscardTile\x12\x04\x08\x01\x10\x12"
    action_4 = b"\x0a\x0aRecordHule\x12\x06\x08\x01\x10\xc8\x01\x18\x00"
    records_payload = (
        b"\x0a" + bytes([len(action_1)]) + action_1 +
        b"\x0a" + bytes([len(action_2)]) + action_2 +
        b"\x0a" + bytes([len(action_3)]) + action_3 +
        b"\x0a" + bytes([len(action_4)]) + action_4
    )
    gzipped_records = gzip.compress(records_payload)

    # 4. Assemble ResGameRecord (field 2 = data, field 3 = head)
    res_record = (
        b"\x12" + encode_varint(len(gzipped_records)) + gzipped_records +
        b"\x1a" + encode_varint(len(head_payload)) + head_payload
    )
    frame = b"\x03\x05\x00" + res_record + (b"\x00" * 12000)

    # Decode frame
    rec = downloader._try_decode_records(frame, test_uuid)
    assert rec is not None, "Failed to decode frame with authentic accounts"
    assert rec["uuid"] == test_uuid

    # Check extracted names
    names = extract_player_names_from_record(rec)
    assert names == expected_players, f"Extracted names {names} does not match {expected_players}"

    # Convert to MJAI events
    events = convert_record_to_mjai(rec, my_seat=0)
    assert len(events) >= 3
    assert events[0]["type"] == "start_game"
    assert events[0]["names"] == expected_players

    # Feed into OpponentProfiler
    profiler = OpponentProfiler()
    profiler.analyze_game_events(events)

    # Verify all 4 players have profiles created with actual round stats
    for name in expected_players:
        p = profiler.get_profile(name)
        assert p is not None, f"Profile for {name} was not created!"
        assert "CPUプレイヤー" not in p.name
        assert p.total_rounds >= 1, f"Total rounds for {name} should be >= 1, got {p.total_rounds}"

    # Verify winner (ブロコフ: seat 1) stats
    p_winner = profiler.get_profile("ブロコフ")
    assert p_winner.hora_count >= 1
    assert p_winner.hora_rate > 0.0


@pytest.mark.asyncio
async def test_paifu_sniffer_child_session_auto_attach_and_frame_logging(tmp_path, monkeypatch, capsys):
    """
    Verifies that PaifuSniffer issues Target.setAutoAttach (with flatten=True) and Network.enable,
    automatically activates Network.enable on newly attached child sessions (Target.attachedToTarget),
    logs every received WebSocket binary frame (len and opcode),
    and successfully processes matches without the 8000-byte length threshold.
    """
    from tools.paifu_to_mjai import PaifuSniffer
    import gzip
    import base64
    import json

    sniffer = PaifuSniffer(cache_dir=str(tmp_path), max_games=1)
    sent_messages = []

    # Construct authentic match frame (around 2KB, under old 8000 byte threshold)
    test_uuid = "261017-child-session-test"
    action_1 = b"\x0a\x0eRecordNewRound\x12\x02\x08\x00"
    action_2 = b"\x0a\x0eActionDealTile\x12\x04\x08\x01\x10\x12"
    records_payload = b"\x0a" + bytes([len(action_1)]) + action_1 + b"\x0a" + bytes([len(action_2)]) + action_2
    gzipped_records = gzip.compress(records_payload)
    head_payload = b"\x0a" + bytes([len(test_uuid)]) + test_uuid.encode("utf-8")
    def encode_varint(n: int) -> bytes:
        res = bytearray()
        while n >= 0x80:
            res.append((n & 0x7F) | 0x80)
            n >>= 7
        res.append(n)
        return bytes(res)
    res_record = (
        b"\x12" + encode_varint(len(gzipped_records)) + gzipped_records +
        b"\x1a" + encode_varint(len(head_payload)) + head_payload
    )
    frame_bytes = b"\x03\x05\x00" + res_record
    assert len(frame_bytes) < 8000, "Should be smaller than 8000 bytes to test limit relaxation"
    frame_b64 = base64.b64encode(frame_bytes).decode("ascii")

    # Sequence of incoming CDP messages:
    # 1. Target.attachedToTarget for child session (e.g. Unity WebGL worker/target)
    # 2. Network.webSocketFrameReceived on child session
    messages_to_receive = [
        {
            "method": "Target.attachedToTarget",
            "params": {
                "sessionId": "SESSION_CHILD_12345",
                "targetInfo": {"type": "worker", "url": "blob:https://mahjongsoul.game.yo-star.com/worker.js"}
            }
        },
        {
            "method": "Network.webSocketFrameReceived",
            "sessionId": "SESSION_CHILD_12345",
            "params": {
                "response": {
                    "opcode": 2,
                    "payloadData": frame_b64
                }
            }
        }
    ]

    class MockWS:
        def __init__(self):
            self.msg_idx = 0

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        async def send(self, msg_text):
            sent_messages.append(json.loads(msg_text))

        async def recv(self):
            if self.msg_idx < len(messages_to_receive):
                m = messages_to_receive[self.msg_idx]
                self.msg_idx += 1
                return json.dumps(m)
            await asyncio.sleep(0.05)
            raise asyncio.TimeoutError()

    monkeypatch.setattr(sniffer, "_find_majsoul_target", lambda: "ws://127.0.0.1:9222/devtools/page/MAIN")
    import websockets
    monkeypatch.setattr(websockets, "connect", lambda *args, **kwargs: MockWS())

    captured_list = []
    sniffer.is_running = True
    await sniffer._listen_async(
        on_captured=lambda rec, cnt, mx, ev: captured_list.append(rec),
        on_status=None,
        on_finished=None,
        on_error=None
    )

    # 1. Verify Target.setAutoAttach was sent with flatten: True
    auto_attach_msgs = [m for m in sent_messages if m.get("method") == "Target.setAutoAttach"]
    assert len(auto_attach_msgs) >= 1
    assert auto_attach_msgs[0]["params"].get("autoAttach") is True
    assert auto_attach_msgs[0]["params"].get("flatten") is True

    # 2. Verify Network.enable was sent both globally and to the child sessionId
    net_enable_msgs = [m for m in sent_messages if m.get("method") == "Network.enable"]
    assert len(net_enable_msgs) >= 2
    assert any(m.get("sessionId") == "SESSION_CHILD_12345" for m in net_enable_msgs)

    # 3. Verify match was successfully captured even though < 8000 bytes
    assert len(captured_list) == 1
    assert captured_list[0]["uuid"] == test_uuid

    # 4. Verify stdout captured received frame log with opcode and len
    captured_stdout = capsys.readouterr().out
    assert "[PaifuSniffer] Received Frame: len=" in captured_stdout
    assert "opcode=2" in captured_stdout


def test_paifu_sniffer_large_frame_38625_diagnostic_logging_and_acceptance(tmp_path, capsys):
    """
    Verifies that a large binary WebSocket frame of exactly len=38625 (matching real-game logs)
    triggers the 5-step diagnostic logs in console and GUI callback:
      ① Protobuf wire-format parse (tags)
      ② gzip decompression (offset, compressed/decompressed length, success)
      ③ Action extraction (action count, sample actions)
      ④ Player name extraction (seat to nicknames)
      ⑤ Final acceptance judgment (adopted N/3 with UUID)
    and successfully accumulates into captured_paifus/captured_records with authentic players.
    """
    from tools.paifu_to_mjai import PaifuSniffer
    import gzip

    sniffer = PaifuSniffer(cache_dir=str(tmp_path), max_games=3)

    def encode_varint(n: int) -> bytes:
        res = bytearray()
        while n >= 0x80:
            res.append((n & 0x7F) | 0x80)
            n >>= 7
        res.append(n)
        return bytes(res)

    # 1. Accounts with authentic nicknames
    players = ["genta2223", "ブロコフ", "龍斗たっちゃん", "横丁の隠居"]
    acc_blobs = []
    for seat, name in enumerate(players):
        nb = name.encode("utf-8")
        payload = (
            b"\x08" + encode_varint(10001 + seat) +
            b"\x10" + bytes([seat]) +
            b"\x1a" + bytes([len(nb)]) + nb
        )
        acc_blobs.append(b"\x1a" + bytes([len(payload)]) + payload)

    head_payload = b"".join(acc_blobs)

    # 2. Rich authentic round actions
    actions_raw = []
    for r in range(4):
        a1 = b"\x0a\x0eRecordNewRound\x12\x04\x08" + bytes([r]) + b"\x10\x00"
        a2 = b"\x0a\x0eActionDealTile\x12\x04\x08\x00\x10\x12"
        a3 = b"\x0a\x12RecordDiscardTile\x12\x04\x08\x00\x10\x12"
        a4 = b"\x0a\x0aRecordHule\x12\x06\x08\x01\x10\xc8\x01\x18\x00"
        actions_raw.extend([
            b"\x0a" + bytes([len(a1)]) + a1,
            b"\x0a" + bytes([len(a2)]) + a2,
            b"\x0a" + bytes([len(a3)]) + a3,
            b"\x0a" + bytes([len(a4)]) + a4,
        ])
    gzipped_records = gzip.compress(b"".join(actions_raw))

    # 3. Assemble ResGameRecord (field 2 data, field 3 head)
    res_record = (
        b"\x12" + encode_varint(len(gzipped_records)) + gzipped_records +
        b"\x1a" + encode_varint(len(head_payload)) + head_payload
    )

    # Liqi header (0x03, 0x05, 0x00) + payload
    frame_base = b"\x03\x05\x00" + res_record

    # Pad frame to exactly 38625 bytes to simulate real-game packet
    TARGET_LEN = 38625
    assert len(frame_base) < TARGET_LEN
    padding = b"\x00" * (TARGET_LEN - len(frame_base))
    large_frame = frame_base + padding
    assert len(large_frame) == TARGET_LEN

    diag_messages = []
    def on_diag(msg):
        diag_messages.append(msg)

    # Execute frame processing
    rec = sniffer.process_frame(large_frame, on_diag=on_diag)

    # Verification:
    # A. Frame was accepted (not skipped)
    assert rec is not None, "Frame len=38625 must be accepted and not skipped!"
    assert len(sniffer.captured_records) == 1
    assert len(sniffer.captured_paifus) == 1
    assert sniffer.captured_paifus is sniffer.captured_records

    # B. UUID was assigned (either extracted or fallback auto_)
    assert rec["uuid"].startswith("auto_") or "38625" in rec["uuid"]

    # C. Actions extracted
    actions = rec["rounds"][0]["actions"]
    assert len(actions) >= 16
    assert actions[0]["action_name"] == "RecordNewRound"

    # D. Authentic accounts preserved
    assert len(rec["accounts"]) == 4
    assert [a["nickname"] for a in rec["accounts"]] == players

    # E. 5-step diagnostic logs emitted
    captured_stdout = capsys.readouterr().out
    combined_log = captured_stdout + "\n" + "\n".join(diag_messages)

    assert "巨大フレーム検知: len=38625" in combined_log
    assert "① Protobuf wire-format パース:" in combined_log
    assert "② gzip 解凍結果:" in combined_log
    assert "③ アクション抽出結果: 検出アクション数=" in combined_log
    assert "④ プレイヤー名抽出結果: 席順別実名=" in combined_log
    assert "⑤ 最終合否判定: 牌譜採用 (1/3件)" in combined_log


def test_paifu_stats_profiler_hora_houju_riichi_and_calls():
    """
    Verifies that:
    1. RecordHule correctly attributes hora (agari) to winner and deal-in (houju) to loser on Ron,
       and only agari without deal-in on Tsumo.
    2. RecordDiscardTile with is_liqi=True generates reach event and increments reach count.
    3. RecordChiPengGang correctly attributes call to actor (seat who called), distinguishing
       yakuhai calls (honors) from tanyao calls (simples 2-8).
    4. OpponentProfiler accurately accumulates both self.stats[seat] and PlayerProfile metrics.
    """
    from tools.paifu_to_mjai import convert_record_to_mjai
    from core.opponent_profiler import OpponentProfiler

    players = ["genta2223", "ブロコフ", "龍斗たっちゃん", "横丁の隠居"]

    record = {
        "uuid": "test-stats-profiler-verification",
        "accounts": [
            {"seat": 0, "nickname": "genta2223"},
            {"seat": 1, "nickname": "ブロコフ"},
            {"seat": 2, "nickname": "龍斗たっちゃん"},
            {"seat": 3, "nickname": "横丁の隠居"},
        ],
        "rounds": [
            # Round 1:
            # - Seat 2 (龍斗たっちゃん) calls Pon on 5z (White Dragon: Yakuhai) from Seat 0
            # - Seat 0 (genta2223) declares Riichi
            # - Seat 1 (ブロコフ) discards 5m and deals in to Seat 0 (genta2223 Ron)
            {
                "round_info": {"chang": 0, "ju": 0, "ben": 0, "liqibang": 0},
                "actions": [
                    {"action_name": "RecordNewRound", "data": {"chang": 0, "ju": 0, "ben": 0, "liqibang": 0, "scores": [25000, 25000, 25000, 25000], "doras": ["1p"]}},
                    {"action_name": "RecordDiscardTile", "data": {"seat": 0, "tile": "5z", "moqie": False}},
                    {"action_name": "RecordChiPengGang", "data": {"seat": 2, "type": 1, "tiles": ["5z", "5z", "5z"], "froms": [2, 2, 0]}},
                    {"action_name": "RecordDiscardTile", "data": {"seat": 2, "tile": "9p", "moqie": False}},
                    {"action_name": "RecordDiscardTile", "data": {"seat": 0, "tile": "1m", "moqie": False, "is_liqi": True}},
                    {"action_name": "RecordDiscardTile", "data": {"seat": 1, "tile": "5m", "moqie": False}},
                    {"action_name": "RecordHule", "data": {"hules": [{"seat": 0, "target": 1, "hu_tile": "5m", "zimo": False}], "delta_scores": [8000, -8000, 0, 0]}},
                ]
            },
            # Round 2:
            # - Seat 3 (横丁の隠居) calls Chi on 3p-4p-5p (All-simples: Tanyao) from Seat 2
            # - Seat 2 (龍斗たっちゃん) wins by Tsumo on 2s
            {
                "round_info": {"chang": 0, "ju": 1, "ben": 0, "liqibang": 0},
                "actions": [
                    {"action_name": "RecordNewRound", "data": {"chang": 0, "ju": 1, "ben": 0, "liqibang": 0, "scores": [33000, 17000, 25000, 25000], "doras": ["5s"]}},
                    {"action_name": "RecordDiscardTile", "data": {"seat": 2, "tile": "3p", "moqie": False}},
                    {"action_name": "RecordChiPengGang", "data": {"seat": 3, "type": 0, "tiles": ["3p", "4p", "5p"], "froms": [3, 3, 2]}},
                    {"action_name": "RecordDiscardTile", "data": {"seat": 3, "tile": "9m", "moqie": False}},
                    {"action_name": "RecordDealTile", "data": {"seat": 2, "tile": "2s"}},
                    {"action_name": "RecordHule", "data": {"hules": [{"seat": 2, "target": 2, "hu_tile": "2s", "zimo": True}], "delta_scores": [-1000, -1000, 3000, -1000]}},
                ]
            }
        ]
    }

    # Convert to standard MJAI events
    events = convert_record_to_mjai(record, my_seat=0)
    assert len(events) >= 10

    # Ensure hora events have correct actor and target
    hora_events = [e for e in events if e.get("type") == "hora"]
    assert len(hora_events) == 2
    # Ron in Round 1: actor=0 (genta2223), target=1 (ブロコフ)
    assert hora_events[0]["actor"] == 0
    assert hora_events[0]["target"] == 1
    # Tsumo in Round 2: actor=2 (龍斗たっちゃん), target=2 (self)
    assert hora_events[1]["actor"] == 2
    assert hora_events[1]["target"] == 2

    # Ensure reach event was generated for Seat 0
    reach_events = [e for e in events if e.get("type") == "reach"]
    assert len(reach_events) == 1
    assert reach_events[0]["actor"] == 0

    # Ensure calls have correct actor
    pon_events = [e for e in events if e.get("type") == "pon"]
    assert len(pon_events) == 1
    assert pon_events[0]["actor"] == 2  # 龍斗たっちゃん
    assert pon_events[0]["target"] == 0

    chi_events = [e for e in events if e.get("type") == "chi"]
    assert len(chi_events) == 1
    assert chi_events[0]["actor"] == 3  # 横丁の隠居
    assert chi_events[0]["target"] == 2

    # Feed into OpponentProfiler
    profiler = OpponentProfiler()
    profiler.analyze_game_events(events)

    # 1. Verify profiler.stats dict directly
    assert profiler.stats[0]["agari"] == 1
    assert profiler.stats[0]["reach"] == 1
    assert profiler.stats[0]["houju"] == 0

    assert profiler.stats[1]["agari"] == 0
    assert profiler.stats[1]["houju"] == 1
    assert profiler.stats[1]["reach"] == 0

    assert profiler.stats[2]["agari"] == 1
    assert profiler.stats[2]["call"] == 1
    assert profiler.stats[2]["yakuhai"] == 1
    assert profiler.stats[2]["tanyao"] == 0
    assert profiler.stats[2]["houju"] == 0

    assert profiler.stats[3]["call"] == 1
    assert profiler.stats[3]["tanyao"] == 1
    assert profiler.stats[3]["yakuhai"] == 0

    # 2. Verify individual PlayerProfile stats
    p0 = profiler.get_profile("genta2223")
    assert p0.hora_count == 1
    assert p0.riichi_count == 1
    assert p0.deal_in_count == 0
    assert p0.hora_rate == 0.5  # 1 win in 2 rounds
    assert p0.riichi_rate == 0.5

    p1 = profiler.get_profile("ブロコフ")
    assert p1.hora_count == 0
    assert p1.deal_in_count == 1
    assert p1.deal_in_rate == 0.5

    p2 = profiler.get_profile("龍斗たっちゃん")
    assert p2.call_count == 1
    assert p2.yakuhai_call_count == 1
    assert p2.tanyao_call_count == 0
    assert p2.open_rate == 0.5
    assert p2.hora_count == 1
    assert p2.hora_rate == 0.5
    assert p2.deal_in_count == 0

    p3 = profiler.get_profile("横丁の隠居")
    assert p3.call_count == 1
    assert p3.tanyao_call_count == 1
    assert p3.yakuhai_call_count == 0
    assert p3.open_rate == 0.5


def _pb_varint(n: int) -> bytes:
    out = bytearray()
    while n >= 0x80:
        out.append((n & 0x7F) | 0x80)
        n >>= 7
    out.append(n)
    return bytes(out)


def _pb_len(tag: int, payload: bytes) -> bytes:
    return _pb_varint((tag << 3) | 2) + _pb_varint(len(payload)) + payload


def _pb_int(tag: int, v: int) -> bytes:
    return _pb_varint(tag << 3) + _pb_varint(v)


def test_paifu_binary_v2_records_numeric_names_and_seat_stats():
    """
    Real-format replay: GameDetailRecords.actions (tag 3) -> GameAction.result (tag 3)
    -> Wrapper{name='.lq.RecordXxx', data}. Numeric strings (200, 3980, 3740) must never
    become nicknames, and hora/houju/reach/naki must land on the correct seats.
    """
    import gzip
    from tools.paifu_to_mjai import PaifuDownloader, extract_player_names_from_record, convert_record_to_mjai
    from core.opponent_profiler import OpponentProfiler

    names = ["genta2223", "ブロコフ", "龍斗たっちゃん", "横丁の隠居"]

    def wrap(name: str, data: bytes) -> bytes:
        wrapper = _pb_len(1, (".lq." + name).encode()) + _pb_len(2, data)
        game_action = _pb_int(1, 1) + _pb_int(2, 1) + _pb_len(3, wrapper)
        return _pb_len(3, game_action)

    def tiles(tag, ts):
        return b"".join(_pb_len(tag, t.encode()) for t in ts)

    recs = b"".join([
        wrap("RecordNewRound", _pb_int(1, 0) + _pb_int(2, 0) + _pb_len(6, b"".join(_pb_varint(25000) for _ in range(4)))),
        wrap("RecordDiscardTile", _pb_int(1, 0) + tiles(2, ["5z"])),
        # Seat 2 (龍斗たっちゃん) pons 5z from seat 0
        wrap("RecordChiPengGang", _pb_int(1, 2) + _pb_int(2, 1) + tiles(3, ["5z", "5z", "5z"]) + _pb_len(4, bytes([2, 2, 0]))),
        wrap("RecordDiscardTile", _pb_int(1, 2) + tiles(2, ["9p"])),
        # Seat 3 riichi
        wrap("RecordDiscardTile", _pb_int(1, 3) + tiles(2, ["1m"]) + _pb_int(3, 1)),
        # Seat 1 discards, seat 3 rons
        wrap("RecordDiscardTile", _pb_int(1, 1) + tiles(2, ["4s"])),
        wrap("RecordHule", _pb_len(1, _pb_int(4, 3) + _pb_int(9, 1) + tiles(3, ["4s"]))),
    ])
    gz = gzip.compress(recs)

    # head: 4 real accounts + noise sub-messages containing numeric strings
    accs = b"".join(
        _pb_len(11, _pb_int(1, 50000 + i) + (_pb_int(2, i) if i else b"") + _pb_len(3, n.encode()))
        for i, n in enumerate(names)
    )
    noise = b"".join(_pb_len(12, _pb_int(2, 1) + _pb_len(3, v)) for v in (b"200", b"3980", b"3740"))
    head = _pb_len(1, b"261017-v2-binary-uuid") + accs + noise
    frame = b"\x03\x05\x00" + _pb_len(2, gz) + _pb_len(3, head) + b"\x00" * 2000

    rec = PaifuDownloader()._try_decode_records(frame, "261017-v2-binary-uuid")
    assert rec is not None
    nicks = [a["nickname"] for a in rec["accounts"]]
    assert not any(n.isdigit() for n in nicks)
    assert extract_player_names_from_record(rec) == names

    acts = rec["rounds"][0]["actions"]
    assert acts[0]["action_name"] == "RecordNewRound"
    assert all(a["data"] for a in acts), "action payloads must be decoded, not empty"

    events = convert_record_to_mjai(rec, my_seat=0)
    profiler = OpponentProfiler()
    profiler.analyze_game_events(events)

    assert set(profiler.profiles) == set(names)
    assert profiler.stats[2]["naki"] == 1 and profiler.stats[2]["yakuhai"] == 1
    assert profiler.stats[0]["naki"] == 0
    assert profiler.stats[3]["reach"] == 1 and profiler.stats[3]["agari"] == 1
    assert profiler.stats[1]["houju"] == 1
    assert profiler.get_profile("龍斗たっちゃん").open_rate == 1.0
    assert profiler.get_profile("genta2223").open_rate == 0.0
    assert profiler.get_profile("横丁の隠居").hora_rate == 1.0
    assert profiler.get_profile("横丁の隠居").riichi_rate == 1.0
    assert profiler.get_profile("ブロコフ").deal_in_rate == 1.0


@pytest.mark.asyncio
async def test_profiler_avg_hora_score_normalization():
    """Verify that OpponentProfiler computes authentic average winning scores without round-end balance inflation."""
    events = [
        {"type": "start_game", "names": ["genta2223", "ブロコフ", "龍斗たっちゃん", "横丁の隠居"]},
        # Round 1: genta wins 8000 pts Ron (score=8000, balance delta=[8000, -8000, 0, 0])
        {"type": "start_kyoku", "bakaze": "E", "kyoku": 1, "honba": 0, "kyotaku": 0, "oya": 0, "scores": [25000, 25000, 25000, 25000], "tehais": [[], [], [], []]},
        {"type": "hora", "actor": 0, "target": 1, "pai": "5m", "score": 8000, "deltas": [8000, -8000, 0, 0]},
        {"type": "end_kyoku"},
        # Round 2: genta wins 2600 pts Tsumo (score=2600, deltas=[2600, -800, -1000, -800])
        {"type": "start_kyoku", "bakaze": "E", "kyoku": 2, "honba": 0, "kyotaku": 0, "oya": 1, "scores": [33000, 17000, 25000, 25000], "tehais": [[], [], [], []]},
        {"type": "hora", "actor": 0, "target": 0, "pai": "2s", "score": 2600, "deltas": [2600, -800, -1000, -800]},
        {"type": "end_kyoku"},
        {"type": "end_game"}
    ]

    from core.opponent_profiler import OpponentProfiler

    profiler = OpponentProfiler()
    profiler.analyze_game_events(events)

    prof = profiler.get_profile("genta2223")
    assert prof is not None
    assert prof.hora_count == 2
    # Average score should be (8000 + 2600) / 2 = 5300.0, NOT 25000+!
    assert prof.avg_hora_score == 5300.0
    assert 5000 <= prof.avg_hora_score <= 8000


@pytest.mark.asyncio
async def test_hud_recommendation_top_candidates_and_display():
    """Verify that ActionController formats 1st/2nd candidate predictions and win probabilities correctly."""
    from core.state_manager import MahjongStateManager
    from core.engine import ActionCandidate
    from io_adapter.action_controller import ActionController

    received_callbacks = []

    controller = ActionController(config={"output_mode": "text_only"})
    controller.on_recommendation_callback = lambda data: received_callbacks.append(data)

    state_mgr = MahjongStateManager(my_actor_id=0)
    state_mgr.reset_round(
        bakaze="E",
        kyoku=1,
        honba=0,
        kyotaku=0,
        oya=0,
        scores=[25000, 25000, 25000, 25000],
        tehais=[["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4s", "7s"], [], [], []],
        dora_marker="1p"
    )

    cand1 = ActionCandidate(
        action_type="dahai",
        tile="4m",
        raw_q_value=185.0,
        modified_q_value=185.0,
        danger_score=0.1,
        shanten_after=0,
        metadata={"ukeire": 6}
    )
    cand2 = ActionCandidate(
        action_type="dahai",
        tile="7s",
        raw_q_value=172.0,
        modified_q_value=172.0,
        danger_score=0.2,
        shanten_after=0,
        metadata={"ukeire": 4}
    )

    action = await controller.execute_action(
        candidate=cand1,
        state_mgr=state_mgr,
        comp_time_ms=12.5,
        all_candidates=[cand1, cand2],
        apply_sleep=False
    )

    assert len(received_callbacks) == 1
    cb = received_callbacks[0]
    top_cands = cb.get("top_candidates", [])
    assert len(top_cands) >= 2
    # Verify 1st candidate
    assert top_cands[0]["tile"] == "4m"
    assert top_cands[1]["tile"] == "7s"
    assert top_cands[1]["diff_q"] == -13.0


def test_action_mjstart_does_not_trigger_vision_fallback(monkeypatch):
    """Verify ActionMJStart does NOT trigger Vision fallback or overwrite hand tiles."""
    from io_adapter.majsoul_bridge import MajsoulBridge
    from io_adapter.vision_hand_reader import VisionHandReader

    called = False
    def _fake_read(self):
        nonlocal called
        called = True
        return ["P"] * 14  # Simulate animation blank screen
    monkeypatch.setattr(VisionHandReader, "read_hand", _fake_read)

    bridge = MajsoulBridge(my_seat=0, vision_fallback_enabled=True)
    events = bridge.translate_action("ActionMJStart", {})
    assert not called, "Vision fallback must NEVER be triggered for ActionMJStart"
    assert len(events) == 1
    assert events[0]["type"] == "start_kyoku"
    # Self hand should be 13 unknown '?' tiles, NOT 14 'P' tiles!
    assert events[0]["tehais"][0] == ["?"] * 13


def test_state_manager_moqie_tsumogiri_preserves_hand_on_unmatched_tile():
    """Verify that discarding a missed tsumo tile with moqie=True preserves the 13 hand tiles."""
    from core.state_manager import MahjongStateManager

    mgr = MahjongStateManager(my_actor_id=0)
    initial_hand = ["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "1z", "1z", "2z", "2z"]
    mgr.reset_round(
        bakaze="E",
        kyoku=1,
        honba=0,
        kyotaku=0,
        oya=0,
        scores=[25000, 25000, 25000, 25000],
        tehais=[list(initial_hand), ["?"]*13, ["?"]*13, ["?"]*13],
        dora_marker="1p"
    )

    # Discard a tile NOT in hand (e.g. drawn tile '9m' that wasn't decoded in deal_tile) with tsumogiri=True
    mgr.update_event({
        "type": "dahai",
        "actor": 0,
        "pai": "9m",
        "tsumogiri": True
    })

    # The original 13 hand tiles must remain untouched
    my_p = mgr.my_player
    assert my_p is not None
    assert my_p.tehai == initial_hand
    assert len(my_p.tehai) == 13


def test_format_tile_label_converts_honor_codes_to_japanese():
    """Verify format_tile_label properly converts honors and number tiles to Japanese names."""
    from io_adapter.action_controller import format_tile_label

    # Winds
    assert format_tile_label("1z") == "東 (1z)"
    assert format_tile_label("E") == "東 (E)"
    assert format_tile_label("2z") == "南 (2z)"
    assert format_tile_label("S") == "南 (S)"
    assert format_tile_label("3z") == "西 (3z)"
    assert format_tile_label("W") == "西 (W)"
    assert format_tile_label("4z") == "北 (4z)"
    assert format_tile_label("N") == "北 (N)"

    # Dragons
    assert format_tile_label("5z") == "白 (5z)"
    assert format_tile_label("P") == "白 (P)"
    assert format_tile_label("6z") == "發 (6z)"
    assert format_tile_label("F") == "發 (F)"
    assert format_tile_label("7z") == "中 (7z)"
    assert format_tile_label("C") == "中 (C)"

    # Suits
    assert format_tile_label("1m") == "1萬 (1m)"
    assert format_tile_label("7s") == "7索 (7s)"
    assert format_tile_label("9p") == "9筒 (9p)"


def test_notify_room_game_start_does_not_trigger_vision_fallback(monkeypatch):
    """Verify NotifyRoomGameStart does NOT trigger Vision fallback or corrupt hand."""
    from io_adapter.majsoul_bridge import MajsoulBridge
    from io_adapter.vision_hand_reader import VisionHandReader

    called = False
    def _fake_read(self):
        nonlocal called
        called = True
        return ["P"] * 14
    monkeypatch.setattr(VisionHandReader, "read_hand", _fake_read)

    bridge = MajsoulBridge(my_seat=0, vision_fallback_enabled=True)
    events = bridge.translate_action("NotifyRoomGameStart", {})
    assert not called, "Vision fallback must NEVER be triggered for NotifyRoomGameStart"
    assert len(events) == 1
    assert events[0]["type"] == "start_kyoku"
    assert events[0]["tehais"][0] == ["?"] * 13


def test_oya_initial_tsumo_emission_brings_hand_to_14_tiles():
    """
    Verify that when self is oya (dealer), ActionNewRound with 13 dealt tiles
    immediately emits an initial tsumo event bringing hand to 14 tiles for initial discard.
    """
    from io_adapter.majsoul_bridge import MajsoulBridge
    from core.state_manager import MahjongStateManager
    from core.engine import DecisionEngine

    bridge = MajsoulBridge(my_seat=0)
    state_mgr = MahjongStateManager(my_actor_id=0)
    engine = DecisionEngine(use_fallback=True)

    nr_payload = {
        "chang": 0,
        "ju": 0, # oya = 0 == my_seat
        "ben": 0,
        "scores": [25000, 25000, 25000, 25000],
        "doras": ["1p"],
        "tiles": ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p"]
    }
    events = bridge.translate_action("ActionNewRound", nr_payload)
    
    # Must contain both start_kyoku and initial tsumo
    assert len(events) == 2
    assert events[0]["type"] == "start_kyoku"
    assert events[1]["type"] == "tsumo"
    assert events[1]["actor"] == 0
    assert events[1]["pai"] == "4p"

    # Update state manager: hand should become 14 tiles
    for ev in events:
        state_mgr.update_event(ev)

    assert len(state_mgr.my_player.tehai) == 14
    # Decision engine must be able to evaluate and offer discards on first turn
    candidates = engine.evaluate(state_mgr)
    discard_candidates = [c for c in candidates if c.action_type in ("dahai", "reach", "none")]
    assert len(discard_candidates) > 0, "Decision engine must trigger on first turn for dealer"


def test_action_deal_tile_self_draw_preserved_and_opponent_hidden():
    """
    Verify that ActionDealTile extracts actual tile when seat == my_seat,
    and obscures tile as '?' when seat != my_seat.
    """
    from io_adapter.majsoul_bridge import MajsoulBridge

    bridge = MajsoulBridge(my_seat=0)
    
    # Self draw on seat 0: tile must be preserved
    self_events = bridge.translate_action("ActionDealTile", {"seat": 0, "tile": "7s"})
    assert len(self_events) == 1
    assert self_events[0]["type"] == "tsumo"
    assert self_events[0]["actor"] == 0
    assert self_events[0]["pai"] == "7s"

    # Opponent draw on seat 1: tile must be obscured
    opp_events = bridge.translate_action("ActionDealTile", {"seat": 1, "tile": "7s"})
    assert len(opp_events) == 1
    assert opp_events[0]["type"] == "tsumo"
    assert opp_events[0]["actor"] == 1
    assert opp_events[0]["pai"] == "?"


def test_state_manager_hand_size_safety_guard_prevents_shrinking_below_13():
    """
    Verify that state_manager strictly forbids reducing self hand below 13 tiles
    even when unmatched dahai events or missed draw tiles occur for 4 consecutive turns.
    """
    from core.state_manager import MahjongStateManager

    state_mgr = MahjongStateManager(my_actor_id=0)
    init_event = {
        "type": "start_kyoku",
        "bakaze": "E",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 0,
        "oya": 1,
        "scores": [25000, 25000, 25000, 25000],
        "tehais": [
            ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p"],
            ["?"] * 13, ["?"] * 13, ["?"] * 13
        ],
        "dora_marker": "1p"
    }
    state_mgr.update_event(init_event)
    assert len(state_mgr.my_player.tehai) == 13

    # Simulate 4 consecutive turns where tsumo was missed (tile '?') and dahai occurs
    discards = ["1m", "2m", "3m", "4m"]
    for d in discards:
        # Missed draw
        state_mgr.update_event({"type": "tsumo", "actor": 0, "pai": "?"})
        assert len(state_mgr.my_player.tehai) == 13, "Missed draw should not add tile"
        # Discard occurs
        state_mgr.update_event({"type": "dahai", "actor": 0, "pai": d, "tsumogiri": False})
        # Hand must NEVER shrink below 13!
        assert len(state_mgr.my_player.tehai) == 13, f"Hand collapsed to {len(state_mgr.my_player.tehai)} tiles after discarding {d}!"


@pytest.mark.asyncio
async def test_user_ignores_recommendation_and_fresh_recalculation_on_next_tsumo():
    """
    Verify that when user ignores AI's 1st recommendation (e.g. recommended 6p, but discarded 1z),
    the previous recommendation cache is purged upon dahai and tsumo, and a fresh recommendation
    is calculated strictly from the actual 14 tiles on the next turn.
    """
    from main import MahjongAIAgent

    agent = MahjongAIAgent(
        my_actor_id=0,
        apply_sleep=False,
        output_mode="text_only"
    )

    received_callbacks = []
    agent.action_controller.on_recommendation_callback = lambda payload: received_callbacks.append(payload)

    # 1. Round starts: South player (oya=1), dealt 13 tiles with isolated 6p and honors 1z, 2z, 3z
    # Hand: 1m 2m 3m 4m 5m 6p 7s 8s 9s 1z 2z 3z 4z
    start_ev = {
        "type": "start_kyoku",
        "bakaze": "E",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 0,
        "oya": 1,
        "scores": [25000, 25000, 25000, 25000],
        "tehais": [
            ["1m", "2m", "3m", "4m", "5m", "6p", "7s", "8s", "9s", "1z", "2z", "3z", "4z"],
            ["?"] * 13, ["?"] * 13, ["?"] * 13
        ],
        "dora_marker": "1p"
    }
    await agent.handle_event(start_ev)

    # 2. Turn 1: Dealer (P1) discards 9m
    await agent.handle_event({"type": "dahai", "actor": 1, "pai": "9m", "tsumogiri": True})

    # 3. Turn 1: Self draws 9p (isolated)
    action1 = await agent.handle_event({"type": "tsumo", "actor": 0, "pai": "9p"})
    assert action1 is not None
    assert action1.action_type in ("dahai", "reach")
    rec1_tile = action1.tile
    assert rec1_tile is not None
    # ActionController must cache this decision
    assert agent.action_controller._last_recommendation is not None
    assert agent.action_controller._last_recommendation.tile == rec1_tile

    # 4. User IGNORES AI's recommendation (rec1_tile) and discards a different tile '1z'
    alt_discard = "1z" if rec1_tile != "1z" else "2z"
    await agent.handle_event({"type": "dahai", "actor": 0, "pai": alt_discard, "tsumogiri": False})

    # Cache MUST be completely purged upon self dahai!
    assert agent.action_controller._last_recommendation is None, "Recommendation cache must be None after self dahai"
    assert agent.action_controller.top_candidates == [], "Top candidates must be cleared after self dahai"
    assert received_callbacks[-1].get("action_type") == "cleared", "Callback must receive 'cleared' signal on dahai"

    # 5. Opponents discard tiles
    await agent.handle_event({"type": "dahai", "actor": 1, "pai": "1s", "tsumogiri": True})
    await agent.handle_event({"type": "dahai", "actor": 2, "pai": "2s", "tsumogiri": True})
    await agent.handle_event({"type": "dahai", "actor": 3, "pai": "3s", "tsumogiri": True})

    # 6. Turn 2: Self draws 5p (which forms a strong block with 6p in hand!)
    action2 = await agent.handle_event({"type": "tsumo", "actor": 0, "pai": "5p"})
    assert action2 is not None
    assert action2.action_type in ("dahai", "reach")
    rec2_tile = action2.tile

    # ActionController cache must be fresh and match the new calculation
    assert agent.action_controller._last_recommendation is not None
    assert agent.action_controller._last_recommendation.tile == rec2_tile

    # Hand currently has [5p, 6p] sequence. 6p must NOT be discarded anymore!
    assert rec2_tile != "6p", f"Tile 6p formed sequence with newly drawn 5p and must not be recommended: got {rec2_tile}"
    # The actual hand has 14 tiles; alt_discard was removed and 5p was added.
    current_hand = agent.state_mgr.my_player.tehai
    assert alt_discard not in current_hand, f"Discarded tile {alt_discard} should no longer be in hand"
    assert "5p" in current_hand, "Newly drawn 5p must be in hand"
    assert "6p" in current_hand, "Preserved 6p must remain in hand"


@pytest.mark.asyncio
@pytest.mark.parametrize("target_seat", [0, 1, 2, 3])
async def test_dynamic_seat_resolution_and_turn_management_for_all_four_seats(target_seat):
    """
    Verify 100% dynamic seat resolution and dealer/non-dealer turn lifecycle across all four seats:
    - Target account_id is dynamically extracted from authGame request.
    - Player list matches account_id to target_seat (0, 1, 2, or 3).
    - Round starts (East 1, oya=0):
      * If target_seat == 0 (dealer / oya): ActionNewRound immediately yields 14th tile tsumo,
        and AIAgent produces a valid discard recommendation on move 1.
      * If target_seat != 0 (non-dealer / child): ActionNewRound yields 13 tiles without tsumo,
        agent remains waiting (no recommendation). When ActionDealTile arrives for target_seat,
        tsumo is triggered, agent produces recommendation, and dahai clears it back to waiting.
    """
    from io_adapter.majsoul_bridge import MajsoulBridge
    from main import MahjongAIAgent

    account_ids = [9990001, 9990002, 9990003, 9990004]
    my_account_id = account_ids[target_seat]

    # Initialize bridge without hardcoded seat (or default 0)
    bridge = MajsoulBridge(my_seat=0)

    # 1. Step 1: Dynamic account_id capture from FastTest.authGame Request
    auth_req_data = {
        "account_id": my_account_id,
        "token": "test_token_abc",
        "game_uuid": "game_12345"
    }
    bridge.translate_action("authGame", auth_req_data)
    assert bridge.my_account_id == my_account_id, "Bridge failed to bind my_account_id from authGame"

    # 2. Step 2: Dynamic seat resolution via NotifyRoomPlayerUpdate / ResAuthGame
    player_update_data = {
        "player_list": [
            {"account_id": account_ids[0], "seat": 0, "nickname": "PlayerEast"},
            {"account_id": account_ids[1], "seat": 1, "nickname": "PlayerSouth"},
            {"account_id": account_ids[2], "seat": 2, "nickname": "PlayerWest"},
            {"account_id": account_ids[3], "seat": 3, "nickname": "PlayerNorth"},
        ]
    }
    bridge.translate_action("NotifyRoomPlayerUpdate", player_update_data)
    assert bridge.my_seat == target_seat, f"Bridge resolved seat {bridge.my_seat}, expected {target_seat}"

    # Setup agent aligned with resolved seat
    agent = MahjongAIAgent(
        my_actor_id=target_seat,
        apply_sleep=False,
        output_mode="text_only"
    )

    # Bridge seat change callback synchronizes agent
    bridge.on_seat_changed = lambda seat: agent.set_my_actor_id(seat)
    assert agent.my_actor_id == target_seat

    # 3. Step 3: ActionNewRound in East 1 (oya = 0)
    tiles_13 = ["1m", "2m", "3m", "4m", "5m", "6p", "7p", "8p", "1s", "2s", "3s", "1z", "2z"]
    new_round_data = {
        "chang": 0,
        "ju": 0,
        "ben": 0,
        "oya": 0,
        "scores": [25000, 25000, 25000, 25000],
        "tiles": list(tiles_13),
        "doras": ["1p"]
    }
    events = bridge.translate_action("ActionNewRound", new_round_data)

    if target_seat == 0:
        # Dealer (oya): must yield start_kyoku AND immediate 14th tile tsumo
        assert len(events) == 2, f"Dealer should have 2 events (start_kyoku + tsumo), got {len(events)}"
        assert events[0]["type"] == "start_kyoku"
        assert events[1]["type"] == "tsumo"
        assert events[1]["actor"] == 0
        from io_adapter.majsoul_bridge import liqi_tile_to_mjai
        assert events[1]["pai"] == liqi_tile_to_mjai(tiles_13[-1])

        # Feed events to agent
        await agent.handle_event(events[0])
        action = await agent.handle_event(events[1])

        # Agent must immediately calculate and emit recommendation for first turn
        assert action is not None, "Dealer agent failed to produce recommendation on first turn"
        assert action.action_type in ("dahai", "reach")
        assert agent.action_controller._last_recommendation is not None
        assert agent.action_controller._last_recommendation.tile == action.tile

        # Dealer plays tile
        dahai_events = bridge.translate_action("ActionDiscardTile", {
            "seat": 0, "tile": action.tile, "is_liqi": False, "moqie": False
        })
        for ev in dahai_events:
            await agent.handle_event(ev)

        # Recommendation must be cleared back to waiting
        assert agent.action_controller._last_recommendation is None
    else:
        # Non-dealer (child): must yield ONLY start_kyoku, NO tsumo
        assert len(events) == 1, f"Child should have 1 event (start_kyoku), got {len(events)}"
        assert events[0]["type"] == "start_kyoku"

        res = await agent.handle_event(events[0])
        # Agent must be in waiting state (no recommendation)
        assert res is None, "Child agent should not produce recommendation on round start"
        assert agent.action_controller._last_recommendation is None

        # Dealer (P0) discards a tile
        p0_discard_events = bridge.translate_action("ActionDiscardTile", {
            "seat": 0, "tile": "9m", "is_liqi": False, "moqie": True
        })
        for ev in p0_discard_events:
            await agent.handle_event(ev)
        assert agent.action_controller._last_recommendation is None

        # If target_seat is West (2) or North (3), intervening players discard
        for seat in range(1, target_seat):
            # intervening draw & discard
            for ev in bridge.translate_action("ActionDealTile", {"seat": seat}):
                await agent.handle_event(ev)
            for ev in bridge.translate_action("ActionDiscardTile", {"seat": seat, "tile": "1z", "is_liqi": False, "moqie": True}):
                await agent.handle_event(ev)
            assert agent.action_controller._last_recommendation is None

        # Now target_seat's turn arrives via ActionDealTile
        deal_events = bridge.translate_action("ActionDealTile", {
            "seat": target_seat,
            "tile": "9p",
            "left_tile_count": 65
        })
        assert len(deal_events) == 1
        assert deal_events[0]["type"] == "tsumo"
        assert deal_events[0]["actor"] == target_seat
        assert deal_events[0]["pai"] == "9p"

        # Agent receives tsumo -> MUST produce discard recommendation
        action = await agent.handle_event(deal_events[0])
        assert action is not None, f"Child seat {target_seat} produced no recommendation on tsumo"
        assert action.action_type in ("dahai", "reach")
        assert agent.action_controller._last_recommendation is not None
        assert agent.action_controller._last_recommendation.tile == action.tile

        # Target seat discards tile -> MUST clear recommendation back to waiting
        dahai_events = bridge.translate_action("ActionDiscardTile", {
            "seat": target_seat, "tile": action.tile, "is_liqi": False, "moqie": False
        })
        for ev in dahai_events:
            await agent.handle_event(ev)
        assert agent.action_controller._last_recommendation is None, "Recommendation must clear on self dahai"


def test_dynamic_seat_resolution_fallback_to_plain_tiles():
    """
    Verify fallback mechanism when account_id matching is unavailable:
    If ActionNewRound provides plain tiles with seat specified, my_seat falls back to it.
    """
    from io_adapter.majsoul_bridge import MajsoulBridge

    # Fallback 1: ActionNewRound carries seat = 2 with plain tiles
    bridge = MajsoulBridge(my_seat=0)
    nr_data = {
        "seat": 2,
        "tiles": ["1m", "2m", "3m", "4m", "5m", "6p", "7p", "8p", "1s", "2s", "3s", "1z", "2z"],
        "doras": ["1p"]
    }
    bridge.translate_action("ActionNewRound", nr_data)
    assert bridge.my_seat == 2, f"Expected fallback to seat 2, got {bridge.my_seat}"

    # Fallback 2: ActionNewRound carries seat = 3 with plain tiles
    bridge2 = MajsoulBridge(my_seat=0)
    nr_data2 = {
        "seat": 3,
        "tiles": ["1m", "2m", "3m", "4m", "5m", "6p", "7p", "8p", "1s", "2s", "3s", "1z", "2z"],
        "doras": ["1p"]
    }
    bridge2.translate_action("ActionNewRound", nr_data2)
    assert bridge2.my_seat == 3, f"Expected fallback to seat 3, got {bridge2.my_seat}"


@pytest.mark.asyncio
async def test_live_stream_opponent_profiling_and_tactical_adaptation():
    """
    Test 1: Verify that real-time live match events (chi/pon, reach, dahai pushes/folds)
    are immediately ingested into OpponentProfiler during live play and dynamically reflected
    in StrategyModifier candidate evaluation.
    """
    from main import MahjongAIAgent
    from core.opponent_profiler import OpponentProfiler

    agent = MahjongAIAgent(my_actor_id=0, apply_sleep=False, output_mode="text_only")
    assert agent.profiler is not None

    # 1. Start game with 3 named opponents
    start_game_ev = {
        "type": "start_game",
        "names": ["Me", "BerserkerEnemy", "NormalEnemy", "WallEnemy"],
        "game_uuid": "live_game_test_001",
        "match_mode": "hanchan"
    }
    await agent.handle_event(start_game_ev)
    assert "BerserkerEnemy" in agent.profiler.profiles
    assert "NormalEnemy" in agent.profiler.profiles

    # 2. Round 1 starts: East 1, Oya = 1 (BerserkerEnemy)
    start_kyoku_ev = {
        "type": "start_kyoku",
        "bakaze": "E",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 0,
        "oya": 1,
        "scores": [25000, 25000, 25000, 25000],
        "tehais": [
            ["1m", "2m", "3m", "4m", "5m", "6p", "7p", "8p", "1s", "2s", "3s", "1z", "2z"],
            ["?"] * 13, ["?"] * 13, ["?"] * 13
        ],
        "dora_marker": "1p"
    }
    await agent.handle_event(start_kyoku_ev)
    assert agent.profiler.live_round_count == 1
    assert agent.profiler.profiles["BerserkerEnemy"].live_rounds == 1

    # 3. P1 (BerserkerEnemy) declares pon of Green Dragon (F)
    pon_ev = {
        "type": "pon",
        "actor": 1,
        "target": 0,
        "pai": "F",
        "consumed": ["F", "F"]
    }
    await agent.handle_event(pon_ev)
    p1_prof = agent.profiler.profiles["BerserkerEnemy"]
    assert p1_prof.call_count == 1
    assert p1_prof.live_call_count == 1
    assert p1_prof.yakuhai_call_count == 1

    # 4. P1 declares Riichi
    reach_ev = {"type": "reach", "actor": 1}
    await agent.handle_event(reach_ev)
    assert p1_prof.riichi_count == 1
    assert p1_prof.live_riichi_count == 1

    # 5. Under P1's riichi, P2 (NormalEnemy) pushes an unsafe tile '5s'
    p2_dahai_unsafe = {"type": "dahai", "actor": 2, "pai": "5s", "tsumogiri": False}
    await agent.handle_event(p2_dahai_unsafe)
    p2_prof = agent.profiler.profiles["NormalEnemy"]
    assert p2_prof.aggressiveness_total_turns == 1
    assert p2_prof.live_aggressiveness_turns == 1
    assert p2_prof.aggressiveness_pushes == 1
    assert p2_prof.live_aggressiveness_pushes == 1

    # 6. Self draws tile -> decision required
    tsumo_ev = {"type": "tsumo", "actor": 0, "pai": "8m"}
    action = await agent.handle_event(tsumo_ev)
    assert action is not None
    assert action.action_type in ("dahai", "reach")

    # 7. End of round via Hora: P1 wins
    hora_ev = {
        "type": "hora",
        "actor": 1,
        "target": 2,
        "score": 3900,
        "deltas": [-3900, 3900, -3900, 0]
    }
    await agent.handle_event(hora_ev)
    assert p1_prof.hora_count == 1
    assert p1_prof.live_hora_count == 1
    assert p2_prof.deal_in_count == 1
    assert p2_prof.live_deal_in_count == 1
    # Check that game_uuid was registered
    assert "live_game_test_001" in p1_prof.recorded_game_uuids


def test_game_uuid_deduplication_between_live_and_paifu_import():
    """
    Test 2: Verify that processing live events and then importing the same match's paifu
    with matching game_uuid does NOT double-count statistics.
    """
    from core.opponent_profiler import OpponentProfiler

    profiler = OpponentProfiler()
    uuid = "match_uuid_dedup_998877"

    # 1. Simulate real-time live events during the match
    live_events = [
        {"type": "start_game", "names": ["Player0", "EnemyA", "EnemyB", "EnemyC"], "game_uuid": uuid},
        {"type": "start_kyoku", "bakaze": "E", "kyoku": 1, "honba": 0, "kyotaku": 0, "oya": 1},
        {"type": "reach", "actor": 1},
        {"type": "dahai", "actor": 1, "pai": "9m", "tsumogiri": True},
        {"type": "hora", "actor": 1, "target": 2, "score": 8000},
        {"type": "end_kyoku"}
    ]
    for ev in live_events:
        profiler.consume_live_event(ev)

    enemy_a = profiler.profiles["EnemyA"]
    assert enemy_a.total_rounds == 1
    assert enemy_a.hora_count == 1
    assert enemy_a.riichi_count == 1
    assert uuid in enemy_a.recorded_game_uuids

    # 2. Later, paifu sniffer imports the exact same game with identical game_uuid
    profiler.analyze_game_events(live_events, game_uuid=uuid)

    # 3. Deduplication guard MUST prevent double-counting
    assert enemy_a.total_rounds == 1, f"Double-counted total_rounds: {enemy_a.total_rounds}"
    assert enemy_a.hora_count == 1, f"Double-counted hora_count: {enemy_a.hora_count}"
    assert enemy_a.riichi_count == 1, f"Double-counted riichi_count: {enemy_a.riichi_count}"


def test_bayesian_smoothing_and_live_session_blending():
    """
    Test 3: Verify that Bayesian smoothing prevents extreme 0%/100% distortions on small sample sizes,
    and live session trends are blended appropriately (30% - 40%) into effective metrics.
    """
    from core.opponent_profiler import PlayerProfile, OpponentProfiler

    # 1. Create a profile with only 1 round played (1 win, 0 calls)
    prof = PlayerProfile(name="SmallSamplePlayer")
    prof.total_rounds = 1
    prof.hora_count = 1
    prof.rounds_with_call = 0
    prof.compute_metrics()

    # Raw observed rates are extreme: 100% win, 0% calls
    assert prof.hora_rate == 1.0
    assert prof.open_rate == 0.0

    # Bayesian smoothing with N_prior=10 mitigates extreme skew:
    # Expected smoothed_hora_rate = (1 + 10 * 0.22) / (1 + 10) = 3.2 / 11 ~= 0.291
    assert 0.25 <= prof.smoothed_hora_rate <= 0.35, f"Smoothed hora rate {prof.smoothed_hora_rate} not smoothed"
    # Expected smoothed_open_rate = (0 + 10 * 0.30) / (1 + 10) = 3.0 / 11 ~= 0.273
    assert 0.20 <= prof.smoothed_open_rate <= 0.32, f"Smoothed open rate {prof.smoothed_open_rate} not smoothed"

    # 2. Test live session trend blending:
    # SmallSamplePlayer historically melds little, but in the current live match, melded in 3 out of 3 rounds!
    prof.live_rounds = 3
    prof.live_rounds_with_call = 3

    # Effective metrics blend historical smoothed (~0.273) with live (1.00) with weight ~0.35
    eff = prof.get_effective_metrics(live_weight=0.35)
    # Expected effective open rate ~= 0.65 * 0.273 + 0.35 * 1.0 ~= 0.177 + 0.35 ~= 0.527
    assert eff["open_rate"] > prof.smoothed_open_rate, "Live high-calling trend was not reflected"
    assert 0.45 <= eff["open_rate"] <= 0.60, f"Blended open_rate {eff['open_rate']} outside expected range"

    # 3. Test Tonpuu (East-only) mode rapid speed exploitation:
    profiler = OpponentProfiler()
    profiler.profiles["SmallSamplePlayer"] = prof
    profiler.set_match_mode("tonpuu")

    # In Tonpuu mode with elevated call rate (eff_open > 0.35), danger penalty must be boosted by 1.3x
    modifiers = profiler.get_exploit_modifiers("SmallSamplePlayer", current_bakaze="E", is_enemy_calling=True)
    assert modifiers["danger_penalty_mult"] >= 1.30, f"Tonpuu call alert multiplier {modifiers['danger_penalty_mult']} < 1.30"
    assert "[東風速攻警戒 1.3x]" in modifiers["exploit_note"]


@pytest.mark.asyncio
async def test_mid_game_join_dynamic_rescue_and_recommendation_trigger(test_config):
    """
    Test Requirement: Verify that attaching mid-game (e.g. South 1 round, Non-dealer)
    without receiving ActionNewRound correctly:
    1. Recovers self seat (my_seat) and synthesizes a rescue start_kyoku event.
    2. Restores hand state and round_state in StateManager.
    3. Successfully triggers AI recommendation (top 1 & 2 candidates) on self draw.
    """
    from io_adapter.majsoul_bridge import MajsoulBridge
    from main import MahjongAIAgent

    # 1. Initialize bridge with default seat 0, but user is actually sitting at seat 2 (Nan 1, West/child)
    bridge = MajsoulBridge(my_seat=0, vision_fallback_enabled=False)
    assert not bridge._in_round

    # 2. Simulate mid-match stream without ActionNewRound:
    # First: Mid-game sync packet arrives on WebSocket connection
    sync_packet = {
        "chang": 1,   # 1 = South round (Nanba)
        "ju": 0,      # 0 = Kyoku 1 (Nan 1)
        "ben": 1,     # Honba = 1
        "scores": [24000, 26000, 25000, 25000],
        "doras": ["1p"]
    }
    events_from_sync = bridge.translate_action("syncGame", sync_packet)

    # Bridge MUST synthesize rescue start_kyoku from syncGame
    assert len(events_from_sync) == 1
    rescue_start = events_from_sync[0]
    assert rescue_start["type"] == "start_kyoku"
    assert rescue_start["bakaze"] == "S"  # South 1 round
    assert rescue_start["kyoku"] == 1
    assert rescue_start["honba"] == 1
    assert bridge._in_round is True

    # Next: Enemy at seat 1 discards 9m
    discard_packet = {
        "seat": 1,
        "tile": "9m",
    }
    events_from_discard = bridge.translate_action("ActionDiscardTile", discard_packet)
    assert len(events_from_discard) == 1
    assert events_from_discard[0]["type"] == "dahai"

    # 3. Next: Turn comes to Player 2 (Self), DealTile arrives with unmasked tile "4s" and full hand tiles
    deal_packet = {
        "seat": 2,
        "tile": "4s",
        "tiles": ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "7s", "8s", "4s"],
        "doras": ["1p"]
    }
    events_from_deal = bridge.translate_action("ActionDealTile", deal_packet)

    # Bridge MUST automatically resolve self seat to 2
    assert bridge.my_seat == 2
    assert any(ev.get("type") == "tsumo" and ev.get("actor") == 2 and ev.get("pai") == "4s" for ev in events_from_deal)

    # 4. Feed events sequentially into MahjongAIAgent
    agent = MahjongAIAgent(config=test_config, my_actor_id=0, apply_sleep=False, output_mode="text_only")

    # Capture recommendations
    recommendations = []
    agent.action_controller.on_recommendation_callback = lambda rec: recommendations.append(rec)

    # Process rescue start_kyoku & initial discard
    for ev in events_from_sync + events_from_discard:
        await agent.handle_event(ev)

    assert agent.state_mgr.round_state is not None
    assert agent.state_mgr.round_state.bakaze == "S"
    assert agent.state_mgr.round_state.kyoku == 1

    # Process self tsumo event
    for ev in events_from_deal:
        action = await agent.handle_event(ev)
        if ev.get("type") == "tsumo" and ev.get("actor") == 2:
            # Self actor ID must be updated to 2
            assert agent.my_actor_id == 2
            assert agent.state_mgr.my_actor_id == 2

            # Self hand must be populated with 14 tiles
            my_hand = agent.state_mgr.my_player.tehai
            assert len(my_hand) == 14
            assert "4s" in my_hand

            # AI decision engine must be triggered and produce an OutputAction
            assert action is not None
            assert action.action_type in ("dahai", "reach")
            assert len(recommendations) >= 1
            rec = recommendations[-1]
            assert "top_candidates" in rec
            assert len(rec["top_candidates"]) >= 2  # 1st and 2nd choices present


def test_gui_exclusive_button_control_guards():
    """
    Test GUI exclusive button control to guarantee:
    1. During assist run or paifu sniffer sniffing, btn_chrome is disabled to prevent duplicate launch.
    2. Assist button toggling prevents duplicate concurrent executions.
    """
    import tkinter as tk
    from gui import MahjongHUDApp

    root = tk.Tk()
    root.withdraw()
    try:
        app = MahjongHUDApp(root)

        # Initially normal
        assert str(app.btn_chrome["state"]) == "normal"
        assert str(app.btn_assist["state"]) == "normal"

        # 1. When assist is running: btn_chrome must be disabled
        app.is_running = True
        app._update_button_states()
        assert str(app.btn_chrome["state"]) == "disabled"

        # When stopped: restored to normal
        app._on_agent_stopped()
        assert str(app.btn_chrome["state"]) == "normal"

        # 2. When paifu sniffer is running: btn_chrome must be disabled
        class MockSniffer:
            is_running = True
        app.paifu_sniffer = MockSniffer()
        app._update_button_states()
        assert str(app.btn_chrome["state"]) == "disabled"

        # When sniffer stops: restored to normal
        app.paifu_sniffer.is_running = False
        app._update_button_states()
        assert str(app.btn_chrome["state"]) == "normal"
    finally:
        root.destroy()












