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
    assert elapsed < 0.005, f"Enemy tsumo took {elapsed:.4f}s, must be instant (< 5ms)!"

    p1_dahai = {"type": "dahai", "actor": 1, "pai": "9m", "tsumogiri": True}
    resp, elapsed = await handler.process_event(p1_dahai)
    # Self has no 9m pair/run, so it's an uncallable discard -> instant pass
    assert resp == {"type": "none"}
    assert elapsed < 0.005, f"Uncallable enemy discard took {elapsed:.4f}s, must be instant (< 5ms)!"

    # Player 2 turn: Tsumo & Discard
    p2_tsumo = {"type": "tsumo", "actor": 2, "pai": "4p"}
    _, elapsed = await handler.process_event(p2_tsumo)
    assert elapsed < 0.005

    p2_dahai = {"type": "dahai", "actor": 2, "pai": "8m", "tsumogiri": True}
    resp, elapsed = await handler.process_event(p2_dahai)
    assert resp == {"type": "none"}
    assert elapsed < 0.005

    # Player 3 turn: Tsumo & Discard
    p3_tsumo = {"type": "tsumo", "actor": 3, "pai": "5p"}
    _, elapsed = await handler.process_event(p3_tsumo)
    assert elapsed < 0.005

    p3_dahai = {"type": "dahai", "actor": 3, "pai": "7m", "tsumogiri": True}
    resp, elapsed = await handler.process_event(p3_dahai)
    assert resp == {"type": "none"}
    assert elapsed < 0.005

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

    assert len(state_mgr.my_player.tehai) == 13
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
    2. Verify that player seating (東家, 南家, 西家, 北家) is properly tracked and reflected in display_name.
    3. Verify that across multiple rounds with distinct actions:
       - Player 0 (genta2223): 1 Hora, 1 Riichi, 0 Deal-in -> hora_rate = 1/3 (0.333), deal_in_rate = 0.0
       - Player 1 (プロコフ): 0 Hora, 0 Riichi, 1 Deal-in -> deal_in_rate = 1/3 (0.333), hora_rate = 0.0
       - Player 2 (佐藤): 1 Hora (Tsumo), 1 Call, 1 Deal-in -> hora_rate = 1/3, open_rate = 1/3, deal_in_rate = 1/3
       - Player 3 (鈴木): 1 Hora (Ron), 1 Riichi, 0 Deal-in -> hora_rate = 1/3, deal_in_rate = 0.0
    4. Confirm that stats are completely independent and NOT duplicated across players.
    """
    from tools.paifu_to_mjai import (
        create_mock_authentic_game_record,
        convert_record_to_mjai,
        extract_player_names_from_record,
    )
    from core.opponent_profiler import OpponentProfiler

    uuid = "261016-64fee530-a705-4f80-b5ab-23d9eb7c6b36"
    custom_names = ["genta2223", "プロコフ", "佐藤", "鈴木"]
    record = create_mock_authentic_game_record(uuid, player_names=custom_names)

    # 1. Name extraction check
    extracted_names = extract_player_names_from_record(record)
    assert extracted_names == custom_names, f"Expected {custom_names}, got {extracted_names}"

    # 2. MJAI conversion check
    events = convert_record_to_mjai(record, my_seat=0)
    assert events[0]["type"] == "start_game"
    assert events[0]["names"] == custom_names

    # 3. Analyze events in profiler
    profiler = OpponentProfiler()
    profiler.analyze_game_events(events)

    # 4. Check each player's independent stats
    p0 = profiler.get_profile("genta2223")
    p1 = profiler.get_profile("プロコフ")
    p2 = profiler.get_profile("佐藤")
    p3 = profiler.get_profile("鈴木")

    assert p0 is not None and p1 is not None and p2 is not None and p3 is not None

    # Check seating
    assert p0.seat_name == "東家"
    assert p0.display_name == "genta2223 (東家)"
    assert p1.seat_name == "南家"
    assert p1.display_name == "プロコフ (南家)"
    assert p2.seat_name == "西家"
    assert p2.display_name == "佐藤 (西家)"
    assert p3.seat_name == "北家"
    assert p3.display_name == "鈴木 (北家)"

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

    # Player 2: 1 Hora (Tsumo), 1 Call (Chi), 1 Deal-in (dealt into p3 in South 4)
    assert p2.hora_count == 1
    assert p2.call_count >= 1
    assert p2.rounds_with_call == 1
    assert p2.open_rate == round(1 / 3, 3)
    assert p2.deal_in_count == 1

    # Player 3: 1 Hora (Ron), 1 Riichi, 0 Deal-in
    assert p3.hora_count == 1
    assert p3.deal_in_count == 0
    assert p3.riichi_count == 1

    # Ensure stats are independent (NOT duplicated like Self/Player1 in the bug report)
    assert p0.hora_count != p1.hora_count
    assert p0.deal_in_count != p1.deal_in_count
    assert p1.deal_in_count != p3.deal_in_count










