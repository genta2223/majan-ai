"""Windows Desktop Majsoul UI Client Area Auto-Detection and Coordinate Calibrator."""

from __future__ import annotations
import argparse
import ctypes
import os
import sys
import time
import yaml
from typing import Optional, Tuple, Dict, Any

# Ensure mahjong_ai_agent root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from io_adapter.action_controller import ScreenPoint, generate_cubic_bezier_trajectory


class RECT(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long)
    ]


class POINT(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_long),
        ("y", ctypes.c_long)
    ]


def find_majsoul_window_win32() -> Optional[Tuple[int, int, int, int]]:
    """
    Finds the Majsoul Windows client window and returns the pure Client Area (X0, Y0, W, H)
    excluding title bars and borders using Win32 API.
    """
    if sys.platform != "win32":
        print("[Calibrate] Win32 API is only supported on Windows.")
        return None

    user32 = ctypes.windll.user32
    target_hwnd = None
    target_keywords = ["雀魂", "mahjongsoul", "majsoul", "じゃんたま"]

    # EnumWindows callback
    WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

    def enum_windows_proc(hwnd, lParam):
        nonlocal target_hwnd
        if not user32.IsWindowVisible(hwnd) or user32.IsIconic(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length > 0:
            buff = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buff, length + 1)
            title = buff.value.lower()
            if any(k in title for k in target_keywords):
                rect = RECT()
                user32.GetClientRect(hwnd, ctypes.byref(rect))
                w = rect.right - rect.left
                h = rect.bottom - rect.top
                if w >= 400 and h >= 300:
                    target_hwnd = hwnd
                    return False
        return True

    user32.EnumWindows(WNDENUMPROC(enum_windows_proc), 0)

    if not target_hwnd:
        return None

    # Get client rectangle (internal drawing area without title bar)
    rect = RECT()
    user32.GetClientRect(target_hwnd, ctypes.byref(rect))
    width = rect.right - rect.left
    height = rect.bottom - rect.top

    # Convert client (0, 0) to absolute screen coordinates
    pt = POINT(0, 0)
    user32.ClientToScreen(target_hwnd, ctypes.byref(pt))
    x0, y0 = pt.x, pt.y

    return (x0, y0, width, height)


def calculate_screen_mapping_from_rect(
    x0: int,
    y0: int,
    w: int,
    h: int
) -> Dict[str, Any]:
    """
    Calculates exact hand slots and action button positions scaled from 1920x1080 design.
    Accounts for 16:9 aspect fit inside the client rectangle.
    """
    aspect_ratio = 16.0 / 9.0
    current_aspect = w / max(1, h)

    # Calculate actual game render viewport (letterbox/pillarbox handling)
    if current_aspect > aspect_ratio:
        # Window is wider than 16:9 (pillarbox black bars on left/right)
        render_h = h
        render_w = int(h * aspect_ratio)
        offset_x = x0 + (w - render_w) // 2
        offset_y = y0
    else:
        # Window is taller than 16:9 (letterbox black bars on top/bottom)
        render_w = w
        render_h = int(w / aspect_ratio)
        offset_x = x0
        offset_y = y0 + (h - render_h) // 2

    scale_x = render_w / 1920.0
    scale_y = render_h / 1080.0

    def _sx(x_1080: int) -> int:
        return int(offset_x + x_1080 * scale_x)

    def _sy(y_1080: int) -> int:
        return int(offset_y + y_1080 * scale_y)

    # Standard 1920x1080 reference coordinates
    mapping = {
        "client_viewport": {
            "x": offset_x,
            "y": offset_y,
            "width": render_w,
            "height": render_h,
            "scale_x": round(scale_x, 4),
            "scale_y": round(scale_y, 4),
        },
        "resolution": [w, h],
        "hand_area": {
            "start_x": _sx(430),
            "end_x": _sx(1490),
            "y": _sy(980),
            "slot_width": int(75 * scale_x),
        },
        "tsumo_slot": {
            "x": _sx(1530),
            "y": _sy(980),
        },
        "button_positions": {
            "riichi": [_sx(1100), _sy(850)],
            "chi": [_sx(1200), _sy(850)],
            "pon": [_sx(1300), _sy(850)],
            "kan": [_sx(1400), _sy(850)],
            "ron": [_sx(1500), _sy(850)],
            "tsumo": [_sx(1500), _sy(850)],
            "pass": [_sx(1600), _sy(850)],
        }
    }
    return mapping


def update_config_file(mapping: Dict[str, Any], config_path: str) -> None:
    """Updates screen_mapping in config/strategy_config.yaml."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["screen_mapping"] = mapping

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    print(f"[Calibrate] Successfully updated {config_path} with calibrated coordinates.")


def test_hover_positions(mapping: Dict[str, Any]) -> None:
    """
    Test hover sequence: moves cursor smoothly to Left tile -> Center tile -> Tsumo tile -> Pass button.
    Does NOT click anything.
    """
    try:
        import pyautogui
    except ImportError:
        print("[TestHover] pyautogui is required for live cursor movement. Install with: pip install pyautogui")
        return

    hand_area = mapping["hand_area"]
    start_x = hand_area["start_x"]
    slot_w = hand_area["slot_width"]
    y_pos = hand_area["y"] - 15  # Slightly elevated hover position

    left_tile = (start_x + slot_w // 2, y_pos)
    center_tile = (start_x + int(6.5 * slot_w), y_pos)
    tsumo_tile = (mapping["tsumo_slot"]["x"], mapping["tsumo_slot"]["y"] - 15)
    pass_btn = tuple(mapping["button_positions"]["pass"])

    targets = [
        ("Left Hand Tile (Slot 0)", left_tile),
        ("Center Hand Tile (Slot 6)", center_tile),
        ("Tsumo Tile (Slot 13)", tsumo_tile),
        ("Pass Button", pass_btn),
    ]

    print("\n[TestHover] Starting hover verification in 2 seconds...")
    print("            Watch your Majsoul screen to verify tile popup animations.")
    time.sleep(2.0)

    curr_x, curr_y = pyautogui.position()
    p_curr = ScreenPoint(x=curr_x, y=curr_y)

    for name, (tx, ty) in targets:
        print(f" -> Moving to: {name} at ({tx}, {ty})")
        p_target = ScreenPoint(x=tx, y=ty)
        traj = generate_cubic_bezier_trajectory(p_curr, p_target, num_samples=18)
        for pt in traj:
            pyautogui.moveTo(pt.x, pt.y)
            time.sleep(0.012)
        p_curr = p_target
        # Hover wait for popup
        time.sleep(1.2)

    # Return cursor to safe neutral area
    viewport = mapping.get("client_viewport", {})
    rest_x = viewport.get("x", 0) + int(viewport.get("width", 1920) * 0.85)
    rest_y = viewport.get("y", 0) + int(viewport.get("height", 1080) * 0.70)
    pyautogui.moveTo(rest_x, rest_y, duration=0.3)
    print("[TestHover] Hover verification completed successfully.")


def main():
    parser = argparse.ArgumentParser(description="Majsoul Windows Client Window Calibration Tool")
    parser.add_argument("--auto-detect", action="store_true", help="Automatically find Majsoul desktop window via Win32 API")
    parser.add_argument("--test-hover", action="store_true", help="Move mouse to test hover on slots and buttons (no clicks)")
    parser.add_argument("--manual", nargs=4, type=int, metavar=("X", "Y", "W", "H"), help="Manually specify Client Area rect")
    parser.add_argument("--config", default=os.path.join(PROJECT_ROOT, "config", "strategy_config.yaml"), help="Config file path")
    args = parser.parse_args()

    client_rect = None
    if args.auto_detect:
        print("[Calibrate] Searching for Majsoul desktop window...")
        client_rect = find_majsoul_window_win32()
        if client_rect:
            x, y, w, h = client_rect
            print(f"[Calibrate] Found Majsoul Window! Client Area: X={x}, Y={y}, Width={w}, Height={h}")
        else:
            print("[Calibrate] Majsoul window not found. Please ensure the app is running, or use --manual X Y W H")
            if not args.manual:
                # Default fallback 1920x1080
                client_rect = (0, 0, 1920, 1080)
                print("[Calibrate] Falling back to default (0, 0, 1920, 1080).")

    elif args.manual:
        client_rect = tuple(args.manual)
        print(f"[Calibrate] Using manual Client Area: {client_rect}")
    else:
        client_rect = (0, 0, 1920, 1080)

    mapping = calculate_screen_mapping_from_rect(*client_rect)
    update_config_file(mapping, args.config)

    if args.test_hover:
        test_hover_positions(mapping)


if __name__ == "__main__":
    main()
