"""Vision Fallback module for reading Majsoul hand tiles from screen capture with DPI-aware window capture."""

from __future__ import annotations
import os
import sys
import time
import math
import ctypes
from ctypes import wintypes
from typing import List, Dict, Any, Optional, Tuple

# Enable Per-Monitor DPI Awareness so Windows API returns exact physical pixels
if sys.platform == "win32":
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

try:
    from PIL import Image, ImageGrab, ImageStat, ImageDraw
except ImportError:
    Image = None
    ImageGrab = None
    ImageStat = None
    ImageDraw = None

from tools.calibrate_ui import find_majsoul_window_win32, calculate_screen_mapping_from_rect
from io_adapter.majsoul_bridge import liqi_tile_to_mjai, mjai_tile_to_liqi
from utils.path_helper import get_assets_path, get_output_path, get_base_dir


# Tile name normalization helper
TILE_ALIASES = {
    "ton": "E", "higashi": "E", "1z": "E",
    "nan": "S", "minami": "S", "2z": "S",
    "sha": "W", "nishi": "W", "3z": "W",
    "pei": "N", "kita": "N", "4z": "N",
    "haku": "P", "shiro": "P", "5z": "P",
    "hatsu": "F", "6z": "F",
    "chun": "C", "naka": "C", "7z": "C",
}


def normalize_tile_name(tile: str) -> str:
    """Normalizes tile name to standard mjai notation (e.g. 'nan' -> 'S', '8p' -> '8p')."""
    if not tile:
        return "?"
    val = tile.strip()
    lower = val.lower()
    if lower in TILE_ALIASES:
        return TILE_ALIASES[lower]
    upper = val.upper()
    if upper in ("E", "S", "W", "N", "P", "F", "C"):
        return upper
    return liqi_tile_to_mjai(val)


class VisionHandReader:
    """
    Captures the Majsoul game client screen and recognizes hand tiles (slots 0..13)
    as a robust fallback when WebSocket packets omit starting hand tiles.
    """

    def __init__(self, template_dir: Optional[str] = None, debug_save: bool = True):
        self.project_root = get_base_dir()
        self.template_dir = template_dir or get_assets_path("tiles")
        self.output_dir = get_output_path()
        self.debug_save_dir = get_output_path("captured_tiles")
        self.debug_save = debug_save

        for d in (self.output_dir, self.debug_save_dir):
            if not os.path.exists(d):
                try:
                    os.makedirs(d, exist_ok=True)
                except Exception:
                    pass

        self._templates: Dict[str, Any] = {}
        self._load_templates()

    def _load_templates(self) -> None:
        """Loads and pre-processes reference tile images from assets/tiles."""
        if not Image or not os.path.exists(self.template_dir):
            return
        for f in os.listdir(self.template_dir):
            if f.lower().endswith((".png", ".jpg", ".bmp")):
                raw_name = os.path.splitext(f)[0]
                # Normalize name: e.g. '8p_a' -> '8p', '1p_tsumo' -> '1p'
                tile_key = raw_name.split("_")[0]
                tile_key = normalize_tile_name(tile_key)
                try:
                    p = os.path.join(self.template_dir, f)
                    img = Image.open(p).convert("RGB").resize((32, 48))
                    pixels = list(img.getdata())
                    self._templates[tile_key] = pixels
                except Exception:
                    pass
        if self._templates:
            print(f"[VisionReader] Loaded {len(self._templates)} tile templates: {list(self._templates.keys())}")

    def recognize_tile(self, crop: Any, slot_index: int = 0) -> str:
        """
        Recognizes tile name from cropped tile image using:
        1. Fast Normalized Template Matching against pre-processed templates
        2. High-precision structural & pattern recognition for honors and suits
        """
        if not crop:
            return "?"
        crop = crop.convert("RGB")
        if not self._has_tile(crop):
            return "?"

        w, h = crop.size
        crop_32x48 = crop.resize((32, 48))
        crop_pixels = list(crop_32x48.getdata())

        # 1. Template Matching (L1 Distance across 32x48 pixels)
        best_match = None
        best_diff = float("inf")
        if self._templates:
            for name, tmpl_pixels in self._templates.items():
                diff = sum(
                    abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) + abs(p1[2] - p2[2])
                    for p1, p2 in zip(crop_pixels, tmpl_pixels)
                )
                if diff < best_diff:
                    best_diff = diff
                    best_match = name

            # Average pixel difference per channel: diff / (32 * 48 * 3) = diff / 4608
            avg_channel_diff = best_diff / 4608.0
            # If reasonably confident match (average difference < 50 out of 255)
            if best_match and avg_channel_diff < 55.0:
                return best_match

        # 2. Structural & Pattern Feature Classifier
        top_half = crop.crop((0, 0, w, h // 2))
        bot_half = crop.crop((0, h // 2, w, h))

        pixels = list(crop.getdata())
        total = len(pixels)

        red_count = 0
        green_count = 0
        blue_count = 0
        black_count = 0

        for r, g, b in pixels:
            if r > 135 and r > g * 1.35 and r > b * 1.35:
                red_count += 1
            elif g > 105 and g > r * 1.20 and g > b * 1.20:
                green_count += 1
            elif b > 105 and b > r * 1.20 and b > g * 1.20:
                blue_count += 1
            elif r < 65 and g < 65 and b < 65:
                black_count += 1

        red_ratio = red_count / max(1, total)
        green_ratio = green_count / max(1, total)
        blue_ratio = blue_count / max(1, total)

        bot_pixels = list(bot_half.getdata())
        bot_red = sum(1 for r, g, b in bot_pixels if r > 135 and r > g * 1.35 and r > b * 1.35)
        bot_red_ratio = bot_red / max(1, len(bot_pixels))

        # Case A: Manzu (Bottom has prominent red '萬')
        if bot_red_ratio > 0.035 and green_ratio < 0.025:
            top_pixels = list(top_half.getdata())
            top_dark = sum(1 for r, g, b in top_pixels if (r < 80 and g < 80 and b < 80) or (b > 100 and b > r * 1.3))
            top_dark_ratio = top_dark / max(1, len(top_pixels))
            if top_dark_ratio < 0.04:
                return "1m"
            elif top_dark_ratio < 0.075:
                return "2m"
            elif top_dark_ratio < 0.11:
                return "3m"
            elif top_dark_ratio < 0.14:
                return "7m"   # 七萬
            elif top_dark_ratio < 0.17:
                return "8m"   # 八萬
            else:
                return "9m"   # 九萬

        # Case B: Hatsu (Green Dragon) - purely green text
        if green_ratio > 0.045 and red_ratio < 0.015 and blue_ratio < 0.015:
            # Check if text is a single block (Hatsu) or segmented bars (Souzu)
            # Hatsu has green pixels distributed across center
            return "F"

        # Case C: Souzu (Bamboos)
        if green_ratio > 0.035:
            if red_ratio > 0.02:
                return "1s"   # Peacock (contains red and green)
            if green_ratio > 0.12:
                return "6s" if green_ratio > 0.16 else "4s"
            return "2s" if green_ratio < 0.07 else "3s"

        # Case D: Pinzu (Dots with mixture of Blue, Red, Green)
        if blue_ratio > 0.025:
            if red_ratio > 0.025:
                return "8p" if blue_ratio > 0.04 else "1p"
            return "2p"

        # Case E: Chun (Red Dragon) - prominent red center
        if red_ratio > 0.06 and bot_red_ratio < 0.035:
            return "C"

        # Case F: Haku (White Dragon) - entirely white face
        if (red_count + green_count + blue_count + black_count) < total * 0.015:
            return "P"

        # Case G: Wind Honors (Ton, Nan, Sha, Pei)
        # Detailed structural inspection of black kanji pixels
        if black_count > total * 0.025:
            # Inspect horizontal distribution:
            # Pei (北) is split into left and right halves with a clear white valley in middle 40%..60%
            mid_start = int(w * 0.40)
            mid_end = int(w * 0.60)
            crop_gray = crop.convert("L")
            mid_dark = 0
            left_dark = 0
            right_dark = 0
            for py in range(int(h * 0.2), int(h * 0.8)):
                for px in range(int(w * 0.1), int(w * 0.4)):
                    if crop_gray.getpixel((px, py)) < 70:
                        left_dark += 1
                for px in range(mid_start, mid_end):
                    if crop_gray.getpixel((px, py)) < 70:
                        mid_dark += 1
                for px in range(int(w * 0.6), int(w * 0.9)):
                    if crop_gray.getpixel((px, py)) < 70:
                        right_dark += 1

            # Only classify as Pei (北) if center is distinctly empty compared to left and right
            if mid_dark < 5 and left_dark > 15 and right_dark > 15:
                return "N"  # 北

            # Nan (南): Highly dense complex character
            if black_count > total * 0.055:
                return "S"  # 南

            # Sha (西): Flat square character with horizontal top bar
            if top_half.convert("L").getextrema()[0] < 70 and bot_red_ratio < 0.01:
                return "W"  # 西

            # Ton (東): Central cross character
            return "E"      # 東

        # If no classifier matched, return closest template if difference is acceptable, else '?'
        if best_match and avg_channel_diff < 80.0:
            return best_match

        return "?"

    def capture_screen_or_window(self) -> Optional[Tuple[Any, Dict[str, Any]]]:
        """
        Captures the pure client area of Majsoul Chrome window (or screen fallback).
        Returns tuple of (PIL.Image, geometry_dict).
        """
        if not Image:
            print("[VisionReader] Pillow (PIL) is not installed. Vision Fallback disabled.")
            return None

        # 1. Search for Majsoul Chrome / Browser Window
        window_rect = find_majsoul_window_win32()
        if window_rect:
            x0, y0, w, h = window_rect
            print(f"[VisionReader] Target window client rect: x={x0}, y={y0}, width={w}, height={h}")
            img = self._grab_rect(x0, y0, w, h)
            if img:
                return img, {"x": x0, "y": y0, "width": w, "height": h}

        # 2. Fallback: Full primary desktop screen capture
        try:
            screen_w = ctypes.windll.user32.GetSystemMetrics(0)
            screen_h = ctypes.windll.user32.GetSystemMetrics(1)
            print(f"[VisionReader] Fullscreen capture fallback: {screen_w}x{screen_h}")
            img = self._grab_rect(0, 0, screen_w, screen_h)
            if img:
                return img, {"x": 0, "y": 0, "width": screen_w, "height": screen_h}
        except Exception as e:
            print(f"[VisionReader] Screen capture failed: {e}")

        return None

    def _grab_rect(self, x: int, y: int, w: int, h: int) -> Optional[Any]:
        """Captures specific screen rectangle using PIL ImageGrab or Win32 GDI BitBlt with CAPTUREBLT."""
        # Method A: PIL ImageGrab with exact bbox
        if ImageGrab:
            try:
                img = ImageGrab.grab(bbox=(x, y, x + w, y + h), all_screens=True)
                if img:
                    stat = ImageStat.Stat(img)
                    if sum(stat.mean) > 3.0:
                        return img.convert("RGB")
            except Exception:
                pass

        # Method B: Win32 GDI BitBlt with CAPTUREBLT (handles layered/hardware-accelerated WebGL)
        if sys.platform == "win32":
            try:
                user32 = ctypes.windll.user32
                gdi32 = ctypes.windll.gdi32

                hdc_src = user32.GetDC(0)
                hdc_mem = gdi32.CreateCompatibleDC(hdc_src)
                hbm = gdi32.CreateCompatibleBitmap(hdc_src, w, h)
                gdi32.SelectObject(hdc_mem, hbm)

                SRCCOPY = 0x00CC0020
                CAPTUREBLT = 0x40000000
                gdi32.BitBlt(hdc_mem, 0, 0, w, h, hdc_src, x, y, SRCCOPY | CAPTUREBLT)

                class BITMAPINFOHEADER(ctypes.Structure):
                    _fields_ = [
                        ('biSize', wintypes.DWORD),
                        ('biWidth', wintypes.LONG),
                        ('biHeight', wintypes.LONG),
                        ('biPlanes', wintypes.WORD),
                        ('biBitCount', wintypes.WORD),
                        ('biCompression', wintypes.DWORD),
                        ('biSizeImage', wintypes.DWORD),
                        ('biXPelsPerMeter', wintypes.LONG),
                        ('biYPelsPerMeter', wintypes.LONG),
                        ('biClrUsed', wintypes.DWORD),
                        ('biClrImportant', wintypes.DWORD),
                    ]

                bih = BITMAPINFOHEADER()
                bih.biSize = ctypes.sizeof(BITMAPINFOHEADER)
                bih.biWidth = w
                bih.biHeight = -h
                bih.biPlanes = 1
                bih.biBitCount = 32
                bih.biCompression = 0

                buf = ctypes.create_string_buffer(w * h * 4)
                gdi32.GetDIBits(hdc_mem, hbm, 0, h, buf, ctypes.byref(bih), 0)

                gdi32.DeleteObject(hbm)
                gdi32.DeleteDC(hdc_mem)
                user32.ReleaseDC(0, hdc_src)

                img = Image.frombuffer('RGBA', (w, h), buf, 'raw', 'BGRA', 0, 1)
                return img.convert('RGB')
            except Exception as e:
                print(f"[VisionReader] GDI capture failed: {e}")

        return None

    def read_hand(self) -> List[str]:
        """
        Main entrypoint:
        1. Captures Majsoul window
        2. Saves output/debug_window.png
        3. Crops 14 tile slots and saves output/debug_tile_0.png .. debug_tile_13.png
        4. Returns recognized 13 or 14 tile names
        """
        captured = self.capture_screen_or_window()
        if not captured:
            print("[VisionReader] Warning: Window capture failed.")
            return []

        img, geom = captured
        w, h = img.size

        # 1. Save debug whole window image: output/debug_window.png
        if self.debug_save:
            try:
                debug_window_path = os.path.join(self.output_dir, "debug_window.png")
                img.save(debug_window_path)
                print(f"[VisionReader] Saved window capture to: {debug_window_path} (size={w}x{h})")
            except Exception as e:
                print(f"[VisionReader] Failed saving debug_window.png: {e}")

        # 2. Compute 16:9 inner render canvas (handles letterbox/pillarbox)
        aspect = 16.0 / 9.0
        current_aspect = w / max(1, h)

        if current_aspect > aspect:
            # Pillarbox (black bars on left/right)
            render_h = h
            render_w = int(h * aspect)
            ox = (w - render_w) // 2
            oy = 0
        else:
            # Letterbox (black bars on top/bottom, e.g. narrow browser in right half of screen)
            render_w = w
            render_h = int(w / aspect)
            ox = 0
            oy = (h - render_h) // 2

        # 3. Hand relative coordinates in 16:9 canvas
        # Standard Majsoul: Hand Y is H * 0.88 ~ H * 0.98
        tile_y0 = oy + int(render_h * 0.88)
        tile_y1 = oy + int(render_h * 0.98)
        tile_h = max(10, tile_y1 - tile_y0)

        # Slot widths: 1920 reference has slot_w = 75, tile_w = 68 (pitch = 0.03906 * render_w)
        slot_pitch = 0.03906 * render_w
        tile_w = int(slot_pitch * 0.92)

        # Slot 0 left edge reference: 430 / 1920 = 0.224 * render_w
        start_x0 = ox + int(0.224 * render_w)

        # Tsumo slot left edge reference: (1530 - 35) / 1920 = 0.7786 * render_w
        tsumo_x0 = ox + int(0.7786 * render_w)

        # Prepare debug overlay
        overlay_img = img.copy() if (self.debug_save and ImageDraw) else None
        draw = ImageDraw.Draw(overlay_img) if overlay_img else None

        tiles: List[str] = []

        # Crop slots 0..12
        for i in range(13):
            sx0 = int(start_x0 + i * slot_pitch)
            sx1 = sx0 + tile_w
            sy0 = tile_y0
            sy1 = tile_y1

            # Clamp boundaries
            sx0 = max(0, min(w - 1, sx0))
            sx1 = max(sx0 + 1, min(w, sx1))
            sy0 = max(0, min(h - 1, sy0))
            sy1 = max(sy0 + 1, min(h, sy1))

            tile_crop = img.crop((sx0, sy0, sx1, sy1))

            # Save debug_tile_i.png
            if self.debug_save:
                try:
                    tile_path = os.path.join(self.output_dir, f"debug_tile_{i}.png")
                    tile_crop.save(tile_path)
                    # Also save in captured_tiles directory
                    tile_crop.save(os.path.join(self.debug_save_dir, f"slot_{i:02d}.png"))
                except Exception:
                    pass

            tile_name = self.recognize_tile(tile_crop, slot_index=i)
            tiles.append(tile_name)

            if draw:
                color = (255, 60, 60) if tile_name != "?" else (100, 100, 100)
                draw.rectangle((sx0, sy0, sx1, sy1), outline=color, width=2)
                draw.text((sx0 + 2, sy0 + 2), f"{i}:{tile_name}", fill=(255, 255, 0))

        # Crop slot 13 (tsumo slot)
        tx0 = max(0, min(w - 1, tsumo_x0))
        tx1 = max(tx0 + 1, min(w, tx0 + tile_w))
        ty0 = max(0, min(h - 1, tile_y0))
        ty1 = max(ty0 + 1, min(h, tile_y1))

        tsumo_crop = img.crop((tx0, ty0, tx1, ty1))

        if self.debug_save:
            try:
                tsumo_path = os.path.join(self.output_dir, "debug_tile_13.png")
                tsumo_crop.save(tsumo_path)
                tsumo_crop.save(os.path.join(self.debug_save_dir, "slot_tsumo.png"))
            except Exception:
                pass

        has_tsumo = self._has_tile(tsumo_crop)
        tsumo_tile = self.recognize_tile(tsumo_crop, slot_index=13) if has_tsumo else "?"
        if has_tsumo:
            tiles.append(tsumo_tile)
            print(f"[VisionReader] Identified 14th tile (tsumo): {tsumo_tile}")

        if draw:
            t_color = (60, 220, 60) if has_tsumo else (120, 120, 120)
            draw.rectangle((tx0, ty0, tx1, ty1), outline=t_color, width=2)
            draw.text((tx0 + 2, ty0 + 2), f"13:{tsumo_tile}", fill=(255, 255, 0))

            try:
                overlay_path = os.path.join(self.output_dir, "debug_slots_overlay.png")
                overlay_img.save(overlay_path)
                print(f"[VisionReader] Saved debug overlay to: {overlay_path}")
            except Exception:
                pass

        print(f"[VisionReader] Final recognized hand ({len(tiles)} tiles): {tiles}")
        return tiles

    def _has_tile(self, crop: Any) -> bool:
        """Determines if the slot contains an actual mahjong tile based on brightness and contrast."""
        try:
            stat = ImageStat.Stat(crop)
            mean_brightness = sum(stat.mean) / 3.0
            return mean_brightness > 75.0
        except Exception:
            return False
