"""Paifu (game record) downloader module for MahjongSoul."""

from __future__ import annotations
import os
import re
import sys
from typing import Dict, Any, List, Optional, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.paifu_to_mjai import (
    PAIFU_UUID_REGEX,
    sanitize_filename,
    parse_paifu_url,
    extract_paifu_uuids,
    extract_player_names_from_record,
    create_mock_authentic_game_record,
    is_cpu_player_name,
    PaifuDownloader,
)

__all__ = [
    "PAIFU_UUID_REGEX",
    "sanitize_filename",
    "parse_paifu_url",
    "extract_paifu_uuids",
    "extract_player_names_from_record",
    "create_mock_authentic_game_record",
    "is_cpu_player_name",
    "PaifuDownloader",
]
