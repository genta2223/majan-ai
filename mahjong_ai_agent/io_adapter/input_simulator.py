"""Input simulator for feeding mjai event streams and game logs into the AI engine."""

from __future__ import annotations
import json
import asyncio
from typing import List, Dict, Any, AsyncIterator, Union
import os


def load_mjai_log(file_path: str) -> List[Dict[str, Any]]:
    """Loads a list of mjai events from a JSON or JSONL file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Log file not found: {file_path}")

    events = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


class InputSimulator:
    """Simulates incoming mjai event stream asynchronously."""

    def __init__(self, events: List[Dict[str, Any]], replay_speed: float = 0.0):
        """
        replay_speed: If > 0, introduces an artificial delay (in seconds) between incoming events.
        """
        self.events = events
        self.replay_speed = replay_speed
        self._current_idx = 0

    async def stream_events(self) -> AsyncIterator[Dict[str, Any]]:
        """Asynchronously stream events one by one."""
        for event in self.events:
            if self.replay_speed > 0:
                await asyncio.sleep(self.replay_speed)
            yield event

    @staticmethod
    def create_mock_ooras_scenario(my_id: int = 0, my_score: int = 38000, enemy_riichi: bool = True) -> List[Dict[str, Any]]:
        """
        Generates a realistic South 4 (Ooras) scenario for testing.
        Player 0 (self) is in 1st place with 38,000 points.
        Player 1 is in 2nd place with 31,000 points and declares Riichi.
        Player 2 has 20,000 points.
        Player 3 has 11,000 points.
        """
        events = [
            {"type": "start_game", "names": ["Self", "Player1", "Player2", "Player3"]},
            {
                "type": "start_kyoku",
                "bakaze": "S",
                "kyoku": 4,
                "honba": 0,
                "kyotaku": 0,
                "oya": 3,
                "scores": [my_score, 31000, 20000, 11000],
                "tehais": [
                    # Self has 1m (Genbutsu if P1 discarded it), 5m (dangerous red dora/middle), 1z, etc.
                    ["1m", "2m", "3m", "7p", "8p", "9p", "2s", "3s", "4s", "5m", "1z", "1z", "9s"],
                    [], [], []
                ],
                "dora_marker": "9p"
            },
            # Turns before riichi
            {"type": "tsumo", "actor": 3, "pai": "3z"},
            {"type": "dahai", "actor": 3, "pai": "3z", "tsumogiri": True},
            {"type": "tsumo", "actor": 0, "pai": "1m"},
            {"type": "dahai", "actor": 0, "pai": "9s", "tsumogiri": False},
            {"type": "tsumo", "actor": 1, "pai": "1m"},
            # Player 1 discards 1m
            {"type": "dahai", "actor": 1, "pai": "1m", "tsumogiri": False},
            {"type": "tsumo", "actor": 2, "pai": "2z"},
            {"type": "dahai", "actor": 2, "pai": "2z", "tsumogiri": True},
            {"type": "tsumo", "actor": 3, "pai": "4z"},
            {"type": "dahai", "actor": 3, "pai": "4z", "tsumogiri": True},
        ]

        if enemy_riichi:
            events.extend([
                # Next turn P1 declares Riichi discarding 6s
                {"type": "tsumo", "actor": 0, "pai": "8m"},
                {"type": "dahai", "actor": 0, "pai": "8m", "tsumogiri": True},
                {"type": "tsumo", "actor": 1, "pai": "6s"},
                {"type": "reach", "actor": 1},
                {"type": "dahai", "actor": 1, "pai": "6s", "tsumogiri": False},
                {"type": "reach_accepted", "actor": 1},
                {"type": "tsumo", "actor": 2, "pai": "7z"},
                {"type": "dahai", "actor": 2, "pai": "7z", "tsumogiri": True},
                {"type": "tsumo", "actor": 3, "pai": "5z"},
                {"type": "dahai", "actor": 3, "pai": "5z", "tsumogiri": True},
                # Now it's Self's turn! Self drew a dangerous 5s
                {"type": "tsumo", "actor": 0, "pai": "5s"},
            ])
        else:
            events.extend([
                {"type": "tsumo", "actor": 0, "pai": "5s"},
            ])

        return events
