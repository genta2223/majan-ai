"""Detailed debug logger and event tracer for Majsoul AI Agent."""

from __future__ import annotations
import datetime
import os
import sys
import threading
import time
from typing import Any, Dict, Optional


class DebugTracer:
    """
    Thread-safe detailed debug logger with millisecond timestamps and component tracking.
    Can be enabled via environment variable MAHJONG_DEBUG=1 or dynamically via toggle.
    """
    _instance: Optional[DebugTracer] = None
    _lock = threading.Lock()

    def __init__(self):
        self.enabled: bool = os.environ.get("MAHJONG_DEBUG", "1") in ("1", "true", "True")
        self.last_event_time: float = time.time()
        self.last_event_type: str = "INIT"
        self.last_state_summary: str = "Initial state"

    @classmethod
    def get_instance(cls) -> DebugTracer:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @staticmethod
    def now_str() -> str:
        return datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]

    def log(self, stage: str, message: str, payload: Optional[Any] = None) -> None:
        """
        Output structured trace log:
        [HH:MM:SS.mmm] [STAGE] message | payload
        """
        if not self.enabled:
            return

        now = self.now_str()
        self.last_event_time = time.time()
        self.last_event_type = stage

        if payload is not None:
            if isinstance(payload, dict):
                # Clean up huge raw fields for readability if present
                p_copy = {k: v for k, v in payload.items() if k not in ("raw_sub_fields", "raw")}
                log_line = f"[{now}] [{stage}] {message} | data={p_copy}"
            elif isinstance(payload, bytes):
                log_line = f"[{now}] [{stage}] {message} | len={len(payload)} hex={payload[:32].hex()}..."
            else:
                log_line = f"[{now}] [{stage}] {message} | {payload}"
        else:
            log_line = f"[{now}] [{stage}] {message}"

        # Write to stdout with flush
        print(log_line, flush=True)

    def mark_event(self, event_type: str, summary: str = "") -> None:
        self.last_event_time = time.time()
        self.last_event_type = event_type
        if summary:
            self.last_state_summary = summary

    def get_idle_seconds(self) -> float:
        return time.time() - self.last_event_time


# Module-level convenience functions
def trace_log(stage: str, message: str, payload: Optional[Any] = None) -> None:
    DebugTracer.get_instance().log(stage, message, payload)


def mark_trace_event(event_type: str, summary: str = "") -> None:
    DebugTracer.get_instance().mark_event(event_type, summary)


def get_tracer() -> DebugTracer:
    return DebugTracer.get_instance()
