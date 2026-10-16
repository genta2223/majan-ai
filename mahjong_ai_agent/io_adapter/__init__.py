"""I/O adapters for input simulation, real-time stream client, Majsoul bridge, and mouse control."""

from .input_simulator import InputSimulator, load_mjai_log
from .action_controller import ActionController, OutputAction, ScreenPoint, MouseStep
from .majsoul_bridge import MajsoulBridge, liqi_tile_to_mjai, mjai_tile_to_liqi
from .stream_client import MjaiStreamHandler, StdioStreamClient, WebSocketStreamClient, is_my_decision_required

__all__ = [
    "InputSimulator",
    "load_mjai_log",
    "ActionController",
    "OutputAction",
    "ScreenPoint",
    "MouseStep",
    "MajsoulBridge",
    "liqi_tile_to_mjai",
    "mjai_tile_to_liqi",
    "MjaiStreamHandler",
    "StdioStreamClient",
    "WebSocketStreamClient",
    "is_my_decision_required",
]
