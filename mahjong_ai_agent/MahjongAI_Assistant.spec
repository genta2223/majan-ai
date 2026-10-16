# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['C:/Users/genta/Dropbox/11_開発/majan-ai/mahjong_ai_agent/gui.py'],
    pathex=[],
    binaries=[],
    datas=[('C:/Users/genta/Dropbox/11_開発/majan-ai/mahjong_ai_agent/config', 'config'), ('C:/Users/genta/Dropbox/11_開発/majan-ai/mahjong_ai_agent/assets', 'assets')],
    hiddenimports=['asyncio', 'websockets', 'websockets.legacy', 'websockets.legacy.client', 'websockets.client', 'pyautogui', 'PIL', 'PIL.Image', 'PIL.ImageGrab', 'PIL.ImageStat', 'PIL.ImageDraw', 'yaml', 'ctypes', 'ctypes.wintypes', 'tkinter', 'tkinter.ttk', 'tkinter.messagebox', 'tkinter.scrolledtext', 'pydantic', 'queue', 'threading', 'json', 'urllib', 'urllib.request', 'utils', 'utils.path_helper', 'core', 'core.engine', 'core.state_manager', 'core.strategy_modifier', 'core.opponent_profiler', 'io_adapter', 'io_adapter.action_controller', 'io_adapter.live_sniffer', 'io_adapter.majsoul_bridge', 'io_adapter.vision_hand_reader', 'io_adapter.input_simulator', 'io_adapter.stream_client', 'tools.calibrate_ui'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='MahjongAI_Assistant',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
