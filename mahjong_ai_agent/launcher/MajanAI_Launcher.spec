# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['C:/Users/genta/Dropbox/11_開発/majan-ai/mahjong_ai_agent/launcher/bootstrap_launcher.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['torch', 'scipy', 'pandas', 'matplotlib', 'numpy', 'cv2', 'PIL', 'pydantic', 'websockets', 'pytest', 'yaml', 'IPython', 'pyautogui'],
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
    name='MajanAI_Launcher',
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
