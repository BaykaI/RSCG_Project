# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
import sys

python_root = Path(sys.base_prefix)
tcl_root = python_root / "tcl"

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=[
        (str(tcl_root / "tcl8.6"), "lib/tcl8.6"),
        (str(tcl_root / "tk8.6"), "lib/tk8.6"),
    ],
    hiddenimports=[],
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
    name='app',
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
