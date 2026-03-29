"""PyInstaller spec for Gel Image Tool — works on macOS and Windows."""

import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files

block_cipher = None

a = Analysis(
    ["src/gel_tool/app.py"],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=["cv2", "numpy"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "scipy", "pandas"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Gel Image Tool",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="Gel Image Tool",
)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="Gel Image Tool.app",
        icon=None,
        bundle_identifier="com.oliveiralab.gel-image-tool",
        info_plist={
            "CFBundleShortVersionString": "0.1.0",
            "NSHighResolutionCapable": True,
        },
    )
