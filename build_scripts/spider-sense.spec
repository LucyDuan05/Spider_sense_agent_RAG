# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for Spider-Sense v2
Build: pyinstaller build_scripts/spider-sense.spec
Output: dist/spider-sense.exe
"""

import os
import sys

block_cipher = None

# Collect all data files
datas = [
    ('knowledge', 'knowledge'),
    ('models', 'models'),
    ('web-frontend/build', 'web-frontend/build'),
    ('requirements.txt', '.'),
]

# Try to add processed data directories
for d in ['processed_cicids', 'processed_cicids2018', 'processed_nslkdd', 'processed_unsw_nb15']:
    if os.path.exists(d):
        datas.append((d, d))

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=[
        'sklearn', 'sklearn.ensemble._forest', 'sklearn.tree',
        'sklearn.utils._typedefs', 'numpy', 'pandas',
        'torch', 'flask', 'flask_cors',
        'backend', 'backend.detection_engine', 'backend.orchestrator',
        'backend.rag_engine', 'backend.xai_engine', 'backend.agent_layer',
        'backend.app',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib', 'scipy', 'PIL', 'cv2', 'tensorflow',
        'jupyter', 'IPython', 'notebook',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='spider-sense',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
