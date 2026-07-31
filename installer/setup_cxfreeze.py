"""cx_Freeze build: a Windows folder install, and an MSI wrapping it.

Run from the repository root with the Windows interpreter:

    python installer/setup_cxfreeze.py build_exe      # folder + exes
    python installer/setup_cxfreeze.py bdist_msi      # installer

Two executables land in the install:

    svrspec.exe       the desktop window (no console, no socket)
    svrspec-cli.exe   the same engine on the command line

The catalogue JSON is copied next to the package rather than zipped, because the
loader reads it with `Path(__file__).parent`, and a delivery tool should let the
customer open those files and check the numbers.
"""

from __future__ import annotations

import sys
from pathlib import Path

from cx_Freeze import Executable, setup

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from svrspec import __version__  # noqa: E402

ICON = ROOT / "installer" / "svrspec.ico"

#: Stable across releases: Windows uses it to recognise an upgrade rather than
#: installing a second copy side by side. Never regenerate this.
UPGRADE_CODE = "{6F2A9C31-4D8B-4E5A-9C77-3B1E0A5D8F42}"

build_exe_options = {
    "packages": ["svrspec", "webview"],
    "excludes": [
        # Nothing here needs the science stack, a GUI toolkit or a test runner;
        # excluding them keeps the install small and the build honest about
        # what it actually depends on.
        "tkinter", "unittest", "pydoc_data", "pytest", "numpy", "pandas",
        "matplotlib", "PIL", "setuptools", "pip",
    ],
    "include_files": [
        (str(ROOT / "svrspec" / "catalog"), "lib/svrspec/catalog"),
        (str(ROOT / "README.md"), "README.md"),
    ],
    # Both of these must stay unzipped, and for the same reason: they read their
    # own data files off the filesystem.
    #
    #   svrspec   the catalogue loader resolves JSON with Path(__file__).parent,
    #             and a delivery tool should let the customer open those files.
    #   webview   pywebview injects its bridge by globbing webview/js/**/*.js
    #             (see webview.util.load_js_files). Inside the zip that glob
    #             matches nothing, so api.js never runs, window.pywebview is
    #             never created, `pywebviewready` never fires -- and the window
    #             renders the page but no dropdown ever fills, because the
    #             bridge the page talks through does not exist. It fails
    #             silently: the import succeeds and the window opens.
    "zip_exclude_packages": ["svrspec", "webview"],
    "include_msvcr": True,
    "optimize": 1,
}

bdist_msi_options = {
    "upgrade_code": UPGRADE_CODE,
    "add_to_path": False,
    "initial_target_dir": r"[ProgramFiles64Folder]\svrspec",
    "install_icon": str(ICON),
    "summary_data": {
        "author": "svrspec",
        "comments": "GPU 서빙 서버 스펙 산정 시뮬레이터",
        "keywords": "server sizing llm capacity planning",
    },
    "all_users": False,
}

setup(
    name="svrspec",
    version=__version__,
    description="GPU 서빙 서버 스펙 산정 시뮬레이터",
    options={"build_exe": build_exe_options, "bdist_msi": bdist_msi_options},
    executables=[
        Executable(
            str(ROOT / "installer" / "app_entry.py"),
            base="gui",                 # no console window
            target_name="svrspec.exe",
            icon=str(ICON),
            shortcut_name="svrspec",
            shortcut_dir="ProgramMenuFolder",
            copyright="Copyright (c) 2026 rokaproj",
        ),
        Executable(
            str(ROOT / "installer" / "cli_entry.py"),
            base="console",
            target_name="svrspec-cli.exe",
            icon=str(ICON),
        ),
    ],
)
