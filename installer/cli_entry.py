"""Entry point for the console executable.

The Windows console defaults to the system code page (cp949 on a Korean
install), which turns every Korean label in the output into mojibake. A frozen
app does not pick up PYTHONUTF8 from the environment, so UTF-8 is forced here
before anything prints.
"""

from __future__ import annotations

import sys


def _force_utf8() -> None:
    if sys.platform == "win32":
        try:
            import ctypes

            # 65001 = CP_UTF8. Without this the console renders the bytes
            # through the old code page even though Python emits UTF-8.
            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
            ctypes.windll.kernel32.SetConsoleCP(65001)
        except Exception:
            pass
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


_force_utf8()

from svrspec.cli import main  # noqa: E402

raise SystemExit(main())
