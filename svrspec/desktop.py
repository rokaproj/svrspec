"""Native desktop window — no HTTP server, no socket, no browser.

The page is handed to an embedded Edge WebView2 as a string, and the JavaScript
talks to Python through pywebview's bridge instead of `fetch`. So the packaged
.exe is a single window that opens and computes: nothing listens on a port,
nothing shows up in the firewall prompt, and there is no localhost URL to
explain to whoever installs it.

The sizing engine is untouched and shared with the CLI and the report writer --
this module is only a shell around it.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import report
from .catalog import Catalog, CatalogError
from .gui import (
    DEFAULT_VOLUMES,
    _params,
    _workload,
    app_html,
    catalog_payload,
    resource_payload,
    size_payload,
)
from .perf import Efficiency
from .sizing import sweep_cpus, tiers

WINDOW_TITLE = "svrspec — GPU 서빙 서버 스펙 산정"
#: Kept above the stylesheet's single-column breakpoint (900px) so the input
#: rail and the results always sit side by side in the app window.
MIN_SIZE = (1100, 760)
DEFAULT_SIZE = (1600, 1000)


class Api:
    """What the page can call. Every entry point validates before computing."""

    def __init__(self, catalog: Catalog | None = None) -> None:
        self._catalog = catalog or Catalog()
        self.window = None  # set once pywebview has created it

    # -- data ------------------------------------------------------------
    def catalog(self) -> dict:
        return catalog_payload(self._catalog)

    def size(self, params: dict | str) -> dict:
        raw = json.loads(params) if isinstance(params, str) else (params or {})
        if not isinstance(raw, dict):
            return {"error": "expected an object of parameters"}
        try:
            return size_payload(self._catalog, _params(raw))
        except CatalogError as exc:
            return {"error": str(exc)}

    def resources(self, params: dict | str) -> dict:
        """Task-manager view for one hardware build across alarm volumes."""
        raw = json.loads(params) if isinstance(params, str) else (params or {})
        if not isinstance(raw, dict):
            return {"error": "expected an object of parameters"}
        volumes = raw.get("volumes") or list(DEFAULT_VOLUMES)
        try:
            clean = [max(1, min(100_000, int(v))) for v in list(volumes)[:6]] or list(DEFAULT_VOLUMES)
            return resource_payload(
                self._catalog, _params(raw), str(raw.get("cpu", "")), tuple(clean)
            )
        except (CatalogError, ValueError, TypeError) as exc:
            return {"error": str(exc)}

    def capacity(self, params: dict | str) -> dict:
        """Where this build breaks, on each load axis.

        Deliberately not part of the live recompute: a four-axis knee search
        simulates whole days at loads far above the configured one and takes
        seconds, where every other call here is milliseconds. The page runs it
        from a button and caches the result until the inputs change.
        """
        raw = json.loads(params) if isinstance(params, str) else (params or {})
        if not isinstance(raw, dict):
            return {"error": "expected an object of parameters"}
        try:
            from .gui import _axes, capacity_payload

            return capacity_payload(
                self._catalog, _params(raw), str(raw.get("cpu", "")), _axes(raw)
            )
        except (CatalogError, ValueError, TypeError) as exc:
            return {"error": str(exc)}

    def lab(self, params: dict | str) -> dict:
        """Assemble one virtual machine and report what is wrong with it.

        Cheap enough to call on every keystroke -- it is arithmetic over the
        catalogue, not a simulation -- which is the point: an under-populated
        DIMM layout should be visible while the operator is still choosing it,
        not after they have bought it.
        """
        raw = json.loads(params) if isinstance(params, str) else (params or {})
        if not isinstance(raw, dict):
            return {"error": "expected an object of parameters"}
        try:
            from .gui import _params, lab_payload

            return lab_payload(
                self._catalog, _params(raw), str(raw.get("cpu", "")),
                str(raw.get("name", "A")),
            )
        except (CatalogError, ValueError, TypeError, KeyError) as exc:
            return {"error": str(exc)}

    def bench(self, params: dict | str) -> dict:
        """Run one load profile against one machine and return replay frames.

        Not part of the live recompute: a full run is cheap in wall clock but
        it is still a whole simulated day, so the page drives it from a button
        and replays the returned frames itself.
        """
        raw = json.loads(params) if isinstance(params, str) else (params or {})
        if not isinstance(raw, dict):
            return {"error": "expected an object of parameters"}
        try:
            from .gui import _params, bench_payload

            return bench_payload(
                self._catalog, _params(raw), str(raw.get("cpu", "")), raw
            )
        except (CatalogError, ValueError, TypeError, KeyError) as exc:
            return {"error": str(exc)}

    def update_check(self) -> dict:
        from .gui import update_payload

        return update_payload()

    def update_install(self) -> str:
        """Download, verify, launch. The window closes so the installer can run."""
        from . import update as up

        release = up.check(timeout=8.0)
        if release is None:
            return "이미 최신 버전이다."
        try:
            path = up.download(release, timeout=180.0)
        except up.UpdateError as exc:
            return f"업데이트 중단: {exc}"
        try:
            up.launch_installer(path)
        except up.UpdateError as exc:
            return f"설치 파일을 실행할 수 없다: {exc}"
        if self.window is not None:
            self.window.destroy()
        return f"설치 프로그램을 실행했다 · {path.name}"

    # -- files -----------------------------------------------------------
    def save_report(self, params: dict | str, fmt: str = "html") -> str:
        """Render a delivery artefact and ask the OS where to put it."""
        raw = json.loads(params) if isinstance(params, str) else (params or {})
        p = _params(raw if isinstance(raw, dict) else {})
        try:
            model = self._catalog.model(p["model"])
            quant = self._catalog.quant(p["quant"])
        except CatalogError as exc:
            return f"저장 실패: {exc}"

        workload = _workload(p)
        candidates = sweep_cpus(
            self._catalog, model, quant, workload,
            sockets=p["sockets"], dimms_per_channel=p["dpc"],
        )
        suggested = f"svrspec-{model.id}-{quant.id}.{'csv' if fmt == 'csv' else 'html'}"
        target = self._ask_where(suggested, fmt)
        if not target:
            return ""

        path = Path(target)
        if fmt == "csv":
            report.write_csv(candidates, path)
        else:
            eff = Efficiency.from_catalog(self._catalog.coefficients)
            path.write_text(
                report.render_html(
                    candidates, tiers(candidates), workload, eff,
                    self._catalog.unverified(),
                    title=f"{model.name} 서버 스펙 산정 리포트",
                ),
                encoding="utf-8",
            )
        return f"저장됨 · {path.name}"

    def _ask_where(self, suggested: str, fmt: str) -> str | None:
        """Native save dialog, with a sane fallback if it is unavailable."""
        import webview

        if self.window is None:
            return str(Path.home() / suggested)
        types = ("CSV (*.csv)",) if fmt == "csv" else ("HTML (*.html)",)
        chosen = self.window.create_file_dialog(
            webview.SAVE_DIALOG, save_filename=suggested, file_types=types
        )
        if not chosen:
            return None
        return chosen if isinstance(chosen, str) else chosen[0]


def run(catalog: Catalog | None = None, debug: bool = False) -> int:
    """Open the window. Blocks until the user closes it."""
    try:
        import webview
    except ImportError:
        raise SystemExit(
            "데스크톱 창을 띄우려면 pywebview가 필요하다.\n"
            "  pip install pywebview\n"
            "서버 방식으로 쓰려면: svrspec gui"
        )

    api = Api(catalog)
    window = webview.create_window(
        WINDOW_TITLE,
        html=app_html("desktop"),
        js_api=api,
        width=DEFAULT_SIZE[0],
        height=DEFAULT_SIZE[1],
        min_size=MIN_SIZE,
        text_select=True,
    )
    api.window = window
    webview.start(debug=debug)
    return 0


def main() -> int:
    """Entry point for the packaged executable."""
    import sys

    if "--version" in sys.argv:
        from . import __version__

        print(f"svrspec {__version__}")
        return 0
    return run(debug="--debug" in sys.argv)


if __name__ == "__main__":
    raise SystemExit(main())
