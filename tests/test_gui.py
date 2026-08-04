"""GUI: payload shapes, input clamping, and the served page.

The server is started for real on an ephemeral port rather than mocked, because
the thing worth testing is that a browser can actually get a sizing out of it.
"""

import importlib.util
import json
import math
import re
import sys
import threading
import types
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from svrspec.catalog import Catalog
from svrspec.gui import (
    DEFAULTS,
    LIMITS,
    _Handler,
    _params,
    _workload,
    app_html,
    catalog_payload,
    size_payload,
)

DATA = Path(__file__).parent / "data"
SERVER_HTML = app_html("server")
DESKTOP_HTML = app_html("desktop")

BASE_REQUEST = {
    "model": "test-3b",
    "quant": "Q4_K_M",
    "alarms_per_day": 150,
    "storm_size": 40,
    "storm_window_s": 30,
    "storms_per_day": 2,
    "slots": 2,
    "sla_seconds": 30,
    "storm_drain_min": 5,
    "system_tokens": 300,
    "fewshot_tokens": 400,
    "alarm_tokens": 250,
    "output_tokens": 250,
    "prompt_cache": True,
    "sockets": 1,
    "dpc": 1,
}


# --------------------------------------------------------------------------
# Input handling
# --------------------------------------------------------------------------


def test_params_clamps_out_of_range_values():
    """A browser can send anything; the server must not trust it."""
    p = _params({"alarms_per_day": 10**9, "slots": -5, "dpc": 7})
    assert p["alarms_per_day"] == LIMITS["alarms_per_day"][1]
    assert p["slots"] == LIMITS["slots"][0]
    assert p["dpc"] == LIMITS["dpc"][1]


def test_params_falls_back_on_garbage():
    p = _params({"alarms_per_day": "많이", "slots": None, "sla_seconds": {}})
    assert p["alarms_per_day"] == DEFAULTS["alarms_per_day"]
    assert p["slots"] == DEFAULTS["slots"]
    assert p["sla_seconds"] == DEFAULTS["sla_seconds"]


def test_params_parses_string_flags_instead_of_using_truthiness():
    p = _params({"prompt_cache": "false", "only_pass": "0"})
    assert p["prompt_cache"] is False
    assert p["only_pass"] is False


def test_params_defaults_are_all_inside_their_limits():
    for key, (low, high) in LIMITS.items():
        assert low <= DEFAULTS[key] <= high, key


def test_workload_reflects_the_request():
    w = _workload(_params({**BASE_REQUEST, "slots": 4, "storm_drain_min": 3}))
    assert w.slots == 4
    assert w.storm_drain_sla_s == 180.0
    assert w.tokens.prefill_tokens == 950
    assert w.tokens.billed_prefill_tokens == 250  # prefix cached


def test_prompt_cache_off_bills_the_whole_prompt():
    w = _workload(_params({**BASE_REQUEST, "prompt_cache": False}))
    assert w.tokens.billed_prefill_tokens == 950


# --------------------------------------------------------------------------
# Payloads
# --------------------------------------------------------------------------


@pytest.fixture
def cat() -> Catalog:
    return Catalog(DATA)


def test_catalog_payload_is_json_serialisable_and_sorted(cat):
    payload = catalog_payload(cat)
    json.dumps(payload)  # must not raise
    sizes = [m["params_b"] for m in payload["models"]]
    assert sizes == sorted(sizes)
    assert payload["counts"]["cpus"] == len(cat.cpus)
    assert all("kv_kib" in m for m in payload["models"])


def test_size_payload_shape(cat):
    d = size_payload(cat, _params(BASE_REQUEST))
    json.dumps(d)
    assert set(d["tiers"]) == {"minimum", "recommended", "comfortable"}
    assert d["total"] == len(cat.cpus)
    assert d["candidates"]
    row = d["candidates"][0]
    for key in ("model", "cores", "bandwidth", "bound", "prefill", "decode",
                "p95_steady", "storm_min", "ram_gb", "verdict", "uncertainty"):
        assert key in row
    # Only the coefficients this sizing actually used.
    kinds = {c["kind"] for c in d["coefficients"]}
    assert kinds <= {"eta_bw", "eta_compute", "per_core_bw_gbs", "dual_socket_efficiency"}
    assert all(c["label"] for c in d["coefficients"])


def test_only_pass_filters_the_table_but_not_the_totals(cat):
    everything = size_payload(cat, _params(BASE_REQUEST))
    filtered = size_payload(cat, _params({**BASE_REQUEST, "only_pass": True}))
    assert filtered["total"] == everything["total"]
    assert len(filtered["candidates"]) <= len(everything["candidates"])
    assert all(r["verdict"] != "fail" for r in filtered["candidates"])


def test_tighter_sla_moves_the_recommendation(cat):
    loose = size_payload(cat, _params({**BASE_REQUEST, "sla_seconds": 120}))
    tight = size_payload(cat, _params({**BASE_REQUEST, "sla_seconds": 2}))
    assert loose["passing"] >= tight["passing"]


def test_unknown_model_raises_for_the_handler_to_report(cat):
    from svrspec.catalog import CatalogError

    with pytest.raises(CatalogError):
        size_payload(cat, _params({**BASE_REQUEST, "model": "nope"}))


# --------------------------------------------------------------------------
# The page
# --------------------------------------------------------------------------


SVG_NAMESPACE = "http://www.w3.org/2000/svg"


def test_page_is_self_contained():
    """No CDN, no external font, no remote image -- it has to work air-gapped.

    The SVG namespace URI is exempt: it is an XML identifier that browsers match
    as a string and never dereference. Every other absolute URL in the page
    would be a request, which an air-gapped install cannot make.
    """
    body = SERVER_HTML.replace(SVG_NAMESPACE, "")
    for forbidden in ("http://", "https://", "<script src", "@import", "//unpkg"):
        assert forbidden not in body, forbidden


def test_every_control_has_a_label():
    ids = set(re.findall(r'<(?:input|select)[^>]*\bid="([^"]+)"', SERVER_HTML))
    labelled = set(re.findall(r'<label[^>]*\bfor="([^"]+)"', SERVER_HTML))
    assert ids and not (ids - labelled)


def test_page_supports_both_themes_and_reduced_motion():
    assert "prefers-color-scheme:dark" in SERVER_HTML
    assert 'data-theme="dark"' in SERVER_HTML
    assert 'data-theme="light"' in SERVER_HTML
    assert "prefers-reduced-motion" in SERVER_HTML


def test_page_uses_tokens_not_hardcoded_colours():
    """Colours live in the token block; component CSS references them."""
    body = SERVER_HTML.split("body{min-height:100vh}")[1].split("</style>")[0]
    assert not re.findall(r"#[0-9a-fA-F]{3,6}", body)


def test_results_region_is_announced():
    assert 'aria-live="polite"' in SERVER_HTML


# --------------------------------------------------------------------------
# Live server
# --------------------------------------------------------------------------


@pytest.fixture
def server():
    handler = type("H", (_Handler,), {"catalog": Catalog(DATA)})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def _get(url: str) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def _post(url: str, payload: dict) -> tuple[int, dict]:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_serves_the_app(server):
    status, body = _get(server + "/")
    assert status == 200
    assert body.startswith(b"<!doctype html>")


def test_serves_the_catalog(server):
    status, body = _get(server + "/api/catalog")
    assert status == 200
    assert json.loads(body)["models"]


def test_sizing_round_trip(server):
    status, d = _post(server + "/api/size", BASE_REQUEST)
    assert status == 200
    assert d["candidates"]
    assert d["model"]["id"] == "test-3b"


def test_unknown_route_is_404(server):
    status, _ = _get(server + "/../etc/passwd")
    assert status == 404


def test_bad_model_returns_an_error_not_a_crash(server):
    status, d = _post(server + "/api/size", {**BASE_REQUEST, "model": "nope"})
    assert status == 400
    assert "unknown model id" in d["error"]


def test_report_download_carries_a_filename(server):
    query = "&".join(
        f"{k}={1 if v is True else 0 if v is False else v}" for k, v in BASE_REQUEST.items()
    )
    with urllib.request.urlopen(f"{server}/api/report.html?{query}", timeout=20) as r:
        assert r.status == 200
        assert "attachment" in r.headers["Content-Disposition"]
        assert r.read().startswith(b"<!doctype html>")


def test_csv_download_has_the_expected_columns(server):
    query = "&".join(
        f"{k}={1 if v is True else 0 if v is False else v}" for k, v in BASE_REQUEST.items()
    )
    with urllib.request.urlopen(f"{server}/api/report.csv?{query}", timeout=20) as r:
        header = r.read().decode("utf-8-sig").splitlines()[0]
    assert "p95_steady_s" in header and "verdict" in header


# --------------------------------------------------------------------------
# Desktop transport
# --------------------------------------------------------------------------


def test_desktop_page_uses_the_bridge_not_a_socket():
    """The packaged app must not depend on anything listening on a port."""
    assert "data-mode='desktop'" in DESKTOP_HTML
    assert "window.pywebview.api" in DESKTOP_HTML
    # Both transports live in the page; the mode attribute picks one at runtime.
    assert 'data-mode=\'server\'' in SERVER_HTML


def test_a_window_whose_bridge_never_loads_rescues_itself():
    """A dead bridge must cost a second, not the whole application.

    Every cause is outside this program's reach -- a missing WebView2 runtime,
    security software that strips the injected script, an install from before
    the packaging fix -- and they all look identical: a window that draws and
    does nothing. Telling the operator to go run `svrspec gui` themselves is a
    handoff, not a fix. The same page over loopback needs no bridge at all.
    """
    import urllib.request

    from svrspec import desktop as dt

    class FakeWindow:
        url = None

        def load_url(self, u):
            self.url = u

    monkey = pytest.MonkeyPatch()
    monkey.setattr(dt, "BRIDGE_GRACE_S", 0.05)
    try:
        api = dt.Api(Catalog(DATA))
        api._window = FakeWindow()
        dt._rescue_if_dead(api)
        assert api._window.url and api._window.url.startswith("http://127.0.0.1:")

        # The rescued page must come up in server mode. Reloading the window
        # onto a page that still waits for the bridge would rescue nothing.
        page = urllib.request.urlopen(api._window.url, timeout=10).read().decode()
        assert "data-mode='server'" in page
        assert json.loads(
            urllib.request.urlopen(api._window.url + "api/catalog", timeout=10).read()
        )["models"]

        # And a window that works is left alone.
        ok = dt.Api(Catalog(DATA))
        ok._window = FakeWindow()
        assert ok.bridge_ok() == "ok"
        dt._rescue_if_dead(ok)
        assert ok._window.url is None
    finally:
        monkey.undo()


def test_nothing_but_methods_is_reachable_from_the_bridge_object():
    """The Api must be a leaf as far as pywebview's bridge walk is concerned.

    To build `window.pywebview.api`, pywebview walks the js_api object with
    `dir()` + `getattr()` and recurses into every public attribute that is not a
    method. A public `self.window` therefore aimed that walk at the pywebview
    Window, then the WinForms form, then the WebView2 COM objects -- .NET
    property reads off the UI thread, every one of them raising, and
    `Rectangle.Empty.Empty.Empty…` recursing to the interpreter's limit. That
    jams the UI thread while the bridge is still being injected: measured on the
    shipped build, the bridge object appeared at 2.2s, at 5.4s, or never, where
    making the attribute private gave 1.4s and an empty error log. "Never" is
    what the operator sees as the app announcing it could not start.

    Walking the object here is the check, because the failure is not in any
    single line -- it is in one attribute being public. Assigning any non-method
    to `self` reintroduces it, and this notices.
    """
    import inspect

    from svrspec.desktop import Api

    api = Api(Catalog(DATA))
    api._window = object()  # what run() parks there; must stay invisible

    public = [n for n in dir(api) if not n.startswith("_")]
    assert public, "the page has to be able to call something"
    non_methods = [n for n in public if not inspect.ismethod(getattr(api, n))]
    assert non_methods == [], (
        f"pywebview will recurse into {non_methods} while building the bridge; "
        "give them leading underscores"
    )


def test_the_page_tells_python_the_bridge_answered():
    """The rescue only cancels if the page reports in, on both paths.

    The `pywebviewready` event may already have fired before this script
    attaches its listener -- that is the whole reason the polling path exists --
    so a signal sent only from the event handler would leave a perfectly
    healthy window getting rescued out from under the operator.
    """
    assert "bridgeSignal" in DESKTOP_HTML
    assert "window.pywebview.api.bridge_ok()" in DESKTOP_HTML
    # Fired from settle() (covers the event path) and from the early-return
    # taken when the bridge was already up (covers the polling path).
    assert DESKTOP_HTML.count("bridgeSignal()") >= 2
    # The dead-bridge message must describe what the app is doing, not hand the
    # operator a command to type.
    assert "자동 전환" in DESKTOP_HTML


def test_desktop_api_answers_without_a_server(monkeypatch):
    from svrspec.desktop import Api

    api = Api(Catalog(DATA))
    assert api.catalog()["models"]

    d = api.size(BASE_REQUEST)
    assert d["candidates"]
    assert d["model"]["id"] == "test-3b"

    # Accepts a JSON string too: some webview bridges marshal objects as text.
    assert api.size(json.dumps(BASE_REQUEST))["candidates"]


def test_desktop_api_reports_bad_input_instead_of_raising():
    from svrspec.desktop import Api

    api = Api(Catalog(DATA))
    assert "error" in api.size({**BASE_REQUEST, "model": "nope"})
    assert "error" in api.size("not json at all")
    assert "error" in api.size([1, 2, 3])


def test_desktop_save_report_writes_both_formats(tmp_path, monkeypatch):
    from svrspec.desktop import Api

    api = Api(Catalog(DATA))
    target = {}

    def fake_dialog(suggested, fmt):
        path = tmp_path / suggested
        target[fmt] = path
        return str(path)

    monkeypatch.setattr(api, "_ask_where", fake_dialog)

    assert "저장됨" in api.save_report(BASE_REQUEST, "html")
    assert "저장됨" in api.save_report(BASE_REQUEST, "csv")
    assert target["html"].read_text(encoding="utf-8").startswith("<!doctype html>")
    assert "verdict" in target["csv"].read_text(encoding="utf-8-sig")


def test_desktop_save_report_cancelled_is_silent(monkeypatch):
    from svrspec.desktop import Api

    api = Api(Catalog(DATA))
    monkeypatch.setattr(api, "_ask_where", lambda suggested, fmt: None)
    assert api.save_report(BASE_REQUEST, "html") == ""


# --------------------------------------------------------------------------
# Token delivery timeline + task-manager resource view
# --------------------------------------------------------------------------


def test_resource_payload_timeline_adds_up(cat):
    from svrspec.gui import resource_payload

    d = resource_payload(cat, _params(BASE_REQUEST), "test-amx-8ch")
    tl = d["timeline"]
    assert [s["name"] for s in tl["stages"]] == ["프롬프트 처리", "토큰 생성", "전송"]
    assert abs(sum(s["seconds"] for s in tl["stages"]) - tl["total_s"]) < 0.01
    # Generation carries the output tokens; prefill carries the billed prompt.
    assert tl["stages"][0]["tokens"] == 250   # prefix cached, so alarm text only
    assert tl["stages"][1]["tokens"] == 250
    assert tl["output_tokens"] == 250
    assert tl["decode_tps"] > 0 and tl["prefill_tps"] > 0


def test_timeline_names_the_binding_bottleneck(cat):
    from svrspec.gui import resource_payload

    d = resource_payload(cat, _params(BASE_REQUEST), "test-amx-8ch")
    assert "바운드" in d["timeline"]["stages"][0]["note"]
    assert "바운드" in d["timeline"]["stages"][1]["note"]


def test_faster_hardware_shortens_the_timeline(cat):
    from svrspec.gui import resource_payload

    slow = resource_payload(cat, _params(BASE_REQUEST), "test-desktop-2ch")
    fast = resource_payload(cat, _params(BASE_REQUEST), "test-avx512-8ch")
    assert fast["timeline"]["total_s"] < slow["timeline"]["total_s"]


def test_resource_rows_cover_the_requested_volumes(cat):
    from svrspec.gui import resource_payload

    d = resource_payload(cat, _params(BASE_REQUEST), "test-amx-8ch", (100, 200, 300))
    assert [r["alarms"] for r in d["rows"]] == [100, 200, 300]


def test_more_alarms_means_more_cpu_time_on_the_same_hardware(cat):
    """The point of the task-manager view: same box, rising load."""
    from svrspec.gui import resource_payload

    d = resource_payload(cat, _params(BASE_REQUEST), "test-amx-8ch", (100, 200, 400))
    cpu = [r["cpu_pct"] for r in d["rows"]]
    work = [r["work_minutes"] for r in d["rows"]]
    assert cpu == sorted(cpu)
    assert work == sorted(work)
    assert work[-1] > work[0]
    # RAM is a function of the model and slot count, not of alarm volume.
    assert len({r["ram_installed_gb"] for r in d["rows"]}) == 1


def test_resource_percentages_stay_in_range(cat):
    from svrspec.gui import resource_payload

    d = resource_payload(cat, _params({**BASE_REQUEST, "alarms_per_day": 5000}),
                         "test-desktop-2ch", (1000, 5000))
    for r in d["rows"]:
        assert 0 <= r["cpu_pct"] <= 100
        assert 0 <= r["ram_pct"] <= 100
        assert r["bandwidth_avg_gbs"] <= r["bandwidth_gbs"] + 0.1


def test_ram_breakdown_sums_to_the_subtotal(cat):
    from svrspec.gui import resource_payload

    b = resource_payload(cat, _params(BASE_REQUEST), "test-amx-8ch")["ram_breakdown"]
    parts = b["weights_gb"] + b["kv_gb"] + b["compute_gb"] + b["os_gb"]
    assert abs(parts - b["subtotal_gb"]) < 0.05
    assert b["installed_gb"] >= b["subtotal_gb"]


def test_resources_route_round_trip(server):
    status, d = _post(server + "/api/resources",
                      {**BASE_REQUEST, "cpu": "test-amx-8ch", "volumes": [100, 250]})
    assert status == 200
    assert [r["alarms"] for r in d["rows"]] == [100, 250]
    assert d["hardware"]["id"] == "test-amx-8ch"


def test_resources_route_rejects_an_unknown_cpu(server):
    status, d = _post(server + "/api/resources", {**BASE_REQUEST, "cpu": "nope"})
    assert status == 400
    assert "unknown cpu id" in d["error"]


def test_resources_route_falls_back_to_default_volumes(server):
    status, d = _post(server + "/api/resources",
                      {**BASE_REQUEST, "cpu": "test-amx-8ch", "volumes": []})
    assert status == 200
    assert [r["alarms"] for r in d["rows"]] == [100, 200, 300]


def test_desktop_api_serves_resources():
    from svrspec.desktop import Api

    api = Api(Catalog(DATA))
    d = api.resources({**BASE_REQUEST, "cpu": "test-amx-8ch"})
    assert d["rows"] and d["timeline"]["total_s"] > 0
    assert "error" in api.resources({**BASE_REQUEST, "cpu": "nope"})


def test_busy_fraction_is_reported_and_bounded():
    from dataclasses import replace as dc_replace

    from svrspec.simulate import simulate
    from svrspec.types import Workload

    w = Workload(alarms_per_day=150, slots=2, teams_rtt_ms=(0.0, 0.0))
    idle, _ = simulate(w, prefill_tps=400.0, decode_by_active={1: 200.0, 2: 320.0})
    loaded, _ = simulate(dc_replace(w, alarms_per_day=4000), prefill_tps=400.0,
                         decode_by_active={1: 200.0, 2: 320.0})
    assert 0.0 <= idle.busy_fraction <= 1.0
    assert loaded.busy_fraction > idle.busy_fraction
    # Busy fraction counts wall clock with any request in flight, so it is never
    # below the per-slot utilisation.
    assert loaded.busy_fraction >= loaded.slot_utilisation - 1e-9


# --------------------------------------------------------------------------
# Slot solving, time series, and the task-manager payload
# --------------------------------------------------------------------------


def test_required_slots_stops_when_concurrency_stops_helping(catalog, eff):
    """Aggregate throughput is capped by compute or bandwidth, not by slots.

    Past that cap another slot changes nothing, and reporting the search ceiling
    would tell the operator to buy concurrency that cannot help.
    """
    from svrspec.sizing import MAX_SEARCHED_SLOTS, required_slots
    from svrspec.types import Workload

    cpu = catalog.cpu("test-desktop-2ch")
    model = catalog.model("test-8b-gqa")
    w = Workload(alarms_per_day=150, slots=1, sla_seconds=1.0, storm_drain_sla_s=1.0)
    slots, candidate = required_slots(
        model, catalog.quant("Q4_K_M"), cpu, catalog.memory_for(cpu), eff, w
    )
    assert candidate.verdict == "fail"          # an impossible SLA cannot pass
    assert slots < MAX_SEARCHED_SLOTS           # ...but it must not claim 32 helps


def test_required_slots_returns_the_first_passing_count(catalog, eff):
    from svrspec.sizing import required_slots
    from svrspec.types import Workload

    cpu = catalog.cpu("test-amx-8ch")
    w = Workload(alarms_per_day=150, slots=1, sla_seconds=60.0, storm_drain_sla_s=3600.0)
    slots, candidate = required_slots(
        catalog.model("test-3b"), catalog.quant("Q4_K_M"), cpu,
        catalog.memory_for(cpu), eff, w,
    )
    assert candidate.verdict in ("pass", "marginal")
    assert slots >= 1


def test_resource_rows_report_the_slots_they_solved_for(cat):
    from svrspec.gui import resource_payload

    d = resource_payload(cat, _params({**BASE_REQUEST, "slots": 1}),
                         "test-amx-8ch", (100, 3000))
    for r in d["rows"]:
        assert r["slots"] >= 1
        assert r["slots_configured"] == 1
        # KV cache follows the slot count, which is how RAM moves with load.
        assert r["kv_gb"] > 0


def test_memory_moves_with_the_slot_count(catalog, eff):
    """The complaint this fixes: RAM that never changes is useless for planning."""
    from svrspec.memory import size_memory
    from svrspec.types import TokenProfile

    model, quant = catalog.model("test-8b-gqa"), catalog.quant("Q4_K_M")
    one = size_memory(model, quant, TokenProfile(), slots=1)
    eight = size_memory(model, quant, TokenProfile(), slots=8)
    assert eight.kv_cache_gb == 8 * one.kv_cache_gb
    assert eight.subtotal_gb > one.subtotal_gb


def test_series_has_a_bucket_per_quarter_hour(cat):
    from svrspec.gui import resource_payload

    s = resource_payload(cat, _params(BASE_REQUEST), "test-amx-8ch")["series"]
    assert len(s["cpu"]) == 96
    assert len(s["queue"]) == 96 and len(s["active"]) == 96
    assert s["bucket_s"] == 900.0
    assert all(0.0 <= v <= 1.0 for v in s["cpu"])


def test_series_shows_the_storms_as_spikes(cat):
    """A thirty-second burst must survive bucketing, or the graph lies."""
    from svrspec.gui import resource_payload

    d = resource_payload(cat, _params({**BASE_REQUEST, "storms_per_day": 2}),
                         "test-desktop-2ch")
    busy = [v for v in d["series"]["cpu"] if v > 0]
    assert busy, "the day cannot be entirely idle"
    assert max(d["series"]["queue"]) > 0     # queueing happened somewhere
    assert max(d["series"]["active"]) >= 1


def test_the_series_conserves_the_days_work(cat):
    """Every busy second must land in exactly one bucket's worth of time.

    Replaces the old `_buckets` unit test: the folding is now done once, by
    `timeline.build_timeline`, and the GUI only reshapes it. What still has to
    hold at this layer is that reshaping loses nothing.
    """
    from svrspec.gui import resource_payload

    d = resource_payload(cat, _params(BASE_REQUEST), "test-desktop-2ch")
    s = d["series"]
    charged = sum(v * s["bucket_s"] for v in s["cpu"])
    # The series is rounded for the wire (4 dp on a fraction), which is worth
    # at most 0.09 s per bucket.
    assert abs(charged - d["bottleneck"]["busy_seconds"]) < 96 * 0.1


def test_live_tiles_carry_what_the_task_manager_shows(cat):
    from svrspec.gui import resource_payload

    live = resource_payload(cat, _params(BASE_REQUEST), "test-amx-8ch")["live"]
    for key in ("cpu_pct", "ram_pct", "ram_used_gb", "ram_installed_gb",
                "bandwidth_avg_gbs", "bandwidth_gbs", "max_queue", "peak_active",
                "alarms", "completed", "verdict"):
        assert key in live
    assert 0 <= live["cpu_pct"] <= 100
    assert live["bandwidth_avg_gbs"] <= live["bandwidth_gbs"] + 0.1


def test_hardware_block_carries_the_identity_fields(cat):
    from svrspec.gui import resource_payload

    hw = resource_payload(cat, _params(BASE_REQUEST), "test-amx-8ch")["hardware"]
    for key in ("label", "cores", "threads", "ghz", "isa", "memory",
                "bandwidth_gbs", "l3_mb", "ram_max_gb", "slots"):
        assert key in hw


# --------------------------------------------------------------------------
# Four resources, four series
# --------------------------------------------------------------------------


def _shape(values: list[float]) -> list[float]:
    """Normalised to its own peak. Two rescaled copies of one scalar match here."""
    top = max(values) or 1.0
    return [round(v / top, 3) for v in values]


def test_the_four_resource_series_are_actually_four_series(cat):
    """The regression this whole view exists to prevent.

    The previous version put `busy_fraction` into the CPU line, copied it into
    the bandwidth line, and drew memory as a constant. Three graphs, one number.
    Prefill saturates the vector units while DRAM idles and decode does the
    reverse, so bandwidth and compute must not even have the same *shape*.
    """
    from itertools import combinations

    from svrspec.gui import resource_payload

    s = resource_payload(cat, _params(BASE_REQUEST), "test-desktop-2ch")["series"]
    names = ("cpu_pct", "bandwidth_pct", "compute_pct", "ram_pct")
    for a, b in combinations(names, 2):
        assert s[a] != s[b], f"{a} and {b} are the same numbers"
        assert _shape(s[a]) != _shape(s[b]), f"{a} and {b} are one series rescaled"

    # Peak and average are different questions too: averaging a storm away is
    # how a sizing tool ends up calling an overloaded box idle.
    assert s["bandwidth_peak_pct"] != s["bandwidth_pct"]
    assert max(s["bandwidth_peak_pct"]) > max(s["bandwidth_pct"])
    assert max(s["compute_peak_pct"]) > max(s["compute_pct"])


def test_the_memory_series_moves_over_the_day(cat):
    """"RAM never changes" was half the complaint. KV residency is not constant."""
    from svrspec.gui import resource_payload

    d = resource_payload(cat, _params(BASE_REQUEST), "test-desktop-2ch")
    used = d["series"]["ram_used_gb"]
    assert len(set(used)) > 1
    assert max(used) > min(used)
    assert max(d["series"]["kv_used_gb"]) > 0

    # ...and the moving line must never be mistaken for the order quantity:
    # llama.cpp reserves the whole context per slot whether or not it is
    # touched, so the reserved figure is the larger one and it is what the
    # server has to be given.
    assert d["ceilings"]["allocated_gb"] >= max(used) - 0.01
    assert d["live"]["ram_reserved_gb"] >= d["live"]["ram_live_peak_gb"]
    assert d["live"]["kv_reserved_gb"] >= d["live"]["kv_live_peak_gb"]
    assert d["ceilings"]["installed_gb"] >= d["ceilings"]["allocated_gb"]


def test_every_series_has_a_bucket_per_quarter_hour(cat):
    from svrspec.gui import resource_payload

    s = resource_payload(cat, _params(BASE_REQUEST), "test-amx-8ch")["series"]
    for key, value in s.items():
        if key == "bucket_s":
            continue
        assert len(value) == 96, key


def test_the_bottleneck_line_names_a_resource_and_a_phase_split(cat):
    """The sentence that decides cores versus memory channels."""
    from svrspec.gui import resource_payload

    b = resource_payload(cat, _params(BASE_REQUEST), "test-amx-8ch")["bottleneck"]
    assert b["resource"] in ("bandwidth", "compute", "none")
    assert b["label"] in ("메모리 대역폭", "연산", "없음")
    assert 0 <= b["prefill_share"] <= 100
    assert abs(b["prefill_share"] + b["decode_share"] - 100) < 1.5
    assert b["sentence"] and b["advice"]
    # An instantaneous ceiling is not overload; a queue is. The payload has to
    # carry that distinction or the UI will paint every healthy box red.
    assert b["overloaded"] == (b["peak_queue"] > 0)
    assert "큐" in b["overload"]


def test_the_task_manager_and_the_row_table_agree_on_their_denominators(cat):
    """One set of numbers. The graph and the table must not disagree."""
    from svrspec.gui import resource_payload

    p = _params({**BASE_REQUEST, "alarms_per_day": 2000})
    row = resource_payload(cat, p, "test-amx-8ch", (2000,))["rows"][0]
    assert row["bandwidth_avg_gbs"] <= row["bandwidth_gbs"] + 0.1
    assert row["bound"] in ("bandwidth", "compute", "none")
    # The table's resource columns are per-resource too, not one busy fraction
    # reprinted three times.
    assert row["cpu_pct"] != row["bandwidth_pct"] != row["compute_pct"]
    assert row["bandwidth_peak_pct"] > row["bandwidth_pct"]
    assert row["ram_live_peak_gb"] <= row["ram_used_gb"] + 0.01


# --------------------------------------------------------------------------
# Where it breaks
# --------------------------------------------------------------------------


def test_capacity_payload_carries_the_knee_and_the_limiter(cat):
    from svrspec.gui import capacity_payload

    d = capacity_payload(cat, _params(BASE_REQUEST), "test-desktop-2ch")
    json.dumps(d, allow_nan=False)      # Infinity would break JSON.parse

    for key in ("knee", "breaks_at", "limiter", "headroom", "weakest_axis", "axes"):
        assert key in d
    assert {a["axis"] for a in d["axes"]} == {"alarms", "storm", "prompt", "output"}
    assert d["weakest_axis"] in {a["axis"] for a in d["axes"]} or d["weakest_axis"] is None
    assert sum(1 for a in d["axes"] if a["weakest"]) <= 1

    for a in d["axes"]:
        assert a["label"] and a["unit"]
        assert a["limiter_label"]
        assert a["points"], "the ramp itself is the evidence; it must be kept"
        assert [pt["value"] for pt in a["points"]] == sorted(
            pt["value"] for pt in a["points"]
        )
        if a["knee"]:
            assert a["knee"]["ok"]
        if a["breaks_at"]:
            assert not a["breaks_at"]["ok"]
            assert a["breaks_at"]["value"] > (a["knee"]["value"] if a["knee"] else -1)


def test_capacity_reports_no_limit_as_null_not_infinity(cat):
    """A storm size of zero makes headroom infinite; JSON cannot carry that."""
    from svrspec.gui import capacity_payload

    d = capacity_payload(
        cat, _params({**BASE_REQUEST, "storm_size": 0}), "test-desktop-2ch", ("storm",)
    )
    json.dumps(d, allow_nan=False)
    assert d["axes"][0]["headroom"] is None


def test_capacity_axis_subset_is_honoured_and_sanitised(cat):
    from svrspec.gui import _axes, capacity_payload

    assert _axes({"axes": ["storm", "nope"]}) == ("storm",)
    assert _axes({"axes": []}) is None
    assert _axes({"axes": "storm"}) is None
    assert _axes({}) is None

    d = capacity_payload(cat, _params(BASE_REQUEST), "test-desktop-2ch", ("storm",))
    assert [a["axis"] for a in d["axes"]] == ["storm"]


def test_capacity_route_round_trip(server):
    status, d = _post(server + "/api/capacity",
                      {**BASE_REQUEST, "cpu": "test-desktop-2ch", "axes": ["storm"]})
    assert status == 200
    assert [a["axis"] for a in d["axes"]] == ["storm"]
    assert "knee" in d and "limiter" in d


def test_capacity_route_rejects_an_unknown_cpu(server):
    status, d = _post(server + "/api/capacity",
                      {**BASE_REQUEST, "cpu": "nope", "axes": ["storm"]})
    assert status == 400
    assert "unknown cpu id" in d["error"]


def test_the_live_recompute_never_runs_a_capacity_search(cat, monkeypatch):
    """Performance guard.

    One axis is a bracketing search over whole simulated days; four of them run
    for seconds to tens of seconds. Hanging that off the keystroke-driven
    recompute would turn the simulator into a spinner, so the two payloads that
    *do* run on every input change must not touch it.
    """
    from svrspec import capacity as capacity_module
    from svrspec.gui import resource_payload

    def explode(*args, **kwargs):
        raise AssertionError("a capacity search ran on the live recompute path")

    monkeypatch.setattr(capacity_module, "find_knee", explode)
    monkeypatch.setattr(capacity_module, "sweep_axes", explode)

    assert size_payload(cat, _params(BASE_REQUEST))["candidates"]
    assert resource_payload(cat, _params(BASE_REQUEST), "test-amx-8ch")["rows"]


def test_the_page_asks_for_capacity_only_from_the_button():
    """The client half of the same guard.

    Two mentions each: the transport plus its single caller, and the handler
    plus the click that wires it. A third would mean something started calling
    it automatically.
    """
    assert SERVER_HTML.count("askCapacity") == 2
    assert SERVER_HTML.count("runCapacity") == 2
    assert 'addEventListener("click", runCapacity)' in SERVER_HTML


def test_the_desktop_page_degrades_when_the_bridge_has_no_capacity_call():
    """The packaged app's bridge is a fixed surface; an older one must not throw."""
    assert "window.pywebview.api.capacity" in DESKTOP_HTML
    assert "데스크톱 빌드에는 과부하 분석이 연결되어 있지 않습니다" in DESKTOP_HTML


def test_the_task_manager_offers_a_tile_per_resource():
    for name in ("CPU", "메모리", "대역폭", "연산", "동시 처리", "큐"):
        assert f'"{name}"' in SERVER_HTML, name
    # Peak and average are drawn as separate lines, not collapsed.
    assert "bandwidth_peak_pct" in SERVER_HTML and "bandwidth_pct" in SERVER_HTML
    assert "compute_peak_pct" in SERVER_HTML and "compute_pct" in SERVER_HTML


def test_two_columns_survive_the_smallest_window():
    """The rail must never jump above the results in the app window.

    The single-column fallback has to sit below the window's minimum width;
    when it sat above, nudging the window smaller broke the side-by-side layout,
    which is what "the columns do not match" was describing.
    """
    import re

    from svrspec.desktop import MIN_SIZE

    match = re.search(
        r"@media \(max-width:(\d+)px\)\{\s*main\{grid-template-columns:minmax",
        DESKTOP_HTML,
    )
    assert match, "the single-column breakpoint is missing"
    assert MIN_SIZE[0] > int(match.group(1))


def test_the_sticky_rail_offset_equals_the_header_height():
    """A hand-tuned offset is how the two columns drifted apart; both read one var."""
    assert "--header-h:60px" in SERVER_HTML
    assert "height:var(--header-h)" in SERVER_HTML
    assert "top:calc(var(--header-h) + var(--s5))" in SERVER_HTML


def test_the_rail_scrolls_on_its_own():
    """A long form must not stretch the results column or get clipped."""
    assert "max-height:calc(100vh - var(--header-h)" in SERVER_HTML
    assert "overflow-y:auto" in SERVER_HTML


def test_desktop_api_answers_a_capacity_request():
    """The desktop build must reach the overload search too.

    The page already degrades gracefully when the bridge lacks the method, but
    degrading is not the goal -- the native app is the primary way this ships
    on Windows, and "run it in server mode instead" is not an answer for an
    air-gapped operator who was given an installer.
    """
    from svrspec.desktop import Api

    api = Api(Catalog(DATA))
    request = {**BASE_REQUEST, "cpu": "test-amx-8ch", "axes": ["storm"]}

    payload = api.capacity(request)
    assert "error" not in payload
    assert payload["axes"], "an axis was requested and must come back"
    axis = payload["axes"][0]
    assert axis["axis"] == "storm"
    assert axis["points"]
    assert "limiter" in axis

    # Marshalled as a JSON string, like some webview bridges do.
    assert "error" not in api.capacity(json.dumps(request))


def test_desktop_capacity_reports_bad_input_instead_of_raising():
    from svrspec.desktop import Api

    api = Api(Catalog(DATA))
    assert "error" in api.capacity({**BASE_REQUEST, "cpu": "no-such-cpu"})
    assert "error" in api.capacity([1, 2, 3])


# --------------------------------------------------------------------------
# Virtual lab
# --------------------------------------------------------------------------
#
# `svrspec.lab` and `svrspec.bench` are landed by other owners in the same wave.
# Everything asserted below is a property of *this* layer -- that the payload
# carries the finding and its remedy, that it reports the bandwidth the engine
# computed instead of rounding the loss away, that a bench never runs on the
# live path, that no infinity reaches the wire. Those are worth holding before
# the engines arrive and worth re-running against the real physics afterwards,
# so the fixture prefers the real module and only stands in when there is none.
#
# The stand-ins are deliberately thin, and the one number that matters -- the
# bandwidth an under-populated board actually achieves -- comes from
# `perf.effective_bandwidth`, which already landed. Nothing here re-derives it.


def _stub_lab() -> types.ModuleType:
    from svrspec.memory import size_memory
    from svrspec.perf import Efficiency, effective_bandwidth, predict_throughput
    from svrspec.types import TokenProfile

    @dataclass(frozen=True)
    class VirtualMachine:
        name: str
        cpu_id: str
        sockets: int
        dimm_gb: int
        dimm_count: int
        model_id: str
        quant_id: str
        slots: int

    @dataclass(frozen=True)
    class Finding:
        level: str
        code: str
        message: str
        remedy: str = ""

    @dataclass(frozen=True)
    class Assembly:
        vm: VirtualMachine
        cpu: object
        memory: object
        model: object
        quant: object
        channels_total: int
        channels_populated: int
        dimms_per_channel: int
        ram_total_gb: int
        ram_used_gb: float
        bandwidth_gbs: float
        bandwidth_full_gbs: float
        prefill_tps: float
        decode_tps_single: float
        decode_tps_full: float
        uncertainty: float
        findings: list = field(default_factory=list)

        @property
        def ok(self) -> bool:
            return not any(f.level == "error" for f in self.findings)

    def _memory_at(cat, cpu, dpc, dimm_gb):
        usable = [
            m
            for m in cat.memory
            if m.ddr_gen == cpu.ddr_gen
            and m.dimms_per_channel == dpc
            and m.rated_mts <= cpu.max_ddr_mts
        ]
        exact = [m for m in usable if m.dimm_gb == dimm_gb]
        pool = exact or usable
        if not pool:
            return cat.memory_for(cpu, dpc)
        return max(pool, key=lambda m: m.effective_mts)

    def assemble(cat, vm, tokens=None):
        tokens = tokens or TokenProfile()
        cpu, model, quant = cat.cpu(vm.cpu_id), cat.model(vm.model_id), cat.quant(vm.quant_id)
        eff = Efficiency.from_catalog(cat.coefficients)
        sockets = max(1, vm.sockets)
        channels_total = cpu.mem_channels * sockets
        count = max(0, vm.dimm_count)
        dpc = max(1, math.ceil(count / channels_total)) if count else 1
        memory = _memory_at(cat, cpu, min(dpc, 2), vm.dimm_gb)
        populated = channels_total if dpc >= 2 else min(count, channels_total)
        per_socket = populated // sockets

        bw, _ = effective_bandwidth(
            cpu, memory, eff, sockets, channels_populated=per_socket
        )
        bw_full, _ = effective_bandwidth(cpu, memory, eff, sockets)
        now = predict_throughput(
            model, quant, cpu, memory, tokens, eff, slots=vm.slots,
            sockets=sockets, channels_populated=per_socket,
        )
        full = predict_throughput(
            model, quant, cpu, memory, tokens, eff, slots=vm.slots, sockets=sockets
        )
        ram = size_memory(model, quant, tokens, slots=vm.slots)
        total_gb = vm.dimm_gb * count

        findings = []
        if count < 1:
            findings.append(Finding("error", "no-dimms", "DIMM이 한 장도 없다.", "메모리를 장착해라."))
        elif populated < channels_total:
            loss = round((1 - bw / bw_full) * 100) if bw_full else 0
            findings.append(Finding(
                "warn", "channels-underfilled",
                f"{populated}/{channels_total} 채널만 장착 — 대역폭 "
                f"{bw_full / 1e9:.0f} → {bw / 1e9:.0f} GB/s ({loss}% 손실)",
                f"같은 {total_gb}GB를 {channels_total}장으로 나눠 꽂으면 전 채널이 찬다. "
                f"장당 {total_gb // channels_total}GB가 필요한데 카탈로그에는 없을 수 있다.",
            ))
        if count and total_gb < ram.subtotal_gb:
            findings.append(Finding(
                "error", "ram-too-small",
                f"모델이 {ram.subtotal_gb:.1f}GB를 요구하는데 {total_gb}GB만 장착했다.",
                "DIMM 용량이나 장수를 늘려라.",
            ))
        if total_gb > cpu.max_mem_gb * sockets:
            findings.append(Finding(
                "error", "ram-exceeds-cpu",
                f"이 CPU의 최대 장착량은 {cpu.max_mem_gb * sockets}GB다.",
                "장수를 줄여라.",
            ))
        if vm.sockets > cpu.sockets_max:
            findings.append(Finding(
                "error", "sockets-exceeded",
                f"이 부품은 최대 {cpu.sockets_max}소켓이다.", "소켓 수를 줄여라."
            ))
        if memory.effective_mts < memory.rated_mts:
            findings.append(Finding(
                "warn", "dpc-derate",
                f"2 DPC로 {memory.rated_mts} → {memory.effective_mts} MT/s로 떨어진다.",
                "채널당 1장으로 구성해라.",
            ))

        return Assembly(
            vm=vm, cpu=cpu, memory=memory, model=model, quant=quant,
            channels_total=channels_total, channels_populated=populated,
            dimms_per_channel=dpc, ram_total_gb=total_gb,
            ram_used_gb=ram.subtotal_gb,
            bandwidth_gbs=bw / 1e9, bandwidth_full_gbs=bw_full / 1e9,
            prefill_tps=now.prefill_tps, decode_tps_single=now.decode_tps_single,
            decode_tps_full=full.decode_tps_single, uncertainty=now.uncertainty,
            findings=findings,
        )

    def dimm_options(cat, cpu, sockets):
        total = cpu.mem_channels * sockets
        caps = sorted({m.dimm_gb for m in cat.memory if m.ddr_gen == cpu.ddr_gen})
        out = []
        for gb in caps:
            for count in range(1, total * 2 + 1):
                if gb * count > cpu.max_mem_gb * sockets:
                    continue
                dpc = math.ceil(count / total)
                out.append({
                    "dimm_gb": gb, "count": count, "ram_total": gb * count,
                    "channels_populated": total if dpc >= 2 else min(count, total),
                    "dpc": dpc,
                })
        return out

    def to_service(cat, asm, workload):  # pragma: no cover - unused by the GUI
        raise NotImplementedError

    mod = types.ModuleType("svrspec.lab")
    mod.VirtualMachine = VirtualMachine
    mod.Finding = Finding
    mod.Assembly = Assembly
    mod.assemble = assemble
    mod.dimm_options = dimm_options
    mod.to_service = to_service
    return mod


def _stub_loadgen() -> types.ModuleType:
    @dataclass(frozen=True)
    class LoadProfile:
        kind: str
        label: str
        span_s: float
        total_alarms: int
        params: dict
        notes: list

    @dataclass(frozen=True)
    class _Alarm:
        id: str
        at_s: float

    def build_load(kind, *, seed=20260730, **params):
        hours = params.get("hours", 24)
        span = float(hours) * 3600.0
        if kind == "ramp":
            lo, hi = params.get("start_rate", 100), params.get("end_rate", 2000)
            label = f"램프 {lo:,} → {hi:,}건/일"
        elif kind == "spike":
            span = 24 * 3600.0
            lo = hi = params.get("base_rate", 165)
            label = f"스파이크 {lo:,} → {params.get('peak_rate', 800):,}건/일"
        elif kind == "soak":
            lo = hi = params.get("rate", 300)
            label = f"소크 {lo:,}건/일 × {hours}시간"
        else:
            span = 24 * 3600.0
            lo = hi = params.get("count") or 359
            label = "실측 하루 재생"
        # Arrivals whose density follows the profile's rate, which is all the
        # frame folding downstream actually reads off them.
        total = max(1, int(round((lo + hi) / 2 * span / 86400.0)))
        alarms = []
        for i in range(total):
            share = (i + 0.5) / total
            if hi != lo:
                # Solve the linear-rate cumulative for this arrival's time.
                a, b = lo, hi - lo
                t = (-a + math.sqrt(a * a + 2 * b * share * (a + b / 2))) / b
            else:
                t = share
            alarms.append(_Alarm(id=f"a{i}", at_s=min(span * 0.999, max(0.0, t * span))))
        alarms.sort(key=lambda a: a.at_s)
        profile = LoadProfile(
            kind=kind, label=label, span_s=span, total_alarms=len(alarms),
            params=dict(params, seed=seed, span_s=span),
            notes=["시간대 분포는 가정이다."],
        )
        return alarms, profile

    mod = types.ModuleType("svrspec.loadgen")
    mod.KINDS = ("replay", "ramp", "spike", "soak")
    mod.LoadProfile = LoadProfile
    mod.build_load = build_load
    return mod


def _stub_bench() -> types.ModuleType:
    from svrspec.pipeline import RunStats

    @dataclass(frozen=True)
    class Frame:
        t_s: float
        queued: int
        active: int
        cpu_pct: float
        bw_gbs: float
        bw_pct: float
        compute_pct: float
        kv_gb: float
        ram_gb: float
        arrived: int
        delivered: int
        offered_rate: float
        p95_so_far_s: float

    @dataclass(frozen=True)
    class BenchResult:
        profile: object
        machine: dict
        stats: RunStats
        frames: list
        worst: list
        findings: list
        breach: dict | None
        notes: list

    def run_bench(cat, asm, alarms, profile, *, workload, frames=600,
                  queue_limit=None, worst_n=10):
        if frames < 1:
            raise ValueError("frames must be positive")
        span = profile.span_s or 1.0
        dt = span / frames
        counts = [0] * frames
        for a in alarms:
            counts[min(frames - 1, int(a.at_s / span * frames))] += 1

        # Crude but monotone: a queue that grows once arrivals outrun service is
        # the only behaviour the playback layer reads off these frames.
        per_frame = max(1.0, asm.decode_tps_single * dt / 250.0)
        rows, queue, delivered_all, worst_s, breach = [], 0.0, 0, 0.0, None
        for i, arrived in enumerate(counts):
            queue += arrived
            served = min(queue, per_frame)
            queue -= served
            delivered_all += int(served)
            p95 = round(queue / max(per_frame, 1e-9) * dt, 3)
            worst_s = max(worst_s, p95)
            active = int(min(workload.slots, math.ceil(served)))
            load = min(1.0, served / max(per_frame, 1e-9))
            rows.append(Frame(
                t_s=round(i * dt, 3), queued=int(queue), active=active,
                cpu_pct=round(100 * load, 1),
                bw_gbs=round(asm.bandwidth_gbs * load, 3),
                bw_pct=round(100 * load, 1), compute_pct=round(60 * load, 1),
                kv_gb=round(0.02 * active, 4),
                ram_gb=round(asm.ram_used_gb * (0.9 + 0.1 * load), 3),
                arrived=arrived, delivered=int(served),
                offered_rate=round(arrived / dt * 86400.0, 1),
                p95_so_far_s=worst_s,
            ))
            if breach is None and worst_s > workload.sla_seconds:
                breach = {"t_s": rows[-1].t_s, "offered_rate": rows[-1].offered_rate,
                          "p95_s": worst_s}

        stats = RunStats(
            received=len(alarms), delivered=delivered_all, dropped=0,
            p50_s=0.5, p95_s=worst_s, p99_s=worst_s, max_s=worst_s,
            p95_steady_s=worst_s, storm_drain_s=0.0, max_queue=int(max(
                (r.queued for r in rows), default=0)),
            mean_queue=0.0, busy_fraction=0.5, slot_utilisation=0.4,
            sla_met=breach is None, storm_sla_met=True,
            tokens_prefill=250 * len(alarms), tokens_generated=250 * delivered_all,
            wall_clock_s=0.0,
        )
        worst = [{"id": f"a{i}", "total_s": round(worst_s, 2), "slot": 0}
                 for i in range(min(worst_n, len(alarms)))]
        return BenchResult(
            profile=profile, machine={"cpu": asm.cpu.id, "ram_gb": asm.ram_total_gb},
            stats=stats, frames=rows, worst=worst, findings=list(asm.findings),
            breach=breach, notes=["가상시간으로 돌렸다 — 실제 모델을 실행하지 않았다."],
        )

    mod = types.ModuleType("svrspec.bench")
    mod.Frame = Frame
    mod.BenchResult = BenchResult
    mod.run_bench = run_bench
    return mod


def _stub_modelbench() -> types.ModuleType:
    """A stand-in for `svrspec.modelbench`, thin on purpose.

    It reproduces the *shape* of the contract and the two monotonic facts the
    screen is built to show -- batching trades per-sequence speed for total
    throughput, a longer context slows decode -- and nothing else. The physics
    belongs to the real engine; asserting it here would only test the stub.

    The one number it does not invent is `asm.decode_tps_single`, which already
    carries the channel population of the assembled board. That is what lets the
    GUI tests check that an under-populated build reaches this screen intact.
    """

    @dataclass(frozen=True)
    class ThroughputPoint:
        batch: int
        ctx_tokens: int
        prefill_tps: float
        decode_tps_single: float
        decode_tps_total: float
        prefill_bound: str
        decode_bound: str

    @dataclass(frozen=True)
    class ConcurrencyPoint:
        users: int
        ttft_s: float
        decode_tps_each: float
        response_s: float
        total_tps: float

    @dataclass(frozen=True)
    class ResourceSplit:
        phase: str
        bandwidth_pct: float
        compute_pct: float
        bound_by: str
        bytes_per_token: float
        flops_per_token: float

    @dataclass(frozen=True)
    class TrainingVerdict:
        kind: str
        feasible: bool
        memory_needed_gb: float
        memory_available_gb: float
        step_seconds: float | None
        epoch_hours: float | None
        reasons: list
        gpu_comparison: str

    @dataclass(frozen=True)
    class ModelBench:
        model_name: str
        quant_id: str
        hardware: str
        throughput: list
        concurrency: list
        resources: list
        training: list
        memory_gb: float
        uncertainty: float
        notes: list
        warnings: list

    def bench_model(asm, *, batches=(1, 2, 4, 8, 16, 32),
                    contexts=(512, 2048, 4096, 8192, 16384),
                    users=(1, 2, 4, 8, 16, 32, 64), output_tokens=256,
                    train_samples=10_000):
        base = max(0.01, float(asm.decode_tps_single))
        prefill = max(0.01, float(asm.prefill_tps))

        throughput = []
        for batch in batches:
            for ctx in contexts:
                single = base / (1.0 + ctx / 65_536.0) / (1.0 + (batch - 1) * 0.25)
                throughput.append(ThroughputPoint(
                    batch=batch, ctx_tokens=ctx, prefill_tps=prefill,
                    decode_tps_single=single, decode_tps_total=single * batch,
                    prefill_bound="compute",
                    decode_bound="bandwidth" if batch < 4 else "compute",
                ))

        concurrency = []
        for count in users:
            each = base / (1.0 + (count - 1) * 0.6)
            ttft = 250.0 * count / prefill
            concurrency.append(ConcurrencyPoint(
                users=count, ttft_s=ttft, decode_tps_each=each,
                response_s=ttft + output_tokens / each, total_tps=each * count,
            ))

        resources = [
            ResourceSplit("prefill", 11.0, 92.0, "compute", 1_180.0, 6.4e9),
            ResourceSplit("decode", 96.0, 18.0, "bandwidth", 2.1e9, 6.4e9),
        ]

        available = float(asm.ram_total_gb)
        weights_gb = float(asm.model.params_b) * 2.0
        needs = {"full": weights_gb * 16.0, "lora": weights_gb * 1.4,
                 "qlora": weights_gb * 0.5}
        training = []
        for kind in ("full", "lora", "qlora"):
            need = needs[kind]
            feasible = need <= available
            training.append(TrainingVerdict(
                kind=kind, feasible=feasible,
                memory_needed_gb=need, memory_available_gb=available,
                step_seconds=None if not feasible else 4.5,
                epoch_hours=None if not feasible else 12.0,
                reasons=(
                    [f"{need:.0f}GB가 필요한데 {available:.0f}GB만 장착했다."]
                    if not feasible else ["장착 메모리 안에 들어간다."]
                ) + ["학습 계수는 카탈로그에 근거가 없는 추정이다."],
                gpu_comparison=f"GPU 1장(80GB)이면 {kind}는 수 시간 규모다.",
            ))

        return ModelBench(
            model_name=asm.model.name, quant_id=asm.quant.id,
            hardware=f"{asm.cpu.vendor} {asm.cpu.model} × {asm.vm.sockets}소켓",
            throughput=throughput, concurrency=concurrency,
            resources=resources, training=training,
            memory_gb=float(asm.ram_used_gb), uncertainty=float(asm.uncertainty),
            notes=["실제 모델을 실행하지 않았다 — 카탈로그 물리로 예측했다.",
                   "학습 축은 계수 근거가 없는 추정이다."],
            warnings=["대역폭 효율 계수가 추정값이다."],
        )

    def to_csv(bench, section):  # pragma: no cover - unused by the GUI
        raise NotImplementedError

    mod = types.ModuleType("svrspec.modelbench")
    mod.ThroughputPoint = ThroughputPoint
    mod.ConcurrencyPoint = ConcurrencyPoint
    mod.ResourceSplit = ResourceSplit
    mod.TrainingVerdict = TrainingVerdict
    mod.ModelBench = ModelBench
    mod.bench_model = bench_model
    mod.to_csv = to_csv
    return mod


def _install_engines(monkeypatch) -> None:
    """Real engines where they exist, stand-ins where they do not."""
    for name, build in (("svrspec.lab", _stub_lab),
                        ("svrspec.loadgen", _stub_loadgen),
                        ("svrspec.bench", _stub_bench),
                        ("svrspec.modelbench", _stub_modelbench)):
        if importlib.util.find_spec(name) is None:
            monkeypatch.setitem(sys.modules, name, build())


@pytest.fixture
def engines(monkeypatch):
    _install_engines(monkeypatch)


#: An eight-channel board with two DIMMs in it: 128 GB of perfectly good memory
#: running at a quarter of the bandwidth. The build this screen exists to catch.
STARVED = {
    **BASE_REQUEST,
    "cpu": "test-amx-8ch",
    "dimm_gb": 64,
    "dimm_count": 2,
    "sockets": 1,
    "slots": 4,
}
#: The same part, with every channel filled. 64 GB is the only DDR5 capacity in
#: the fixture catalogue, which is the point the remedy has to make: you cannot
#: always spread the *same* gigabytes across every channel with what is stocked.
FULL = {**STARVED, "dimm_count": 8}


def _lab(cat, request):
    from svrspec.gui import lab_payload

    return lab_payload(cat, _params(request), request["cpu"], request.get("name", "A"))


def test_catalog_payload_carries_the_cpus_the_lab_picks_from(cat):
    """The lab picks one part by hand; it cannot sweep, so it needs the list."""
    payload = catalog_payload(cat)
    json.dumps(payload, allow_nan=False)
    assert payload["cpus"], "the lab dropdown has nothing to offer without this"
    assert {c["id"] for c in payload["cpus"]} == {c.id for c in cat.cpus}
    for key in ("label", "cores", "mem_channels", "sockets_max", "ddr_gen", "max_mem_gb"):
        assert key in payload["cpus"][0], key


def test_lab_payload_carries_every_finding_with_its_remedy(cat, engines):
    """A problem without a fix is worse than silence: it strands the reader."""
    d = _lab(cat, STARVED)
    json.dumps(d, allow_nan=False)

    assert d["findings"], "a two-DIMM eight-channel board is not a clean build"
    for f in d["findings"]:
        assert f["level"] in ("error", "warn", "info")
        assert f["code"] and f["message"]
    # Every warning and error has to say how to fix itself.
    assert all(f["remedy"] for f in d["findings"] if f["level"] in ("error", "warn"))
    assert d["errors"] + d["warnings"] <= len(d["findings"])


def test_two_dimms_in_an_eight_channel_board_cost_three_quarters_of_the_bandwidth(
    cat, engines
):
    """The whole point of the screen, asserted as arithmetic.

    Capacity looks fine and decode collapses with the bandwidth, because decode
    reads the whole weight set out of DRAM per token. If the payload ever
    reports the full-channel figure here, the warning becomes decoration.
    """
    d = _lab(cat, STARVED)

    codes = {f["code"] for f in d["findings"]}
    assert "channels-underfilled" in codes

    assert d["channels"] == {"total": 8, "populated": 2, "dimms_per_channel": 1,
                             "fill_pct": 25.0}
    ratio = d["bandwidth"]["gbs"] / d["bandwidth"]["full_gbs"]
    assert abs(ratio - 0.25) < 0.01, (d["bandwidth"], "bandwidth is not a quarter")
    assert d["bandwidth"]["loss_pct"] == pytest.approx(75.0, abs=1.0)

    # Decode is bandwidth bound, so it falls with it -- that is the sentence the
    # operator has to read, and it has to be in the payload to be readable.
    decode = d["throughput"]
    assert decode["decode_tps_single"] < decode["decode_tps_full"]
    assert decode["decode_loss_pct"] >= 50


def test_a_fully_populated_board_raises_no_channel_warning(cat, engines):
    d = _lab(cat, FULL)
    assert "channels-underfilled" not in {f["code"] for f in d["findings"]}
    assert d["channels"]["populated"] == d["channels"]["total"]
    assert d["bandwidth"]["gbs"] == d["bandwidth"]["full_gbs"]
    assert d["bandwidth"]["loss_pct"] in (None, 0.0)
    assert d["ok"] is True


def test_lab_payload_offers_the_dimm_combinations_the_dropdown_needs(cat, engines):
    d = _lab(cat, STARVED)
    assert d["options"], "the capacity dropdown is built from this"
    for o in d["options"]:
        assert o["dimm_gb"] > 0 and o["count"] > 0
        assert o["dpc"] in (1, 2)
        assert o["ram_total_gb"] == o["dimm_gb"] * o["count"]


def test_the_live_assembly_never_runs_a_bench(cat, engines, monkeypatch):
    """Performance guard, the same one the capacity search has.

    Assembly runs on every dropdown change. A bench builds a load and pushes it
    through the queue engine; hanging that off the live path would turn a
    simulator into a spinner. So the immediate payload must not touch it.
    """
    import svrspec.bench as bench_module
    import svrspec.loadgen as loadgen_module

    def explode(*args, **kwargs):
        raise AssertionError("a bench ran on the live assembly path")

    monkeypatch.setattr(bench_module, "run_bench", explode)
    monkeypatch.setattr(loadgen_module, "build_load", explode)

    assert _lab(cat, STARVED)["headline"]
    assert _lab(cat, FULL)["ok"] is True


def test_bench_payload_carries_frames_breach_stats_and_worst(cat, engines):
    from svrspec.gui import bench_payload

    request = {**FULL, "kind": "ramp",
               "profile": {"start_rate": 100, "end_rate": 4000, "hours": 24},
               "frames": 240}
    d = bench_payload(cat, _params(request), request["cpu"], request)
    json.dumps(d, allow_nan=False)

    for key in ("frames", "breach", "stats", "worst", "profile", "machine", "findings"):
        assert key in d, key
    assert d["blocked"] is None
    assert len(d["frames"]) == 240
    assert d["profile"]["kind"] == "ramp" and d["profile"]["span_s"] == 86400.0
    assert d["stats"]["received"] > 0

    first = d["frames"][0]
    for key in ("t_s", "queued", "active", "cpu_pct", "bw_gbs", "bw_pct",
                "compute_pct", "kv_gb", "ram_gb", "arrived", "delivered",
                "offered_rate", "p95_so_far_s"):
        assert key in first, key
    times = [f["t_s"] for f in d["frames"]]
    assert times == sorted(times)

    # A ramp exists to find the breaking point; the payload has to carry it or
    # the chart has nothing to mark.
    if d["breach"]:
        assert set(d["breach"]) >= {"t_s", "offered_rate", "p95_s"}
        before = [f for f in d["frames"] if f["t_s"] < d["breach"]["t_s"]]
        assert all(f["p95_so_far_s"] <= d["sla"]["sla_seconds"] for f in before)


def test_bench_refuses_a_build_that_cannot_be_ordered(cat, engines):
    """An error-level finding is not a machine, so numbers about it are fiction."""
    from svrspec.gui import bench_payload

    request = {**BASE_REQUEST, "cpu": "test-amx-8ch", "dimm_gb": 32,
               "dimm_count": 0, "kind": "ramp"}
    d = bench_payload(cat, _params(request), request["cpu"], request)
    json.dumps(d, allow_nan=False)
    assert d["ok"] is False
    assert d["blocked"]
    assert d["frames"] == [] and d["breach"] is None
    assert any(f["level"] == "error" for f in d["findings"])


def test_bench_parameters_are_clamped_and_scoped_to_their_profile():
    from svrspec.gui import BENCH_DEFAULTS, _bench_kind, _bench_params

    # A ramp never sees a spike's knobs, so a stale field cannot reach a
    # keyword the engine does not take.
    assert set(_bench_params("ramp", {"peak_rate": 9, "start_rate": 5})) == {
        "start_rate", "end_rate", "hours"
    }
    assert _bench_params("ramp", {"hours": 10**6})["hours"] == 336
    assert _bench_params("ramp", {"start_rate": -4})["start_rate"] == 1
    assert _bench_params("ramp", {"end_rate": "많이"})["end_rate"] == \
        BENCH_DEFAULTS["end_rate"]
    assert _bench_params("replay", {"date": "; rm -rf /"})["date"] == "2026-06-01"
    assert _bench_params("replay", {"count": 0})["count"] is None
    assert _bench_kind({"kind": "nope"}) == "replay"
    assert _bench_kind({"kind": "soak"}) == "soak"


def test_no_payload_leaks_an_infinity_to_the_browser(cat, engines):
    """`JSON.parse` rejects the literal Python emits, so it must never be sent."""
    from svrspec.gui import _clean, _r

    assert _r(float("inf")) is None
    assert _r(float("nan")) is None
    assert _clean({"a": float("inf"), "b": [1.0, float("-inf")], "c": {"d": float("nan")}}) \
        == {"a": None, "b": [1.0, None], "c": {"d": None}}

    request = {**FULL, "kind": "soak", "profile": {"rate": 300, "hours": 72},
               "frames": 60}
    from svrspec.gui import bench_payload

    d = bench_payload(cat, _params(request), request["cpu"], request)
    json.dumps(d, allow_nan=False)   # the guard the handler applies for real


def test_lab_limits_clamp_the_board_the_browser_asked_for():
    p = _params({"dimm_gb": 10**6, "dimm_count": -3})
    assert p["dimm_gb"] == LIMITS["dimm_gb"][1]
    assert p["dimm_count"] == 0        # zero is a build the lab has to describe


# --------------------------------------------------------------------------
# Lab routes
# --------------------------------------------------------------------------


@pytest.fixture
def lab_server(monkeypatch):
    """A live server with the engines resolved the same way the tests are."""
    _install_engines(monkeypatch)
    handler = type("H", (_Handler,), {"catalog": Catalog(DATA)})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def test_lab_route_round_trip(lab_server):
    status, d = _post(lab_server + "/api/lab", STARVED)
    assert status == 200
    assert d["channels"]["populated"] == 2
    assert "channels-underfilled" in {f["code"] for f in d["findings"]}


def test_lab_route_rejects_an_unknown_cpu(lab_server):
    status, d = _post(lab_server + "/api/lab", {**STARVED, "cpu": "nope"})
    assert status == 400
    assert "unknown cpu id" in d["error"]


def test_bench_route_round_trip(lab_server):
    status, d = _post(lab_server + "/api/bench", {
        **FULL, "kind": "ramp", "frames": 120,
        "profile": {"start_rate": 100, "end_rate": 3000, "hours": 24},
    })
    assert status == 200
    assert len(d["frames"]) == 120
    assert d["profile"]["kind"] == "ramp"
    assert "stats" in d and "worst" in d


def test_bench_route_survives_a_garbage_profile(lab_server):
    status, d = _post(lab_server + "/api/bench", {
        **FULL, "kind": "구름", "frames": "많이", "profile": "not a dict",
    })
    assert status == 200
    assert d["kind"] == "replay"           # unknown kinds fall back, they do not 500
    assert len(d["frames"]) == 600


def test_an_unroutable_lab_path_is_still_404(lab_server):
    status, _ = _post(lab_server + "/api/labs", {})
    assert status == 404


# --------------------------------------------------------------------------
# The lab screen
# --------------------------------------------------------------------------


def test_the_page_runs_a_bench_only_from_the_button():
    """The client half of the live-path guard.

    Two mentions each: the transport plus its single caller, and the handler
    plus the click that wires it. A third would mean something started running
    a load test automatically.
    """
    assert SERVER_HTML.count("askBench") == 2
    assert SERVER_HTML.count("runBench") == 2
    assert 'byId("lab-run").addEventListener("click", runBench)' in SERVER_HTML
    # Assembly is the opposite: cheap, and wired to every change on the form.
    assert SERVER_HTML.count("askLab") == 2
    assert 'railForm.addEventListener("input", onLabInput)' in SERVER_HTML


def test_the_lab_screen_offers_a_gauge_per_resource():
    for name in ("CPU 가동", "대기 큐", "메모리 대역폭", "RAM 실사용"):
        assert f'"{name}"' in SERVER_HTML, name
    # Four different columns of the frame, not one number wearing four labels.
    for column in ("cpu_pct", "queued", "bw_pct", "ram_gb"):
        assert f"f.{column}" in SERVER_HTML, column


def test_the_player_offers_speeds_and_a_scrub():
    assert '[[1, "1×"], [60, "60×"], [600, "600×"], [0, "즉시"]]' in SERVER_HTML
    assert 'range.type = "range"' in SERVER_HTML
    assert 'range.setAttribute("aria-label", "재생 위치")' in SERVER_HTML
    assert "requestAnimationFrame(step)" in SERVER_HTML


def test_reduced_motion_shows_the_final_state_without_animating():
    assert 'window.matchMedia("(prefers-reduced-motion: reduce)")' in SERVER_HTML
    assert "if(STILL){" in SERVER_HTML
    assert "controls.hidden = true" in SERVER_HTML


def test_the_breach_is_drawn_as_a_marker_not_only_a_sentence():
    """The ramp's whole output is "where does it break". It has to be visible."""
    assert "breachLine" in SERVER_HTML
    assert "SLA 붕괴 지점" in SERVER_HTML
    assert "c.run.breach" in SERVER_HTML


def test_the_lab_keeps_two_builds_side_by_side():
    assert 'id="mach-a"' in SERVER_HTML and 'id="mach-b"' in SERVER_HTML
    assert 'id="lab-compare"' in SERVER_HTML
    assert "asm-pair" in SERVER_HTML
    assert 'stroke-dasharray"] = "5 3"' in SERVER_HTML   # machine B, overlaid


def test_the_desktop_page_degrades_when_the_bridge_has_no_lab_call():
    assert "window.pywebview.api.lab" in DESKTOP_HTML
    assert "window.pywebview.api.bench" in DESKTOP_HTML
    assert "데스크톱 빌드에는 가상 랩이 연결되어 있지 않습니다" in DESKTOP_HTML


def test_every_lab_control_carries_a_visible_label():
    """Covered globally too, but this is the screen that just added fifteen."""
    ids = set(re.findall(r'<(?:input|select)[^>]*\bid="(lab-[^"]+|bp-[^"]+)"', SERVER_HTML))
    labelled = set(re.findall(r'<label[^>]*\bfor="([^"]+)"', SERVER_HTML))
    assert len(ids) >= 15
    assert not (ids - labelled)


def test_the_lab_findings_carry_a_word_not_only_a_colour():
    assert '{error: "오류", warn: "경고", info: "참고"}' in SERVER_HTML
    assert "고치는 법 · " in SERVER_HTML       # the remedy, always rendered


def test_desktop_api_serves_the_lab_and_the_bench(engines):
    """Degrading gracefully is the fallback, not the goal.

    The packaged app is how this ships on Windows, and "run it in server mode
    instead" is not an answer for an operator who was handed an installer.
    """
    from svrspec.desktop import Api

    api = Api(Catalog(DATA))
    d = api.lab(STARVED)
    assert "error" not in d
    assert "channels-underfilled" in {f["code"] for f in d["findings"]}

    b = api.bench({**FULL, "kind": "ramp", "frames": 90,
                   "profile": {"start_rate": 100, "end_rate": 3000, "hours": 24}})
    assert "error" not in b
    assert len(b["frames"]) == 90

    # Marshalled as a JSON string, like some webview bridges do.
    assert "error" not in api.lab(json.dumps(STARVED))


def test_desktop_lab_reports_bad_input_instead_of_raising(engines):
    from svrspec.desktop import Api

    api = Api(Catalog(DATA))
    assert "error" in api.lab({**STARVED, "cpu": "no-such-cpu"})
    assert "error" in api.lab([1, 2, 3])
    assert "error" in api.bench({**FULL, "cpu": "no-such-cpu"})


def test_the_desktop_bridge_cannot_hang_the_window_forever():
    """A missed `pywebviewready` event must not leave the page pending.

    The original wait attached a listener and nothing else. If pywebview had
    already fired the event before this script ran, the listener never fired,
    the promise never settled, and the window sat on "카탈로그를 불러오는 중…"
    with no error -- a pending promise cannot reach a .catch(). The fix is to
    poll for the object as well and to reject after a bounded wait.
    """
    page = app_html("desktop")

    assert "BRIDGE_TIMEOUT_MS" in page
    assert "setInterval" in page          # the poll that covers the missed event
    assert "clearInterval" in page        # and it must be torn down
    assert "reject(" in page              # a bounded wait that actually fails

    # The remedy has to travel with the failure -- and the remedy is now
    # something the app does, not something it asks the operator to go do.
    # Naming WebView2 and a shell command was the old text: correct diagnosis,
    # wrong owner. See test_a_window_whose_bridge_never_loads_rescues_itself.
    assert "서버 방식으로 자동 전환" in page

    # The guard must check the method it is about to call, not just the object:
    # pywebview populates `window.pywebview` before the api is usable.
    assert "window.pywebview.api.catalog" in page


def test_a_boot_failure_clears_the_loading_placeholder():
    """The window may not keep claiming it is loading after it has given up."""
    page = app_html("server")
    assert 'id="empty"' in page
    assert "empty.remove()" in page
    # The default screen is the model one now, so the failure has to land there
    # too -- reporting it only on a hidden tab is the same bug wearing a hat.
    assert '["mb-results", "results", "lab-results"]' in page


# --------------------------------------------------------------------------
# Model performance: three tabs, model performance first
# --------------------------------------------------------------------------
#
# The complaint this answers: the tool had grown into one application -- alarm
# load -- and had no screen for the question underneath it, "put this model on
# this server and what does it do". So: three tabs, the model one in front, and
# everything that already existed rehoused rather than dropped.


def _mb(cat, request, raw=None):
    from svrspec.gui import modelbench_payload

    merged = {**request, **(raw or {})}
    return modelbench_payload(cat, _params(merged), merged["cpu"], merged)


def test_the_default_view_is_model_performance():
    """Not the alarm screen. That was the whole point of the rework."""
    # Prefixed: "model" belonged to the LLM <select> first, and sharing it
    # silently broke the whole page. See test_no_two_elements_share_an_id.
    assert '<main id="screen-model">' in SERVER_HTML
    assert '<main id="screen-size" hidden>' in SERVER_HTML
    assert '<main id="screen-lab" hidden>' in SERVER_HTML
    assert '<button id="view-model" type="button" aria-pressed="true">모델 성능' in SERVER_HTML
    for other in ("view-size", "view-lab"):
        assert re.search(rf'id="{other}"[^>]*aria-pressed="false"', SERVER_HTML), other
    assert 'setView("model")' in SERVER_HTML


def test_the_three_tabs_are_the_ones_the_structure_asked_for():
    nav = SERVER_HTML.split('<nav class="views"')[1].split("</nav>")[0]
    assert "모델 성능" in nav
    assert "자원" in nav
    # Named for what it does. It was briefly "적용 사례: 관제 알람", which named
    # one of its four load profiles as though it were the whole screen --
    # the screen assembles a machine and drives load at it.
    assert "부하 테스트" in nav
    assert "관제 알람" not in nav
    # Toggles, not links: assistive tech has to be able to read the state.
    assert nav.count("aria-pressed") == 3


def test_nothing_that_already_existed_disappeared():
    """Rehoused, not removed. Every earlier screen still has to be reachable."""
    for marker in (
        '<main id="screen-size"',     # sizing: tiers, candidate table, downloads
        "권장 스펙",
        "CPU 후보",
        "토큰 전달 시뮬레이터",
        "작업관리자",
        "개수별 리소스",
        "과부하 지점",
        '<main id="screen-lab"',      # the virtual lab and its load profiles
        "가상 서버 조립",
        "부하 테스트",
        "효율 계수",
    ):
        assert marker in SERVER_HTML, marker
    # And the rails and result columns of all three screens are still wired.
    for host in ("mb-results", "results", "lab-results"):
        assert f'id="{host}"' in SERVER_HTML, host
    for rail in ("mb-rail", "rail", "lab-rail"):
        assert f'id="{rail}"' in SERVER_HTML, rail


def test_the_page_measures_a_model_only_from_the_button():
    """Same guard, same reason as the capacity search and the load bench.

    Two mentions each: the transport plus its single caller, and the handler
    plus the click that wires it. A third would mean the grid started
    recomputing on every keystroke.
    """
    assert SERVER_HTML.count("askModelBench") == 2
    assert SERVER_HTML.count("runModelBench") == 2
    assert 'byId("mb-run").addEventListener("click", runModelBench)' in SERVER_HTML


def test_the_desktop_page_degrades_when_the_bridge_has_no_modelbench_call():
    """The packaged bridge is a fixed surface; an older installer must not throw."""
    assert "window.pywebview.api.modelbench" in DESKTOP_HTML
    assert "데스크톱 빌드에는 모델 성능 측정이 연결되어 있지 않습니다" in DESKTOP_HTML


def test_every_model_screen_control_carries_a_visible_label():
    ids = set(re.findall(r'<(?:input|select)[^>]*\bid="(mb-[^"]+)"', SERVER_HTML))
    labelled = set(re.findall(r'<label[^>]*\bfor="([^"]+)"', SERVER_HTML))
    assert len(ids) >= 7
    assert not (ids - labelled)


def test_the_model_screen_shows_all_four_axes():
    for section in ("추론 처리량", "동시 사용자", "연산 자원 분해", "학습·파인튜닝"):
        assert section in SERVER_HTML, section
    # The batch trade-off and the context cost are stated, not left to be
    # inferred from the grid.
    assert "배치는 처리량을 사고 응답 속도를 판다" in SERVER_HTML
    assert "KV 캐시를 다시 읽어야 하기 때문이다" in SERVER_HTML
    # Prefill and decode read against each other, which is the point.
    assert "기계의 반대쪽이다" in SERVER_HTML


def test_the_concurrency_curve_marks_the_unreadable_point():
    """"몇 명까지 쓸 만한가" is a threshold question, so the threshold is drawn."""
    assert "readable_tps" in SERVER_HTML
    assert "답답함" in SERVER_HTML and "쓸 만함" in SERVER_HTML   # colour + word
    assert 'role: "img"' in SERVER_HTML
    assert "읽기 편한 하한" in SERVER_HTML


def test_the_training_refusal_shows_its_reasons_and_the_gpu_comparison():
    """A bare "안 된다" is not a result anybody can act on."""
    assert "t.reasons" in SERVER_HTML
    assert "GPU 비교 · " in SERVER_HTML
    assert "t.gpu_comparison" in SERVER_HTML
    assert "엔진이 비교값을 내지 않았다" in SERVER_HTML     # never silently dropped
    assert "학습 축의 계수는 카탈로그에 근거가 없다" in SERVER_HTML


def test_modelbench_payload_carries_every_axis(cat, engines):
    from svrspec.gui import MB_BATCHES, MB_CONTEXTS, MB_USERS

    d = _mb(cat, FULL)
    json.dumps(d, allow_nan=False)
    assert d["blocked"] is None
    for key in ("throughput", "concurrency", "resources", "training", "notes"):
        assert d[key], key
    assert d["warnings"] is not None
    assert len(d["throughput"]) == len(MB_BATCHES) * len(MB_CONTEXTS)
    assert len(d["concurrency"]) == len(MB_USERS)
    assert d["model_name"] and d["quant_id"] and d["hardware"]
    assert d["memory_gb"] is not None and d["uncertainty"] is not None

    row = d["throughput"][0]
    for key in ("batch", "ctx_tokens", "prefill_tps", "decode_tps_single",
                "decode_tps_total", "prefill_bound_label", "decode_bound_label"):
        assert key in row, key
    point = d["concurrency"][0]
    for key in ("users", "ttft_s", "decode_tps_each", "response_s", "total_tps",
                "readable"):
        assert key in point, key
    assert {s["phase"] for s in d["resources"]} <= {"prefill", "decode"}
    assert all(s["bound_label"] and s["phase_label"] for s in d["resources"])
    assert {t["kind"] for t in d["training"]} == {"full", "lora", "qlora"}


def test_the_training_verdict_never_arrives_without_its_grounds(cat, engines):
    """AC: an infeasible verdict must carry reasons and a GPU comparison."""
    d = _mb(cat, FULL)
    for t in d["training"]:
        assert t["verdict"] in ("가능", "불가")          # a word, not only a colour
        assert isinstance(t["reasons"], list)
        if not t["feasible"]:
            assert t["reasons"], t["kind"]
            assert t["gpu_comparison"], t["kind"]


def test_an_under_populated_board_reaches_the_model_screen_intact(cat, engines):
    """The lab's channel population has to survive into this payload.

    Two DIMMs in an eight-channel board is the build the whole tool exists to
    catch. If the model screen quoted full-bus numbers for it, it would be
    advertising hardware nobody is buying.
    """
    starved = _mb(cat, STARVED)
    full = _mb(cat, FULL)
    assert starved["machine"]["channels"]["populated"] == 2
    assert starved["throughput"][0]["decode_tps_single"] < \
        full["throughput"][0]["decode_tps_single"]
    assert "channels-underfilled" in {f["code"] for f in starved["findings"]}


def test_a_build_that_cannot_be_ordered_is_refused_not_measured(cat, engines):
    """An error-level finding means the numbers would be about fiction."""
    d = _mb(cat, {**FULL, "dimm_count": 0})
    assert d["ok"] is False
    assert d["blocked"]
    assert d["throughput"] == [] and d["training"] == []
    assert d["findings"]


def test_the_live_recompute_never_runs_a_model_bench(cat, engines, monkeypatch):
    """Performance guard, server half.

    The grid is dozens of predictions plus a training verdict. The two payloads
    that do run on every input change must not touch it.
    """
    import importlib

    from svrspec.gui import resource_payload

    def explode(*args, **kwargs):
        raise AssertionError("a model bench ran on the live recompute path")

    engine = importlib.import_module("svrspec.modelbench")
    monkeypatch.setattr(engine, "bench_model", explode)
    assert size_payload(cat, _params(BASE_REQUEST))["candidates"]
    assert resource_payload(cat, _params(BASE_REQUEST), "test-amx-8ch")["rows"]
    from svrspec.gui import lab_payload

    assert lab_payload(cat, _params(FULL), FULL["cpu"])["channels"]


def test_modelbench_clamps_the_training_sample_count():
    from svrspec.gui import MODELBENCH_DEFAULTS, MODELBENCH_LIMITS, _mb_number

    low, high = MODELBENCH_LIMITS["train_samples"]
    assert _mb_number({"train_samples": 10**12}, "train_samples") == high
    assert _mb_number({"train_samples": -1}, "train_samples") == low
    assert _mb_number({"train_samples": "많이"}, "train_samples") == \
        MODELBENCH_DEFAULTS["train_samples"]
    assert low <= MODELBENCH_DEFAULTS["train_samples"] <= high


def test_no_model_axis_leaks_an_infinity_to_the_browser():
    """`allow_nan=False` rejects `Infinity`, and `JSON.parse` refuses it too.

    "이 서버에서는 끝나지 않는다" is a real answer for an epoch time, and the
    engine is entitled to express it as an infinity. It has to cross the wire as
    null, and the row builders are where that happens.
    """
    from types import SimpleNamespace

    from svrspec.gui import _concurrency_row, _throughput_row, _training_row

    inf, nan = float("inf"), float("nan")
    t = _training_row(SimpleNamespace(
        kind="full", feasible=False, memory_needed_gb=inf, memory_available_gb=nan,
        step_seconds=inf, epoch_hours=inf, reasons=["메모리가 모자란다."],
        gpu_comparison="GPU 1장이면 몇 시간이다.",
    ))
    assert t["memory_needed_gb"] is None and t["epoch_hours"] is None
    assert t["step_seconds"] is None and t["memory_available_gb"] is None
    assert t["reasons"] and t["gpu_comparison"]

    p = _throughput_row(SimpleNamespace(
        batch=1, ctx_tokens=512, prefill_tps=inf, decode_tps_single=nan,
        decode_tps_total=-inf, prefill_bound="compute", decode_bound="bandwidth",
    ))
    c = _concurrency_row(SimpleNamespace(
        users=1, ttft_s=inf, decode_tps_each=nan, response_s=inf, total_tps=nan,
    ))
    assert p["prefill_tps"] is None and c["ttft_s"] is None
    # A speed that is not a number cannot be called readable.
    assert c["readable"] is False
    for row in (t, p, c):
        json.dumps(row, allow_nan=False)


def test_unknown_bound_names_are_passed_through_not_dropped():
    """A ceiling this layer has no Korean word for still has to be reported."""
    from types import SimpleNamespace

    from svrspec.gui import MB_BOUND_LABEL, _split_row, _throughput_row

    assert "core-bandwidth" in MB_BOUND_LABEL   # perf reports it; the alarm view folds it
    row = _throughput_row(SimpleNamespace(
        batch=1, ctx_tokens=512, prefill_tps=1.0, decode_tps_single=1.0,
        decode_tps_total=1.0, prefill_bound="something-new", decode_bound="compute",
    ))
    assert row["prefill_bound_label"] == "something-new"
    split = _split_row(SimpleNamespace(
        phase="prefill", bandwidth_pct=1.0, compute_pct=2.0, bound_by="core-bandwidth",
        bytes_per_token=1.0, flops_per_token=2.0,
    ))
    assert split["bound_label"] == "코어당 대역폭"
    assert split["advice"], "a named bottleneck without advice is half an answer"


def test_modelbench_route_round_trip(lab_server):
    status, d = _post(lab_server + "/api/modelbench", {**FULL, "train_samples": 5_000})
    assert status == 200
    assert d["train_samples"] == 5_000
    assert d["throughput"] and d["concurrency"] and d["training"]
    assert d["blocked"] is None


def test_modelbench_route_rejects_an_unknown_cpu(lab_server):
    status, d = _post(lab_server + "/api/modelbench", {**FULL, "cpu": "nope"})
    assert status == 400
    assert "unknown cpu id" in d["error"]


def test_modelbench_route_survives_a_garbage_sample_count(lab_server):
    from svrspec.gui import MODELBENCH_DEFAULTS

    status, d = _post(lab_server + "/api/modelbench",
                      {**FULL, "train_samples": "아무거나"})
    assert status == 200
    assert d["train_samples"] == MODELBENCH_DEFAULTS["train_samples"]


def test_an_unroutable_model_path_is_still_404(lab_server):
    status, _ = _post(lab_server + "/api/modelbenchmark", {})
    assert status == 404


def test_the_model_tables_scroll_inside_their_own_box():
    """Wide content scrolls in its own box; the page body never scrolls sideways."""
    grid = SERVER_HTML.split("function mbThroughput(d){")[1].split("function mbTradeoff")[0]
    assert 'el("div", "scroll")' in grid
    users = SERVER_HTML.split("function mbConcurrency(d){")[1].split("function mbGauge")[0]
    assert 'el("div", "scroll")' in users


# --------------------------------------------------------------------------
# Generation tok/s and TTFT: the two numbers a person feels
# --------------------------------------------------------------------------


def test_the_top_summary_leads_with_generation_speed_and_ttft():
    """These two are what somebody feels, so they are the first thing drawn."""
    assert "mbSummaryTiles" in SERVER_HTML
    assert 'el("div", "mb-head")' in SERVER_HTML
    # Drawn into the head card, above every table on the screen.
    head = SERVER_HTML.split("function renderModel(){")[1] \
                      .split("host.appendChild(card);")[0]
    assert "mbSummaryTiles(d)" in head
    assert head.index("mbSummaryTiles(d)") < head.index("추론 시 실사용")
    # The comfort threshold is stated in a sentence, not only drawn as a line.
    assert "기다린다는 느낌" in SERVER_HTML


def test_the_felt_numbers_are_labelled_in_both_languages():
    """Two kinds of reader: one says TTFT, one says 첫 토큰까지. Both get both."""
    for label in ("생성 속도(Generation tok/s)", "첫 토큰까지(TTFT)",
                  "전체 합계(Total tok/s)", "프롬프트 처리(Prefill tok/s)"):
        assert label in SERVER_HTML, label
    # Every metric the grid can draw is a (engine key, 한국어(원어), unit) triple,
    # so a raw field name cannot reach a column header by accident.
    block = SERVER_HTML.split("var MB_METRICS = [")[1].split("\n  ];")[0]
    keys = ("decode_tps_single", "decode_tps_total", "ttft_s", "prefill_tps",
            # RAM is a grid metric too: it is the axis that decides whether a
            # cell can run at all, so it has to be selectable next to the
            # speeds rather than living only in the summary.
            "ram_gb")
    for key in keys:
        assert key in block, key
    # One bracket per triple: a raw field name cannot reach a column header by
    # accident, and an untriplet entry would not balance.
    assert block.count("[") == len(keys)


def test_generation_speed_is_never_quoted_as_the_server_total():
    """The one confusion this screen exists to prevent, spelled out on screen."""
    assert "배치는 처리량을 사고 응답 속도를 판다" in SERVER_HTML
    assert "한 사용자가 보는 " in SERVER_HTML
    assert "서버 전체 합계 — 한 사용자가 보는 속도가 아니다" in SERVER_HTML


def test_the_summary_carries_the_two_felt_numbers_and_their_condition(cat, engines):
    d = _mb(cat, FULL)
    s = d["summary"]
    for key in ("gen_tps", "ttft_s", "total_tps", "batch", "ctx_tokens",
                "condition", "readable"):
        assert key in s, key
    assert s["gen_tps"] and s["ttft_s"] and s["total_tps"]
    # A headline number without its condition is a number nobody can check.
    assert s["batch"] == 1
    assert "배치 1" in s["condition"]
    assert f"{s['ctx_tokens']:,}" in s["condition"]
    assert str(d["output_tokens"]) in s["condition"]
    # The batched contrast travels with it: the total climbs, the per-user speed
    # falls, and a longer context costs generation speed.
    assert s["busy_total_tps"] >= s["total_tps"]
    assert s["busy_gen_tps"] <= s["gen_tps"]
    assert s["long_gen_tps"] <= s["gen_tps"]


def test_a_blocked_build_still_answers_with_an_empty_summary(cat, engines):
    """The page reads `summary` before it reads `blocked`; it must exist either way."""
    d = _mb(cat, {**FULL, "dimm_count": 0})
    assert d["blocked"]
    assert d["summary"]["gen_tps"] is None and d["summary"]["ttft_s"] is None
    json.dumps(d, allow_nan=False)


def test_ttft_is_derived_when_the_engine_does_not_report_it():
    """TTFT is the prompt over the prefill rate. A blank cell is not an option.

    The field is being added on the engine side in this same wave, so this layer
    prefers it where it exists and computes it where it does not -- and the
    switch-over must not change what the page shows.
    """
    from types import SimpleNamespace

    from svrspec.gui import _mb_ttft, _throughput_row

    pt = SimpleNamespace(batch=1, ctx_tokens=2048, prefill_tps=1024.0,
                         decode_tps_single=40.0, decode_tps_total=40.0,
                         prefill_bound="compute", decode_bound="bandwidth")
    assert _mb_ttft(pt) == pytest.approx(2.0)
    assert _throughput_row(pt)["ttft_s"] == pytest.approx(2.0)

    pt.ttft_s = 0.75                     # the engine's own figure wins
    assert _mb_ttft(pt) == pytest.approx(0.75)
    assert _throughput_row(pt)["ttft_s"] == pytest.approx(0.75)

    # A machine that processes nothing has no TTFT to quote, and "infinity" is
    # not something JSON.parse will accept.
    stalled = _throughput_row(SimpleNamespace(
        batch=1, ctx_tokens=512, prefill_tps=0.0, decode_tps_single=1.0,
        decode_tps_total=1.0, prefill_bound="none", decode_bound="none",
    ))
    assert stalled["ttft_s"] is None


def test_every_grid_row_carries_a_ttft(cat, engines):
    d = _mb(cat, FULL)
    assert all(r["ttft_s"] is not None for r in d["throughput"])
    # Longer prompt, longer wait -- at a fixed batch.
    first = [r for r in d["throughput"] if r["batch"] == 1]
    waits = [r["ttft_s"] for r in sorted(first, key=lambda r: r["ctx_tokens"])]
    assert waits == sorted(waits)


def test_the_grid_reports_ram_per_row_not_just_once():
    """One memory figure for the whole grid hid a tenfold range.

    A reader who sized the box from the summary would find the bottom-right of
    the table would not load, so each row carries what it actually needs and
    whether the assembled machine has it.
    """
    # 8 x 8GB fills every channel of this board but is only 64 GB, so the
    # default grid's heaviest corners genuinely do not load on it.
    payload = _mb(
        Catalog(DATA),
        {**BASE_REQUEST, "model": "test-8b-gqa", "cpu": "test-amx-8ch",
         "dimm_gb": 8, "dimm_count": 8},
    )
    rows = payload["throughput"]
    assert rows

    for row in rows:
        assert row["ram_gb"] is not None and row["ram_gb"] > 0
        assert isinstance(row["fits"], bool)

    rams = [r["ram_gb"] for r in rows]
    assert max(rams) > min(rams), rams
    # The heaviest cell on a 64 GB board is the one that must be flagged.
    assert any(r["fits"] is False for r in rows), rams


def test_the_caller_can_choose_its_own_operating_point():
    """The whole point of the screen is to see *your* batch and context.

    The axes were fixed on the server to stop a browser asking for a
    four-hundred point sweep. That concern is real, but refusing every
    override was too strict: 16k context at batch 48 is something people
    actually run, and the grid could not show it.
    """
    payload = _mb(
        Catalog(DATA),
        {**BASE_REQUEST, "model": "test-8b-gqa", "cpu": "test-amx-8ch",
         "dimm_gb": 8, "dimm_count": 8},
        {"batches": [1, 48], "contexts": [4096, 65536]},
    )
    assert payload["batches"] == [1, 48]
    assert payload["contexts"] == [4096, 65536]
    assert {(r["batch"], r["ctx_tokens"]) for r in payload["throughput"]} == {
        (1, 4096), (1, 65536), (48, 4096), (48, 65536)
    }


def test_an_axis_override_is_clamped_deduplicated_and_capped():
    """Accepting overrides may not mean accepting a denial of service."""
    from svrspec.gui import MB_AXIS_MAX_POINTS, MB_BATCHES, _mb_axis

    # Out of range on both ends, duplicated, unordered, and far too long.
    got = _mb_axis({"batches": [0, -5, 4, 4, 99999] + list(range(1, 40))},
                   "batches", MB_BATCHES)
    assert len(got) <= MB_AXIS_MAX_POINTS
    assert got == tuple(sorted(set(got)))
    assert min(got) >= 1 and max(got) <= 512

    # Junk falls back rather than raising: this runs behind a form.
    assert _mb_axis({"batches": "nonsense"}, "batches", MB_BATCHES) == MB_BATCHES
    assert _mb_axis({"batches": []}, "batches", MB_BATCHES) == MB_BATCHES
    assert _mb_axis({"batches": ["x", None]}, "batches", MB_BATCHES) == MB_BATCHES
    assert _mb_axis({}, "batches", MB_BATCHES) == MB_BATCHES


def test_the_os_profile_is_selectable_and_changes_the_memory():
    """OS is a sizing input, not a constant baked into the engine."""
    request = {**BASE_REQUEST, "model": "test-8b-gqa", "cpu": "test-amx-8ch",
               "dimm_gb": 64, "dimm_count": 8}
    axes = {"batches": [1], "contexts": [4096]}
    lean = _mb(Catalog(DATA), request, {**axes, "os_name": "linux-container"})
    fat = _mb(Catalog(DATA), request, {**axes, "os_name": "windows-desktop"})

    assert lean["os_name"] == "linux-container"
    assert lean["throughput"][0]["ram_gb"] < fat["throughput"][0]["ram_gb"]
    # Silicon is unchanged by the operating system.
    assert lean["throughput"][0]["gen_tps"] == fat["throughput"][0]["gen_tps"]

    # An unknown name falls back instead of failing the run.
    unknown = _mb(Catalog(DATA), request, {**axes, "os_name": "plan9"})
    assert unknown["os_name"] is None


def test_the_grid_shows_which_cells_cannot_run_not_just_the_payload():
    """Adding a field to the payload is not the same as showing it.

    `ram_gb` and `fits` were in the response before anything drew them, which
    is worse than not having them: the data says the bottom-right corner will
    not load and the screen still invites you to pick it.
    """
    # Selectable as a metric...
    assert '"필요 메모리(RAM)"' in SERVER_HTML

    # ...and marked in place, on every metric view, not only the RAM one.
    assert "mb-nofit" in SERVER_HTML
    assert "row.fits === false" in SERVER_HTML
    assert "장착 메모리에 들어가지 않는다" in SERVER_HTML

    # The mark carries the meaning without colour, for print and for readers
    # who cannot rely on it.
    assert "mb-nofit-mark" in SERVER_HTML
    assert "var(--error)" in SERVER_HTML


def test_the_operating_system_assumption_is_visible_where_the_memory_is():
    """A memory figure that hides which OS it assumed is not a figure."""
    payload = _mb(
        Catalog(DATA),
        {**BASE_REQUEST, "model": "test-8b-gqa", "cpu": "test-amx-8ch",
         "dimm_gb": 64, "dimm_count": 8},
        {"batches": [1], "contexts": [4096], "os_name": "linux-container"},
    )
    block = payload["os"]
    assert block["label"] and block["note"]
    assert block["hard_limit"] is True
    assert "OOM" in block["overrun"]
    assert block["chosen"] is True

    # An unchosen profile still reports what it fell back to.
    default = _mb(
        Catalog(DATA),
        {**BASE_REQUEST, "model": "test-8b-gqa", "cpu": "test-amx-8ch",
         "dimm_gb": 64, "dimm_count": 8},
        {"batches": [1], "contexts": [4096]},
    )
    assert default["os"]["chosen"] is False
    assert default["os"]["label"]


def test_the_operator_can_actually_choose_the_operating_system():
    """The engine sizing per OS is worthless if the screen cannot ask for one.

    The payload accepted `os_name` for a release before any control sent it, so
    every reading silently used the default. Three things have to line up: the
    catalogue offers the list, the markup has the control, and the request
    carries the choice.
    """
    from svrspec.gui import catalog_payload
    from svrspec.memory import OS_PROFILES

    cat = catalog_payload(Catalog(DATA))
    assert [p["id"] for p in cat["os_profiles"]] == list(OS_PROFILES)
    assert cat["os_default"] in OS_PROFILES
    # The list has to carry what distinguishes the profiles, not just names --
    # a container that gets OOM-killed is a different purchase decision.
    assert any(p["hard_limit"] for p in cat["os_profiles"])
    assert all(p["runtime_gb"] > 0 for p in cat["os_profiles"])

    assert 'id="mb-os"' in SERVER_HTML
    assert 'p.os_name = labVal("mb-os")' in SERVER_HTML
    assert 'fillSelect(byId("mb-os")' in SERVER_HTML


def test_the_operating_system_moves_the_memory_the_grid_reports():
    """Choosing an OS has to change the numbers, or the control is decoration."""
    def low(os_name):
        payload = _mb(
            Catalog(DATA),
            {**BASE_REQUEST, "model": "test-8b-gqa", "cpu": "test-amx-8ch",
             "dimm_gb": 64, "dimm_count": 8},
            {"batches": [1], "contexts": [4096], "os_name": os_name},
        )
        return payload["throughput"][0]["ram_gb"]

    lean, fat = low("linux-container"), low("windows-desktop")
    assert fat > lean, f"windows-desktop should need more than a container: {fat} vs {lean}"


def test_the_tool_carries_its_own_evidence_instead_of_asking_for_a_log():
    """It sizes servers nobody can touch, so "go measure it" is not an answer.

    Asking the operator to run llama-bench on the hardware is only possible for
    hardware they have. The whole premise here is that they do not have it yet
    -- that is why they are sizing it. So every coefficient has to ship with
    its own source, and the page has to show them.
    """
    from svrspec.gui import catalog_payload

    payload = catalog_payload(Catalog(DATA))
    ev = payload["evidence"]
    assert ev, "예측 계수가 화면에 하나도 노출되지 않는다"
    for row in ev:
        assert row["kind"] and row["confidence"]
        assert set(row) >= {"value", "source", "source_url", "notes"}

    # The panel and its renderer have to exist, not just the payload.
    assert 'id="mb-evidence"' in SERVER_HTML
    assert "renderEvidence" in SERVER_HTML
    assert "이 숫자들의 근거" in SERVER_HTML


def test_the_shipped_catalogue_has_no_sourceless_rows():
    """A row with no source is a number the reader cannot check or challenge.

    They were all memory rows -- DDR4 grades and 2DPC derates -- and every one
    of them is published platform data, so carrying them unsourced was a gap in
    this catalogue rather than a limit on what is knowable.
    """
    from svrspec.catalog import Catalog as RealCatalog

    left = RealCatalog().unverified()
    assert left == [], f"출처 없는 행이 남아 있다: {left}"


def test_the_page_does_not_send_the_operator_off_to_find_data():
    """Handing over a research task is the tool declining to answer."""
    assert "벤더 데이터시트로 대조" not in SERVER_HTML
    assert "출처 없는 스펙" in SERVER_HTML


def test_no_two_elements_share_an_id():
    """A duplicate id fails silently and looks like the whole page broke.

    `<main id="model">` and `<select id="model">` both existed, so
    getElementById("model") returned the <main> -- the boot code poured 32
    <optgroup>s into the screen container, the LLM dropdown stayed empty, and
    the option groups rendered as loose text over the layout. Nothing threw.
    Reported as "항목별 UI 전부 깨지고 정상작동 안함", which is exactly right.
    """
    import collections
    import re

    for mode in ("server", "desktop"):
        ids = re.findall(r"""\sid=["']([^"']+)["']""", app_html(mode))
        dupes = {i: n for i, n in collections.Counter(ids).items() if n > 1}
        assert not dupes, f"{mode}: 중복 id {dupes}"


def test_every_id_the_script_looks_up_actually_exists():
    """Renaming a container must not leave the code reaching for the old name."""
    import re

    page = SERVER_HTML
    present = set(re.findall(r"""\sid=["']([^"']+)["']""", page))
    # Ids built by string concatenation ("view-" + v, "screen-" + v) are not
    # literals, so check the pieces the loop can produce.
    for v in ("model", "size", "lab"):
        assert f"screen-{v}" in present, f"screen-{v} 가 없다"
        assert f"view-{v}" in present, f"view-{v} 가 없다"


def test_the_model_dropdown_groups_are_named_not_keyed():
    """`sub-2B` is a catalogue key; a dropdown needs a caption."""
    assert "SIZE_CLASS_LABEL" in SERVER_HTML
    assert "2B 미만" in SERVER_HTML
    assert "70B 이상" in SERVER_HTML


def test_the_rail_is_legible_before_anything_loads():
    """An empty form reads as a broken window, not a slow one.

    Every number in the rail used to be written by script after the catalogue
    arrived. With a dead desktop bridge that is a full timeout of a form with
    nothing in it -- which is what the screenshot of "UI 깨짐" actually showed.
    Markup defaults cost nothing and are there before the first byte of JSON.
    """
    import re

    for key, want in DEFAULTS.items():
        m = re.search(rf'<input type="number" id="{key}"[^>]*>', SERVER_HTML)
        if m is None:          # not every default is a rail input
            continue
        assert f'value="{want}"' in m.group(0), f"{key}: 마크업 기본값 없음 또는 불일치"

    # And the script must not carry a second copy that can drift from it.
    assert "storm_window_s:30" not in SERVER_HTML.replace(" ", "")


def test_the_two_column_rows_cannot_be_knocked_out_of_line():
    """Per-child margins tilted every paired row by 8px.

    Only one field in a `.row` matched `.field:first-of-type`, so it sat higher
    than its neighbour. Spacing belongs to the container.
    """
    assert ".field:first-of-type" not in SERVER_HTML
    assert ".field{display:flex; flex-direction:column; gap:var(--s1); margin:0" in SERVER_HTML
    # One control height, not two: number inputs and selects sat 4px apart.
    assert "select{min-height:40px}" not in SERVER_HTML


def test_the_interface_speaks_politely():
    """It is a tool people are asked to trust, not a machine issuing orders."""
    import re

    for mode in ("server", "desktop"):
        page = app_html(mode)
        # Strip script comments, which are English but can quote old wording.
        # The ending must end a word: "부하라면" contains 하라 and is innocent.
        rude = re.findall(r"[가-힣][^\"'\n<>]{0,40}?(?:해라|하라|기다려라|열어라|봐라)(?![가-힣])", page)
        assert not rude, f"{mode}: 명령형 반말 {rude[:5]}"


def test_the_bridge_wait_is_short_enough_to_sit_through():
    """A working bridge answers in milliseconds; a long ceiling only delays the
    admission of failure, and that wait is the whole of the reported hang."""
    import re

    from svrspec import desktop as dt

    ms = int(re.search(r"var BRIDGE_TIMEOUT_MS = (\d+)", SERVER_HTML).group(1))
    assert ms <= 6000, f"브리지 대기 {ms}ms 는 너무 길다"
    # Python must give up just after the page does, so the rescue follows the
    # message instead of leaving an error with nothing behind it.
    assert ms / 1000 < dt.BRIDGE_GRACE_S <= ms / 1000 + 3


# --------------------------------------------------------------------------
# The hardware sweep: the load-test screen's buying question
# --------------------------------------------------------------------------


def test_the_load_test_screen_leads_with_hardware_not_alarms():
    """The alarm replay was demoted, not deleted, and the order says so.

    "알람 건수 기능은 별로 필요없다" was answered by changing what the screen
    leads with: the buying question (what does each box top out at) is first,
    and the measured-day replay stays underneath for the delivery report --
    the one place a specific alarm count really is the question.
    """
    lead = SERVER_HTML.index("하드웨어별 성능 한계")
    replay = SERVER_HTML.index("알람 부하 재생")
    assert lead < replay, "the alarm replay must not lead this screen"
    # And the sweep says plainly that it does not depend on an alarm count.
    assert "알람 건수와 무관" in SERVER_HTML


def test_the_sweep_states_why_channels_are_populated_per_cpu():
    """A hardware comparison that half-populates the wide parts is not one.

    The catalogue mixes 6-, 8- and 12-channel CPUs, so a fixed DIMM count
    would hand the wide parts a crippled memory subsystem. The screen has to
    say it is filling per channel, because otherwise the ranking looks wrong
    to anybody who knows the parts.
    """
    assert "채널당 1개씩" in SERVER_HTML
    assert "6·8·12" in SERVER_HTML


def test_the_sweep_explains_when_cores_beat_bandwidth():
    """The one sentence on this screen that changes what somebody buys."""
    assert "대역폭보다 코어 수가 동시 한계를 정한다" in SERVER_HTML
    assert "프롬프트는 동시 사용자가" in SERVER_HTML


def test_the_desktop_page_degrades_when_the_bridge_has_no_hwsweep_call():
    """Same fixed-surface rule as every other added bridge call."""
    assert "window.pywebview.api.hwsweep" in DESKTOP_HTML


def test_hwsweep_payload_answers_and_echoes_its_setup(cat):
    from svrspec.gui import _params, hwsweep_payload

    p = _params({"model": "test-3b", "quant": "Q4_K_M"})
    d = hwsweep_payload(cat, p, {"ctx_tokens": 2048, "output_tokens": 128,
                                 "target_s": 20, "sockets": 1})
    assert d["rows"], "no CPU was ranked"
    # The rows are meaningless without what was asked to produce them.
    assert d["ctx_tokens"] == 2048 and d["output_tokens"] == 128
    assert d["target_s"] == 20
    row = d["rows"][0]
    for key in ("cpu", "cores", "channels", "max_users", "limited_by", "gen_tps"):
        assert key in row, key
    # Must survive strict JSON: the engine can hand back an infinity.
    json.dumps(d, allow_nan=False)


def test_hwsweep_payload_clamps_a_hostile_setup(cat):
    """A form field may not ask the server for an unbounded search."""
    from svrspec.gui import HW_LIMITS, _params, hwsweep_payload

    p = _params({"model": "test-3b", "quant": "Q4_K_M"})
    d = hwsweep_payload(cat, p, {"ctx_tokens": 10 ** 9, "output_tokens": -5,
                                 "target_s": 10 ** 6, "sockets": 999})
    assert d["ctx_tokens"] == HW_LIMITS["ctx_tokens"][1]
    assert d["output_tokens"] == HW_LIMITS["output_tokens"][0]
    assert d["target_s"] == HW_LIMITS["target_s"][1]
    assert d["sockets"] == HW_LIMITS["sockets"][1]


def test_hwsweep_payload_reports_a_bad_model_instead_of_raising(cat):
    from svrspec.gui import _params, hwsweep_payload

    p = _params({"model": "no-such-model", "quant": "Q4_K_M"})
    d = hwsweep_payload(cat, p, {})
    assert d.get("error")
