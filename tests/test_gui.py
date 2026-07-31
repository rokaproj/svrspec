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
    assert "error" in api.size("not json at all" if False else {"model": "nope"})
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
    assert "데스크톱 빌드에는 과부하 분석이 연결되어 있지 않다" in DESKTOP_HTML


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


def _install_engines(monkeypatch) -> None:
    """Real engines where they exist, stand-ins where they do not."""
    for name, build in (("svrspec.lab", _stub_lab),
                        ("svrspec.loadgen", _stub_loadgen),
                        ("svrspec.bench", _stub_bench)):
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
    assert "데스크톱 빌드에는 가상 랩이 연결되어 있지 않다" in DESKTOP_HTML


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
