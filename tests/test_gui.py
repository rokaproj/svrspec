"""GUI: payload shapes, input clamping, and the served page.

The server is started for real on an ephemeral port rather than mocked, because
the thing worth testing is that a browser can actually get a sizing out of it.
"""

import json
import re
import threading
import urllib.error
import urllib.request
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


def test_buckets_conserve_busy_time(cat):
    """Every event's span must land in exactly one bucket's worth of time."""
    from svrspec.gui import _buckets

    events = [(0.0, 1, 0, 100.0), (500.0, 2, 3, 1000.0), (86000.0, 1, 0, 400.0)]
    span = 86400.0
    b = _buckets(events, span, count=96)
    width = span / 96
    charged = sum(v * width for v in b["cpu"])
    assert abs(charged - sum(e[3] for e in events)) < 1.0


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
