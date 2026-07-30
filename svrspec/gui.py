"""Local web GUI, served by the standard library.

Why a browser and not a desktop toolkit: tkinter is not installed here and would
add a system package to a tool whose whole point is running unmodified on an
air-gapped server. `http.server` ships with Python, so the GUI costs nothing in
dependencies and inherits the design tokens the delivery report already uses.

A full 28-CPU sweep takes about 45 ms, so the UI recomputes on every keystroke
and slider move. That is what makes it a simulator rather than a report
generator: you change the alarm volume and watch the recommendation move.

Binds to localhost by default. It executes no user input -- requests only pick
catalogue ids and numeric workload fields, and everything is validated against
the catalogue before use.
"""

from __future__ import annotations

import json
import threading
import webbrowser
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from . import report
from .catalog import Catalog, CatalogError
from .memory import kv_bytes_per_token
from .perf import Efficiency, widest_isa
from .sizing import sweep_cpus, tiers
from .theme import stylesheet
from .types import TokenProfile, Workload

# --------------------------------------------------------------------------
# Request -> Workload
# --------------------------------------------------------------------------

#: Numeric workload fields the client may set, with sane bounds. Anything
#: outside the range is clamped rather than trusted.
LIMITS = {
    "alarms_per_day": (1, 100_000),
    "storm_size": (0, 5_000),
    "storm_window_s": (1, 3_600),
    "storms_per_day": (0, 100),
    "slots": (1, 64),
    "sla_seconds": (1, 3_600),
    "storm_drain_min": (1, 1_440),
    "system_tokens": (0, 32_000),
    "fewshot_tokens": (0, 32_000),
    "alarm_tokens": (1, 32_000),
    "output_tokens": (1, 32_000),
    "sockets": (1, 8),
    "dpc": (1, 2),
}

DEFAULTS = {
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
    "sockets": 1,
    "dpc": 1,
}


def _number(raw: dict, key: str) -> int:
    low, high = LIMITS[key]
    try:
        value = int(float(raw.get(key, DEFAULTS[key])))
    except (TypeError, ValueError):
        value = DEFAULTS[key]
    return max(low, min(high, value))


def _params(raw: dict) -> dict:
    out = {k: _number(raw, k) for k in LIMITS}
    out["prompt_cache"] = bool(raw.get("prompt_cache", True))
    out["only_pass"] = bool(raw.get("only_pass", False))
    out["model"] = str(raw.get("model", ""))
    out["quant"] = str(raw.get("quant", "Q4_K_M"))
    return out


def _workload(p: dict) -> Workload:
    return Workload(
        alarms_per_day=p["alarms_per_day"],
        storm_size=p["storm_size"],
        storm_window_s=float(p["storm_window_s"]),
        storms_per_day=p["storms_per_day"],
        slots=p["slots"],
        sla_seconds=float(p["sla_seconds"]),
        storm_drain_sla_s=p["storm_drain_min"] * 60.0,
        tokens=TokenProfile(
            system_tokens=p["system_tokens"],
            fewshot_tokens=p["fewshot_tokens"],
            alarm_tokens=p["alarm_tokens"],
            output_tokens=p["output_tokens"],
            prompt_cache=p["prompt_cache"],
        ),
    )


# --------------------------------------------------------------------------
# Payloads
# --------------------------------------------------------------------------


def catalog_payload(cat: Catalog) -> dict:
    return {
        "models": [
            {
                "id": m.id,
                "name": m.name,
                "family": m.family,
                "params_b": round(m.params_b, 2),
                "active_params_b": round(m.active_params_b, 2) if m.active_params_b else None,
                "size_class": m.size_class,
                "korean": m.korean,
                "kv_kib": round(kv_bytes_per_token(m) / 1024, 1),
                "n_head": m.n_head,
                "n_kv_head": m.n_kv_head,
                "ctx_train": m.ctx_train,
            }
            for m in sorted(cat.models, key=lambda m: m.params_b)
        ],
        "quants": [
            {"id": q.id, "bpw": q.bits_per_weight, "quality": q.quality} for q in cat.quants
        ],
        "counts": {
            "models": len(cat.models),
            "cpus": len(cat.cpus),
            "memory": len(cat.memory),
            "coefficients": len(cat.coefficients),
        },
    }


def size_payload(cat: Catalog, p: dict) -> dict:
    model = cat.model(p["model"]) if p["model"] else cat.models[0]
    quant = cat.quant(p["quant"])
    workload = _workload(p)
    eff = Efficiency.from_catalog(cat.coefficients)

    candidates = sweep_cpus(
        cat, model, quant, workload, sockets=p["sockets"], dimms_per_channel=p["dpc"]
    )
    shown = [c for c in candidates if c.verdict != "fail"] if p["only_pass"] else candidates
    tier_map = tiers(candidates)

    def tier(key: str) -> dict | None:
        c = tier_map.get(key)
        if not c:
            return None
        sim = c.sim_pessimistic or c.sim
        return {
            "cpu": f"{c.cpu.vendor} {c.cpu.model}",
            "cores": c.cpu.cores * c.sockets,
            "sockets": c.sockets,
            "memory_gb": c.memory_gb,
            "memory": f"{c.memory.ddr_gen}-{c.memory.effective_mts}",
            "channels": c.cpu.mem_channels * c.sockets,
            "headroom": round(c.headroom, 1),
            "p95_steady": round(sim.p95_steady_s, 1),
            "storm_min": round(sim.storm_drain_s / 60, 1),
            "verdict": c.verdict,
            "tdp_w": c.cpu.tdp_w * c.sockets,
            "price_usd": (c.cpu.price_usd * c.sockets) if c.cpu.price_usd else None,
        }

    rows = []
    for c in shown:
        sim = c.sim_pessimistic or c.sim
        rows.append(
            {
                "id": c.cpu.id,
                "vendor": c.cpu.vendor,
                "model": c.cpu.model,
                "family": c.cpu.family,
                "cores": c.cpu.cores * c.sockets,
                "ghz": c.cpu.all_core_turbo_ghz,
                "isa": widest_isa(c.cpu),
                "memory": f"{c.cpu.mem_channels * c.sockets}ch {c.memory.ddr_gen}-{c.memory.effective_mts}",
                "bandwidth": round(c.throughput.effective_bandwidth_gbs),
                "bound": c.throughput.decode_bound_by,
                "prefill": round(c.throughput.prefill_tps),
                "decode": round(c.throughput.decode_tps_single, 1),
                "latency": round(c.latency.total_s, 1),
                "p95_steady": round(sim.p95_steady_s, 1),
                "storm_min": round(sim.storm_drain_s / 60, 1),
                "ram_gb": c.memory_gb,
                "uncertainty": round(c.throughput.uncertainty * 100),
                "verdict": c.verdict,
                "reasons": c.reasons,
                "price_usd": (c.cpu.price_usd * c.sockets) if c.cpu.price_usd else None,
                "tdp_w": c.cpu.tdp_w * c.sockets,
            }
        )

    used = report._used_coefficients(eff, candidates)
    return {
        "model": {
            "id": model.id,
            "name": model.name,
            "params_b": round(model.params_b, 2),
            "active_params_b": round(model.active_params_b, 2) if model.active_params_b else None,
            "kv_kib": round(kv_bytes_per_token(model) / 1024, 1),
            "quant": quant.id,
            "bpw": quant.bits_per_weight,
        },
        "workload": {
            "prefill_tokens": workload.tokens.prefill_tokens,
            "billed_prefill": workload.tokens.billed_prefill_tokens,
            "ctx": workload.tokens.peak_ctx_tokens,
        },
        "tiers": {k: tier(k) for k in ("minimum", "recommended", "comfortable")},
        "candidates": rows,
        "total": len(candidates),
        "passing": sum(1 for c in candidates if c.verdict == "pass"),
        "coefficients": [
            {
                "id": c.id,
                "kind": c.kind,
                "key": c.key,
                "value": c.value,
                "confidence": c.confidence,
                "label": report.CONFIDENCE_LABEL.get(c.confidence, c.confidence),
                "notes": c.notes,
                "source_url": c.source_url,
            }
            for c in used
        ],
        "warnings": report._collect_warnings(candidates),
        "unverified": [{"kind": k, "id": i} for k, i in cat.unverified()],
    }


#: Alarm volumes the resource view compares side by side.
DEFAULT_VOLUMES = (100, 200, 300)


def resource_payload(cat: Catalog, p: dict, cpu_id: str, volumes=DEFAULT_VOLUMES) -> dict:
    """One fixed hardware build, several alarm volumes.

    Two questions answered together. First, what does a single alarm actually
    look like in time -- how long until the first token, how long the generation
    takes at the predicted rate, how long the delivery adds. Second, what load
    does the box carry as volume grows: the task-manager view.

    The per-alarm timeline is the no-queueing case, so it is the floor. The
    resource rows come from the full simulated day, where queueing is included.
    """
    from .sizing import evaluate

    model = cat.model(p["model"]) if p["model"] else cat.models[0]
    quant = cat.quant(p["quant"])
    cpu = cat.cpu(cpu_id)
    memory = cat.memory_for(cpu, p["dpc"])
    eff = Efficiency.from_catalog(cat.coefficients)
    sockets = min(p["sockets"], cpu.sockets_max)

    base = _workload(p)
    tokens = base.tokens

    rows = []
    for volume in volumes:
        c = evaluate(model, quant, cpu, memory, eff, replace(base, alarms_per_day=volume), sockets)
        sim = c.sim_pessimistic or c.sim
        installed = c.memory_gb
        return_bw = c.throughput.effective_bandwidth_gbs
        rows.append(
            {
                "alarms": volume,
                # A task manager's CPU graph: share of the day the box is working.
                "cpu_pct": round(sim.busy_fraction * 100, 1),
                "slot_pct": round(sim.slot_utilisation * 100, 1),
                "ram_used_gb": round(c.ram.subtotal_gb, 1),
                "ram_installed_gb": installed,
                "ram_pct": round(min(c.ram.subtotal_gb / installed, 1.0) * 100, 1) if installed else 0,
                "bandwidth_gbs": round(return_bw, 1),
                "bandwidth_avg_gbs": round(return_bw * sim.busy_fraction, 1),
                "max_queue": sim.max_queue_depth,
                "p95_steady": round(sim.p95_steady_s, 1),
                "storm_min": round(sim.storm_drain_s / 60, 1),
                "verdict": c.verdict,
                "reasons": c.reasons,
                # Server-hours of work the day contains, for a sanity read.
                "work_minutes": round(sim.busy_fraction * 24 * 60, 1),
            }
        )

    reference = evaluate(model, quant, cpu, memory, eff, base, sockets)
    t = reference.throughput
    lat = reference.latency
    stages = [
        {
            "name": "프롬프트 처리",
            "seconds": round(lat.ttft_s, 3),
            "tokens": tokens.billed_prefill_tokens,
            "tps": round(t.prefill_tps, 1),
            "note": "prefill · 연산 바운드" if t.prefill_bound_by == "compute" else "prefill",
        },
        {
            "name": "토큰 생성",
            "seconds": round(lat.generate_s, 3),
            "tokens": tokens.output_tokens,
            "tps": round(t.decode_tps_single, 2),
            "note": "decode · 대역폭 바운드"
            if t.decode_bound_by in ("bandwidth", "core-bandwidth")
            else "decode · 연산 바운드",
        },
        {
            "name": "전송",
            "seconds": round(lat.teams_s, 3),
            "tokens": 0,
            "tps": 0,
            "note": "네트워크 왕복",
        },
    ]

    return {
        "hardware": {
            "id": cpu.id,
            "label": f"{cpu.vendor} {cpu.model}",
            "family": cpu.family,
            "cores": cpu.cores * sockets,
            "threads": cpu.threads * sockets,
            "ghz": cpu.all_core_turbo_ghz,
            "sockets": sockets,
            "isa": widest_isa(cpu),
            "memory": f"{cpu.mem_channels * sockets}ch {memory.ddr_gen}-{memory.effective_mts}",
            "bandwidth_gbs": round(t.effective_bandwidth_gbs, 1),
            "tdp_w": cpu.tdp_w * sockets,
            "ram_installed_gb": reference.memory_gb,
            "slots": base.slots,
        },
        "timeline": {
            "stages": stages,
            "total_s": round(lat.total_s, 3),
            "prefill_tps": round(t.prefill_tps, 1),
            "decode_tps": round(t.decode_tps_single, 2),
            "output_tokens": tokens.output_tokens,
            "uncertainty": round(t.uncertainty * 100),
        },
        "ram_breakdown": {
            "weights_gb": round(reference.ram.weights_gb, 2),
            "kv_gb": round(reference.ram.kv_cache_gb, 2),
            "compute_gb": round(reference.ram.compute_buffer_gb, 2),
            "os_gb": round(reference.ram.runtime_os_gb, 2),
            "subtotal_gb": round(reference.ram.subtotal_gb, 2),
            "installed_gb": reference.memory_gb,
        },
        "rows": rows,
    }


def update_payload() -> dict:
    """Whether a newer release exists. Never raises, never blocks for long."""
    from . import __version__, update

    release = update.check()
    if release is None:
        return {"available": False, "current": __version__}
    return {
        "available": True,
        "current": __version__,
        "version": release.version,
        "tag": release.tag,
        "page_url": release.page_url,
        "notes": release.notes[:600],
        "installer": release.installer_name,
    }


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------


class _Handler(BaseHTTPRequestHandler):
    server_version = "svrspec"
    catalog: Catalog

    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        # The default handler prints a line per request to stderr, which buries
        # the startup banner the operator actually needs to read.
        pass

    # -- helpers ---------------------------------------------------------
    def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: dict, code: int = 200) -> None:
        self._send(code, json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _error(self, message: str, code: int = 400) -> None:
        self._json({"error": message}, code)

    # -- routes ----------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        route = urlparse(self.path)
        if route.path in ("/", "/index.html"):
            self._send(200, app_html("server").encode("utf-8"), "text/html; charset=utf-8")
        elif route.path == "/api/catalog":
            self._json(catalog_payload(self.catalog))
        elif route.path == "/api/update":
            self._json(update_payload())
        elif route.path in ("/api/report.html", "/api/report.csv"):
            self._download(route)
        else:
            self._error("not found", 404)

    def do_POST(self) -> None:  # noqa: N802
        route = urlparse(self.path).path
        if route not in ("/api/size", "/api/resources"):
            self._error("not found", 404)
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            raw = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            self._error("malformed request body")
            return
        if not isinstance(raw, dict):
            self._error("expected a JSON object")
            return
        try:
            if route == "/api/resources":
                volumes = raw.get("volumes") or list(DEFAULT_VOLUMES)
                volumes = [
                    max(1, min(100_000, int(v))) for v in volumes[:6] if str(v).strip()
                ] or list(DEFAULT_VOLUMES)
                self._json(resource_payload(
                    self.catalog, _params(raw), str(raw.get("cpu", "")), tuple(volumes)
                ))
            else:
                self._json(size_payload(self.catalog, _params(raw)))
        except (CatalogError, ValueError, TypeError) as exc:
            self._error(str(exc))

    def _download(self, route) -> None:
        query = {k: v[0] for k, v in parse_qs(route.query).items()}
        query["prompt_cache"] = query.get("prompt_cache", "1") not in ("0", "false")
        p = _params(query)
        try:
            model = self.catalog.model(p["model"])
            quant = self.catalog.quant(p["quant"])
        except CatalogError as exc:
            self._error(str(exc))
            return

        workload = _workload(p)
        eff = Efficiency.from_catalog(self.catalog.coefficients)
        candidates = sweep_cpus(
            self.catalog, model, quant, workload,
            sockets=p["sockets"], dimms_per_channel=p["dpc"],
        )
        stem = f"svrspec-{model.id}-{quant.id}"

        if route.path.endswith(".csv"):
            # write_csv owns the column order, so round-trip through a temp file
            # rather than duplicating that list here and letting the two drift.
            import tempfile
            from pathlib import Path

            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "r.csv"
                report.write_csv(candidates, path)
                body = path.read_bytes()
            self._send(200, body, "text/csv; charset=utf-8",
                       {"Content-Disposition": f'attachment; filename="{stem}.csv"'})
            return

        html = report.render_html(
            candidates, tiers(candidates), workload, eff, self.catalog.unverified(),
            title=f"{model.name} 서버 스펙 산정 리포트",
        )
        self._send(200, html.encode("utf-8"), "text/html; charset=utf-8",
                   {"Content-Disposition": f'attachment; filename="{stem}.html"'})


def serve(
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
    catalog: Catalog | None = None,
) -> None:
    handler = type("Handler", (_Handler,), {"catalog": catalog or Catalog()})
    httpd = ThreadingHTTPServer((host, port), handler)
    url = f"http://{host}:{port}/"

    cat = handler.catalog
    print(f"svrspec GUI  {url}")
    print(f"  카탈로그 {len(cat.models)} 모델 · {len(cat.cpus)} CPU · "
          f"{len(cat.memory)} 메모리 · {len(cat.coefficients)} 계수")
    print("  Ctrl+C 로 종료")

    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n종료")
    finally:
        httpd.server_close()


# --------------------------------------------------------------------------
# The page
# --------------------------------------------------------------------------

_GUI_CSS = """
body{min-height:100vh}
header{
  display:flex; align-items:baseline; gap:var(--s3);
  padding:var(--s4) var(--s5); border-bottom:1px solid var(--border);
  background:var(--bg-secondary); position:sticky; top:0; z-index:10;
}
header h1{font-size:var(--fs-md)}
header .tag{font-size:var(--fs-xs); color:var(--text-tertiary)}
header .spacer{flex:1}
main{
  display:grid; grid-template-columns:320px minmax(0,1fr);
  gap:var(--s5); max-width:1440px; margin:0 auto;
  padding:var(--s5) var(--s5) var(--s7);
  align-items:start;
}
@media (max-width:960px){
  main{grid-template-columns:minmax(0,1fr)}
  #rail{position:static}
}
#rail{position:sticky; top:76px; display:flex; flex-direction:column; gap:var(--s3)}
fieldset{
  border:1px solid var(--border); border-radius:var(--radius);
  background:var(--bg-secondary); padding:var(--s3) var(--s4) var(--s4); margin:0;
}
legend{
  font-size:var(--fs-xs); font-weight:600; letter-spacing:0.06em;
  text-transform:uppercase; color:var(--text-tertiary); padding:0 var(--s1);
}
.field{display:flex; flex-direction:column; gap:var(--s1); margin-top:var(--s3)}
.field:first-of-type{margin-top:var(--s1)}
.field label{font-size:var(--fs-sm); color:var(--text-secondary)}
.row{display:grid; grid-template-columns:1fr 1fr; gap:var(--s3)}
input[type=number],select{
  width:100%; min-height:36px; padding:var(--s1) var(--s2);
  font:400 var(--fs-sm) var(--font); color:var(--text-primary);
  background:var(--bg-tertiary); border:1px solid var(--border);
  border-radius:var(--radius-sm);
}
select{min-height:40px}
input[type=number]:hover,select:hover{border-color:var(--text-tertiary)}
.check{display:flex; align-items:center; gap:var(--s2); margin-top:var(--s3);
  font-size:var(--fs-sm); color:var(--text-secondary); min-height:24px}
.check input{width:16px; height:16px; accent-color:var(--accent)}
.hint{font-size:var(--fs-xs); color:var(--text-tertiary)}
button{
  font:600 var(--fs-sm) var(--font); color:var(--text-primary);
  background:var(--bg-tertiary); border:1px solid var(--border);
  border-radius:var(--radius-sm); padding:var(--s2) var(--s3);
  min-height:36px; cursor:pointer;
}
button:hover{border-color:var(--text-tertiary)}
.actions{display:flex; gap:var(--s2)}
.actions a{
  flex:1; text-align:center; font:600 var(--fs-sm) var(--font);
  color:var(--text-primary); background:var(--bg-tertiary);
  border:1px solid var(--border); border-radius:var(--radius-sm);
  padding:var(--s2) var(--s3); min-height:44px;
  display:flex; align-items:center; justify-content:center;
  text-decoration:none;
}
.actions a:hover{border-color:var(--accent); color:var(--accent); text-decoration:none}

#results{display:flex; flex-direction:column; gap:var(--s5); min-width:0}
#results.stale{opacity:0.55}
@media (prefers-reduced-motion:no-preference){
  #results{transition:opacity 120ms ease-out}
}
.tiers{display:grid; gap:var(--s3);
  grid-template-columns:repeat(auto-fit,minmax(240px,1fr))}
/* The one deliberate flourish: the headroom multiple, set large. It is the
   number somebody opened this tool to find. */
.tier .multiple{
  font-size:var(--fs-xl); font-weight:600; line-height:1.15;
  letter-spacing:-0.02em; margin:var(--s2) 0 0;
}
.tier .cpu{font-size:var(--fs-md); font-weight:600; margin-top:var(--s2)}
.tier .detail{font-size:var(--fs-xs); color:var(--text-secondary);
  margin-top:var(--s2); line-height:1.55}
.tier.recommended{border-left:3px solid var(--accent)}
.tier.empty .multiple{color:var(--text-tertiary); font-size:var(--fs-lg)}
.section-head{display:flex; align-items:baseline; gap:var(--s3); margin-bottom:var(--s2)}
.section-head h2{font-size:var(--fs-md)}
.section-head .spacer{flex:1}
.bound{font-size:var(--fs-xs); color:var(--text-tertiary)}
tbody tr:hover{background:var(--bg-tertiary)}
tbody tr.sel{background:var(--selected-bg);
  box-shadow:inset 3px 0 0 var(--selected-border)}
.why{font-size:var(--fs-sm); color:var(--text-secondary)}
.why li{margin:var(--s1) 0}
#empty{padding:var(--s5); text-align:center; color:var(--text-secondary)}

/* Token delivery timeline — the "how fast does this feel" view. */
.tl{display:flex; height:34px; border-radius:var(--radius-sm);
  overflow:hidden; border:1px solid var(--border); background:var(--bg-tertiary)}
.tl div{display:flex; align-items:center; justify-content:center;
  font-size:var(--fs-xs); font-weight:600; color:var(--on-fill); min-width:2px;
  white-space:nowrap; overflow:hidden}
.tl .s-prefill{background:var(--accent)}
.tl .s-decode{background:var(--success)}
.tl .s-send{background:var(--text-secondary)}
.tl-wrap{position:relative}
.tl-head{position:absolute; top:-4px; bottom:-4px; width:2px;
  background:var(--text-primary); left:0; display:none}
.tl-legend{display:flex; gap:var(--s4); flex-wrap:wrap;
  margin-top:var(--s2); font-size:var(--fs-xs); color:var(--text-secondary)}
.tl-legend b{color:var(--text-primary); font-variant-numeric:tabular-nums}
.play{display:flex; align-items:center; gap:var(--s3); margin-top:var(--s3)}
.counter{font:600 var(--fs-lg)/1 var(--mono); font-variant-numeric:tabular-nums}
.counter small{font:400 var(--fs-xs) var(--font); color:var(--text-tertiary)}
.stream{margin-top:var(--s3); padding:var(--s3) var(--s4);
  border:1px solid var(--border); border-radius:var(--radius-sm);
  background:var(--bg-tertiary); font:var(--fs-sm)/1.7 var(--font);
  color:var(--text-secondary); min-height:5.2em; max-height:5.2em; overflow:hidden}

/* Task-manager meters. */
.meter{position:relative; height:16px; min-width:110px;
  border-radius:3px; background:var(--bg-tertiary);
  border:1px solid var(--border); overflow:hidden}
.meter span{position:absolute; inset:0 auto 0 0; background:var(--accent)}
.meter.warn span{background:var(--warning)}
.meter.hot span{background:var(--error)}
.meter b{position:absolute; inset:0; display:flex; align-items:center;
  justify-content:center; font-size:10px; font-weight:600;
  color:var(--text-primary); font-variant-numeric:tabular-nums}
.hw{display:flex; flex-wrap:wrap; gap:var(--s2) var(--s4);
  font-size:var(--fs-xs); color:var(--text-secondary); margin-top:var(--s2)}
.hint-row{font-size:var(--fs-xs); color:var(--text-tertiary); margin-top:var(--s2)}
"""

def app_html(mode: str = "server") -> str:
    """The page, wired for one of two transports.

    "server"  fetch() against this module's HTTP routes
    "desktop" window.pywebview.api, so the packaged app needs no socket at all
    """
    return (
        "<!doctype html>\n<html lang='ko'>\n<head>\n<meta charset='utf-8'>\n"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>\n"
        "<title>svrspec — GPU 서빙 서버 스펙 산정</title>\n"
        f"<style>{stylesheet(_GUI_CSS)}</style>\n</head>\n"
        f"<body data-mode='{mode}'>\n"
        """
<header>
  <h1>svrspec</h1>
  <span class="tag">GPU 서빙 서버 스펙 산정</span>
  <span class="spacer"></span>
  <button id="update" type="button" hidden></button>
  <button id="theme" type="button" aria-label="화면 테마 전환">테마: 자동</button>
</header>

<main>
  <form id="rail" aria-label="산정 조건">
    <fieldset>
      <legend>모델</legend>
      <div class="field">
        <label for="model">LLM</label>
        <select id="model" name="model"></select>
        <span class="hint" id="model-hint"></span>
      </div>
      <div class="field">
        <label for="quant">양자화</label>
        <select id="quant" name="quant"></select>
      </div>
    </fieldset>

    <fieldset>
      <legend>알람 부하</legend>
      <div class="field">
        <label for="alarms_per_day">개수</label>
        <input type="number" id="alarms_per_day" min="1" max="100000" step="10">
      </div>
      <div class="row">
        <div class="field">
          <label for="storm_size">스톰 크기</label>
          <input type="number" id="storm_size" min="0" max="5000">
        </div>
        <div class="field">
          <label for="storm_window_s">스톰 창(초)</label>
          <input type="number" id="storm_window_s" min="1" max="3600">
        </div>
      </div>
      <div class="row">
        <div class="field">
          <label for="storms_per_day">스톰 횟수/일</label>
          <input type="number" id="storms_per_day" min="0" max="100">
        </div>
        <div class="field">
          <label for="slots">동시 슬롯</label>
          <input type="number" id="slots" min="1" max="64">
        </div>
      </div>
      <div class="row">
        <div class="field">
          <label for="sla_seconds">지연 SLA(초)</label>
          <input type="number" id="sla_seconds" min="1" max="3600">
        </div>
        <div class="field">
          <label for="storm_drain_min">스톰 소진(분)</label>
          <input type="number" id="storm_drain_min" min="1" max="1440">
        </div>
      </div>
      <span class="hint">평상시 알람은 지연 SLA로, 스톰은 소진 시간으로 판정한다.</span>
    </fieldset>

    <fieldset>
      <legend>프롬프트 토큰</legend>
      <div class="row">
        <div class="field">
          <label for="system_tokens">시스템</label>
          <input type="number" id="system_tokens" min="0" max="32000" step="50">
        </div>
        <div class="field">
          <label for="fewshot_tokens">few-shot</label>
          <input type="number" id="fewshot_tokens" min="0" max="32000" step="50">
        </div>
      </div>
      <div class="row">
        <div class="field">
          <label for="alarm_tokens">알람 원문</label>
          <input type="number" id="alarm_tokens" min="1" max="32000" step="50">
        </div>
        <div class="field">
          <label for="output_tokens">출력</label>
          <input type="number" id="output_tokens" min="1" max="32000" step="50">
        </div>
      </div>
      <div class="check">
        <input type="checkbox" id="prompt_cache" checked>
        <label for="prompt_cache">프리픽스 캐시 사용</label>
      </div>
      <span class="hint" id="token-hint"></span>
    </fieldset>

    <fieldset>
      <legend>서버 구성</legend>
      <div class="row">
        <div class="field">
          <label for="sockets">소켓</label>
          <select id="sockets"><option value="1">1</option><option value="2">2</option></select>
        </div>
        <div class="field">
          <label for="dpc">채널당 DIMM</label>
          <select id="dpc"><option value="1">1</option><option value="2">2</option></select>
        </div>
      </div>
      <div class="check">
        <input type="checkbox" id="only_pass">
        <label for="only_pass">통과 후보만 표시</label>
      </div>
      <div class="field">
        <label for="volumes">리소스 비교 개수</label>
        <input type="text" id="volumes" value="100, 200, 300" inputmode="numeric">
        <span class="hint">표의 CPU를 클릭하면 그 하드웨어의 리소스 사용량을 이 개수들로 비교한다.</span>
      </div>
    </fieldset>

    <div class="actions">
      <a id="dl-html" href="#" download>리포트 저장</a>
      <a id="dl-csv" href="#" download>CSV</a>
    </div>
    <span class="hint" id="save-status" aria-live="polite"></span>
  </form>

  <div id="results" aria-live="polite">
    <div id="empty" class="card">카탈로그를 불러오는 중…</div>
  </div>
</main>

<script>
(function(){
  "use strict";
  var NUM = ["alarms_per_day","storm_size","storm_window_s","storms_per_day","slots",
             "sla_seconds","storm_drain_min","system_tokens","fewshot_tokens",
             "alarm_tokens","output_tokens","sockets","dpc"];
  var VERDICT = {pass:"통과", marginal:"여유부족", fail:"미달"};
  var TIER_LABEL = {minimum:"최소 스펙", recommended:"권장 스펙", comfortable:"여유 스펙"};
  var BOUND = {"bandwidth":"대역폭", "core-bandwidth":"코어", "compute":"연산"};
  var root = document.documentElement, results = document.getElementById("results");
  var catalog = null, seq = 0, seqRes = 0, selectedCpu = null, player = null;

  // Two transports, one page. In the packaged desktop app there is no socket:
  // the window talks straight to Python through pywebview's bridge.
  var MODE = document.body.getAttribute("data-mode") || "server";
  var DESKTOP = MODE === "desktop";

  function bridgeReady(){
    if(!DESKTOP || (window.pywebview && window.pywebview.api)) return Promise.resolve();
    return new Promise(function(resolve){
      window.addEventListener("pywebviewready", function(){ resolve(); }, {once:true});
    });
  }
  function askCatalog(){
    return DESKTOP ? window.pywebview.api.catalog()
                   : fetch("/api/catalog").then(function(r){ return r.json(); });
  }
  function askSize(p){
    if(DESKTOP) return window.pywebview.api.size(p);
    return fetch("/api/size", {method:"POST",
                               headers:{"Content-Type":"application/json"},
                               body: JSON.stringify(p)})
      .then(function(r){ return r.json(); });
  }
  function askUpdate(){
    if(DESKTOP) return window.pywebview.api.update_check();
    return fetch("/api/update").then(function(r){ return r.json(); });
  }
  function askResources(p){
    if(DESKTOP) return window.pywebview.api.resources(p);
    return fetch("/api/resources", {method:"POST",
                                    headers:{"Content-Type":"application/json"},
                                    body: JSON.stringify(p)})
      .then(function(r){ return r.json(); });
  }

  function el(tag, cls, text){
    var n = document.createElement(tag);
    if(cls) n.className = cls;
    if(text !== undefined && text !== null) n.textContent = String(text);
    return n;
  }

  // ---- theme -------------------------------------------------------
  var THEMES = ["auto","light","dark"], NAME = {auto:"자동", light:"라이트", dark:"다크"};
  var themeBtn = document.getElementById("theme");
  function setTheme(t){
    if(t === "auto") root.removeAttribute("data-theme");
    else root.setAttribute("data-theme", t);
    themeBtn.textContent = "테마: " + NAME[t];
    try{ localStorage.setItem("svrspec-theme", t); }catch(e){}
  }
  var saved = "auto";
  try{ saved = localStorage.getItem("svrspec-theme") || "auto"; }catch(e){}
  setTheme(THEMES.indexOf(saved) < 0 ? "auto" : saved);
  themeBtn.addEventListener("click", function(){
    var next = THEMES[(THEMES.indexOf(
      root.getAttribute("data-theme") || "auto") + 1) % THEMES.length];
    setTheme(next);
  });

  // ---- params ------------------------------------------------------
  function params(){
    var p = {model: document.getElementById("model").value,
             quant: document.getElementById("quant").value,
             prompt_cache: document.getElementById("prompt_cache").checked,
             only_pass: document.getElementById("only_pass").checked};
    NUM.forEach(function(k){ p[k] = Number(document.getElementById(k).value); });
    p.cpu = selectedCpu || "";
    p.volumes = (document.getElementById("volumes").value || "")
      .split(/[,\\s]+/).map(Number).filter(function(n){ return n > 0; }).slice(0, 6);
    return p;
  }
  function refreshDownloads(p){
    if(DESKTOP) return;   // desktop saves through a native file dialog instead
    var q = new URLSearchParams();
    Object.keys(p).forEach(function(k){
      if(k === "only_pass") return;
      q.set(k, typeof p[k] === "boolean" ? (p[k] ? "1" : "0") : p[k]);
    });
    document.getElementById("dl-html").href = "/api/report.html?" + q.toString();
    document.getElementById("dl-csv").href = "/api/report.csv?" + q.toString();
  }

  function wireDesktopSaves(){
    if(!DESKTOP) return;
    var status = document.getElementById("save-status");
    [["dl-html","html"],["dl-csv","csv"]].forEach(function(pair){
      var a = document.getElementById(pair[0]);
      a.removeAttribute("download");
      a.setAttribute("href", "#");
      a.addEventListener("click", function(ev){
        ev.preventDefault();
        status.textContent = "저장 중\u2026";
        window.pywebview.api.save_report(params(), pair[1]).then(function(msg){
          status.textContent = msg || "";
        }).catch(function(err){ status.textContent = "저장 실패: " + err; });
      });
    });
  }

  // ---- render ------------------------------------------------------
  function tierCard(key, t){
    var card = el("div", "card tier" + (key === "recommended" ? " recommended" : ""));
    card.appendChild(el("div", "label", TIER_LABEL[key]));
    if(!t){
      card.classList.add("empty");
      card.appendChild(el("p", "multiple", "해당 없음"));
      card.appendChild(el("p", "detail",
        "카탈로그의 어떤 CPU도 이 조건을 만족하지 못했다."));
      return card;
    }
    card.appendChild(el("p", "cpu", t.cpu));
    var mult = el("p", "multiple", t.headroom.toFixed(1) + "\\u00d7");
    mult.className += " v-" + t.verdict;
    card.appendChild(mult);
    card.appendChild(el("p", "hint", "SLA 대비 여유 · 평상시 p95 " + t.p95_steady + "초"));
    var bits = [t.cores + "코어", t.sockets + "소켓",
                t.memory_gb + " GB " + t.memory + " (" + t.channels + "ch)",
                "스톰 " + t.storm_min + "분", t.tdp_w + "W"];
    if(t.price_usd) bits.push("$" + t.price_usd.toLocaleString());
    card.appendChild(el("p", "detail", bits.join(" · ")));
    return card;
  }

  var COLS = [
    ["CPU", function(r){ return r.vendor + " " + r.model; }, ""],
    ["코어", function(r){ return r.cores + " @ " + r.ghz + "GHz"; }, "num"],
    ["ISA", function(r){ return r.isa; }, ""],
    ["메모리", function(r){ return r.memory; }, ""],
    ["대역폭 GB/s", function(r){ return r.bandwidth; }, "num"],
    ["병목", function(r){ return BOUND[r.bound] || r.bound; }, ""],
    ["prefill t/s", function(r){ return r.prefill; }, "num"],
    ["decode t/s", function(r){ return r.decode; }, "num"],
    ["1건 지연 s", function(r){ return r.latency; }, "num"],
    ["평상시 p95 s", function(r){ return r.p95_steady; }, "num"],
    ["스톰 분", function(r){ return r.storm_min; }, "num"],
    ["RAM GB", function(r){ return r.ram_gb; }, "num"],
    ["오차", function(r){ return "\\u00b1" + r.uncertainty + "%"; }, "num"]
  ];

  function table(rows){
    var box = el("div", "scroll"), tbl = el("table");
    var thead = el("thead"), htr = el("tr");
    COLS.forEach(function(c){
      var th = el("th", c[2], c[0]); htr.appendChild(th);
    });
    htr.appendChild(el("th", "", "판정"));
    thead.appendChild(htr); tbl.appendChild(thead);

    var tbody = el("tbody");
    rows.forEach(function(r){
      var tr = el("tr");
      tr.tabIndex = 0;
      COLS.forEach(function(c){ tr.appendChild(el("td", c[2], c[1](r))); });
      var v = el("td", "v-" + r.verdict, VERDICT[r.verdict] || r.verdict);
      if(r.reasons && r.reasons.length) v.title = r.reasons.join(" / ");
      tr.appendChild(v);
      if(r.id === selectedCpu) tr.classList.add("sel");
      tr.setAttribute("role", "button");
      tr.title = (tr.title ? tr.title + " · " : "") + "클릭하면 리소스 사용량을 본다";
      function pick(){ selectedCpu = r.id; run(); }
      tr.addEventListener("click", pick);
      tr.addEventListener("keydown", function(ev){
        if(ev.key === "Enter" || ev.key === " "){ ev.preventDefault(); pick(); }
      });
      tbody.appendChild(tr);
    });
    tbl.appendChild(tbody); box.appendChild(tbl);
    return box;
  }

  function section(title, aside, node){
    var wrap = el("div");
    var head = el("div", "section-head");
    head.appendChild(el("h2", "", title));
    head.appendChild(el("span", "spacer"));
    if(aside) head.appendChild(el("span", "bound", aside));
    wrap.appendChild(head);
    wrap.appendChild(node);
    return wrap;
  }

  function coefficientTable(list){
    var box = el("div", "scroll"), tbl = el("table");
    var thead = el("thead"), htr = el("tr");
    ["종류","적용","값","근거","비고"].forEach(function(h, i){
      htr.appendChild(el("th", i === 2 ? "num" : "", h));
    });
    thead.appendChild(htr); tbl.appendChild(thead);
    var tbody = el("tbody");
    list.forEach(function(c){
      var tr = el("tr");
      tr.appendChild(el("td", "", c.kind));
      tr.appendChild(el("td", "", c.key));
      tr.appendChild(el("td", "num", c.value));
      tr.appendChild(el("td", c.confidence === "estimate" ? "v-marginal" : "", c.label));
      var note = el("td", "wrap");
      note.appendChild(document.createTextNode(c.notes));
      if(c.source_url){
        note.appendChild(document.createTextNode(" "));
        var a = el("a", "", "출처");
        a.href = c.source_url; a.target = "_blank"; a.rel = "noopener noreferrer";
        note.appendChild(a);
      }
      tr.appendChild(note);
      tbody.appendChild(tr);
    });
    tbl.appendChild(tbody); box.appendChild(tbl);
    return box;
  }


  // ---- token delivery timeline + task manager ----------------------
  function meter(pct, label){
    var box = el("div", "meter" + (pct >= 85 ? " hot" : pct >= 60 ? " warn" : ""));
    var fill = el("span"); fill.style.width = Math.min(100, Math.max(0, pct)) + "%";
    box.appendChild(fill);
    box.appendChild(el("b", "", label !== undefined ? label : pct.toFixed(1) + "%"));
    box.setAttribute("role", "img");
    box.setAttribute("aria-label", (label || pct.toFixed(1) + "%"));
    return box;
  }

  function stopPlayback(){
    if(player){ cancelAnimationFrame(player.raf); player = null; }
  }

  function timelinePanel(tl){
    var wrap = el("div", "card");
    wrap.appendChild(el("div", "label", "알람 1건 전달 시간"));
    wrap.appendChild(el("p", "hint",
      "대기 없이 한 건만 처리할 때. prefill " + tl.prefill_tps + " tok/s, decode " +
      tl.decode_tps + " tok/s 예측 기준 · 오차 \u00b1" + tl.uncertainty + "%"));

    var total = tl.total_s || 0.001;
    var tlWrap = el("div", "tl-wrap"); tlWrap.style.marginTop = "12px";
    var bar = el("div", "tl");
    var classes = ["s-prefill", "s-decode", "s-send"];
    tl.stages.forEach(function(st, i){
      var seg = el("div", classes[i]);
      seg.style.width = (100 * st.seconds / total) + "%";
      seg.title = st.name + " " + st.seconds + "초 · " + st.note;
      if(st.seconds / total > 0.13) seg.textContent = st.seconds.toFixed(2) + "s";
      bar.appendChild(seg);
    });
    var head = el("div", "tl-head");
    tlWrap.appendChild(bar); tlWrap.appendChild(head);
    wrap.appendChild(tlWrap);

    var legend = el("div", "tl-legend");
    tl.stages.forEach(function(st){
      var item = el("span");
      item.appendChild(document.createTextNode(st.name + " "));
      item.appendChild(el("b", "", st.seconds.toFixed(2) + "s"));
      if(st.tokens) item.appendChild(document.createTextNode(" / " + st.tokens + " tok"));
      legend.appendChild(item);
    });
    var totalItem = el("span");
    totalItem.appendChild(document.createTextNode("합계 "));
    totalItem.appendChild(el("b", "", total.toFixed(2) + "s"));
    legend.appendChild(totalItem);
    wrap.appendChild(legend);

    // Real-time playback: the point is to feel the speed, not to read a number.
    var controls = el("div", "play");
    var btn = el("button", "", "\u25b6 재생");
    btn.type = "button";
    var counter = el("div", "counter", "0");
    var unit = el("small", "", " / " + tl.output_tokens + " 토큰");
    counter.appendChild(unit);
    controls.appendChild(btn);
    controls.appendChild(counter);
    wrap.appendChild(controls);

    var stream = el("div", "stream");
    stream.setAttribute("aria-hidden", "true");
    wrap.appendChild(stream);

    var SAMPLE = "장애 요약: 코어 스위치 업링크 다운으로 지사 3개 회선이 동시 단절되었습니다. " +
      "영향 범위는 지사망 전체이며 우회 경로가 없어 즉시 조치가 필요합니다. " +
      "권고: 광 트랜시버 상태 확인 후 예비 포트로 절체하고, 회선사에 장애 접수하십시오. ";

    btn.addEventListener("click", function(){
      if(player){ stopPlayback(); btn.textContent = "\u25b6 재생"; head.style.display = "none"; return; }
      btn.textContent = "\u25a0 정지";
      head.style.display = "block";
      var t0 = performance.now();
      var prefill = tl.stages[0].seconds, decode = tl.stages[1].seconds;
      function frame(now){
        var t = (now - t0) / 1000;
        head.style.left = (100 * Math.min(t / total, 1)) + "%";
        var produced = t <= prefill ? 0
          : Math.min(tl.output_tokens, Math.round(tl.output_tokens * (t - prefill) / decode));
        counter.textContent = String(produced);
        counter.appendChild(unit);
        var chars = Math.round(SAMPLE.length * produced / Math.max(tl.output_tokens, 1));
        stream.textContent = SAMPLE.slice(0, chars);
        if(t < total){ player.raf = requestAnimationFrame(frame); }
        else { stopPlayback(); btn.textContent = "\u25b6 다시 재생"; }
      }
      player = {raf: requestAnimationFrame(frame)};
    });
    return wrap;
  }

  var RES_COLS = [
    ["개수", function(r){ return r.alarms; }, "num"],
    ["CPU 사용률", null, ""],
    ["RAM", null, ""],
    ["RAM 사용", function(r){ return r.ram_used_gb + " / " + r.ram_installed_gb + " GB"; }, "num"],
    ["평균 대역폭", function(r){ return r.bandwidth_avg_gbs + " / " + r.bandwidth_gbs + " GB/s"; }, "num"],
    ["최대 큐", function(r){ return r.max_queue; }, "num"],
    ["평상시 p95", function(r){ return r.p95_steady + " s"; }, "num"],
    ["스톰 소진", function(r){ return r.storm_min + " 분"; }, "num"],
    ["하루 작업시간", function(r){ return r.work_minutes + " 분"; }, "num"]
  ];

  function resourcePanel(d){
    var wrap = el("div", "card");
    wrap.appendChild(el("div", "label", "작업관리자 · 하드웨어 고정, 개수별 리소스"));
    var hw = d.hardware;
    wrap.appendChild(el("p", "cpu", hw.label));
    var bits = el("div", "hw");
    [hw.cores + "코어 / " + hw.threads + "스레드 @ " + hw.ghz + "GHz",
     hw.sockets + "소켓", hw.isa.toUpperCase(), hw.memory,
     hw.bandwidth_gbs + " GB/s", "RAM " + hw.ram_installed_gb + "GB",
     hw.slots + " 슬롯", hw.tdp_w + "W"].forEach(function(x){
      bits.appendChild(el("span", "", x));
    });
    wrap.appendChild(bits);

    var box = el("div", "scroll"); box.style.marginTop = "12px";
    var tbl = el("table"), thead = el("thead"), htr = el("tr");
    RES_COLS.forEach(function(c){ htr.appendChild(el("th", c[2], c[0])); });
    htr.appendChild(el("th", "", "판정"));
    thead.appendChild(htr); tbl.appendChild(thead);

    var tbody = el("tbody");
    d.rows.forEach(function(r){
      var tr = el("tr");
      RES_COLS.forEach(function(c, i){
        if(c[1]){ tr.appendChild(el("td", c[2], c[1](r))); return; }
        var td = el("td", c[2]);
        td.appendChild(i === 1 ? meter(r.cpu_pct, r.cpu_pct.toFixed(1) + "%")
                               : meter(r.ram_pct, r.ram_pct.toFixed(0) + "%"));
        tr.appendChild(td);
      });
      var v = el("td", "v-" + r.verdict, VERDICT[r.verdict] || r.verdict);
      if(r.reasons && r.reasons.length) v.title = r.reasons.join(" / ");
      tr.appendChild(v);
      tbody.appendChild(tr);
    });
    tbl.appendChild(tbody); box.appendChild(tbl);
    wrap.appendChild(box);

    var b = d.ram_breakdown;
    wrap.appendChild(el("p", "hint-row",
      "RAM 내역: 가중치 " + b.weights_gb + " + KV " + b.kv_gb + " + 컴퓨트 " +
      b.compute_gb + " + OS·런타임 " + b.os_gb + " = " + b.subtotal_gb +
      " GiB → 장착 권장 " + b.installed_gb + " GB"));
    wrap.appendChild(el("p", "hint-row",
      "CPU 사용률은 하루 중 서버가 실제로 일한 시간의 비율이다. " +
      "llama.cpp는 요청이 있으면 모든 스레드를 점유하므로 유휴가 아니면 사실상 포화 상태다."));
    return wrap;
  }

  function loadResources(){
    if(!selectedCpu) return;
    var mine = ++seqRes;
    askResources(params()).then(function(d){
      if(mine !== seqRes) return;
      var host = document.getElementById("resources");
      if(!host) return;
      host.textContent = "";
      if(d.error){
        host.appendChild(el("div", "note", "리소스 산정 실패: " + d.error));
        return;
      }
      host.appendChild(section("토큰 전달 시뮬레이터", d.hardware.label,
                               timelinePanel(d.timeline)));
      host.appendChild(section("리소스 사용량", "개수별", resourcePanel(d)));
    });
  }

  function render(d){
    stopPlayback();
    results.textContent = "";

    var m = d.model;
    var head = el("div", "card");
    head.appendChild(el("div", "label", "산정 대상"));
    var line = m.name + " · " + m.params_b + "B";
    if(m.active_params_b) line += " (활성 " + m.active_params_b + "B MoE)";
    line += " · " + m.quant + " " + m.bpw + " bpw · KV " + m.kv_kib + " KiB/토큰";
    head.appendChild(el("p", "cpu", line));
    head.appendChild(el("p", "detail",
      "프롬프트 " + d.workload.prefill_tokens + " 토큰 (실제 처리 " +
      d.workload.billed_prefill + ") · 컨텍스트 " + d.workload.ctx +
      " · 후보 " + d.total + "종 중 " + d.passing + "종 통과"));
    results.appendChild(head);

    var tiers = el("div", "tiers");
    ["minimum","recommended","comfortable"].forEach(function(k){
      tiers.appendChild(tierCard(k, d.tiers[k]));
    });
    results.appendChild(section("권장 스펙", "판정은 오차를 반영한 불리한 추정 기준", tiers));

    if(d.candidates.length){
      results.appendChild(section(
        "CPU 후보", d.candidates.length + "종", table(d.candidates)));
    } else {
      results.appendChild(section("CPU 후보", "", el("div", "card", "표시할 후보가 없다.")));
    }

    var resHost = el("div"); resHost.id = "resources";
    resHost.style.display = "flex";
    resHost.style.flexDirection = "column";
    resHost.style.gap = "24px";
    results.appendChild(resHost);
    if(!selectedCpu){
      resHost.appendChild(el("div", "note",
        "위 표에서 CPU를 클릭하면 그 하드웨어의 토큰 전달 시간과 개수별 리소스 사용량이 여기 나온다."));
    }

    results.appendChild(section("효율 계수", "이 산정에 쓰인 것만",
      coefficientTable(d.coefficients)));

    if(d.warnings.length){
      var ul = el("ul", "notes");
      d.warnings.forEach(function(w){ ul.appendChild(el("li", "", w)); });
      results.appendChild(section("주의사항", "", ul));
    }
    if(d.unverified.length){
      var n = el("div", "note");
      n.appendChild(el("strong", "", "미확인 스펙 " + d.unverified.length + "건. "));
      n.appendChild(document.createTextNode(
        "납품 문서로 확정하기 전에 벤더 데이터시트로 대조해야 한다: " +
        d.unverified.slice(0, 14).map(function(u){ return u.kind + ":" + u.id; }).join(", ") +
        (d.unverified.length > 14 ? " 외 " + (d.unverified.length - 14) + "건" : "")));
      results.appendChild(n);
    }
  }

  // ---- update ------------------------------------------------------
  function checkForUpdate(){
    var btn = document.getElementById("update");
    askUpdate().then(function(u){
      if(!u || !u.available) return;
      btn.hidden = false;
      btn.textContent = "\u2191 업데이트 " + u.tag;
      btn.title = (u.notes || "").slice(0, 200);
      btn.addEventListener("click", function(){
        if(!DESKTOP){ window.open(u.page_url, "_blank", "noopener"); return; }
        btn.disabled = true;
        btn.textContent = "내려받는 중\u2026";
        window.pywebview.api.update_install().then(function(msg){
          btn.disabled = false;
          btn.textContent = msg || ("\u2191 업데이트 " + u.tag);
          btn.title = msg || "";
        });
      });
    }).catch(function(){ /* offline is not an error worth showing */ });
  }

  // ---- run ---------------------------------------------------------
  function run(){
    var p = params();
    refreshDownloads(p);
    results.classList.add("stale");
    var mine = ++seq;
    askSize(p)
      .then(function(d){
        if(mine !== seq) return;           // a newer request already answered
        results.classList.remove("stale");
        if(d.error){
          results.textContent = "";
          results.appendChild(el("div", "note", "산정 실패: " + d.error));
          return;
        }
        render(d);
        loadResources();
      })
      .catch(function(err){
        if(mine !== seq) return;
        results.classList.remove("stale");
        results.textContent = "";
        results.appendChild(el("div", "note",
          (DESKTOP ? "산정 엔진 호출에 실패했다. (" : "서버에 연결하지 못했다. (") + err + ")"));
      });
  }

  function modelHint(){
    var id = document.getElementById("model").value;
    var m = catalog.models.filter(function(x){ return x.id === id; })[0];
    var hint = document.getElementById("model-hint");
    if(!m){ hint.textContent = ""; return; }
    var bits = [m.params_b + "B"];
    if(m.active_params_b) bits.push("활성 " + m.active_params_b + "B");
    bits.push("GQA " + m.n_head + "/" + m.n_kv_head);
    bits.push("KV " + m.kv_kib + " KiB/토큰");
    if(m.korean) bits.push("한국어");
    hint.textContent = bits.join(" · ");
  }
  function tokenHint(){
    var p = params();
    var total = p.system_tokens + p.fewshot_tokens + p.alarm_tokens;
    var billed = p.prompt_cache ? p.alarm_tokens : total;
    document.getElementById("token-hint").textContent =
      "프롬프트 " + total + " 토큰 중 매 요청 " + billed + " 토큰을 실제로 처리한다.";
  }

  bridgeReady().then(askCatalog).then(function(c){
    catalog = c;
    var sel = document.getElementById("model"), groups = {};
    c.models.forEach(function(m){
      if(!groups[m.size_class]){
        groups[m.size_class] = document.createElement("optgroup");
        groups[m.size_class].label = m.size_class;
        sel.appendChild(groups[m.size_class]);
      }
      var o = document.createElement("option");
      o.value = m.id;
      o.textContent = m.name + " (" + m.params_b + "B" +
        (m.active_params_b ? ", 활성 " + m.active_params_b + "B" : "") + ")";
      groups[m.size_class].appendChild(o);
    });
    var pick = c.models.filter(function(m){ return m.korean && m.params_b < 5; })[0]
            || c.models[0];
    sel.value = pick.id;

    var qs = document.getElementById("quant");
    c.quants.forEach(function(q){
      var o = document.createElement("option");
      o.value = q.id;
      o.textContent = q.id + " (" + q.bpw + " bpw, " + q.quality + ")";
      qs.appendChild(o);
    });
    qs.value = "Q4_K_M";

    var d = {alarms_per_day:150, storm_size:40, storm_window_s:30, storms_per_day:2,
             slots:2, sla_seconds:30, storm_drain_min:5, system_tokens:300,
             fewshot_tokens:400, alarm_tokens:250, output_tokens:250, sockets:1, dpc:1};
    NUM.forEach(function(k){ document.getElementById(k).value = d[k]; });

    document.getElementById("rail").addEventListener("input", function(){
      modelHint(); tokenHint(); run();
    });
    document.getElementById("rail").addEventListener("change", function(){
      modelHint(); tokenHint(); run();
    });
    wireDesktopSaves();
    modelHint(); tokenHint(); run();
    checkForUpdate();
  }).catch(function(err){
    results.textContent = "";
    results.appendChild(el("div", "note", "카탈로그를 불러오지 못했다: " + err));
  });
})();
</script>
</body>
</html>
"""
    )
