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
from .memory import DEFAULT_OS_PROFILE, OS_PROFILES, kv_bytes_per_token
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
    # The virtual lab builds a board by hand, so DIMM capacity and count are
    # inputs rather than something the sizer derives. Zero DIMMs is allowed on
    # purpose: "no DIMMs" is a configuration the lab has to be able to report
    # on, not one it should refuse to represent.
    "dimm_gb": (1, 1_024),
    "dimm_count": (0, 64),
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
    "dimm_gb": 32,
    "dimm_count": 8,
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


def _axes(raw: dict) -> tuple[str, ...] | None:
    """Which load axes the capacity request wants. Unknown names are dropped."""
    from .capacity import AXES

    wanted = raw.get("axes")
    if not isinstance(wanted, (list, tuple)):
        return None
    clean = tuple(a for a in AXES if a in {str(x) for x in wanted})
    return clean or None


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
        # The virtual lab picks a part by hand instead of sweeping the whole
        # catalogue, so the page needs the CPU list up front -- the same list
        # the sizing table draws, in the same order it sorts by.
        "cpus": [
            {
                "id": c.id,
                "label": f"{c.vendor} {c.model}",
                "vendor": c.vendor,
                "model": c.model,
                "family": c.family,
                "cores": c.cores,
                "threads": c.threads,
                "ghz": c.all_core_turbo_ghz,
                "isa": widest_isa(c),
                "mem_channels": c.mem_channels,
                "ddr_gen": c.ddr_gen,
                "max_ddr_mts": c.max_ddr_mts,
                "sockets_max": c.sockets_max,
                "max_mem_gb": c.max_mem_gb,
                "tdp_w": c.tdp_w,
                "price_usd": c.price_usd,
            }
            for c in sorted(cat.cpus, key=lambda c: (c.vendor, -c.mem_channels, c.model))
        ],
        # The operating system is a sizing input, not a footnote: its resident
        # set and the slack it wants move the memory answer by gigabytes, and
        # whether overrunning it swaps or kills changes how much slack to buy.
        # The page cannot offer that choice unless it knows the list.
        "os_profiles": [
            {
                "id": p.id,
                "label": p.label,
                "runtime_gb": _r(p.runtime_gb, 2),
                "headroom": _r(p.headroom, 2),
                "hard_limit": bool(p.hard_limit),
            }
            for p in OS_PROFILES.values()
        ],
        # Where every prediction on every screen actually comes from. This tool
        # sizes servers nobody can put their hands on, so "run a benchmark and
        # send me the log" is not a path it can offer -- it has to carry its own
        # evidence and show it. A number whose source the reader cannot see is
        # indistinguishable from one that was made up.
        "evidence": [
            {
                "id": c.id,
                "kind": c.kind,
                "key": c.key,
                "value": c.value,
                "confidence": c.confidence,
                "source": c.source,
                "source_url": c.source_url,
                "notes": c.notes,
            }
            for c in sorted(cat.coefficients, key=lambda c: (c.kind, c.key))
        ],
        "os_default": DEFAULT_OS_PROFILE,
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

#: Korean names for the two ceilings a run can sit against.
BOUND_LABEL = {"bandwidth": "메모리 대역폭", "compute": "연산", "none": "없음"}

#: What buying more of the binding resource actually means at the order form.
BOUND_ADVICE = {
    "bandwidth": "메모리 채널 수와 DDR 등급이 돈이 되는 곳이다. 코어를 더 사도 이 벽은 그대로다.",
    "compute": "코어 수와 벡터 ISA(AMX/AVX-512)가 돈이 되는 곳이다. 메모리를 더 빠르게 해도 이 벽은 그대로다.",
    "none": "이 부하에서는 어느 천장에도 오래 붙지 않았다. 더 큰 모델이나 더 긴 프롬프트를 감당할 여지가 있다.",
}


def _finite(value) -> float | None:
    """JSON has no infinity, and JSON.parse rejects the literal Python emits.

    "no limit found" has to cross the wire as null, not as `Infinity`.
    """
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return f


def _r(value, digits: int = 2) -> float | None:
    """`round`, but an infinity or a NaN becomes null instead of poisoning JSON."""
    f = _finite(value)
    return round(f, digits) if f is not None else None


def _clean(value):
    """Deep-sanitise a payload built by another module.

    The lab and bench engines are free to put whatever they like in `params`,
    `worst` and the machine summary. Anything that reaches `json.dumps` with a
    non-finite float would either be rejected (`allow_nan=False`) or emitted as
    a bare `Infinity`, which `JSON.parse` refuses. Neither is a failure the
    browser can do anything with, so it is dealt with here instead.
    """
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_clean(v) for v in value]
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        return _finite(value)
    return str(value)


def _run_timeline(model, quant, cpu, memory, eff, workload, sockets, reference):
    """One nominal simulated day, folded into the phase-resolved series.

    `reference` supplies the throughput prediction so the timeline's ceilings
    are the same ones the sizing table quoted; the two views cannot disagree.
    """
    from .simulate import simulate
    from .sizing import decode_table
    from .timeline import DAY_SECONDS, build_timeline, ceilings_for

    _, trace = simulate(
        workload,
        prefill_tps=reference.throughput.prefill_tps,
        decode_by_active=decode_table(model, quant, cpu, memory, workload, sockets, eff),
    )
    ceilings = ceilings_for(
        model, quant, cpu, memory, eff, workload,
        reference.throughput, sockets, reference.memory_gb,
    )
    return trace, ceilings, build_timeline(trace, ceilings, span_s=DAY_SECONDS)


def _series(timeline) -> dict:
    """Four resources with four different shapes.

    Every column here comes out of one `Timeline`, so the CPU line, the
    bandwidth line, the compute line and the memory line cannot be one scalar
    wearing four labels -- which is exactly what the previous version drew.
    Prefill burns the vector units while DRAM idles and decode does the
    opposite, so the bandwidth and compute lines move against each other; that
    is the information the single-line version threw away.
    """
    buckets = timeline.buckets
    width = buckets[0].span_s if buckets else 0.0

    def col(fn, digits: int = 1) -> list:
        return [round(fn(b), digits) for b in buckets]

    return {
        "bucket_s": round(width, 1),
        # 0..1, kept as a fraction because the CPU tile draws on its own scale.
        # Clamped: a day that overran charges the backlog to the last bucket,
        # and no graph can honestly draw 300% of a bucket. `overran_s` in the
        # bottleneck block is where that spill is reported instead.
        "cpu": [round(min(b.cpu_pct / 100.0, 1.0), 4) for b in buckets],
        "cpu_pct": [round(min(b.cpu_pct, 100.0), 1) for b in buckets],
        "bandwidth_pct": col(lambda b: b.bandwidth_avg_pct),
        "bandwidth_peak_pct": col(lambda b: b.bandwidth_peak_pct),
        "bandwidth_gbs": col(lambda b: b.bandwidth_avg_gbs, 2),
        "compute_pct": col(lambda b: b.compute_avg_pct),
        "compute_peak_pct": col(lambda b: b.compute_peak_pct),
        "compute_tflops": col(lambda b: b.compute_avg_tflops, 3),
        "ram_pct": col(lambda b: b.ram_pct),
        "ram_used_gb": col(lambda b: b.ram_used_gb, 2),
        "kv_used_gb": col(lambda b: b.kv_used_gb, 3),
        "prefill_tps": col(lambda b: b.prefill_tps),
        "decode_tps": col(lambda b: b.decode_tps),
        "queue": [b.queued for b in buckets],
        "active": [b.active for b in buckets],
        "arrived": [b.arrived for b in buckets],
        "completed": [b.completed for b in buckets],
        "saturated": [b.saturated or "" for b in buckets],
    }


def _day_average(timeline, fn) -> float:
    """Mean of a per-bucket average over the whole day. Buckets are equal width."""
    buckets = timeline.buckets
    return sum(fn(b) for b in buckets) / len(buckets) if buckets else 0.0


def _ceilings_block(ceilings, reference) -> dict:
    from .memory import GB

    return {
        "bandwidth_gbs": round(ceilings.bandwidth_bytes_s / 1e9, 1),
        "compute_tflops": round(ceilings.compute_flops / 1e12, 2),
        # Static + reserved is what the box must actually be given. The live
        # series only shows what was *touched*, and quoting that alone would
        # under-order the RAM.
        #
        # The reserved half is taken from `size_memory`, not from
        # `Ceilings.kv_reserved_bytes`: the two differ because sizing rounds the
        # context up the way `llama-server -c` is actually set, and the report,
        # the CSV and the RAM column all quote the rounded one. One question,
        # one number. The static half is identical either way.
        "static_gb": round(ceilings.static_bytes / GB, 2),
        "kv_reserved_gb": round(reference.ram.kv_cache_gb, 2),
        "allocated_gb": round(reference.ram.subtotal_gb, 2),
        "installed_gb": round(ceilings.installed_bytes / GB, 1),
        "bandwidth_confidence": ceilings.bandwidth_confidence,
        "compute_confidence": ceilings.compute_confidence,
    }


def _bottleneck(timeline) -> dict:
    """The one line that decides whether to buy cores or memory channels."""
    busy = timeline.busy_seconds
    bound = timeline.binding_resource
    held = {
        "bandwidth": timeline.seconds_bandwidth_bound,
        "compute": timeline.seconds_compute_bound,
    }.get(bound, 0.0)
    share_pct = (held / busy * 100.0) if busy else 0.0

    if bound == "none":
        sentence = "이 부하에서는 어느 천장에도 오래 붙지 않았다 — 대역폭도 연산도 여유가 있다."
    else:
        sentence = (
            f"이 부하에서 이 서버는 {BOUND_LABEL[bound]} 바운드다 — 작업시간의 "
            f"{share_pct:.0f}%. 그 작업시간은 prefill {timeline.prefill_share * 100:.0f}% / "
            f"decode {timeline.decode_share * 100:.0f}%로 갈린다."
        )

    # Saying this plainly matters: a decoding request sits at the bandwidth
    # ceiling by definition. Painting that red would call every healthy server
    # overloaded. Overload is work arriving faster than it drains -- a queue.
    if timeline.peak_queue > 0:
        overload = (
            f"최대 {timeline.peak_queue}건이 큐에 쌓였다 — 도착이 소진보다 빨랐던 구간이 있다."
        )
    else:
        overload = "큐가 한 번도 쌓이지 않았다 — 순간적으로 천장에 붙는 것 자체는 과부하가 아니다."

    return {
        "resource": bound,
        "label": BOUND_LABEL.get(bound, bound),
        "advice": BOUND_ADVICE.get(bound, ""),
        "sentence": sentence,
        "overload": overload,
        "overloaded": timeline.peak_queue > 0,
        "bound_share_pct": round(share_pct, 1),
        "prefill_share": round(timeline.prefill_share * 100, 1),
        "decode_share": round(timeline.decode_share * 100, 1),
        "busy_seconds": round(busy, 1),
        "busy_minutes": round(busy / 60.0, 1),
        "seconds_bandwidth_bound": round(timeline.seconds_bandwidth_bound, 1),
        "seconds_compute_bound": round(timeline.seconds_compute_bound, 1),
        "peak_bandwidth_pct": round(timeline.peak_bandwidth_pct, 1),
        "peak_compute_pct": round(timeline.peak_compute_pct, 1),
        "peak_kv_used_gb": round(timeline.peak_kv_used_gb, 2),
        "peak_queue": timeline.peak_queue,
        "overran_s": round(timeline.overran_s, 1),
        "overran_h": round(timeline.overran_s / 3600.0, 2),
        "notes": list(timeline.notes),
    }


def resource_payload(cat: Catalog, p: dict, cpu_id: str, volumes=DEFAULT_VOLUMES) -> dict:
    """One fixed hardware build, several alarm volumes.

    Two questions answered together. First, what does a single alarm look like in
    time -- how long until the first token, how long generation takes at the
    predicted rate, how long delivery adds. Second, what load does the box carry
    as volume grows, and how much concurrency (and therefore memory) that needs.

    Each volume is solved for the slot count it actually requires, so the memory
    figure moves with the workload instead of restating the configured setting.
    """
    from .sizing import evaluate, required_slots
    from .timeline import DAY_SECONDS

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
        want = replace(base, alarms_per_day=volume)
        slots, c = required_slots(model, quant, cpu, memory, eff, want, sockets)
        sim = c.sim_pessimistic or c.sim
        installed = c.memory_gb
        bw = c.throughput.effective_bandwidth_gbs
        # Resource columns come from this volume's own timeline. The previous
        # version put `busy_fraction` in the CPU column and then again in the
        # bandwidth column, so the two could never differ.
        _, _, row_tl = _run_timeline(
            model, quant, cpu, memory, eff, replace(want, slots=slots), sockets, c
        )
        rows.append(
            {
                "alarms": volume,
                "slots": slots,
                "slots_configured": base.slots,
                # A task manager's CPU graph: share of the day spent working.
                "cpu_pct": round(min(row_tl.busy_seconds / DAY_SECONDS, 1.0) * 100, 1),
                "slot_pct": round(sim.slot_utilisation * 100, 1),
                "ram_used_gb": round(c.ram.subtotal_gb, 1),
                "ram_installed_gb": installed,
                "ram_pct": round(min(c.ram.subtotal_gb / installed, 1.0) * 100, 1)
                if installed else 0,
                "ram_live_peak_gb": round(
                    max((b.ram_used_gb for b in row_tl.buckets), default=0.0), 2
                ),
                "kv_gb": round(c.ram.kv_cache_gb, 2),
                "kv_live_peak_gb": round(row_tl.peak_kv_used_gb, 2),
                "bandwidth_gbs": round(bw, 1),
                "bandwidth_avg_gbs": round(
                    _day_average(row_tl, lambda b: b.bandwidth_avg_gbs), 2
                ),
                "bandwidth_pct": round(_day_average(row_tl, lambda b: b.bandwidth_avg_pct), 1),
                "bandwidth_peak_pct": round(row_tl.peak_bandwidth_pct, 1),
                "compute_pct": round(_day_average(row_tl, lambda b: b.compute_avg_pct), 1),
                "compute_peak_pct": round(row_tl.peak_compute_pct, 1),
                "bound": row_tl.binding_resource,
                "bound_label": BOUND_LABEL.get(row_tl.binding_resource, "-"),
                "prefill_share": round(row_tl.prefill_share * 100, 1),
                "max_queue": sim.max_queue_depth,
                "p95_steady": round(sim.p95_steady_s, 1),
                "storm_min": round(sim.storm_drain_s / 60, 1),
                "verdict": c.verdict,
                "reasons": c.reasons,
                "work_minutes": round(row_tl.busy_seconds / 60.0, 1),
                "completed": sim.completed,
                "overran_s": round(row_tl.overran_s, 1),
            }
        )

    # The graph and the per-alarm timeline describe the configured setting.
    reference = evaluate(model, quant, cpu, memory, eff, base, sockets)
    _, ceilings, timeline = _run_timeline(
        model, quant, cpu, memory, eff, base, sockets, reference
    )
    series = _series(timeline)

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

    sim_ref = reference.sim_pessimistic or reference.sim
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
            "ram_max_gb": cpu.max_mem_gb * sockets,
            "slots": base.slots,
            "l3_mb": cpu.l3_mb * sockets,
            "socket": cpu.socket,
            "passmark": cpu.passmark_multithread,
            "price_usd": (cpu.price_usd * sockets) if cpu.price_usd else None,
        },
        "live": {
            # Headline tiles, all read off the same timeline the graphs draw.
            "cpu_pct": round(min(timeline.busy_seconds / DAY_SECONDS, 1.0) * 100, 1),
            "slot_pct": round(sim_ref.slot_utilisation * 100, 1),
            # Reserved, not touched: llama.cpp allocates the full context for
            # every slot up front. This is the number to order RAM against.
            "ram_pct": round(
                min(reference.ram.subtotal_gb / reference.memory_gb, 1.0) * 100, 1
            ) if reference.memory_gb else 0,
            "ram_used_gb": round(reference.ram.subtotal_gb, 1),
            "ram_reserved_gb": round(reference.ram.subtotal_gb, 2),
            "ram_installed_gb": reference.memory_gb,
            # What the run actually touched, at its worst moment. Always the
            # smaller of the two, and on its own it under-orders the server.
            "ram_live_peak_gb": round(
                max((b.ram_used_gb for b in timeline.buckets), default=0.0), 2
            ),
            "kv_reserved_gb": round(reference.ram.kv_cache_gb, 2),
            "kv_live_peak_gb": round(timeline.peak_kv_used_gb, 2),
            "bandwidth_avg_gbs": round(
                _day_average(timeline, lambda b: b.bandwidth_avg_gbs), 2
            ),
            "bandwidth_busy_gbs": round(
                (
                    sum(b.bandwidth_avg_gbs * b.span_s for b in timeline.buckets)
                    / timeline.busy_seconds
                )
                if timeline.busy_seconds else 0.0,
                2,
            ),
            "bandwidth_pct": round(_day_average(timeline, lambda b: b.bandwidth_avg_pct), 1),
            "bandwidth_peak_pct": round(timeline.peak_bandwidth_pct, 1),
            "bandwidth_gbs": round(t.effective_bandwidth_gbs, 1),
            "compute_pct": round(_day_average(timeline, lambda b: b.compute_avg_pct), 1),
            "compute_peak_pct": round(timeline.peak_compute_pct, 1),
            "compute_tflops": round(ceilings.compute_flops / 1e12, 2),
            # Nominal run, because that is what the graph beside it draws. The
            # pessimistic figure the verdict turns on is carried separately.
            "max_queue": timeline.peak_queue,
            "max_queue_pessimistic": sim_ref.max_queue_depth,
            "peak_active": max((b.active for b in timeline.buckets), default=0),
            "alarms": base.alarms_per_day,
            "completed": sim_ref.completed,
            "verdict": reference.verdict,
        },
        "series": series,
        "ceilings": _ceilings_block(ceilings, reference),
        "bottleneck": _bottleneck(timeline),
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


# --------------------------------------------------------------------------
# Where it breaks
# --------------------------------------------------------------------------


def _load_point(point) -> dict | None:
    if point is None:
        return None
    return {
        "value": point.value,
        "verdict": point.verdict,
        "ok": point.ok,
        "p95_steady_s": round(point.p95_steady_s, 2),
        "storm_drain_s": round(point.storm_drain_s, 1),
        "storm_min": round(point.storm_drain_s / 60.0, 2),
        "max_queue": point.max_queue,
        "busy_pct": round(point.busy_fraction * 100, 1),
        "bound": point.binding_resource,
        "bound_label": BOUND_LABEL.get(point.binding_resource, point.binding_resource),
        "prefill_share": round(point.prefill_share * 100, 1),
        "ram_needed_gb": point.ram_needed_gb,
        "reasons": list(point.reasons),
    }


def capacity_payload(cat: Catalog, p: dict, cpu_id: str, axes=None) -> dict:
    """How much load this build takes before the SLA goes, and what gives first.

    Deliberately *not* wired to the live recompute. A four-axis sweep costs
    seconds to tens of seconds because each probe is a full simulated day, and
    hanging that off every keystroke is how a simulator turns into a spinner.
    The page asks for this once, on a button.
    """
    from .capacity import AXES, AXIS_LABEL, LIMITER_LABEL, sweep_axes, weakest_axis

    model = cat.model(p["model"]) if p["model"] else cat.models[0]
    quant = cat.quant(p["quant"])
    cpu = cat.cpu(cpu_id)
    memory = cat.memory_for(cpu, p["dpc"])
    sockets = min(p["sockets"], cpu.sockets_max)
    base = _workload(p)

    wanted = tuple(a for a in (axes or AXES) if a in AXES) or AXES
    curves = sweep_axes(cat, model, quant, cpu, memory, base, wanted, sockets)
    weakest = weakest_axis(curves)

    out = []
    for axis in wanted:
        curve = curves[axis]
        label, unit = AXIS_LABEL[axis]
        out.append(
            {
                "axis": axis,
                "label": label,
                "unit": unit,
                "baseline": curve.baseline,
                "knee": _load_point(curve.knee),
                "breaks_at": _load_point(curve.breaks_at),
                "limiter": curve.limiter,
                "limiter_label": LIMITER_LABEL.get(curve.limiter, curve.limiter),
                # None means "no limit found inside the search range" -- the
                # honest answer, and the one JSON can actually carry.
                "headroom": _finite(curve.headroom),
                "hit_ceiling": curve.hit_ceiling,
                "notes": list(curve.notes),
                "points": [_load_point(pt) for pt in curve.points],
                "weakest": axis == weakest,
            }
        )

    # The axis the screen opens on: the one that breaks first if we know it.
    selected = weakest or wanted[0]
    focus = next(a for a in out if a["axis"] == selected)

    return {
        "hardware": {
            "id": cpu.id,
            "label": f"{cpu.vendor} {cpu.model}",
            "cores": cpu.cores * sockets,
            "sockets": sockets,
            "memory": f"{cpu.mem_channels * sockets}ch {memory.ddr_gen}-{memory.effective_mts}",
            "isa": widest_isa(cpu),
        },
        "model": {"id": model.id, "name": model.name, "quant": quant.id},
        "sla": {
            "sla_seconds": base.sla_seconds,
            "storm_drain_min": round(base.storm_drain_sla_s / 60.0, 1),
            "slots": base.slots,
        },
        "axes": out,
        "weakest_axis": weakest,
        "axis": selected,
        # Repeated at the top level so a caller that only wants the headline
        # does not have to walk the axis list to find it.
        "knee": focus["knee"],
        "breaks_at": focus["breaks_at"],
        "limiter": focus["limiter"],
        "limiter_label": focus["limiter_label"],
        "headroom": focus["headroom"],
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
# Virtual lab: assemble a board by hand, then load-test it
# --------------------------------------------------------------------------
#
# Two payloads with deliberately different costs, and the split is the whole
# design. `lab_payload` is arithmetic over the catalogue -- microseconds -- so
# it runs on every dropdown change and the warning about a half-populated board
# appears while the operator is still turning the knob. `bench_payload` replays
# a whole simulated load through the queue engine, so it runs on a button and
# its result goes stale when the inputs move, exactly like `capacity_payload`.


def _findings(items) -> list[dict]:
    """`Finding` -> wire. The remedy travels with the problem, always.

    A warning that names a broken build and stops there is worse than none: it
    tells the operator something is wrong and leaves them to guess. Every
    finding the lab raises knows how to fix itself, so the field is carried
    here rather than left to the page to invent.
    """
    out = []
    for f in items or ():
        out.append(
            {
                "level": str(getattr(f, "level", "info")),
                "code": str(getattr(f, "code", "")),
                "message": str(getattr(f, "message", "")),
                "remedy": str(getattr(f, "remedy", "") or ""),
            }
        )
    return out


def _dimm_option(option: dict) -> dict:
    """One buildable DIMM combination, normalised for the dropdown.

    Accepts the engine's naming loosely on purpose: this is the only place the
    two modules meet, and a key rename on the far side should widen a dropdown
    entry, not blank the panel out.
    """
    o = dict(option)
    gb = int(o.get("dimm_gb") or 0)
    count = int(o.get("count") or o.get("dimm_count") or 0)
    return {
        "dimm_gb": gb,
        "count": count,
        "ram_total_gb": int(o.get("ram_total") or o.get("ram_total_gb") or gb * count),
        "channels_populated": int(o.get("channels_populated") or 0),
        "dpc": int(o.get("dpc") or o.get("dimms_per_channel") or 1),
    }


def _assemble(cat: Catalog, p: dict, cpu_id: str, name: str = "A"):
    """Build the machine the request describes. Never raises on a bad *build*.

    Imported lazily so that a partially installed tree still serves the sizing
    screen: the lab is an added surface, not a load-time dependency of it.
    """
    from .lab import VirtualMachine, assemble

    vm = VirtualMachine(
        name=str(name)[:40] or "A",
        cpu_id=cpu_id,
        sockets=p["sockets"],
        dimm_gb=p["dimm_gb"],
        dimm_count=p["dimm_count"],
        model_id=p["model"] or cat.models[0].id,
        quant_id=p["quant"],
        slots=p["slots"],
    )
    return vm, assemble(cat, vm, _workload(p).tokens)


def _pct_loss(now: float | None, full: float | None) -> float | None:
    """How much of the full-channel figure this build gave up, as a percent."""
    a, b = _finite(now), _finite(full)
    if a is None or b is None or b <= 0:
        return None
    return round(max(0.0, (1.0 - a / b)) * 100, 1)


def lab_payload(cat: Catalog, p: dict, cpu_id: str, name: str = "A") -> dict:
    """One hand-built machine, and everything wrong with it.

    Cheap by construction -- `assemble` is catalogue lookups and a roofline, no
    simulation, about a third of a millisecond -- which is what lets the page
    recompute on every keystroke. The expensive question ("what load does it
    survive") is a separate route on a separate button.
    """
    vm, asm = _assemble(cat, p, cpu_id, name)
    return _lab_block(cat, vm, asm)


def _lab_block(cat: Catalog, vm, asm) -> dict:
    """The assembled machine, flattened for the wire.

    Split out from `lab_payload` so the bench route can report the machine it
    actually ran without assembling it a second time -- one `Assembly`, one
    description of it, no chance of the two disagreeing.
    """
    cpu, mem, model, quant = asm.cpu, asm.memory, asm.model, asm.quant
    sockets = vm.sockets

    channels_total = int(asm.channels_total)
    populated = int(asm.channels_populated)
    findings = _findings(asm.findings)
    errors = sum(1 for f in findings if f["level"] == "error")
    warnings = sum(1 for f in findings if f["level"] == "warn")

    # The headline the mock-up asked for: part, socket count, channel fill and
    # the DIMM arithmetic on one line, because that is the sentence somebody
    # reads back to a vendor when they order the box.
    headline = (
        f"{cpu.vendor} {cpu.model} × {sockets}소켓 ({channels_total}채널) · "
        f"{vm.dimm_count} × {vm.dimm_gb}GB = {asm.ram_total_gb}GB"
    )

    return {
        "machine": {
            "name": vm.name,
            "cpu_id": vm.cpu_id,
            "sockets": vm.sockets,
            "dimm_gb": vm.dimm_gb,
            "dimm_count": vm.dimm_count,
            "model_id": vm.model_id,
            "quant_id": vm.quant_id,
            "slots": vm.slots,
        },
        "headline": headline,
        "cpu": {
            "id": cpu.id,
            "label": f"{cpu.vendor} {cpu.model}",
            "family": cpu.family,
            "cores": cpu.cores * sockets,
            "threads": cpu.threads * sockets,
            "ghz": cpu.all_core_turbo_ghz,
            "isa": widest_isa(cpu),
            "sockets": sockets,
            "sockets_max": cpu.sockets_max,
            "channels_per_socket": cpu.mem_channels,
            "max_mem_gb": cpu.max_mem_gb * sockets,
            "tdp_w": cpu.tdp_w * sockets,
            "l3_mb": cpu.l3_mb * sockets,
            "socket": cpu.socket,
            "price_usd": (cpu.price_usd * sockets) if cpu.price_usd else None,
        },
        "memory": {
            "ddr_gen": mem.ddr_gen,
            "rated_mts": mem.rated_mts,
            "effective_mts": mem.effective_mts,
            "dimms_per_channel": int(asm.dimms_per_channel),
            "dimm_gb": vm.dimm_gb,
            "kind": mem.kind,
            "label": f"{mem.ddr_gen}-{mem.effective_mts} {mem.kind}",
            "derated": mem.effective_mts < mem.rated_mts,
        },
        "model": {
            "id": model.id,
            "name": model.name,
            "params_b": _r(model.params_b, 2),
            "active_params_b": _r(model.active_params_b, 2) if model.active_params_b else None,
            "quant": quant.id,
            "bpw": quant.bits_per_weight,
            "kv_kib": _r(kv_bytes_per_token(model) / 1024, 1),
            "slots": vm.slots,
        },
        "channels": {
            "total": channels_total,
            "populated": populated,
            "dimms_per_channel": int(asm.dimms_per_channel),
            "fill_pct": _r(100.0 * populated / channels_total, 1) if channels_total else None,
        },
        "ram": {
            "total_gb": _r(asm.ram_total_gb, 1),
            "used_gb": _r(asm.ram_used_gb, 2),
            "free_gb": _r(max(0.0, asm.ram_total_gb - asm.ram_used_gb), 2),
            "pct": _r(min(100.0, 100.0 * asm.ram_used_gb / asm.ram_total_gb), 1)
            if asm.ram_total_gb else None,
        },
        "bandwidth": {
            "gbs": _r(asm.bandwidth_gbs, 1),
            "full_gbs": _r(asm.bandwidth_full_gbs, 1),
            "loss_pct": _pct_loss(asm.bandwidth_gbs, asm.bandwidth_full_gbs),
        },
        "throughput": {
            "prefill_tps": _r(asm.prefill_tps, 1),
            "decode_tps_single": _r(asm.decode_tps_single, 2),
            "decode_tps_full": _r(asm.decode_tps_full, 2),
            "decode_loss_pct": _pct_loss(asm.decode_tps_single, asm.decode_tps_full),
            "uncertainty_pct": _r((asm.uncertainty or 0.0) * 100, 0),
        },
        "findings": findings,
        "ok": bool(asm.ok),
        "errors": errors,
        "warnings": warnings,
        "options": [_dimm_option(o) for o in _dimm_options(cat, cpu, sockets)],
    }


def _dimm_options(cat: Catalog, cpu, sockets: int) -> list:
    from .lab import dimm_options

    return list(dimm_options(cat, cpu, sockets))


#: Per-profile numeric parameters the client may set, with bounds. Same rule as
#: `LIMITS`: clamped, never trusted.
BENCH_LIMITS = {
    "count": (0, 100_000),
    "storm_size": (0, 5_000),
    "storms_per_day": (0, 100),
    "start_rate": (1, 200_000),
    "end_rate": (1, 200_000),
    "base_rate": (1, 200_000),
    "peak_rate": (1, 200_000),
    "rate": (1, 200_000),
    "hours": (1, 336),
    "spike_at_h": (0, 335),
    "spike_minutes": (1, 1_440),
}

BENCH_DEFAULTS = {
    "count": 0,          # 0 means "the whole generated day"
    "storm_size": 40,
    "storms_per_day": 2,
    "start_rate": 100,
    "end_rate": 2_000,
    "base_rate": 165,
    "peak_rate": 800,
    "rate": 300,
    "hours": 24,
    "spike_at_h": 12,
    "spike_minutes": 30,
}

#: Which parameters each profile actually takes. Sending a ramp's fields to a
#: soak run would be silently ignored by the engine; dropping them here means
#: the cache key and the reproduction record only ever carry live inputs.
BENCH_KEYS = {
    "replay": ("count", "storm_size", "storms_per_day"),
    "ramp": ("start_rate", "end_rate", "hours"),
    "spike": ("base_rate", "peak_rate", "spike_at_h", "spike_minutes"),
    "soak": ("rate", "hours"),
}

DEFAULT_BENCH_DATE = "2026-06-01"
DEFAULT_BENCH_SEED = 20260730
#: 600 frames is the resolution the browser plays back. Raw events are hundreds
#: of kilobytes; folded frames are tens, and a display cannot resolve more.
DEFAULT_FRAMES = 600


def _bench_kind(raw: dict) -> str:
    from .loadgen import KINDS

    kind = str(raw.get("kind", "replay"))
    return kind if kind in KINDS else "replay"


def _bench_params(kind: str, raw: dict) -> dict:
    """The profile's own knobs, clamped. Unknown keys never reach the engine."""
    source = raw if isinstance(raw, dict) else {}
    out: dict = {}
    for key in BENCH_KEYS.get(kind, ()):
        low, high = BENCH_LIMITS[key]
        try:
            value = int(float(source.get(key, BENCH_DEFAULTS[key])))
        except (TypeError, ValueError):
            value = BENCH_DEFAULTS[key]
        out[key] = max(low, min(high, value))
    if kind == "replay":
        date = str(source.get("date", DEFAULT_BENCH_DATE))
        parts = date.split("-")
        ok = len(parts) == 3 and all(p.isdigit() for p in parts) and len(parts[0]) == 4
        out["date"] = date if ok else DEFAULT_BENCH_DATE
        # The engine reads None as "however many the day generates".
        out["count"] = out["count"] or None
    return out


def _frame_row(f) -> dict:
    return {
        "t_s": _r(f.t_s, 2),
        "queued": int(f.queued),
        "active": int(f.active),
        "cpu_pct": _r(f.cpu_pct, 1),
        "bw_gbs": _r(f.bw_gbs, 2),
        "bw_pct": _r(f.bw_pct, 1),
        "compute_pct": _r(f.compute_pct, 1),
        "kv_gb": _r(f.kv_gb, 3),
        "ram_gb": _r(f.ram_gb, 2),
        "arrived": int(f.arrived),
        "delivered": int(f.delivered),
        "offered_rate": _r(f.offered_rate, 1),
        "p95_so_far_s": _r(f.p95_so_far_s, 2),
    }


def _stats_block(stats) -> dict | None:
    from dataclasses import asdict, is_dataclass

    if stats is None:
        return None
    raw = asdict(stats) if is_dataclass(stats) else dict(stats)
    return _clean({k: (_r(v, 3) if isinstance(v, float) else v) for k, v in raw.items()})


def _profile_block(profile) -> dict:
    return {
        "kind": str(getattr(profile, "kind", "")),
        "label": str(getattr(profile, "label", "")),
        "span_s": _r(getattr(profile, "span_s", 0.0), 1),
        "total_alarms": int(getattr(profile, "total_alarms", 0) or 0),
        "params": _clean(dict(getattr(profile, "params", {}) or {})),
        "notes": [str(n) for n in (getattr(profile, "notes", ()) or ())],
    }


def bench_payload(cat: Catalog, p: dict, cpu_id: str, raw: dict) -> dict:
    """Run one load profile against one hand-built machine.

    Never wired to the live recompute. The run itself is tens of milliseconds
    of virtual time, but the load generation in front of it is not free and the
    result is a 600-frame recording the page then plays back -- recomputing all
    of that per keystroke would throw away a finished playback for nothing.
    Same discipline, and for the same reason, as `capacity_payload`.
    """
    from . import bench as bench_engine
    from .loadgen import build_load

    name = str(raw.get("name", "A"))
    vm, asm = _assemble(cat, p, cpu_id, name)
    machine = _lab_block(cat, vm, asm)

    kind = _bench_kind(raw)
    profile_params = _bench_params(kind, raw.get("profile") or {})
    try:
        seed = int(raw.get("seed", DEFAULT_BENCH_SEED))
    except (TypeError, ValueError):
        seed = DEFAULT_BENCH_SEED
    seed = max(0, min(2**31 - 1, seed))

    base = {
        "name": name,
        "kind": kind,
        "seed": seed,
        "profile_params": _clean(profile_params),
        "machine": machine,
        "findings": machine["findings"],
        "ok": machine["ok"],
        "frames": [],
        "breach": None,
        "stats": None,
        "worst": [],
        "notes": [],
        "profile": None,
    }

    # A build with an error-level finding is not a machine; running load
    # against it would produce numbers for hardware that cannot be ordered.
    # The findings already say what to fix, so say that and stop.
    if not machine["ok"]:
        base["blocked"] = "구성에 오류가 있다 — 아래 문제를 먼저 고쳐야 부하 테스트를 돌릴 수 있다."
        return base

    try:
        frames = int(raw.get("frames", DEFAULT_FRAMES))
    except (TypeError, ValueError):
        frames = DEFAULT_FRAMES
    frames = max(60, min(1_200, frames))

    alarms, profile = build_load(kind, seed=seed, **profile_params)
    result = bench_engine.run_bench(
        cat, asm, alarms, profile, workload=_workload(p), frames=frames, worst_n=10
    )

    base["blocked"] = None
    base["profile"] = _profile_block(result.profile)
    base["frames"] = [_frame_row(f) for f in result.frames]
    base["breach"] = _clean(result.breach) if result.breach else None
    base["stats"] = _stats_block(result.stats)
    base["worst"] = _clean(list(result.worst or ()))
    base["notes"] = [str(n) for n in (result.notes or ())]
    base["machine_summary"] = _clean(dict(result.machine or {}))
    base["sla"] = {
        "sla_seconds": _r(_workload(p).sla_seconds, 1),
        "slots": vm.slots,
    }
    return base


# --------------------------------------------------------------------------
# Model performance
# --------------------------------------------------------------------------
#
# The question underneath every other screen in this tool. The alarm pipeline is
# one application of a model on a machine, not the only question worth asking
# about it, and until now there was no surface that answered "put this model on
# this server and how fast is it" without first inventing an alarm volume.
#
# Nothing here re-derives physics: `modelbench` owns that, and it owns it on top
# of `perf`. This layer picks the axes, names the bounds in Korean, and refuses
# to let a non-finite float reach the browser.

#: Korean names for the two phases a request is spent in. They are worth naming
#: because they use opposite halves of the machine, which is the whole point of
#: the resource-split view.
PHASE_LABEL = {"prefill": "프롬프트 처리", "decode": "토큰 생성"}

#: Every ceiling `perf` can report, including the per-core one, which the alarm
#: screens fold into "대역폭" and this one does not: "코어를 더 사라" and "채널을
#: 더 사라" are different orders.
MB_BOUND_LABEL = {
    "bandwidth": "메모리 대역폭",
    "core-bandwidth": "코어당 대역폭",
    "compute": "연산",
    "none": "없음",
}

#: What being held by that ceiling means at the order form.
MB_BOUND_ADVICE = {
    "bandwidth": "메모리 채널 수와 DDR 등급이 돈이 되는 곳이다. 코어를 더 사도 이 벽은 그대로다.",
    "core-bandwidth": "코어 하나가 끌어올 수 있는 대역폭에 걸렸다. 채널을 더 채우기보다 "
                      "코어를 늘리거나 슬롯을 늘려 여러 코어가 함께 읽게 해야 한다.",
    "compute": "코어 수와 벡터 ISA(AMX/AVX-512)가 돈이 되는 곳이다. 메모리를 더 빠르게 해도 "
               "이 벽은 그대로다.",
    "none": "이 위상에서는 어느 천장에도 붙지 않았다.",
}

TRAIN_LABEL = {"full": "full 파인튜닝", "lora": "LoRA", "qlora": "QLoRA"}

#: Below this, generated text arrives slower than a person reads it, so "몇 명까지
#: 쓸 만한가"에 답하려면 이 선을 그려야 한다. A round number on purpose: it is a
#: reading-comfort threshold, not a measurement.
READABLE_TPS = 10.0

#: The grid axes the page draws when it asks for nothing else.
MB_BATCHES = (1, 2, 4, 8, 16, 32)
MB_CONTEXTS = (512, 2048, 4096, 8192, 16384)
MB_USERS = (1, 2, 4, 8, 16, 32, 64)

#: A caller may replace an axis, but not with anything it likes. Each value is
#: clamped and each axis is capped in length, because the grid is a product of
#: two of them and a browser asking for a four-hundred point sweep costs the
#: server four hundred predictions. Refusing overrides outright was the earlier
#: rule and it was too strict: the whole point of the screen is to see *your*
#: operating point, and 16k context at batch 48 is a real one somebody runs.
MB_AXIS_LIMITS = {
    "batches": (1, 512),
    "contexts": (128, 1_048_576),
    "users": (1, 4096),
}
MB_AXIS_MAX_POINTS = 12


def _mb_os(raw: dict) -> str | None:
    """Which operating system profile to size against.

    None means "whatever the engine defaults to", which keeps a request that
    never heard of this field behaving exactly as before.
    """
    from .memory import OS_PROFILES

    name = raw.get("os_name")
    return str(name) if name in OS_PROFILES else None


def _mb_os_block(os_name: str | None) -> dict:
    """The operating system assumption, stated where the numbers are read.

    Carries `overrun` because the two failure modes are not interchangeable:
    a host that runs out of memory gets slow, a container that runs out is
    killed, and a reader deciding how much slack to buy needs to know which
    one they are buying against.
    """
    from .memory import os_profile

    profile = os_profile(os_name)
    return {
        "id": profile.id,
        "label": profile.label,
        "runtime_gb": _r(profile.runtime_gb, 2),
        "headroom": _r(profile.headroom, 2),
        "hard_limit": bool(profile.hard_limit),
        "overrun": profile.overrun_consequence,
        "note": profile.note,
        "chosen": os_name is not None,
    }


def _mb_axis(raw: dict, key: str, fallback: tuple[int, ...]) -> tuple[int, ...]:
    """One axis from the request, clamped, deduplicated and length-capped.

    Anything unusable falls back to the default rather than raising: this runs
    behind a form, and a typo in one field should not lose the whole run.
    """
    wanted = raw.get(key)
    if not isinstance(wanted, (list, tuple)):
        return fallback
    low, high = MB_AXIS_LIMITS[key]
    seen: list[int] = []
    for item in wanted:
        try:
            value = int(float(item))
        except (TypeError, ValueError):
            continue
        value = max(low, min(value, high))
        if value not in seen:
            seen.append(value)
        if len(seen) >= MB_AXIS_MAX_POINTS:
            break
    return tuple(sorted(seen)) or fallback

#: Same rule as `LIMITS`: clamped, never trusted. `output_tokens` is not here on
#: purpose -- it is already a workload field, and the generation length that
#: sizes the KV cache has to be the same one the concurrency curve times.
MODELBENCH_LIMITS = {"train_samples": (100, 10_000_000)}
MODELBENCH_DEFAULTS = {"train_samples": 10_000}


def _mb_number(raw: dict, key: str) -> int:
    low, high = MODELBENCH_LIMITS[key]
    try:
        value = int(float(raw.get(key, MODELBENCH_DEFAULTS[key])))
    except (TypeError, ValueError):
        value = MODELBENCH_DEFAULTS[key]
    return max(low, min(high, value))


def _mb_ttft(pt) -> float | None:
    """Time to first token for one grid point.

    The engine reports it where it has it. Where it does not, TTFT *is* the
    prompt divided by the rate the prompt is processed at, so it is derived here
    rather than left blank -- it is the number a person waits through, and a
    blank cell would be the one thing this screen must not print.
    """
    direct = _finite(getattr(pt, "ttft_s", None))
    if direct is not None:
        return direct
    rate, ctx = _finite(pt.prefill_tps), _finite(pt.ctx_tokens)
    if rate is None or ctx is None or rate <= 0:
        return None
    return ctx / rate


def _throughput_row(pt) -> dict:
    prefill, decode = str(pt.prefill_bound), str(pt.decode_bound)
    return {
        "batch": int(pt.batch),
        "ctx_tokens": int(pt.ctx_tokens),
        "ttft_s": _r(_mb_ttft(pt), 2),
        "prefill_tps": _r(pt.prefill_tps, 1),
        "decode_tps_single": _r(pt.decode_tps_single, 2),
        # The same number under the name the page prints. Both keys ship so the
        # existing client code keeps working, and so nobody reading the payload
        # has to know that "Generation tok/s" and `decode_tps_single` are one
        # thing.
        "gen_tps": _r(getattr(pt, "gen_tps", None) or pt.decode_tps_single, 2),
        "decode_tps_total": _r(pt.decode_tps_total, 2),
        "prefill_bound": prefill,
        "decode_bound": decode,
        "prefill_bound_label": MB_BOUND_LABEL.get(prefill, prefill),
        "decode_bound_label": MB_BOUND_LABEL.get(decode, decode),
        # RAM belongs on the row, not only in the summary: llama.cpp reserves
        # the full context per slot, so the bottom-right of this grid can want
        # ten times what the top-left does. A cell the machine cannot load has
        # to say so where it is read.
        "ram_gb": _r(getattr(pt, "ram_gb", None), 2),
        "fits": bool(getattr(pt, "fits", True)),
    }


def _concurrency_row(pt) -> dict:
    each = _finite(pt.decode_tps_each)
    return {
        "users": int(pt.users),
        "ttft_s": _r(pt.ttft_s, 2),
        "decode_tps_each": _r(pt.decode_tps_each, 2),
        "response_s": _r(pt.response_s, 2),
        "total_tps": _r(pt.total_tps, 1),
        # A word, not only a colour: the page prints this next to the number.
        "readable": bool(each is not None and each >= READABLE_TPS),
    }


def _split_row(s) -> dict:
    phase, bound = str(s.phase), str(s.bound_by)
    return {
        "phase": phase,
        "phase_label": PHASE_LABEL.get(phase, phase),
        "bandwidth_pct": _r(s.bandwidth_pct, 1),
        "compute_pct": _r(s.compute_pct, 1),
        "bound_by": bound,
        "bound_label": MB_BOUND_LABEL.get(bound, bound),
        "advice": MB_BOUND_ADVICE.get(bound, ""),
        "bytes_per_token": _r(s.bytes_per_token, 1),
        "flops_per_token": _r(s.flops_per_token, 1),
    }


def _training_row(t) -> dict:
    """One verdict, with its reasons attached.

    `reasons` and `gpu_comparison` are not decoration. For a CPU server the
    answer is usually "no", and a bare "no" is not a result anybody can act on
    -- the reasons say what was short and the GPU line says what it would take.
    """
    kind = str(t.kind)
    feasible = bool(t.feasible)
    return {
        "kind": kind,
        "kind_label": TRAIN_LABEL.get(kind, kind),
        "feasible": feasible,
        "verdict": "가능" if feasible else "불가",
        "memory_needed_gb": _r(t.memory_needed_gb, 1),
        "memory_available_gb": _r(t.memory_available_gb, 1),
        "step_seconds": _r(t.step_seconds, 2),
        "epoch_hours": _r(t.epoch_hours, 1),
        "reasons": [str(r) for r in (t.reasons or ())],
        "gpu_comparison": str(t.gpu_comparison or ""),
    }


def _mb_pick(rows: list, batch: int, ctx: int) -> dict | None:
    """One grid point by coordinate. The engine owns the row order; this does not."""
    for row in rows:
        if row["batch"] == batch and row["ctx_tokens"] == ctx:
            return row
    return rows[0] if rows else None


def _mb_summary(rows: list, output_tokens: int) -> dict:
    """The two numbers a person actually feels, and the one they get confused with.

    Generation tok/s is what one sequence sees. The server total is what the box
    adds up to across every sequence in the batch. Quoting the second as if it
    were the first is exactly the mistake this screen exists to prevent, so the
    summary carries both, names the condition they hold for, and keeps the
    batched figures beside them for contrast.

    TTFT is the other half of "how does this feel": generation speed says how
    fast the answer streams, TTFT says how long you stare at nothing first.
    """
    ref = _mb_pick(rows, MB_BATCHES[0], MB_CONTEXTS[0])
    if ref is None:
        return {
            "batch": None, "ctx_tokens": None, "gen_tps": None, "ttft_s": None,
            "total_tps": None, "readable": False, "condition": "",
            "busy_batch": None, "busy_gen_tps": None, "busy_total_tps": None,
            "long_ctx_tokens": None, "long_gen_tps": None,
        }
    busiest = _mb_pick(rows, MB_BATCHES[-1], MB_CONTEXTS[0])
    longest = _mb_pick(rows, MB_BATCHES[0], MB_CONTEXTS[-1])
    gen = _finite(ref["decode_tps_single"])
    return {
        "batch": ref["batch"],
        "ctx_tokens": ref["ctx_tokens"],
        "gen_tps": ref["decode_tps_single"],
        "ttft_s": ref["ttft_s"],
        "total_tps": ref["decode_tps_total"],
        "readable": bool(gen is not None and gen >= READABLE_TPS),
        "condition": (
            f"배치 {ref['batch']} · 컨텍스트 {ref['ctx_tokens']:,} 토큰 "
            f"· 생성 {output_tokens} 토큰 기준"
        ),
        "busy_batch": busiest["batch"] if busiest else None,
        "busy_gen_tps": busiest["decode_tps_single"] if busiest else None,
        "busy_total_tps": busiest["decode_tps_total"] if busiest else None,
        "long_ctx_tokens": longest["ctx_tokens"] if longest else None,
        "long_gen_tps": longest["decode_tps_single"] if longest else None,
    }


def modelbench_payload(cat: Catalog, p: dict, cpu_id: str, raw: dict | None = None) -> dict:
    """Four axes of one model on one machine: inference, concurrency, resources, training.

    Button-driven, for the same reason as `capacity_payload` and `bench_payload`:
    the grid is dozens of predictions plus a training verdict, and none of it
    changes usefully while somebody is still typing a DIMM count.

    Imported lazily so a partially installed tree still serves the other
    screens: this is an added surface, not a load-time dependency of them.
    """
    raw = raw or {}
    vm, asm = _assemble(cat, p, cpu_id, str(raw.get("name", "M")))
    machine = _lab_block(cat, vm, asm)
    output_tokens = int(p["output_tokens"])
    train_samples = _mb_number(raw, "train_samples")
    batches = _mb_axis(raw, "batches", MB_BATCHES)
    contexts = _mb_axis(raw, "contexts", MB_CONTEXTS)
    users = _mb_axis(raw, "users", MB_USERS)
    os_name = _mb_os(raw)

    base = {
        "machine": machine,
        "findings": machine["findings"],
        "ok": machine["ok"],
        "blocked": None,
        "model_name": machine["model"]["name"],
        "quant_id": machine["model"]["quant"],
        "hardware": machine["headline"],
        "batches": list(batches),
        "contexts": list(contexts),
        "users": list(users),
        "os_name": os_name,
        "os": _mb_os_block(os_name),
        "output_tokens": output_tokens,
        "train_samples": train_samples,
        "readable_tps": READABLE_TPS,
        "throughput": [],
        "concurrency": [],
        "resources": [],
        "training": [],
        "summary": _mb_summary([], output_tokens),
        "memory_gb": None,
        "uncertainty": None,
        "notes": [],
        "warnings": [],
    }

    # A build with an error-level finding is not a machine. Quoting tok/s for
    # hardware nobody can order is worse than saying what is wrong with it.
    if not machine["ok"]:
        base["blocked"] = "구성에 오류가 있다 — 아래 문제를 먼저 고쳐야 모델 성능을 낼 수 있다."
        return base

    from .modelbench import bench_model

    mb = bench_model(
        asm, batches=batches, contexts=contexts, users=users,
        output_tokens=output_tokens, train_samples=train_samples,
        os_name=os_name,
    )
    base["model_name"] = str(mb.model_name)
    base["quant_id"] = str(mb.quant_id)
    base["hardware"] = str(mb.hardware)
    base["throughput"] = [_throughput_row(x) for x in mb.throughput]
    base["concurrency"] = [_concurrency_row(x) for x in mb.concurrency]
    base["resources"] = [_split_row(x) for x in mb.resources]
    base["training"] = [_training_row(x) for x in mb.training]
    base["summary"] = _mb_summary(base["throughput"], output_tokens)
    base["memory_gb"] = _r(mb.memory_gb, 2)
    base["uncertainty"] = _r((_finite(mb.uncertainty) or 0.0) * 100, 0)
    base["notes"] = [str(n) for n in (mb.notes or ())]
    base["warnings"] = [str(w) for w in (mb.warnings or ())]
    # The engine is free to hand back an infinity for a step time it could not
    # bound; `allow_nan=False` would reject the body and the page would see a
    # failure instead of a null. Deal with it here, once, for every axis.
    return _clean(base)


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
        # allow_nan=False on purpose: Python happily writes a bare `Infinity`,
        # and `JSON.parse` rejects it, so the page would fail on a body that
        # looked fine on this side. Better to raise here, where the handler
        # turns it into an error the browser can actually display.
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False)
        self._send(code, body.encode("utf-8"), "application/json; charset=utf-8")

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
        if route not in ("/api/size", "/api/resources", "/api/capacity",
                         "/api/lab", "/api/bench", "/api/modelbench"):
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
            elif route == "/api/capacity":
                self._json(capacity_payload(
                    self.catalog, _params(raw), str(raw.get("cpu", "")), _axes(raw)
                ))
            elif route == "/api/lab":
                self._json(lab_payload(
                    self.catalog, _params(raw), str(raw.get("cpu", "")),
                    str(raw.get("name", "A")),
                ))
            elif route == "/api/bench":
                self._json(bench_payload(
                    self.catalog, _params(raw), str(raw.get("cpu", "")), raw
                ))
            elif route == "/api/modelbench":
                self._json(modelbench_payload(
                    self.catalog, _params(raw), str(raw.get("cpu", "")), raw
                ))
            else:
                self._json(size_payload(self.catalog, _params(raw)))
        except (CatalogError, ValueError, TypeError, AttributeError) as exc:
            # AttributeError is in here for the engines: a payload built against
            # a dataclass that shipped a field short must report a sentence, not
            # hand the browser a 500 with a traceback in the log.
            self._error(str(exc))
        except ImportError as exc:
            # The lab, the bench and the model bench are optional engines. A tree
            # missing one must say so in a sentence, not hand back a traceback.
            self._error(f"이 빌드에는 그 엔진이 없다: {exc}")

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


def serve_background(catalog: Catalog | None = None) -> str:
    """Start the same server on a loopback port nobody has to choose, and return its URL.

    The desktop window's rescue path. When the pywebview bridge does not come
    up -- a missing WebView2 runtime, security software that blocks the
    injected script, an install that predates the fix -- the window would sit
    there dead and tell the operator to go run a command themselves. A tool
    that cannot open its own window should repair itself, not delegate.

    Loopback and an ephemeral port on purpose: nothing is reachable from
    another machine and no fixed port can already be taken.
    """
    handler = type("Handler", (_Handler,), {"catalog": catalog or Catalog()})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{httpd.server_address[1]}/"


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
/* The rail is sticky, so its offset has to equal the header height exactly.
   Deriving that from content is how the two columns drifted out of line, so the
   header height is pinned here and both sides read the same variable. */
:root{--header-h:60px}
body{min-height:100vh}
header{
  display:flex; align-items:center; gap:var(--s3);
  height:var(--header-h); padding:0 var(--s5); box-sizing:border-box;
  border-bottom:1px solid var(--border);
  background:var(--bg-secondary); position:sticky; top:0; z-index:10;
}
header h1{font-size:var(--fs-md); line-height:1}
header .tag{font-size:var(--fs-xs); color:var(--text-tertiary)}
header .spacer{flex:1}
main{
  display:grid; grid-template-columns:var(--rail-w) minmax(0,1fr);
  gap:var(--s5); max-width:1680px; margin:0 auto;
  padding:var(--s5) var(--s5) var(--s7);
  align-items:start;
}
/* The single-column fallback sits BELOW the app window's minimum width (1100),
   so in the desktop app the two columns are always side by side. Putting the
   breakpoint above the window minimum was what made the inputs jump above the
   results as soon as the window was nudged smaller. */
:root{--rail-w:340px}
@media (max-width:1340px){:root{--rail-w:300px}}
@media (max-width:900px){
  main{grid-template-columns:minmax(0,1fr)}
  #rail,#lab-rail,#mb-rail{position:static; max-height:none; overflow:visible}
}
#rail,#lab-rail,#mb-rail{
  position:sticky; top:calc(var(--header-h) + var(--s5));
  display:flex; flex-direction:column; gap:var(--s3);
  /* Own scrollbar: a long form must not be clipped by the viewport, and the
     results column must not be dragged taller to accommodate it. */
  max-height:calc(100vh - var(--header-h) - var(--s5) * 2);
  overflow-y:auto; overscroll-behavior:contain;
  padding-right:var(--s1);
}
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

#results,#lab-results,#mb-results{
  display:flex; flex-direction:column; gap:var(--s5); min-width:0;
}
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

/* --- Task manager, laid out like Windows' Performance tab: a column of
   selectable resource tiles on the left, the chosen one graphed large. --- */
.tm{display:grid; grid-template-columns:186px minmax(0,1fr); gap:var(--s4)}
@media (max-width:760px){.tm{grid-template-columns:minmax(0,1fr)}}
.tm-tiles{display:flex; flex-direction:column; gap:var(--s2)}
.tile{
  display:grid; grid-template-columns:1fr auto; align-items:center;
  gap:var(--s1) var(--s2); text-align:left; width:100%;
  padding:var(--s2) var(--s3); min-height:60px;
  background:var(--bg-tertiary); border:1px solid var(--border);
  border-left:3px solid var(--border); border-radius:var(--radius-sm);
  cursor:pointer; font:inherit; color:inherit;
}
.tile:hover{border-color:var(--text-tertiary)}
.tile[aria-pressed="true"]{
  background:var(--selected-bg); border-color:var(--selected-border);
  border-left-color:var(--accent);
}
.tile .t-name{font-size:var(--fs-xs); font-weight:600; color:var(--text-secondary)}
.tile .t-val{font-size:var(--fs-md); font-weight:600; font-variant-numeric:tabular-nums}
.tile .t-sub{grid-column:1/-1; font-size:10px; color:var(--text-tertiary)}
.tile svg{grid-column:1/-1; width:100%; height:20px; display:block}

.graph{
  border:1px solid var(--border); border-radius:var(--radius-sm);
  background:var(--bg-tertiary); padding:var(--s3);
}
.graph-head{display:flex; align-items:baseline; gap:var(--s3);
  margin-bottom:var(--s2)}
.graph-head .g-title{font-size:var(--fs-sm); font-weight:600}
.graph-head .g-scale{font-size:var(--fs-xs); color:var(--text-tertiary)}
.graph-head .spacer{flex:1}
.graph svg{width:100%; height:190px; display:block}
.graph .axis{font-size:10px; fill:var(--text-tertiary)}
.axis-x{display:flex; justify-content:space-between;
  font-size:10px; color:var(--text-tertiary); margin-top:var(--s1)}

.stats{display:grid; gap:var(--s2) var(--s4); margin-top:var(--s3);
  grid-template-columns:repeat(auto-fit,minmax(112px,1fr))}
.stat .s-name{font-size:10px; text-transform:uppercase; letter-spacing:.05em;
  color:var(--text-tertiary)}
.stat .s-val{font-size:var(--fs-sm); font-weight:600;
  font-variant-numeric:tabular-nums}

/* Memory composition, the way Task Manager shows it. */
.compose{display:flex; height:22px; border:1px solid var(--border);
  border-radius:3px; overflow:hidden; margin-top:var(--s2)}
.compose div{display:flex; align-items:center; justify-content:center;
  font-size:10px; font-weight:600; color:var(--on-fill); min-width:1px;
  white-space:nowrap; overflow:hidden}
.compose .c-weights{background:var(--accent)}
.compose .c-kv{background:var(--success)}
.compose .c-compute{background:var(--warning)}
.compose .c-os{background:var(--text-secondary)}
.compose .c-free{background:var(--bg-tertiary); color:var(--text-tertiary)}
.compose-legend{display:flex; flex-wrap:wrap; gap:var(--s1) var(--s3);
  margin-top:var(--s2); font-size:10px; color:var(--text-secondary)}
.compose-legend i{display:inline-block; width:8px; height:8px; border-radius:2px;
  margin-right:4px; vertical-align:middle}

/* The bottleneck sentence: the line that decides cores vs memory channels. */
.bn{
  display:flex; flex-wrap:wrap; align-items:baseline; gap:var(--s1) var(--s3);
  margin-top:var(--s3); padding:var(--s2) var(--s3);
  border:1px solid var(--border); border-left:3px solid var(--accent);
  border-radius:var(--radius-sm); background:var(--bg-tertiary);
  font-size:var(--fs-sm);
}
.bn .why{color:var(--text-secondary); font-size:var(--fs-xs)}
.g-legend{display:flex; flex-wrap:wrap; gap:var(--s1) var(--s3);
  margin-top:var(--s2); font-size:10px; color:var(--text-secondary)}
.g-legend i{display:inline-block; width:12px; height:2px; margin-right:5px;
  vertical-align:middle; border-radius:1px; background:currentColor}
.g-legend i.dash{height:0; background:none; border-top:2px dashed currentColor}
.g-legend i.tick{width:3px; height:8px; border-radius:1px}

/* Where it breaks. Run on demand: a four-axis sweep is seconds of simulation,
   so it must never hang off the live recompute. */
.cap-run{display:flex; flex-wrap:wrap; align-items:center;
  gap:var(--s3); margin-top:var(--s3)}
.cap-axes{display:flex; flex-wrap:wrap; gap:var(--s2); margin-top:var(--s3)}
.cap-axes button[aria-pressed="true"]{border-color:var(--accent); color:var(--accent)}
.cap-axes .spacer{flex:1; min-width:var(--s2)}
.badge{
  display:inline-block; font-size:10px; font-weight:600; letter-spacing:.04em;
  padding:1px var(--s2); border-radius:999px; border:1px solid var(--border);
  background:var(--bg-tertiary); color:var(--text-secondary);
}
.badge.weak{border-color:var(--error); color:var(--error)}
tbody tr.weak{background:var(--selected-bg); box-shadow:inset 3px 0 0 var(--error)}
.curve{
  border:1px solid var(--border); border-radius:var(--radius-sm);
  background:var(--bg-tertiary); padding:var(--s3); margin-top:var(--s3);
}
/* Uniform scaling here, unlike the time-series charts: this one has point
   markers, and a stretched circle reads as a different mark. */
.curve svg{width:100%; height:auto; display:block}
.curve .axis{font-size:10px; fill:var(--text-tertiary)}

/* --- Virtual lab -------------------------------------------------------
   A second screen, not a second product: it reuses the rail geometry, the
   card, the table and every token the sizing screen already uses. The only
   things that are new here are the ones the sizing screen has no equivalent
   for -- a channel strip, a findings list, and a transport for playing a
   recorded run back. */
nav.views{display:flex; gap:var(--s1); min-width:0; overflow-x:auto}
nav.views button{white-space:nowrap}
nav.views button[aria-pressed="true"]{
  border-color:var(--accent); color:var(--accent); background:var(--selected-bg);
}
main[hidden]{display:none}

.mach-tabs{display:flex; gap:var(--s2); margin-top:var(--s1)}
.mach-tabs button{flex:1}
.mach-tabs button[aria-pressed="true"]{
  border-color:var(--accent); color:var(--accent); background:var(--selected-bg);
}

.asm-pair{display:grid; gap:var(--s4);
  grid-template-columns:repeat(auto-fit,minmax(320px,1fr))}
.asm .headline{font:600 var(--fs-md)/1.45 var(--font); margin-top:var(--s2)}
.asm.editing{border-left:3px solid var(--accent)}
.asm dl{display:grid; grid-template-columns:auto minmax(0,1fr);
  gap:var(--s1) var(--s3); margin:var(--s3) 0 0; font-size:var(--fs-sm)}
.asm dt{font-size:var(--fs-xs); text-transform:uppercase; letter-spacing:.05em;
  color:var(--text-tertiary); align-self:center; white-space:nowrap}
.asm dd{margin:0; font-variant-numeric:tabular-nums; min-width:0}
.asm .was{color:var(--text-tertiary)}
.asm .loss{color:var(--error); font-weight:600}

/* Eight boxes, filled ones lit. The picture a datasheet draws of a board,
   because "2/8 채널" as a number is easy to read past. */
.chan{display:flex; gap:3px; margin-top:var(--s2)}
.chan i{flex:1; height:14px; border-radius:2px;
  background:var(--bg-tertiary); border:1px solid var(--border)}
.chan i.on{background:var(--accent); border-color:var(--accent)}

.find{list-style:none; margin:var(--s3) 0 0; padding:0;
  display:flex; flex-direction:column; gap:var(--s2)}
.find li{
  border:1px solid var(--border); border-left:3px solid var(--text-tertiary);
  border-radius:var(--radius-sm); background:var(--bg-tertiary);
  padding:var(--s2) var(--s3); font-size:var(--fs-sm);
}
.find li.f-error{border-left-color:var(--error)}
.find li.f-warn{border-left-color:var(--warning)}
.find li.f-info{border-left-color:var(--accent)}
.find .lv{font-size:10px; font-weight:600; letter-spacing:.05em;
  text-transform:uppercase; margin-right:var(--s2)}
.find li.f-error .lv{color:var(--error)}
.find li.f-warn .lv{color:var(--warning)}
.find li.f-info .lv{color:var(--accent)}
/* The remedy is not optional decoration: a warning without one just tells
   somebody their build is broken and leaves them there. */
.find .fix{display:block; margin-top:var(--s1); font-size:var(--fs-xs);
  color:var(--text-secondary)}

.gauges{display:grid; gap:var(--s3); margin-top:var(--s3);
  grid-template-columns:repeat(auto-fit,minmax(150px,1fr))}
.gauge{border:1px solid var(--border); border-radius:var(--radius-sm);
  background:var(--bg-tertiary); padding:var(--s2) var(--s3)}
.gauge .g-name{font-size:var(--fs-xs); font-weight:600; color:var(--text-secondary)}
.gauge .g-val{font-size:var(--fs-lg); font-weight:600; line-height:1.25;
  font-variant-numeric:tabular-nums}
.gauge .g-sub{font-size:10px; color:var(--text-tertiary)}
.bar{height:10px; margin-top:var(--s1); border-radius:2px; overflow:hidden;
  background:var(--bg-primary); border:1px solid var(--border)}
.bar span{display:block; height:100%; width:0; background:var(--accent)}
/* Four resources, four fills. Same rule as the task manager: if two meters
   are the same colour the reader assumes they are the same number. */
.g-queue span{background:var(--warning)}
.g-bw span{background:var(--success)}
.g-ram span{background:var(--text-secondary)}
.bar.hot span{background:var(--error)}

.transport{display:flex; flex-wrap:wrap; align-items:center; gap:var(--s2);
  margin-top:var(--s4)}
.transport .spacer{flex:1; min-width:var(--s2)}
.speeds{display:flex; gap:var(--s1)}
.speeds button[aria-pressed="true"]{border-color:var(--accent); color:var(--accent)}
.clock{font:600 var(--fs-md)/1 var(--mono); font-variant-numeric:tabular-nums}
.scrub{display:flex; align-items:center; gap:var(--s3); margin-top:var(--s3)}
.scrub input[type=range]{flex:1; min-width:120px; height:26px;
  accent-color:var(--accent); background:transparent; border:0; padding:0}
.progress{height:6px; border-radius:3px; margin-top:var(--s2);
  background:var(--bg-tertiary); border:1px solid var(--border); overflow:hidden}
.progress span{display:block; height:100%; width:0; background:var(--accent)}

.breach{
  display:flex; flex-wrap:wrap; align-items:baseline; gap:var(--s1) var(--s3);
  margin-top:var(--s3); padding:var(--s2) var(--s3);
  border:1px solid var(--border); border-left:3px solid var(--error);
  border-radius:var(--radius-sm); background:var(--bg-tertiary);
  font-size:var(--fs-sm);
}
.breach.clear{border-left-color:var(--success)}
.breach .why{color:var(--text-secondary); font-size:var(--fs-xs)}
.bench-run{display:flex; flex-wrap:wrap; align-items:center; gap:var(--s3);
  margin-top:var(--s3)}
.bench-metrics{display:flex; flex-wrap:wrap; gap:var(--s2); margin-top:var(--s3)}
.bench-metrics button[aria-pressed="true"]{border-color:var(--accent); color:var(--accent)}
.bench-chart svg{width:100%; height:220px; display:block}
.bench-chart .axis{font-size:10px; fill:var(--text-tertiary)}
.pgroup[hidden]{display:none}

/* --- Model performance -------------------------------------------------
   The default screen. It borrows the card, the table, the gauge and the
   metric picker the other two screens already use -- a new tab that looked
   like a different product would just be a second tool in the same window.
   The only new geometry is a two-up pair for things that only make sense
   read against each other: prefill against decode, and the three training
   verdicts against one another. */
.mb-pair{display:grid; gap:var(--s4);
  grid-template-columns:repeat(auto-fit,minmax(300px,1fr))}
.mb-metrics{display:flex; flex-wrap:wrap; gap:var(--s2); margin-bottom:var(--s3)}
.mb-metrics button[aria-pressed="true"]{border-color:var(--accent); color:var(--accent)}
.mb-chart{
  border:1px solid var(--border); border-radius:var(--radius-sm);
  background:var(--bg-tertiary); padding:var(--s3); margin-bottom:var(--s3);
}
.mb-chart svg{width:100%; height:210px; display:block}
.mb-chart .axis{font-size:10px; fill:var(--text-tertiary)}
/* The grid's first column is a row label, not a number, so it does not get the
   tabular alignment the measurements do. */
.mb-grid td:first-child,.mb-grid th:first-child{font-weight:600}
/* A cell whose operating point does not fit the assembled machine. The
   colour is the semantic error token, and the mark carries the meaning
   on its own so it survives a monochrome print or a colour deficiency. */
.mb-nofit{color:var(--error)}
.mb-nofit-mark{font-weight:700}
.mb-verdict{display:flex; flex-wrap:wrap; align-items:baseline;
  gap:var(--s1) var(--s2); margin-top:var(--s2); font-size:var(--fs-md)}
.mb-verdict .num{font-variant-numeric:tabular-nums; font-size:var(--fs-sm);
  color:var(--text-secondary)}

/* The two numbers a person actually feels -- how fast the answer streams, and
   how long they stare at nothing first -- set at the top and set large. The
   server total sits beside them on purpose: it is the figure those two get
   confused with, and putting it anywhere else invites quoting it as if it were
   what one user sees. */
.mb-head{display:grid; gap:var(--s3); margin-top:var(--s4);
  grid-template-columns:repeat(auto-fit,minmax(196px,1fr))}
.h-tile{
  border:1px solid var(--border); border-left:3px solid var(--accent);
  border-radius:var(--radius-sm); background:var(--bg-tertiary);
  padding:var(--s3) var(--s4);
}
.h-tile.total{border-left-color:var(--text-tertiary)}
.h-name{font-size:var(--fs-xs); font-weight:600; color:var(--text-secondary)}
.h-val{
  font-size:var(--fs-xl); font-weight:600; line-height:1.15;
  letter-spacing:-0.02em; font-variant-numeric:tabular-nums;
  margin-top:var(--s1);
}
.h-val .h-unit{font-size:var(--fs-sm); font-weight:600; letter-spacing:0;
  color:var(--text-secondary); margin-left:var(--s1)}
.h-sub{font-size:var(--fs-xs); color:var(--text-tertiary); margin-top:var(--s1)}
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
  <nav class="views" aria-label="화면 전환">
    <button id="view-model" type="button" aria-pressed="true">모델 성능</button>
    <button id="view-size" type="button" aria-pressed="false">자원</button>
    <button id="view-lab" type="button" aria-pressed="false">부하 테스트</button>
  </nav>
  <span class="spacer"></span>
  <button id="update" type="button" hidden></button>
  <button id="theme" type="button" aria-label="화면 테마 전환">테마: 자동</button>
</header>

<main id="screen-model">
  <form id="mb-rail" aria-label="모델 성능 조건">
    <fieldset>
      <legend>모델</legend>
      <div class="field">
        <label for="mb-model">LLM</label>
        <select id="mb-model"></select>
        <span class="hint" id="mb-model-hint"></span>
      </div>
      <div class="field">
        <label for="mb-quant">양자화</label>
        <select id="mb-quant"></select>
      </div>
    </fieldset>

    <fieldset>
      <legend>이 모델을 올릴 서버</legend>
      <div class="field">
        <label for="mb-cpu">CPU</label>
        <select id="mb-cpu"></select>
      </div>
      <div class="row">
        <div class="field">
          <label for="mb-sockets">소켓</label>
          <select id="mb-sockets"></select>
        </div>
        <div class="field">
          <label for="mb-slots">동시 슬롯</label>
          <input type="number" id="mb-slots" min="1" max="64" value="4">
        </div>
      </div>
      <div class="row">
        <div class="field">
          <label for="mb-dimm-gb">DIMM 용량(GB)</label>
          <input type="number" id="mb-dimm-gb" min="1" max="1024" step="8" value="32">
        </div>
        <div class="field">
          <label for="mb-dimm-count">DIMM 개수</label>
          <input type="number" id="mb-dimm-count" min="0" max="64" value="8">
        </div>
      </div>
      <div class="field">
        <label for="mb-os">운영체제</label>
        <select id="mb-os"></select>
      </div>
      <span class="hint" id="mb-hw-hint"></span>
    </fieldset>

    <fieldset>
      <legend>측정 조건</legend>
      <div class="row">
        <div class="field">
          <label for="mb-output-tokens">생성 토큰 수</label>
          <input type="number" id="mb-output-tokens" min="1" max="32000" step="32" value="256">
        </div>
        <div class="field">
          <label for="mb-train-samples">학습 샘플 수</label>
          <input type="number" id="mb-train-samples" min="100" max="10000000"
                 step="1000" value="10000">
        </div>
      </div>
      <span class="hint">배치·컨텍스트·동시 사용자 축은 서버가 고정한다 — 표의 열이 그 축이다.
        프롬프트 토큰 구성은 자원 화면의 값을 그대로 쓴다.</span>
      <div class="bench-run">
        <button id="mb-run" type="button">▶ 모델 성능 측정</button>
      </div>
      <span class="hint" id="mb-run-hint">실제 모델을 돌리지 않는다 — 카탈로그의 물리와 계수로 예측한다.</span>
    </fieldset>

    <fieldset>
      <legend>이 숫자들의 근거</legend>
      <span class="hint">이 도구는 손에 없는 서버를 산정한다. 그래서 실측을 받아오라고
        요구하지 않고 근거를 직접 들고 있다 — 아래가 예측에 쓰인 계수 전부다.</span>
      <div id="mb-evidence"></div>
    </fieldset>
  </form>

  <div id="mb-results" aria-live="polite">
    <div class="card">카탈로그를 불러오는 중…</div>
  </div>
</main>

<main id="screen-size" hidden>
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

<main id="screen-lab" hidden>
  <form id="lab-rail" aria-label="가상 서버 조립">
    <fieldset>
      <legend>머신</legend>
      <div class="mach-tabs" role="group" aria-label="편집할 머신">
        <button id="mach-a" type="button" aria-pressed="true">A</button>
        <button id="mach-b" type="button" aria-pressed="false" disabled>B</button>
      </div>
      <div class="check">
        <input type="checkbox" id="lab-compare">
        <label for="lab-compare">B와 비교</label>
      </div>
      <span class="hint">두 벌을 조립해 같은 부하로 돌리면 프레임이 겹쳐 그려진다.</span>
    </fieldset>

    <fieldset>
      <legend>가상 서버 조립</legend>
      <div class="field">
        <label for="lab-cpu">CPU</label>
        <select id="lab-cpu"></select>
      </div>
      <div class="row">
        <div class="field">
          <label for="lab-sockets">소켓</label>
          <select id="lab-sockets"></select>
        </div>
        <div class="field">
          <label for="lab-dimm-gb">DIMM 용량</label>
          <select id="lab-dimm-gb"></select>
        </div>
      </div>
      <div class="row">
        <div class="field">
          <label for="lab-dimm-count">DIMM 개수</label>
          <select id="lab-dimm-count"></select>
        </div>
        <div class="field">
          <label for="lab-slots">동시 슬롯</label>
          <input type="number" id="lab-slots" min="1" max="64">
        </div>
      </div>
      <span class="hint" id="lab-build-hint"></span>
    </fieldset>

    <fieldset>
      <legend>모델</legend>
      <div class="field">
        <label for="lab-model">LLM</label>
        <select id="lab-model"></select>
      </div>
      <div class="field">
        <label for="lab-quant">양자화</label>
        <select id="lab-quant"></select>
      </div>
      <span class="hint">프롬프트 토큰과 지연 SLA는 산정 화면의 값을 그대로 쓴다.</span>
    </fieldset>

    <fieldset>
      <legend>부하 테스트</legend>
      <div class="field">
        <label for="lab-profile">프로파일</label>
        <select id="lab-profile">
          <option value="replay">실측 재생 — 하루치를 그대로</option>
          <option value="ramp" selected>램프 — 부하를 올려 무너지는 지점을 찾는다</option>
          <option value="spike">스파이크 — 평시에서 급증</option>
          <option value="soak">소크 — 균일 부하 장시간</option>
        </select>
      </div>

      <div class="pgroup" id="pg-replay" hidden>
        <div class="field">
          <label for="bp-date">날짜</label>
          <input type="date" id="bp-date" value="2026-06-01">
        </div>
        <div class="row">
          <div class="field">
            <label for="bp-count">건수(0=하루 전체)</label>
            <input type="number" id="bp-count" min="0" max="100000" step="10" value="0">
          </div>
          <div class="field">
            <label for="bp-storm-size">스톰 크기</label>
            <input type="number" id="bp-storm-size" min="0" max="5000" value="40">
          </div>
        </div>
        <div class="field">
          <label for="bp-storms">스톰 횟수/일</label>
          <input type="number" id="bp-storms" min="0" max="100" value="2">
        </div>
      </div>

      <div class="pgroup" id="pg-ramp">
        <div class="row">
          <div class="field">
            <label for="bp-start-rate">시작 부하(건/일)</label>
            <input type="number" id="bp-start-rate" min="1" max="200000" step="50" value="100">
          </div>
          <div class="field">
            <label for="bp-end-rate">끝 부하(건/일)</label>
            <input type="number" id="bp-end-rate" min="1" max="200000" step="50" value="2000">
          </div>
        </div>
        <div class="field">
          <label for="bp-hours">구간(시간)</label>
          <input type="number" id="bp-hours" min="1" max="336" value="24">
        </div>
      </div>

      <div class="pgroup" id="pg-spike" hidden>
        <div class="row">
          <div class="field">
            <label for="bp-base-rate">평시 부하(건/일)</label>
            <input type="number" id="bp-base-rate" min="1" max="200000" step="10" value="165">
          </div>
          <div class="field">
            <label for="bp-peak-rate">급증 부하(건/일)</label>
            <input type="number" id="bp-peak-rate" min="1" max="200000" step="50" value="800">
          </div>
        </div>
        <div class="row">
          <div class="field">
            <label for="bp-spike-at">급증 시각(시)</label>
            <input type="number" id="bp-spike-at" min="0" max="335" value="12">
          </div>
          <div class="field">
            <label for="bp-spike-min">급증 길이(분)</label>
            <input type="number" id="bp-spike-min" min="1" max="1440" value="30">
          </div>
        </div>
      </div>

      <div class="pgroup" id="pg-soak" hidden>
        <div class="row">
          <div class="field">
            <label for="bp-rate">부하(건/일)</label>
            <input type="number" id="bp-rate" min="1" max="200000" step="50" value="300">
          </div>
          <div class="field">
            <label for="bp-soak-hours">구간(시간)</label>
            <input type="number" id="bp-soak-hours" min="1" max="336" value="72">
          </div>
        </div>
      </div>

      <div class="bench-run">
        <button id="lab-run" type="button">▶ 부하 테스트 실행</button>
      </div>
      <span class="hint" id="lab-run-hint">가상시간으로 즉시 완주한 뒤, 재생은 브라우저가 한다.</span>
    </fieldset>
  </form>

  <div id="lab-results" aria-live="polite">
    <div class="card">카탈로그를 불러오는 중…</div>
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

  // How long to keep waiting for the desktop bridge before giving up, and how
  // often to look for it. Waiting on the `pywebviewready` event alone is not
  // enough: if the event fired before this script attached its listener, the
  // listener never runs and the promise stays pending forever -- the window sits
  // on "카탈로그를 불러오는 중…" with nothing to click and no error, because a
  // promise that never settles cannot reach a .catch(). So poll for the object
  // as well, and fail loudly rather than hang.
  var BRIDGE_TIMEOUT_MS = 15000, BRIDGE_POLL_MS = 50;

  function bridgeUp(){
    return !!(window.pywebview && window.pywebview.api && window.pywebview.api.catalog);
  }

  // Tell Python the bridge answered. Python is waiting on this: if the signal
  // never arrives it starts a loopback server and reloads this window onto it,
  // so a dead bridge costs the operator a second instead of the whole app. Both
  // the event path and the polling path go through here, because the event is
  // exactly the thing that may already have fired.
  function bridgeSignal(){
    try { window.pywebview.api.bridge_ok(); } catch(e){}
  }

  function bridgeReady(){
    if(!DESKTOP) return Promise.resolve();
    if(bridgeUp()){ bridgeSignal(); return Promise.resolve(); }
    return new Promise(function(resolve, reject){
      var done = false, timer = null;
      function settle(ok){
        if(done) return;
        done = true;
        if(timer) clearInterval(timer);
        window.removeEventListener("pywebviewready", onReady);
        if(ok){ bridgeSignal(); resolve(); }
        // Python is already switching this window over to a loopback server by
        // the time this fires -- it stopped waiting before we did. Say what is
        // happening rather than handing the operator a command to type.
        else reject(new Error(
          "데스크톱 브리지가 " + (BRIDGE_TIMEOUT_MS / 1000) + "초 안에 준비되지 않았다. " +
          "서버 방식으로 자동 전환하는 중이다 — 잠시 기다려라. " +
          "화면이 그대로면 창을 닫고 다시 열어라."));
      }
      function onReady(){ settle(true); }
      window.addEventListener("pywebviewready", onReady, {once:true});
      // The event may already have fired. Polling is what actually covers that.
      var waited = 0;
      timer = setInterval(function(){
        if(bridgeUp()) return settle(true);
        waited += BRIDGE_POLL_MS;
        if(waited >= BRIDGE_TIMEOUT_MS) settle(false);
      }, BRIDGE_POLL_MS);
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
  function askCapacity(p){
    if(DESKTOP){
      // The bridge is a fixed surface in the packaged app. If this build of it
      // predates the capacity engine, say so plainly rather than throwing.
      if(!(window.pywebview.api && window.pywebview.api.capacity))
        return Promise.resolve({error:
          "이 데스크톱 빌드에는 과부하 분석이 연결되어 있지 않다. 서버 모드(svrspec gui)에서 실행해라."});
      return window.pywebview.api.capacity(p);
    }
    return fetch("/api/capacity", {method:"POST",
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

  // ---- charts (SVG, no libraries) -----------------------------------
  var SVGNS = "http://www.w3.org/2000/svg";
  function svgEl(tag, attrs){
    var n = document.createElementNS(SVGNS, tag);
    Object.keys(attrs || {}).forEach(function(k){ n.setAttribute(k, attrs[k]); });
    return n;
  }
  function areaPath(values, max, w, h){
    if(!values.length) return "";
    var span = values.length - 1 || 1;
    var top = max > 0 ? max : 1;
    var d = "M 0," + h.toFixed(2);
    values.forEach(function(v, i){
      var x = (i / span) * w, y = h - Math.min(v / top, 1) * h;
      d += " L " + x.toFixed(2) + "," + y.toFixed(2);
    });
    return d + " L " + w.toFixed(2) + "," + h.toFixed(2) + " Z";
  }
  function linePath(values, max, w, h){
    if(!values.length) return "";
    var span = values.length - 1 || 1;
    var top = max > 0 ? max : 1;
    return values.map(function(v, i){
      var x = (i / span) * w, y = h - Math.min(v / top, 1) * h;
      return (i ? "L" : "M") + x.toFixed(2) + "," + y.toFixed(2);
    }).join(" ");
  }
  function sparkline(values, max){
    var W = 100, H = 20;
    var svg = svgEl("svg", {viewBox: "0 0 " + W + " " + H, preserveAspectRatio: "none",
                            "aria-hidden": "true"});
    svg.appendChild(svgEl("path", {d: areaPath(values, max, W, H),
                                   fill: "var(--accent)", opacity: "0.22"}));
    svg.appendChild(svgEl("path", {d: linePath(values, max, W, H), fill: "none",
                                   stroke: "var(--accent)", "stroke-width": "1"}));
    return svg;
  }
  function flatLine(value, max, w, h){
    var y = (h - Math.min(Math.max(value, 0) / (max > 0 ? max : 1), 1) * h).toFixed(2);
    return "M0," + y + " L" + w.toFixed(2) + "," + y;
  }

  // One chart, up to four things on it: the bucket average, the peak inside the
  // bucket, a fixed reference level (reserved memory, say), and a tick wherever
  // work was actually queued. Average alone hides the storms; peak alone paints
  // a day that was idle 97% of the time as a solid wall of 100%.
  function bigChart(m){
    var W = 600, H = 180, PAD = 30, FOOT = 6;
    var svg = svgEl("svg", {viewBox: "0 0 " + W + " " + H,
                            preserveAspectRatio: "none", role: "img",
                            "aria-label": m.title + " 24시간 추이"});
    var iw = W - PAD, ih = H - 14 - FOOT;
    for(var i = 0; i <= 4; i++){
      var y = (ih / 4) * i;
      svg.appendChild(svgEl("line", {x1: PAD, y1: y.toFixed(1), x2: W, y2: y.toFixed(1),
                                     stroke: "var(--border)", "stroke-width": "1"}));
      var t = svgEl("text", {x: 0, y: (y + 4).toFixed(1), class: "axis"});
      t.textContent = fmtNum(m.max * (1 - i / 4));
      svg.appendChild(t);
    }
    var g = svgEl("g", {transform: "translate(" + PAD + ",0)"});
    if(m.peak){
      g.appendChild(svgEl("path", {d: areaPath(m.peak, m.max, iw, ih),
                                   fill: "var(--warning)", opacity: "0.14"}));
      g.appendChild(svgEl("path", {d: linePath(m.peak, m.max, iw, ih), fill: "none",
                                   stroke: "var(--warning)", "stroke-width": "1"}));
    }
    g.appendChild(svgEl("path", {d: areaPath(m.values, m.max, iw, ih),
                                 fill: "var(--accent)", opacity: "0.20"}));
    g.appendChild(svgEl("path", {d: linePath(m.values, m.max, iw, ih), fill: "none",
                                 stroke: "var(--accent)", "stroke-width": "1.5"}));
    if(m.reference !== null && m.reference !== undefined){
      g.appendChild(svgEl("path", {d: flatLine(m.reference, m.max, iw, ih), fill: "none",
                                   stroke: "var(--text-tertiary)", "stroke-width": "1.5",
                                   "stroke-dasharray": "5 4"}));
    }
    // Queueing, marked along the floor. Sitting at a ceiling is not overload;
    // a queue is. Keeping them visually separate is the whole point.
    if(m.queue){
      var span = m.queue.length - 1 || 1;
      m.queue.forEach(function(q, i){
        if(!q) return;
        var x = (i / span) * iw;
        g.appendChild(svgEl("rect", {x: x.toFixed(2), y: (ih + 2).toFixed(1),
                                     width: "3", height: String(FOOT - 2),
                                     fill: "var(--error)"}));
      });
    }
    svg.appendChild(g);
    return svg;
  }

  function chartLegend(items){
    var box = el("div", "g-legend");
    items.forEach(function(item){
      var span = el("span");
      span.style.color = "var(--" + item.token + ")";
      var swatch = el("i", item.kind || "");
      span.appendChild(swatch);
      var text = el("span", "", item.name);
      text.style.color = "var(--text-secondary)";
      span.appendChild(text);
      box.appendChild(span);
    });
    return box;
  }
  function fmtNum(v){
    if(v >= 100) return String(Math.round(v));
    if(v >= 10) return v.toFixed(0);
    if(v >= 1) return v.toFixed(1);
    return v.toFixed(2);
  }

  // ---- task manager ------------------------------------------------
  var tmMetric = "cpu";

  var AVG_LEGEND = {token: "accent", name: "버킷 평균"};
  var PEAK_LEGEND = {token: "warning", name: "버킷 내 피크"};
  var QUEUE_LEGEND = {token: "error", name: "큐 발생", kind: "tick"};

  // Every metric reads a different column of the same timeline. Nothing here
  // reuses another metric's numbers -- that reuse is what made the CPU, memory
  // and bandwidth graphs one line wearing three labels.
  function tmSeries(d, metric){
    var s = d.series, live = d.live, ceil = d.ceilings;
    if(metric === "queue"){
      return {key: metric, values: s.queue, queue: s.queue,
              max: Math.max(1, Math.max.apply(null, s.queue)),
              unit: "건", title: "큐 깊이", now: live.max_queue,
              tileValue: String(live.max_queue), tileText: "건 (구간 최대)",
              nowText: "최대 " + live.max_queue + " 건",
              legend: [{token: "accent", name: "대기 건수(구간 최대)"}],
              stats: [["최대 큐(예측)", live.max_queue + " 건"],
                      ["최대 큐(불리한 추정)", live.max_queue_pessimistic + " 건"],
                      ["도착", live.alarms + " 건/일"],
                      ["완료", live.completed + " 건"]],
              sub: "과부하의 정의는 이것이다 — 도착이 소진보다 빨라 대기가 쌓인 구간."};
    }
    if(metric === "active"){
      return {key: metric, values: s.active, queue: s.queue,
              max: Math.max(1, d.hardware.slots, Math.max.apply(null, s.active)),
              unit: "개", title: "동시 처리", now: live.peak_active,
              tileValue: String(live.peak_active),
              tileText: "/ " + d.hardware.slots + " 슬롯",
              nowText: "최대 " + live.peak_active + " / " + d.hardware.slots + " 슬롯",
              legend: [{token: "accent", name: "동시 처리 중(구간 최대)"}, QUEUE_LEGEND],
              stats: [["슬롯 점유율", live.slot_pct + " %"],
                      ["구성 슬롯", d.hardware.slots + " 개"],
                      ["최대 동시", live.peak_active + " 개"]],
              sub: "동시에 처리 중인 요청. 슬롯을 다 쓰고도 남으면 그 초과분이 큐로 간다."};
    }
    if(metric === "bandwidth"){
      return {key: metric, values: s.bandwidth_pct, peak: s.bandwidth_peak_pct,
              queue: s.queue,
              max: Math.max(100, Math.max.apply(null, s.bandwidth_peak_pct),
                            Math.max.apply(null, s.bandwidth_pct)),
              unit: "%", title: "메모리 대역폭", now: live.bandwidth_pct,
              tileValue: live.bandwidth_pct + "%",
              tileText: "피크 " + live.bandwidth_peak_pct + "% · " + live.bandwidth_avg_gbs + " GB/s",
              nowText: "하루 평균 " + live.bandwidth_avg_gbs + " GB/s · 천장 " +
                       ceil.bandwidth_gbs + " GB/s",
              legend: [AVG_LEGEND, PEAK_LEGEND, QUEUE_LEGEND],
              stats: [["하루 평균", live.bandwidth_avg_gbs + " GB/s"],
                      ["일하는 동안 평균", live.bandwidth_busy_gbs + " GB/s"],
                      ["피크", live.bandwidth_peak_pct + " %"],
                      ["천장", ceil.bandwidth_gbs + " GB/s"],
                      ["근거", ceil.bandwidth_confidence]],
              sub: "decode가 태우는 자원이다. 토큰 하나마다 가중치 전체를 DRAM에서 한 번 훑는다 — " +
                   "생성 중인 요청은 정의상 이 천장에 붙어 있다."};
    }
    if(metric === "compute"){
      return {key: metric, values: s.compute_pct, peak: s.compute_peak_pct,
              queue: s.queue,
              max: Math.max(100, Math.max.apply(null, s.compute_peak_pct),
                            Math.max.apply(null, s.compute_pct)),
              unit: "%", title: "연산(벡터 유닛)", now: live.compute_pct,
              tileValue: live.compute_pct + "%",
              tileText: "피크 " + live.compute_peak_pct + "%",
              nowText: "피크 " + live.compute_peak_pct + " % · 천장 " +
                       ceil.compute_tflops + " TFLOP/s",
              legend: [AVG_LEGEND, PEAK_LEGEND, QUEUE_LEGEND],
              stats: [["피크", live.compute_peak_pct + " %"],
                      ["천장", ceil.compute_tflops + " TFLOP/s"],
                      ["prefill 비중", d.bottleneck.prefill_share + " %"],
                      ["근거", ceil.compute_confidence]],
              sub: "prefill이 태우는 자원이다. 대역폭 그래프와 반대로 움직인다 — " +
                   "프롬프트를 처리하는 동안 DRAM은 거의 놀고, 생성이 시작되면 뒤집힌다."};
    }
    if(metric === "ram"){
      // Two lines on purpose. The moving one is what the run touched; the flat
      // one is what llama.cpp reserved up front and therefore what the server
      // has to be given. Drawing only the moving line under-orders the RAM.
      return {key: metric, values: s.ram_used_gb, queue: s.queue,
              reference: ceil.allocated_gb,
              max: Math.max(ceil.installed_gb, ceil.allocated_gb),
              unit: "GB", title: "메모리", now: live.ram_pct,
              tileValue: live.ram_pct + "%",
              tileText: "예약 " + ceil.allocated_gb + " / " + live.ram_installed_gb + " GB",
              nowText: "실사용 최대 " + live.ram_live_peak_gb + " GB · 예약 " +
                       ceil.allocated_gb + " GB · 장착 " + live.ram_installed_gb + " GB",
              legend: [{token: "accent", name: "실사용(가중치+실제 KV)"},
                       {token: "text-tertiary", name: "예약(전체 컨텍스트 × 슬롯)", kind: "dash"},
                       QUEUE_LEGEND],
              stats: [["실사용 최대", live.ram_live_peak_gb + " GB"],
                      ["예약", ceil.allocated_gb + " GB"],
                      ["KV 실사용 최대", live.kv_live_peak_gb + " GB"],
                      ["KV 예약", ceil.kv_reserved_gb + " GB"],
                      ["장착", live.ram_installed_gb + " GB"]],
              sub: "실사용은 하루 동안 움직인다(진행 중인 요청의 KV). 서버에 꽂아야 할 양은 " +
                   "그 아래가 아니라 예약선이다."};
    }
    return {key: "cpu", values: s.cpu_pct, queue: s.queue, max: 100, unit: "%",
            title: "CPU", now: live.cpu_pct, tileValue: live.cpu_pct + "%",
            tileText: "하루 " + d.bottleneck.busy_minutes + "분 작업",
            nowText: live.cpu_pct + " % · 하루 " + d.bottleneck.busy_minutes + "분",
            legend: [{token: "accent", name: "일하고 있던 시간 비율"}, QUEUE_LEGEND],
            stats: [["하루 작업시간", d.bottleneck.busy_minutes + " 분"],
                    ["슬롯 점유율", live.slot_pct + " %"],
                    ["완료", live.completed + " 건"],
                    ["판정", VERDICT[live.verdict] || live.verdict]],
            sub: "llama.cpp는 일하는 동안 스레드를 전부 물고 있다 — 이 선은 '바쁜 시간의 비율'이지 " +
                 "코어 여유가 아니다."};
  }

  var TM_METRICS = [
    ["cpu", "CPU"], ["ram", "메모리"], ["bandwidth", "대역폭"],
    ["compute", "연산"], ["active", "동시 처리"], ["queue", "큐"]
  ];

  function bottleneckLine(d){
    var b = d.bottleneck;
    var box = el("div", "bn");
    var strong = el("strong", "", "병목: " + b.label +
      (b.resource === "none" ? "" : " 바운드"));
    box.appendChild(strong);
    box.appendChild(el("span", "", b.sentence));
    box.appendChild(el("span", "why", b.advice));
    return box;
  }

  function taskManager(d){
    var wrap = el("div", "card");
    wrap.appendChild(el("div", "label", "작업관리자 · 성능"));
    var hw = d.hardware;
    wrap.appendChild(el("p", "cpu", hw.label));
    var bits = el("div", "hw");
    [hw.cores + "코어 / " + hw.threads + "스레드 @ " + hw.ghz + "GHz",
     hw.sockets + "소켓", hw.isa.toUpperCase(), hw.memory,
     hw.bandwidth_gbs + " GB/s", "L3 " + hw.l3_mb + "MB",
     hw.slots + " 슬롯", hw.tdp_w + "W",
     "알람 " + d.live.alarms + "개/일"].forEach(function(x){
      bits.appendChild(el("span", "", x));
    });
    if(hw.passmark) bits.appendChild(el("span", "", "CPU Mark " + hw.passmark.toLocaleString()));
    wrap.appendChild(bits);
    wrap.appendChild(bottleneckLine(d));
    wrap.appendChild(el("p", "hint-row", d.bottleneck.overload));
    if(d.bottleneck.overran_s > 0){
      wrap.appendChild(el("div", "note",
        "하루가 끝난 뒤에도 " + d.bottleneck.overran_h +
        "시간 더 일했다 — 백로그가 다음 날로 넘어간다. 이 그래프의 마지막 구간에 그 초과분이 몰려 있다."));
    }
    if(d.bottleneck.notes.length){
      var ul = el("ul", "notes");
      d.bottleneck.notes.forEach(function(n){ ul.appendChild(el("li", "", n)); });
      wrap.appendChild(ul);
    }

    var grid = el("div", "tm"); grid.style.marginTop = "16px";
    var tiles = el("div", "tm-tiles");
    var pane = el("div");

    function draw(){
      var m = tmSeries(d, tmMetric);
      pane.textContent = "";
      var box = el("div", "graph");
      var head = el("div", "graph-head");
      head.appendChild(el("span", "g-title", m.title));
      head.appendChild(el("span", "spacer"));
      head.appendChild(el("span", "g-scale",
        "24시간 · " + (d.series.bucket_s / 60) + "분 단위 · 최대 " +
        fmtNum(m.max) + " " + m.unit));
      box.appendChild(head);
      box.appendChild(bigChart(m));
      var ax = el("div", "axis-x");
      ["00:00", "06:00", "12:00", "18:00", "24:00"].forEach(function(t){
        ax.appendChild(el("span", "", t));
      });
      box.appendChild(ax);
      box.appendChild(chartLegend(m.legend));
      pane.appendChild(box);

      var stats = el("div", "stats");
      m.stats.concat([["현재", m.nowText]]).forEach(function(pair){
        var st = el("div", "stat");
        st.appendChild(el("div", "s-name", pair[0]));
        st.appendChild(el("div", "s-val", pair[1]));
        stats.appendChild(st);
      });
      pane.appendChild(stats);
      pane.appendChild(el("p", "hint-row", m.sub));

      if(tmMetric === "ram") pane.appendChild(composition(d));
    }

    TM_METRICS.forEach(function(pair){
      var key = pair[0];
      var m = tmSeries(d, key);
      var tile = el("button", "tile");
      tile.type = "button";
      tile.setAttribute("aria-pressed", String(key === tmMetric));
      tile.appendChild(el("div", "t-name", pair[1]));
      tile.appendChild(el("div", "t-val", m.tileValue));
      tile.appendChild(sparkline(m.peak || m.values, m.max));
      tile.appendChild(el("div", "t-sub", m.tileText));
      tile.addEventListener("click", function(){
        tmMetric = key;
        Array.prototype.forEach.call(tiles.children, function(t, i){
          t.setAttribute("aria-pressed", String(TM_METRICS[i][0] === tmMetric));
        });
        draw();
      });
      tiles.appendChild(tile);
    });

    grid.appendChild(tiles);
    grid.appendChild(pane);
    wrap.appendChild(grid);
    draw();
    return wrap;
  }

  function composition(d){
    var b = d.ram_breakdown, box = el("div");
    box.appendChild(el("div", "s-name", "메모리 구성"));
    var bar = el("div", "compose");
    var total = b.installed_gb || b.subtotal_gb;
    var parts = [["c-weights", "가중치", b.weights_gb], ["c-kv", "KV", b.kv_gb],
                 ["c-compute", "컴퓨트", b.compute_gb], ["c-os", "OS", b.os_gb],
                 ["c-free", "여유", Math.max(0, total - b.subtotal_gb)]];
    parts.forEach(function(part){
      var seg = el("div", part[0]);
      seg.style.width = (100 * part[2] / total) + "%";
      seg.title = part[1] + " " + part[2].toFixed(2) + " GiB";
      if(part[2] / total > 0.12) seg.textContent = part[2].toFixed(1);
      bar.appendChild(seg);
    });
    box.appendChild(bar);
    var legend = el("div", "compose-legend");
    parts.forEach(function(part){
      var item = el("span");
      var swatch = el("i");
      swatch.style.background = "var(--" + ({
        "c-weights": "accent", "c-kv": "success", "c-compute": "warning",
        "c-os": "text-secondary", "c-free": "border"}[part[0]]) + ")";
      item.appendChild(swatch);
      item.appendChild(document.createTextNode(part[1] + " " + part[2].toFixed(2) + " GiB"));
      legend.appendChild(item);
    });
    box.appendChild(legend);
    box.appendChild(el("p", "hint-row",
      "이 막대는 예약량이다 — llama.cpp가 슬롯마다 전체 컨텍스트를 미리 잡는다. 위 그래프의 " +
      "움직이는 선은 그중 실제로 건드린 부분(진행 중인 요청의 KV)이고, 서버에 꽂아야 할 양은 " +
      "이 막대 쪽이다. 알람 개수는 더 많은 슬롯이 필요해질 때 이 막대를 끌어올린다."));
    return box;
  }

  var RES_COLS = [
    ["개수", function(r){ return r.alarms; }, "num"],
    ["필요 슬롯", function(r){ return r.slots; }, "num"],
    ["CPU 사용률", null, ""],
    ["RAM", null, ""],
    ["RAM 예약", function(r){ return r.ram_used_gb + " / " + r.ram_installed_gb + " GB"; }, "num"],
    ["RAM 실사용", function(r){ return r.ram_live_peak_gb + " GB"; }, "num"],
    ["KV 예약", function(r){ return r.kv_gb + " GiB"; }, "num"],
    ["대역폭 평균", function(r){ return r.bandwidth_avg_gbs + " / " + r.bandwidth_gbs; }, "num"],
    ["대역폭 피크", function(r){ return r.bandwidth_peak_pct + " %"; }, "num"],
    ["연산 피크", function(r){ return r.compute_peak_pct + " %"; }, "num"],
    ["병목", function(r){ return r.bound_label; }, ""],
    ["최대 큐", function(r){ return r.max_queue; }, "num"],
    ["평상시 p95", function(r){ return r.p95_steady + " s"; }, "num"],
    ["스톰 소진", function(r){ return r.storm_min + " 분"; }, "num"],
    ["작업시간", function(r){ return r.work_minutes + " 분"; }, "num"]
  ];

  function resourceTable(d){
    var box = el("div", "scroll");
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
        td.appendChild(i === 2 ? meter(r.cpu_pct, r.cpu_pct.toFixed(1) + "%")
                               : meter(r.ram_pct, r.ram_pct.toFixed(0) + "%"));
        tr.appendChild(td);
      });
      var v = el("td", "v-" + r.verdict, VERDICT[r.verdict] || r.verdict);
      if(r.reasons && r.reasons.length) v.title = r.reasons.join(" / ");
      tr.appendChild(v);
      tbody.appendChild(tr);
    });
    tbl.appendChild(tbody); box.appendChild(tbl);

    var wrap = el("div");
    wrap.appendChild(box);
    wrap.appendChild(el("p", "hint-row",
      "각 개수마다 SLA를 지키는 데 필요한 최소 슬롯을 따로 풀어서 낸 값이다. " +
      "슬롯을 늘려도 개선이 멈추면 그 지점에서 멈춘다 — 총 처리량이 연산이나 대역폭에 " +
      "걸린 것이고, 그때는 동시성이 아니라 더 빠른 CPU가 답이다."));
    wrap.appendChild(el("p", "hint-row",
      "판정 · p95 · 스톰 소진 · 최대 큐는 오차를 반영한 불리한 추정 기준이고, " +
      "리소스 계열(CPU · RAM · 대역폭 · 연산)은 예측 중앙값 기준이다 — 위 그래프와 같은 값이다."));
    return wrap;
  }

  // ---- where it breaks ---------------------------------------------
  // Request-driven on purpose. One axis is a bracketing search over whole
  // simulated days and the four together run for seconds to tens of seconds;
  // wiring that to the live recompute would turn every keystroke into a stall.
  // So: a button, a cache keyed on the inputs, and nothing automatic.
  var capacity = {key: null, data: null, error: null, axis: null,
                  metric: null, busy: false};
  var capHost = null;

  function capKey(p){
    var out = {};
    Object.keys(p).forEach(function(k){
      // Neither of these changes the answer, and letting them invalidate the
      // cache would throw away a ten-second analysis for nothing.
      if(k === "volumes" || k === "only_pass") return;
      out[k] = p[k];
    });
    return JSON.stringify(out);
  }

  function axisOf(d, name){
    return d.axes.filter(function(a){ return a.axis === name; })[0] || d.axes[0];
  }
  function metricOf(a){
    return capacity.metric || (a.limiter === "storm-drain" ? "storm" : "p95");
  }

  function curveChart(d, a, metric){
    var sla = metric === "storm" ? d.sla.storm_drain_min : d.sla.sla_seconds;
    var unit = metric === "storm" ? "분" : "초";
    var title = metric === "storm" ? "스톰 소진" : "평상시 p95";
    var W = 640, H = 250, L = 46, R = 14, T = 14, B = 38;
    var iw = W - L - R, ih = H - T - B;

    var pts = a.points.slice().sort(function(x, y){ return x.value - y.value; });
    var xs = pts.map(function(p){ return p.value; });
    var lo = Math.min.apply(null, xs), hi = Math.max.apply(null, xs);
    // The search ramps geometrically, so a linear axis squashes everything
    // below the knee into the left edge. Log it once the span is wide.
    var logX = lo > 0 && hi / lo >= 20;
    function fx(v){
      if(hi <= lo) return L + iw / 2;   // a single evaluated point sits centred
      var t = logX
        ? (Math.log(Math.max(v, lo)) - Math.log(lo)) / (Math.log(hi) - Math.log(lo))
        : (v - lo) / (hi - lo);
      return L + Math.min(Math.max(t, 0), 1) * iw;
    }
    function value(p){ return metric === "storm" ? p.storm_min : p.p95_steady_s; }
    var worst = Math.max.apply(null, pts.map(value));
    // Failing points can run to hours. Clamp the scale so the SLA line and the
    // knee stay legible; clamped markers sit on the top edge and say so.
    var top = Math.max(sla * 1.25, Math.min(worst, sla * 4)) || 1;
    function fy(v){ return T + ih - Math.min(Math.max(v, 0) / top, 1) * ih; }

    var svg = svgEl("svg", {viewBox: "0 0 " + W + " " + H, role: "img",
      "aria-label": a.label + "을(를) 올리며 측정한 " + title + " 곡선"});
    var i, y;
    for(i = 0; i <= 4; i++){
      y = T + (ih / 4) * i;
      svg.appendChild(svgEl("line", {x1: L, y1: y.toFixed(1), x2: (W - R), y2: y.toFixed(1),
                                     stroke: "var(--border)", "stroke-width": "1"}));
      var lbl = svgEl("text", {x: L - 6, y: (y + 3.5).toFixed(1), class: "axis",
                               "text-anchor": "end"});
      lbl.textContent = fmtNum(top * (1 - i / 4));
      svg.appendChild(lbl);
    }
    var yName = svgEl("text", {x: L - 6, y: (T - 4).toFixed(1), class: "axis",
                               "text-anchor": "end"});
    yName.textContent = unit;
    svg.appendChild(yName);

    // The SLA, drawn once. Everything above it is a miss.
    svg.appendChild(svgEl("line", {x1: L, y1: fy(sla).toFixed(1), x2: (W - R),
                                   y2: fy(sla).toFixed(1), stroke: "var(--warning)",
                                   "stroke-width": "1.5", "stroke-dasharray": "5 4"}));
    var slaText = svgEl("text", {x: (W - R), y: (fy(sla) - 5).toFixed(1), class: "axis",
                                 "text-anchor": "end"});
    slaText.textContent = "목표 " + sla + unit;
    svg.appendChild(slaText);

    if(a.knee){
      svg.appendChild(svgEl("line", {x1: fx(a.knee.value).toFixed(1), y1: T,
                                     x2: fx(a.knee.value).toFixed(1), y2: (T + ih),
                                     stroke: "var(--accent)", "stroke-width": "1",
                                     "stroke-dasharray": "3 3"}));
      var kneeText = svgEl("text", {x: (fx(a.knee.value) + 4).toFixed(1),
                                    y: (T + 10), class: "axis"});
      kneeText.textContent = "무릎 " + a.knee.value + a.unit;
      svg.appendChild(kneeText);
    }

    svg.appendChild(svgEl("path", {
      d: pts.map(function(p, k){
        return (k ? "L" : "M") + fx(p.value).toFixed(2) + "," + fy(value(p)).toFixed(2);
      }).join(" "),
      fill: "none", stroke: "var(--text-tertiary)", "stroke-width": "1.5"}));

    pts.forEach(function(p){
      var clipped = value(p) > top;
      var dot = svgEl("circle", {cx: fx(p.value).toFixed(2), cy: fy(value(p)).toFixed(2),
        r: clipped ? "3" : "4",
        fill: p.ok ? "var(--success)" : "var(--error)",
        stroke: "var(--bg-tertiary)", "stroke-width": "1"});
      var tip = svgEl("title", {});
      tip.textContent = a.label + " " + p.value + a.unit + " · " +
        (VERDICT[p.verdict] || p.verdict) + " · p95 " + p.p95_steady_s + "초 · 스톰 " +
        p.storm_min + "분 · 최대 큐 " + p.max_queue + " · 병목 " + p.bound_label +
        (clipped ? " · 축 범위를 넘어 위쪽 끝에 붙여 그렸다" : "");
      dot.appendChild(tip);
      svg.appendChild(dot);
    });

    var marks = [lo, a.knee ? a.knee.value : null, hi];
    marks.forEach(function(v, k){
      if(v === null) return;
      var tx = svgEl("text", {x: fx(v).toFixed(1), y: (T + ih + 14), class: "axis",
                              "text-anchor": k === 0 ? "start" : k === 2 ? "end" : "middle"});
      tx.textContent = v + a.unit;
      svg.appendChild(tx);
    });
    var xName = svgEl("text", {x: (L + iw / 2), y: (T + ih + 30), class: "axis",
                               "text-anchor": "middle"});
    xName.textContent = a.label + (logX ? " (로그 눈금)" : "");
    svg.appendChild(xName);
    return svg;
  }

  function capacityTable(d){
    var box = el("div", "scroll"), tbl = el("table");
    var thead = el("thead"), htr = el("tr");
    [["축", ""], ["현재 부하", "num"], ["무릎(통과 상한)", "num"], ["첫 실패", "num"],
     ["여유 배수", "num"], ["한계요인", ""], ["무릎에서의 병목", ""]
    ].forEach(function(h){ htr.appendChild(el("th", h[1], h[0])); });
    thead.appendChild(htr); tbl.appendChild(thead);

    var tbody = el("tbody");
    d.axes.forEach(function(a){
      var tr = el("tr");
      tr.tabIndex = 0;
      tr.setAttribute("role", "button");
      if(a.weakest) tr.classList.add("weak");
      if(a.axis === capacity.axis) tr.classList.add("sel");

      var name = el("td");
      name.appendChild(document.createTextNode(a.label + " "));
      if(a.weakest) name.appendChild(el("span", "badge weak", "먼저 무너짐"));
      tr.appendChild(name);
      tr.appendChild(el("td", "num", a.baseline + a.unit));
      tr.appendChild(el("td", "num", a.knee ? a.knee.value + a.unit : "없음"));
      tr.appendChild(el("td", "num",
        a.breaks_at ? a.breaks_at.value + a.unit : (a.hit_ceiling ? "상한까지 없음" : "-")));
      var head = el("td", "num " + (!a.knee || a.headroom < 1 ? "v-fail" :
        a.headroom < 2 ? "v-marginal" : "v-pass"));
      head.textContent = !a.knee ? "이미 미달"
        : a.headroom === null ? "제한 없음" : a.headroom.toFixed(2) + "배";
      tr.appendChild(head);
      tr.appendChild(el("td", "", a.limiter_label));
      tr.appendChild(el("td", "", a.knee ? a.knee.bound_label : "-"));

      function choose(){ capacity.axis = a.axis; capacity.metric = null; renderCapacity(); }
      tr.addEventListener("click", choose);
      tr.addEventListener("keydown", function(ev){
        if(ev.key === "Enter" || ev.key === " "){ ev.preventDefault(); choose(); }
      });
      tbody.appendChild(tr);
    });
    tbl.appendChild(tbody); box.appendChild(tbl);
    return box;
  }

  function capacityBody(d){
    var box = el("div");
    var weakest = d.weakest_axis ? axisOf(d, d.weakest_axis) : null;
    var head = el("div", "bn");
    if(weakest){
      head.appendChild(el("strong", "", "먼저 무너지는 축: " + weakest.label));
      var room = weakest.headroom;
      head.appendChild(el("span", "",
        room !== null && room < 1
          ? "현재 " + weakest.baseline + weakest.unit + "는 이미 한계 " +
            weakest.knee.value + weakest.unit + "를 넘었다 (여유 " + room.toFixed(2) +
            "배) · " + weakest.limiter_label
          : "현재 " + weakest.baseline + weakest.unit + "에서 " + weakest.knee.value +
            weakest.unit + "까지 버틴다" +
            (room === null ? "" : " (여유 " + room.toFixed(2) + "배)") +
            " · " + weakest.limiter_label));
      head.appendChild(el("span", "why",
        "무릎은 SLA를 지키는 마지막 부하다. 그 위는 오차를 반영한 불리한 추정에서 미달로 판정된다."));
    } else if(d.axes.every(function(x){ return x.hit_ceiling; })){
      head.appendChild(el("strong", "", "탐색 상한까지 무너지지 않았다"));
      head.appendChild(el("span", "why",
        "이 부하에 대해서는 이 빌드가 과할 만큼 여유가 있다는 뜻이다."));
    } else {
      // The opposite case, and the one worth being blunt about: no knee exists
      // because the *current* setting already misses.
      var broken = d.axes.filter(function(x){ return !x.knee; });
      head.appendChild(el("strong", "", "현재 부하에서 이미 SLA를 지키지 못한다"));
      head.appendChild(el("span", "",
        broken.map(function(x){ return x.label; }).join(" · ") +
        " 축은 지금 설정에서 이미 미달이라 무릎이 없다."));
      head.appendChild(el("span", "why",
        "더 빠른 CPU나 더 작은 모델·짧은 프롬프트로 통과 구간에 들어간 다음 다시 분석해라."));
    }
    box.appendChild(head);

    var a = axisOf(d, capacity.axis || d.axis);
    capacity.axis = a.axis;
    box.appendChild(capacityTable(d));

    if(d.axes.some(function(x){ return !x.knee; })){
      box.appendChild(el("p", "hint-row",
        "무릎이 없는 축은 지금 설정에서 이미 미달이라 여유를 잴 기준 자체가 없다. " +
        "먼저 통과 구간으로 들어간 다음 다시 보면 그 축의 한계도 나온다."));
    }

    var picker = el("div", "cap-axes");
    d.axes.forEach(function(x){
      var b = el("button", "", x.label);
      b.type = "button";
      b.setAttribute("aria-pressed", String(x.axis === a.axis));
      b.addEventListener("click", function(){
        capacity.axis = x.axis; capacity.metric = null; renderCapacity();
      });
      picker.appendChild(b);
    });
    picker.appendChild(el("span", "spacer"));
    [["p95", "평상시 p95"], ["storm", "스톰 소진"]].forEach(function(pair){
      var b = el("button", "", pair[1]);
      b.type = "button";
      b.setAttribute("aria-pressed", String(metricOf(a) === pair[0]));
      b.addEventListener("click", function(){
        capacity.metric = pair[0]; renderCapacity();
      });
      picker.appendChild(b);
    });
    box.appendChild(picker);

    var curve = el("div", "curve");
    var chead = el("div", "graph-head");
    chead.appendChild(el("span", "g-title", a.label + " 부하 곡선"));
    chead.appendChild(el("span", "spacer"));
    chead.appendChild(el("span", "g-scale", a.points.length + "개 지점 평가"));
    curve.appendChild(chead);
    curve.appendChild(curveChart(d, a, metricOf(a)));
    curve.appendChild(chartLegend([
      {token: "success", name: "SLA 통과"}, {token: "error", name: "미달"},
      {token: "warning", name: "목표선", kind: "dash"},
      {token: "accent", name: "무릎", kind: "dash"}
    ]));
    box.appendChild(curve);

    box.appendChild(el("p", "hint-row",
      a.breaks_at
        ? a.label + "이(가) " + a.breaks_at.value + a.unit + "에 닿으면 " + a.limiter_label +
          ". 그때 병목은 " + a.breaks_at.bound_label + "이고 필요 RAM은 " +
          a.breaks_at.ram_needed_gb + " GB로 올라간다."
        : a.label + " 축에서는 탐색 상한까지 SLA가 깨지지 않았다."));
    if(a.breaks_at && a.breaks_at.reasons.length){
      var ul = el("ul", "notes");
      a.breaks_at.reasons.forEach(function(r){ ul.appendChild(el("li", "", r)); });
      box.appendChild(ul);
    }
    if(a.notes.length){
      var un = el("ul", "notes");
      a.notes.forEach(function(n){ un.appendChild(el("li", "", n)); });
      box.appendChild(un);
    }
    return box;
  }

  function renderCapacity(){
    if(!capHost) return;
    capHost.textContent = "";
    var fresh = !!capacity.data && capacity.key === capKey(params());
    var card = el("div", "card");
    card.appendChild(el("div", "label", "과부하 지점"));
    card.appendChild(el("p", "prose",
      "이 하드웨어에 이 모델을 올렸을 때 부하를 어디까지 올릴 수 있는지, 그리고 무엇이 먼저 " +
      "무너지는지를 축별로 찾는다. 각 축을 기하급수적으로 올려 SLA가 깨지는 구간을 잡고 " +
      "그 사이를 이분 탐색한다."));
    var run = el("div", "cap-run");
    var btn = el("button", "",
      capacity.busy ? "분석 중…" : fresh ? "다시 분석" : "과부하 분석 실행");
    btn.type = "button";
    btn.disabled = capacity.busy;
    btn.addEventListener("click", runCapacity);
    run.appendChild(btn);
    run.appendChild(el("span", "hint",
      "축 4개 전수 탐색 · 수 초에서 수십 초 · 입력을 바꿔도 자동으로 다시 돌지 않는다"));
    card.appendChild(run);
    if(capacity.busy){
      card.appendChild(el("p", "hint-row",
        "축마다 하루치 시뮬레이션을 여러 번 돌린다. 그동안 나머지 화면은 계속 쓸 수 있다."));
    }
    if(capacity.error){
      card.appendChild(el("div", "note", "과부하 분석 실패: " + capacity.error));
    }
    if(fresh){
      card.appendChild(capacityBody(capacity.data));
    } else if(capacity.data && !capacity.busy){
      card.appendChild(el("p", "hint-row",
        "입력이 바뀌었다. 이전 분석 결과는 지금 조건과 맞지 않으므로 다시 실행해야 한다."));
    }
    capHost.appendChild(card);
  }

  function runCapacity(){
    if(capacity.busy || !selectedCpu) return;
    var p = params(), key = capKey(p);
    capacity.busy = true;
    capacity.error = null;
    renderCapacity();
    askCapacity(p).then(function(d){
      capacity.busy = false;
      if(!d || d.error){
        capacity.error = (d && d.error) || "알 수 없는 오류";
        capacity.data = null;
      } else {
        capacity.key = key; capacity.data = d;
        capacity.axis = d.axis; capacity.metric = null; capacity.error = null;
      }
      renderCapacity();
    }).catch(function(err){
      capacity.busy = false;
      capacity.error = String(err);
      renderCapacity();
    });
  }

  function loadResources(){
    if(!selectedCpu) return;
    var mine = ++seqRes;
    askResources(params()).then(function(d){
      if(mine !== seqRes) return;
      var host = document.getElementById("resources");
      if(!host) return;
      host.textContent = "";
      capHost = null;
      if(d.error){
        host.appendChild(el("div", "note", "리소스 산정 실패: " + d.error));
        return;
      }
      host.appendChild(section("토큰 전달 시뮬레이터", d.hardware.label,
                               timelinePanel(d.timeline)));
      host.appendChild(section("작업관리자", d.hardware.label, taskManager(d)));
      host.appendChild(section("개수별 리소스", d.rows.length + "개 구간",
                               resourceTable(d)));
      capHost = el("div");
      host.appendChild(section("과부하 지점", d.hardware.label, capHost));
      renderCapacity();
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
    // Only speak up when a row really has no source behind it. Sending the
    // operator off to collect vendor datasheets is not a caveat, it is the
    // tool declining to answer -- and this tool exists to size servers nobody
    // has access to, so "go measure it" is never an available answer.
    if(d.unverified.length){
      var n = el("div", "note");
      n.appendChild(el("strong", "", "출처 없는 스펙 " + d.unverified.length + "건. "));
      n.appendChild(document.createTextNode(
        "아래 수치는 그만큼 넓게 읽어야 한다: " +
        d.unverified.slice(0, 14).map(function(u){ return u.kind + ":" + u.id; }).join(", ") +
        (d.unverified.length > 14 ? " 외 " + (d.unverified.length - 14) + "건" : "")));
      results.appendChild(n);
    }
  }

  // ---- virtual lab -------------------------------------------------
  // Two costs, two paths, and the split is the whole design of this screen.
  // Assembly is arithmetic over the catalogue, so it runs on every change of
  // every dropdown -- that immediacy is the point, because the operator has to
  // see the bandwidth collapse while their hand is still on the control. The
  // bench replays a whole load and hands back a 600-frame recording, so it runs
  // on a button and goes stale when the inputs move. Same rule, and the same
  // reason, as the capacity search on the other screen.
  var STILL = false;
  try{ STILL = window.matchMedia("(prefers-reduced-motion: reduce)").matches; }catch(e){}

  var MACHINES = {A: null, B: null};
  var lab = {
    active: "A", compare: false,
    asm: {A: null, B: null}, asmError: {A: null, B: null},
    bench: {A: null, B: null},
    benchKey: null, benchBusy: false, benchError: null, benchStale: false,
    metric: "cpu_pct", optionSig: "", seq: 0, benchSeq: 0
  };
  var bench = null;   // live playback state, or null

  function num(v){ return (typeof v === "number" && isFinite(v)) ? v : 0; }
  function byId(id){ return document.getElementById(id); }
  function labVal(id){ var n = byId(id); return n ? n.value : ""; }
  function labNum(id, dflt){
    var v = Number(labVal(id));
    return isFinite(v) ? v : dflt;
  }
  function fillSelect(sel, pairs){
    sel.textContent = "";
    pairs.forEach(function(pair){
      var o = document.createElement("option");
      o.value = String(pair[0]);
      o.textContent = pair[1];
      sel.appendChild(o);
    });
  }
  function cpuById(id){
    return (catalog.cpus || []).filter(function(c){ return c.id === id; })[0] || null;
  }
  function fmtClock(t){
    // Hours can pass 24 here: a soak profile is days long, and wrapping the
    // clock would make the second day look like the first.
    var s = Math.max(0, Math.round(num(t)));
    var h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
    return (h < 10 ? "0" : "") + h + ":" + (m < 10 ? "0" : "") + m;
  }

  // ---- machine state ----
  function defaultMachine(){
    var cpus = catalog.cpus || [];
    var cpu = cpus[0] || {id: "", mem_channels: 8, sockets_max: 1};
    // Open on the widest memory bus in the catalogue: it is the build where an
    // under-populated board costs the most, and therefore the one worth showing.
    cpus.forEach(function(c){ if(c.mem_channels > cpu.mem_channels) cpu = c; });
    return {cpu: cpu.id, sockets: 1, dimm_gb: 32,
            dimm_count: cpu.mem_channels || 8,
            model: byId("model").value, quant: byId("quant").value, slots: 4};
  }
  function readMachine(){
    var m = MACHINES[lab.active];
    if(!m) return;
    m.cpu = labVal("lab-cpu") || m.cpu;
    m.sockets = labNum("lab-sockets", m.sockets);
    m.dimm_gb = labNum("lab-dimm-gb", m.dimm_gb);
    m.dimm_count = labNum("lab-dimm-count", m.dimm_count);
    m.model = labVal("lab-model") || m.model;
    m.quant = labVal("lab-quant") || m.quant;
    m.slots = labNum("lab-slots", m.slots);
  }
  function writeMachine(){
    var m = MACHINES[lab.active];
    byId("lab-cpu").value = m.cpu;
    fillSockets();
    byId("lab-sockets").value = String(m.sockets);
    byId("lab-model").value = m.model;
    byId("lab-quant").value = m.quant;
    byId("lab-slots").value = String(m.slots);
    lab.optionSig = "";     // the DIMM lists belong to a machine, not to the form
  }
  function fillSockets(){
    var m = MACHINES[lab.active], cpu = cpuById(labVal("lab-cpu"));
    var most = Math.max(1, (cpu && cpu.sockets_max) || 1);
    var list = [];
    for(var i = 1; i <= most; i++) list.push([i, i + "소켓"]);
    fillSelect(byId("lab-sockets"), list);
    if(m.sockets > most) m.sockets = most;
    byId("lab-sockets").value = String(m.sockets);
  }

  // The DIMM count list is built from the board, not from the engine's list of
  // "sensible" builds: two DIMMs in an eight-channel board is exactly the
  // mistake this screen exists to show, and it must stay reachable from the
  // dropdown. Capacities do come from the engine, because those are whatever
  // the catalogue actually stocks.
  function fillDimmSelects(d){
    var m = MACHINES[lab.active];
    var caps = [];
    (d.options || []).forEach(function(o){
      if(o.dimm_gb > 0 && caps.indexOf(o.dimm_gb) < 0) caps.push(o.dimm_gb);
    });
    if(!caps.length) caps = [m.dimm_gb];
    caps.sort(function(a, b){ return a - b; });
    var total = Math.max(1, num(d.channels.total) || 8);
    var most = Math.max(1, Math.min(64, total * 2));
    var changed = false;
    // Snap first, then label: the count labels quote the chosen capacity, so
    // building them against a capacity this board cannot take would print one
    // wrong list before correcting itself.
    if(caps.indexOf(m.dimm_gb) < 0){ m.dimm_gb = caps[0]; changed = true; }
    if(m.dimm_count > most){ m.dimm_count = most; changed = true; }
    if(m.dimm_count < 1){ m.dimm_count = 1; changed = true; }
    var sig = lab.active + "|" + caps.join(",") + "|" + most + "|" + m.dimm_gb;

    if(sig !== lab.optionSig){
      lab.optionSig = sig;
      fillSelect(byId("lab-dimm-gb"), caps.map(function(gb){ return [gb, gb + " GB"]; }));
      var counts = [];
      for(var i = 1; i <= most; i++){
        var pop = Math.min(i, total), dpc = Math.ceil(i / total);
        // Capacity and channel fill on the same line, because reading those two
        // apart is exactly how somebody orders 128 GB at a quarter of the speed.
        counts.push([i, i + "장 · " + (i * m.dimm_gb) + "GB · " + pop + "/" + total +
                        "채널" + (dpc > 1 ? " · " + dpc + " DPC" : "")]);
      }
      fillSelect(byId("lab-dimm-count"), counts);
    }
    byId("lab-dimm-gb").value = String(m.dimm_gb);
    byId("lab-dimm-count").value = String(m.dimm_count);
    return changed;
  }

  // ---- transports ----
  function labParams(key){
    var m = MACHINES[key], p = params();
    p.cpu = m.cpu; p.sockets = m.sockets;
    p.dimm_gb = m.dimm_gb; p.dimm_count = m.dimm_count;
    p.model = m.model; p.quant = m.quant; p.slots = m.slots;
    p.name = key;
    delete p.volumes; delete p.only_pass;   // neither reaches the lab engine
    return p;
  }
  var NO_LAB_BRIDGE =
    "이 데스크톱 빌드에는 가상 랩이 연결되어 있지 않다. 서버 모드(svrspec gui)에서 실행해라.";
  function askLab(p){
    if(DESKTOP){
      if(!(window.pywebview.api && window.pywebview.api.lab))
        return Promise.resolve({error: NO_LAB_BRIDGE});
      return window.pywebview.api.lab(p);
    }
    return fetch("/api/lab", {method:"POST",
                              headers:{"Content-Type":"application/json"},
                              body: JSON.stringify(p)})
      .then(function(r){ return r.json(); });
  }
  function askBench(p){
    if(DESKTOP){
      if(!(window.pywebview.api && window.pywebview.api.bench))
        return Promise.resolve({error: NO_LAB_BRIDGE});
      return window.pywebview.api.bench(p);
    }
    return fetch("/api/bench", {method:"POST",
                                headers:{"Content-Type":"application/json"},
                                body: JSON.stringify(p)})
      .then(function(r){ return r.json(); });
  }

  function activeKeys(){ return lab.compare ? ["A", "B"] : ["A"]; }

  function runLab(){
    if(!catalog || !MACHINES.A) return;
    var mine = ++lab.seq, keys = activeKeys();
    Promise.all(keys.map(function(k){
      return askLab(labParams(k))
        .then(function(d){ return {key: k, d: d}; })
        .catch(function(err){ return {key: k, d: {error: String(err)}}; });
    })).then(function(list){
      if(mine !== lab.seq) return;      // a newer request already answered
      var snapped = false;
      list.forEach(function(item){
        if(!item.d || item.d.error){
          lab.asmError[item.key] = (item.d && item.d.error) || "알 수 없는 오류";
          lab.asm[item.key] = null;
        } else {
          lab.asmError[item.key] = null;
          lab.asm[item.key] = item.d;
          if(item.key === lab.active && fillDimmSelects(item.d)) snapped = true;
        }
      });
      if(!lab.compare){ lab.asm.B = null; lab.asmError.B = null; }
      markBenchStale();
      renderLab();
      // The dropdown snapped to a value this board can take. Ask once more with
      // the corrected build; the second answer cannot snap again, so this ends.
      if(snapped) runLab();
    });
  }

  // ---- bench request ----
  function showProfileGroup(){
    var kind = labVal("lab-profile");
    ["replay", "ramp", "spike", "soak"].forEach(function(k){
      byId("pg-" + k).hidden = (k !== kind);
    });
  }
  function benchRequest(key){
    var p = labParams(key), kind = labVal("lab-profile"), profile = {};
    if(kind === "replay"){
      profile.date = labVal("bp-date") || "2026-06-01";
      profile.count = labNum("bp-count", 0);
      profile.storm_size = labNum("bp-storm-size", 40);
      profile.storms_per_day = labNum("bp-storms", 2);
    } else if(kind === "spike"){
      profile.base_rate = labNum("bp-base-rate", 165);
      profile.peak_rate = labNum("bp-peak-rate", 800);
      profile.spike_at_h = labNum("bp-spike-at", 12);
      profile.spike_minutes = labNum("bp-spike-min", 30);
    } else if(kind === "soak"){
      profile.rate = labNum("bp-rate", 300);
      profile.hours = labNum("bp-soak-hours", 72);
    } else {
      profile.start_rate = labNum("bp-start-rate", 100);
      profile.end_rate = labNum("bp-end-rate", 2000);
      profile.hours = labNum("bp-hours", 24);
    }
    p.kind = kind; p.profile = profile; p.frames = 600;
    return p;
  }
  function benchSig(){
    return JSON.stringify(activeKeys().map(benchRequest));
  }
  function markBenchStale(){
    if(lab.benchKey && lab.benchKey !== benchSig()) lab.benchStale = true;
  }

  function runBench(){
    if(lab.benchBusy) return;
    var keys = activeKeys();
    // An error-level finding means this is not a machine anybody can order, so
    // numbers about its behaviour would be fiction. The server refuses too.
    if(!keys.every(function(k){ return lab.asm[k] && lab.asm[k].ok; })) return;
    var sig = benchSig(), mine = ++lab.benchSeq;
    lab.benchBusy = true; lab.benchError = null;
    renderLab();
    Promise.all(keys.map(function(k){
      return askBench(benchRequest(k))
        .then(function(d){ return {key: k, d: d}; })
        .catch(function(err){ return {key: k, d: {error: String(err)}}; });
    })).then(function(list){
      if(mine !== lab.benchSeq) return;
      lab.benchBusy = false;
      lab.bench = {A: null, B: null};
      list.forEach(function(item){
        if(!item.d || item.d.error) lab.benchError = (item.d && item.d.error) || "알 수 없는 오류";
        else if(item.d.blocked) lab.benchError = item.d.blocked;
        else lab.bench[item.key] = item.d;
      });
      if(lab.bench.A){ lab.benchKey = sig; lab.benchStale = false; }
      renderLab();
    });
  }

  // ---- assembly rendering ----
  var LEVEL_WORD = {error: "오류", warn: "경고", info: "참고"};

  function findingList(items){
    var ul = el("ul", "find");
    items.forEach(function(f){
      var li = el("li", "f-" + f.level);
      // The word as well as the colour: a red bar alone is not a judgement
      // anybody can read out loud, and it is invisible to a colour-blind eye.
      li.appendChild(el("span", "lv", LEVEL_WORD[f.level] || f.level));
      li.appendChild(document.createTextNode(f.message));
      if(f.remedy) li.appendChild(el("span", "fix", "고치는 법 · " + f.remedy));
      ul.appendChild(li);
    });
    return ul;
  }

  function channelStrip(c){
    var box = el("div", "chan");
    box.setAttribute("role", "img");
    box.setAttribute("aria-label",
      "메모리 채널 " + c.total + "개 중 " + c.populated + "개 장착");
    var total = Math.max(0, Math.min(64, num(c.total)));
    for(var i = 0; i < total; i++) box.appendChild(el("i", i < c.populated ? "on" : ""));
    return box;
  }

  function lossy(now, full, unit, lossPct){
    var span = el("span");
    span.appendChild(document.createTextNode(num(now).toLocaleString() + unit));
    if(lossPct !== null && lossPct !== undefined && lossPct >= 0.5){
      span.appendChild(el("span", "was", " (전 채널 " + num(full).toLocaleString() + unit + " "));
      span.appendChild(el("span", "loss", "−" + lossPct + "%"));
      span.appendChild(el("span", "was", ")"));
    }
    return span;
  }

  function assemblyCard(key, d){
    var card = el("div", "card asm" + (key === lab.active ? " editing" : ""));
    card.appendChild(el("div", "label",
      "머신 " + key + (key === lab.active ? " · 편집 중" : "")));
    card.appendChild(el("p", "headline", d.headline));
    card.appendChild(channelStrip(d.channels));
    card.appendChild(el("p", "hint",
      d.channels.populated + "/" + d.channels.total + " 채널 장착 · " +
      d.bandwidth.gbs + " GB/s · " + d.memory.label +
      (d.channels.dimms_per_channel > 1
        ? " · " + d.channels.dimms_per_channel + " DPC" : "")));

    var dl = el("dl");
    function row(name, value){
      dl.appendChild(el("dt", "", name));
      var dd = el("dd");
      if(typeof value === "string") dd.textContent = value;
      else dd.appendChild(value);
      dl.appendChild(dd);
    }
    var cpu = d.cpu;
    row("CPU", cpu.label + " · " + cpu.cores + "코어/" + cpu.threads + "스레드 @ " +
               cpu.ghz + "GHz · " + String(cpu.isa).toUpperCase());
    row("DIMM", d.machine.dimm_count + " × " + d.machine.dimm_gb + "GB = " +
                d.ram.total_gb + " GB");
    row("대역폭", lossy(d.bandwidth.gbs, d.bandwidth.full_gbs, " GB/s",
                      d.bandwidth.loss_pct));
    row("prefill", num(d.throughput.prefill_tps).toLocaleString() + " tok/s");
    row("decode", lossy(d.throughput.decode_tps_single, d.throughput.decode_tps_full,
                       " tok/s", d.throughput.decode_loss_pct));
    row("모델", d.model.name + " · " + d.model.quant + " · " + d.model.slots + " 슬롯 · " +
                "오차 ±" + d.throughput.uncertainty_pct + "%");
    row("모델 RAM", d.ram.used_gb + " / " + d.ram.total_gb + " GB" +
                    (d.ram.pct === null ? "" : " (" + d.ram.pct + "%)"));
    card.appendChild(dl);

    if(d.findings.length) card.appendChild(findingList(d.findings));
    else card.appendChild(el("p", "hint-row",
      "지적 사항 없음 — 채널이 모두 차 있고 용량도 맞는다."));
    return card;
  }

  // ---- bench rendering ----
  var BENCH_METRICS = [
    ["cpu_pct", "CPU", "%"], ["queued", "대기 큐", "건"],
    ["bw_pct", "대역폭", "%"], ["ram_gb", "RAM", "GB"],
    ["p95_so_far_s", "p95", "초"], ["offered_rate", "부하율", "건/일"]
  ];
  function metricLabel(key){
    var hit = BENCH_METRICS.filter(function(m){ return m[0] === key; })[0];
    return hit ? hit[1] : key;
  }
  function metricUnit(key){
    var hit = BENCH_METRICS.filter(function(m){ return m[0] === key; })[0];
    return hit ? hit[2] : "";
  }

  var STAT_ROWS = [
    ["received", "도착", "건"], ["delivered", "전달", "건"], ["dropped", "드롭", "건"],
    ["p50_s", "p50", "초"], ["p95_s", "p95", "초"], ["p99_s", "p99", "초"],
    ["max_s", "최대 지연", "초"], ["max_queue", "최대 큐", "건"],
    ["storm_drain_s", "스톰 소진", "초"], ["tokens_generated", "생성 토큰", "tok"]
  ];

  function statsGrid(d){
    var box = el("div", "stats"), s = d.stats || {};
    STAT_ROWS.forEach(function(row){
      if(s[row[0]] === undefined || s[row[0]] === null) return;
      var cell = el("div", "stat");
      cell.appendChild(el("div", "s-name", row[1]));
      var v = s[row[0]];
      // Counts keep their thousands separator; seconds are quoted to two
      // decimals, because the third one is well inside the model's error bar.
      var text = typeof v !== "number" ? String(v)
        : (v % 1 === 0 ? v.toLocaleString() : v.toFixed(2));
      cell.appendChild(el("div", "s-val", text + (row[2] ? " " + row[2] : "")));
      box.appendChild(cell);
    });
    if(s.slot_utilisation !== undefined && s.slot_utilisation !== null){
      var u = el("div", "stat");
      u.appendChild(el("div", "s-name", "슬롯 점유"));
      u.appendChild(el("div", "s-val", Math.round(num(s.slot_utilisation) * 100) + " %"));
      box.appendChild(u);
    }
    var verdict = el("div", "stat");
    verdict.appendChild(el("div", "s-name", "SLA"));
    verdict.appendChild(el("div", "s-val " + (d.breach ? "v-fail" : "v-pass"),
      d.breach ? "미달" : "통과"));
    box.appendChild(verdict);
    return box;
  }

  function breachLine(d){
    var b = d.breach;
    var box = el("div", "breach" + (b ? "" : " clear"));
    if(!b){
      box.appendChild(el("strong", "", "끝까지 SLA를 지켰다"));
      box.appendChild(el("span", "",
        "이 프로파일이 덮는 구간 내내 p95가 " + d.sla.sla_seconds + "초를 넘지 않았다."));
      box.appendChild(el("span", "why",
        "무너지는 지점을 보려면 램프의 끝 부하를 올리거나 더 큰 모델로 다시 돌려라."));
      return box;
    }
    box.appendChild(el("strong", "", "무너지는 지점"));
    box.appendChild(el("span", "",
      fmtClock(b.t_s) + " · 부하 " + Math.round(num(b.offered_rate)).toLocaleString() +
      "건/일 · p95 " + num(b.p95_s).toFixed(1) + "초 (목표 " + d.sla.sla_seconds + "초)"));
    box.appendChild(el("span", "why",
      "여기서 처음으로 지연 SLA를 넘겼다. 아래 차트의 세로선이 그 시각이다."));
    return box;
  }

  // Whole run at a glance: one line per machine, the SLA where it applies, and
  // a vertical mark at each breach. The playhead is a separate element on
  // purpose -- moving one attribute per frame is far cheaper than rebuilding
  // 600 points sixty times a second.
  function benchChart(runs, metric, span){
    var W = 640, H = 200, L = 46, R = 12, T = 12, B = 24;
    var iw = W - L - R, ih = H - T - B;
    var cols = runs.map(function(r){
      return {key: r.key, run: r.d,
              values: r.d.frames.map(function(f){ return num(f[metric]); })};
    });
    var max = 1;
    cols.forEach(function(c){
      c.values.forEach(function(v){ if(v > max) max = v; });
    });
    if(metric === "cpu_pct" || metric === "bw_pct") max = Math.max(max, 100);
    var sla = runs[0].d.sla ? num(runs[0].d.sla.sla_seconds) : 0;
    if(metric === "p95_so_far_s" && sla) max = Math.max(max, sla * 1.2);

    var svg = svgEl("svg", {viewBox: "0 0 " + W + " " + H,
      preserveAspectRatio: "none", role: "img",
      "aria-label": metricLabel(metric) + " 전체 구간 추이"});
    var i, y;
    for(i = 0; i <= 4; i++){
      y = T + (ih / 4) * i;
      svg.appendChild(svgEl("line", {x1: L, y1: y.toFixed(1), x2: (W - R),
        y2: y.toFixed(1), stroke: "var(--border)", "stroke-width": "1"}));
      var lbl = svgEl("text", {x: (L - 6), y: (y + 3.5).toFixed(1), class: "axis",
                               "text-anchor": "end"});
      lbl.textContent = fmtNum(max * (1 - i / 4));
      svg.appendChild(lbl);
    }
    var g = svgEl("g", {transform: "translate(" + L + "," + T + ")"});
    var TONE = {A: "var(--accent)", B: "var(--warning)"};
    cols.forEach(function(c){
      var attrs = {d: linePath(c.values, max, iw, ih), fill: "none",
                   stroke: TONE[c.key] || "var(--accent)", "stroke-width": "1.5"};
      if(c.key === "B") attrs["stroke-dasharray"] = "5 3";
      g.appendChild(svgEl("path", attrs));
    });
    if(metric === "p95_so_far_s" && sla){
      g.appendChild(svgEl("path", {d: flatLine(sla, max, iw, ih), fill: "none",
        stroke: "var(--warning)", "stroke-width": "1.5", "stroke-dasharray": "5 4"}));
    }
    cols.forEach(function(c){
      var b = c.run.breach;
      if(!b) return;
      var x = Math.max(0, Math.min(1, num(b.t_s) / span)) * iw;
      g.appendChild(svgEl("line", {x1: x.toFixed(1), y1: 0, x2: x.toFixed(1),
        y2: ih.toFixed(1), stroke: "var(--error)", "stroke-width": "1.5",
        "stroke-dasharray": "4 3"}));
      var tag = svgEl("text", {x: (x + 4).toFixed(1), y: 10, class: "axis"});
      tag.textContent = c.key + " 붕괴 " + fmtClock(b.t_s);
      g.appendChild(tag);
    });
    var head = svgEl("line", {x1: 0, y1: 0, x2: 0, y2: ih.toFixed(1),
      stroke: "var(--text-primary)", "stroke-width": "1.5", opacity: "0.75"});
    g.appendChild(head);
    svg.appendChild(g);

    var legend = [{token: "accent", name: "머신 A"}];
    if(cols.length > 1) legend.push({token: "warning", name: "머신 B", kind: "dash"});
    if(metric === "p95_so_far_s" && sla)
      legend.push({token: "warning", name: "지연 SLA", kind: "dash"});
    if(cols.some(function(c){ return c.run.breach; }))
      legend.push({token: "error", name: "SLA 붕괴 지점", kind: "tick"});

    return {svg: svg, head: head, legend: legend,
            toX: function(t){ return Math.max(0, Math.min(1, t / span)) * iw; }};
  }

  // Column names are translated where the engine's field is known and passed
  // through where it is not: the row shape belongs to the bench, so a field it
  // adds later must show up as an extra column rather than disappear.
  var WORST_NAME = {
    alarm_id: "알람", severity: "심각도", storm_id: "스톰",
    arrived_s: "도착", queue_wait_s: "대기", ttft_s: "첫 토큰",
    generate_s: "생성", deliver_s: "전송", total_s: "합계", slot: "슬롯"
  };

  function worstTable(rows){
    var keys = Object.keys(rows[0]).slice(0, 12);
    var box = el("div", "scroll"), tbl = el("table");
    var thead = el("thead"), htr = el("tr");
    keys.forEach(function(k){
      htr.appendChild(el("th", typeof rows[0][k] === "number" ? "num" : "",
                         WORST_NAME[k] || k));
    });
    thead.appendChild(htr); tbl.appendChild(thead);
    var tbody = el("tbody");
    rows.forEach(function(r){
      var tr = el("tr");
      keys.forEach(function(k){
        var v = r[k];
        if(v === null || v === undefined){ tr.appendChild(el("td", "", "-")); return; }
        if(typeof v !== "number"){ tr.appendChild(el("td", "", String(v))); return; }
        // Arrival is a clock reading in a run that can span days; the rest are
        // durations, and two decimals is as fine as any of them are known.
        tr.appendChild(el("td", "num",
          k === "arrived_s" ? fmtClock(v)
            : /_s$/.test(k) ? v.toFixed(2) + "s"
            : String(Math.round(v * 100) / 100)));
      });
      tbody.appendChild(tr);
    });
    tbl.appendChild(tbody); box.appendChild(tbl);
    return box;
  }

  // The player. The server finished the whole run in milliseconds, so nothing
  // is streaming and nothing is polled -- the browser is what makes time pass
  // here, which is why 600x and a scrub bar are possible at all.
  function playbackPanel(runs){
    var primary = runs[0].d, frames = primary.frames;
    var span = num(primary.profile && primary.profile.span_s) ||
               num(frames.length ? frames[frames.length - 1].t_s : 0) || 1;
    var box = el("div");

    var maxQueue = 1, installed = Math.max(1, num(primary.machine.ram.total_gb));
    var deliveredTo = [], acc = 0;
    frames.forEach(function(f){
      if(f.queued > maxQueue) maxQueue = f.queued;
      acc += num(f.delivered); deliveredTo.push(acc);
    });
    var total = num(primary.profile && primary.profile.total_alarms) || acc;

    // Four resources, four different columns of the frame. Nothing here is
    // another gauge's number rescaled -- that reuse is what made the old
    // resource graphs one series wearing three labels.
    var GAUGES = [
      {name: "CPU 가동", cls: "",
       pct: function(f){ return num(f.cpu_pct); },
       val: function(f){ return Math.round(num(f.cpu_pct)) + "%"; },
       sub: function(f){ return "동시 " + f.active + " / " + primary.sla.slots + " 슬롯"; }},
      {name: "대기 큐", cls: "g-queue",
       pct: function(f){ return 100 * num(f.queued) / maxQueue; },
       val: function(f){ return num(f.queued).toLocaleString() + "건"; },
       sub: function(){ return "구간 최대 " + maxQueue.toLocaleString() + "건"; }},
      {name: "메모리 대역폭", cls: "g-bw",
       pct: function(f){ return num(f.bw_pct); },
       val: function(f){ return num(f.bw_gbs).toFixed(1) + " GB/s"; },
       sub: function(f){ return Math.round(num(f.bw_pct)) + "% · 연산 " +
                                Math.round(num(f.compute_pct)) + "%"; }},
      {name: "RAM 실사용", cls: "g-ram",
       pct: function(f){ return 100 * num(f.ram_gb) / installed; },
       val: function(f){ return num(f.ram_gb).toFixed(1) + " GB"; },
       sub: function(f){ return "장착 " + installed + " GB · KV " +
                                num(f.kv_gb).toFixed(2) + " GB"; }}
    ];
    var gauges = el("div", "gauges");
    var widgets = GAUGES.map(function(gd){
      var cell = el("div", "gauge");
      cell.setAttribute("role", "img");
      cell.appendChild(el("div", "g-name", gd.name));
      var value = el("div", "g-val", "-");
      var bar = el("div", "bar " + gd.cls), fill = el("span");
      bar.appendChild(fill);
      var sub = el("div", "g-sub", "");
      cell.appendChild(value); cell.appendChild(bar); cell.appendChild(sub);
      gauges.appendChild(cell);
      return {d: gd, cell: cell, value: value, bar: bar, fill: fill, sub: sub};
    });
    box.appendChild(gauges);

    var controls = el("div", "transport");
    var playBtn = el("button", "", "▶ 재생");
    playBtn.type = "button";
    playBtn.setAttribute("aria-pressed", "false");
    controls.appendChild(playBtn);
    var speeds = el("div", "speeds");
    controls.appendChild(speeds);
    controls.appendChild(el("span", "spacer"));
    var clock = el("div", "clock", "00:00");
    controls.appendChild(clock);
    var counter = el("span", "hint", "");
    controls.appendChild(counter);
    box.appendChild(controls);

    var progress = el("div", "progress"), progressFill = el("span");
    progress.appendChild(progressFill);
    progress.setAttribute("role", "img");
    box.appendChild(progress);

    var scrub = el("div", "scrub");
    scrub.appendChild(el("span", "hint", "재생 위치"));
    var range = document.createElement("input");
    range.type = "range"; range.min = "0"; range.max = "1000"; range.step = "1";
    range.value = "0";
    range.setAttribute("aria-label", "재생 위치");
    scrub.appendChild(range);
    box.appendChild(scrub);

    var picker = el("div", "bench-metrics");
    BENCH_METRICS.forEach(function(m){
      var b = el("button", "", m[1]);
      b.type = "button";
      b.setAttribute("aria-pressed", String(m[0] === lab.metric));
      b.addEventListener("click", function(){
        lab.metric = m[0];
        Array.prototype.forEach.call(picker.children, function(x, i){
          x.setAttribute("aria-pressed", String(BENCH_METRICS[i][0] === lab.metric));
        });
        drawChart();
        paint(state.t, false);
      });
      picker.appendChild(b);
    });
    box.appendChild(picker);

    var chartHost = el("div", "graph bench-chart");
    chartHost.style.marginTop = "12px";
    box.appendChild(chartHost);
    var chart = null;
    function drawChart(){
      chartHost.textContent = "";
      var head = el("div", "graph-head");
      head.appendChild(el("span", "g-title",
        metricLabel(lab.metric) + " · 전체 구간"));
      head.appendChild(el("span", "spacer"));
      head.appendChild(el("span", "g-scale",
        fmtClock(span) + " · " + frames.length + "프레임 · " + metricUnit(lab.metric)));
      chartHost.appendChild(head);
      chart = benchChart(runs, lab.metric, span);
      chartHost.appendChild(chart.svg);
      chartHost.appendChild(chartLegend(chart.legend));
    }
    drawChart();

    function indexAt(t){
      if(!frames.length) return 0;
      var i = Math.round((t / span) * (frames.length - 1));
      return Math.max(0, Math.min(frames.length - 1, i));
    }
    function paint(t, moveScrub){
      var i = indexAt(t), f = frames[i];
      if(!f) return;
      widgets.forEach(function(w){
        var pct = Math.max(0, Math.min(100, w.d.pct(f)));
        w.fill.style.width = pct + "%";
        if(pct >= 90) w.bar.classList.add("hot"); else w.bar.classList.remove("hot");
        w.value.textContent = w.d.val(f);
        w.sub.textContent = w.d.sub(f);
        w.cell.setAttribute("aria-label",
          w.d.name + " " + w.d.val(f) + " · " + w.d.sub(f));
      });
      clock.textContent = fmtClock(t);
      counter.textContent = "전달 " + deliveredTo[i].toLocaleString() + " / " +
        total.toLocaleString() + "건 · p95 " + num(f.p95_so_far_s).toFixed(1) +
        "초 · 부하 " + Math.round(num(f.offered_rate)).toLocaleString() + "건/일";
      var frac = Math.max(0, Math.min(1, t / span));
      progressFill.style.width = (100 * frac) + "%";
      progress.setAttribute("aria-label",
        "재생 진행 " + Math.round(frac * 100) + "%");
      if(moveScrub) range.value = String(Math.round(1000 * frac));
      if(chart){
        var x = chart.toX(t).toFixed(1);
        chart.head.setAttribute("x1", x);
        chart.head.setAttribute("x2", x);
      }
    }

    var state = {t: 0, speed: 60, playing: false, raf: 0, last: 0};
    bench = state;
    function stop(){
      if(state.raf) cancelAnimationFrame(state.raf);
      state.raf = 0; state.playing = false;
      playBtn.textContent = "▶ 재생";
      playBtn.setAttribute("aria-pressed", "false");
    }
    function step(now){
      var dt = (now - state.last) / 1000;
      state.last = now;
      state.t += dt * state.speed;
      if(state.t >= span){ state.t = span; paint(state.t, true); stop(); return; }
      paint(state.t, true);
      state.raf = requestAnimationFrame(step);
    }
    playBtn.addEventListener("click", function(){
      if(state.playing){ stop(); return; }
      if(state.t >= span) state.t = 0;
      state.playing = true;
      playBtn.textContent = "■ 일시정지";
      playBtn.setAttribute("aria-pressed", "true");
      state.last = performance.now();
      state.raf = requestAnimationFrame(step);
    });

    var SPEEDS = [[1, "1×"], [60, "60×"], [600, "600×"], [0, "즉시"]];
    SPEEDS.forEach(function(pair){
      var b = el("button", "", pair[1]);
      b.type = "button";
      b.setAttribute("aria-pressed", String(pair[0] === state.speed));
      b.addEventListener("click", function(){
        if(pair[0] === 0){ stop(); state.t = span; paint(state.t, true); }
        else { state.speed = pair[0]; }
        Array.prototype.forEach.call(speeds.children, function(x, i){
          x.setAttribute("aria-pressed",
            String(SPEEDS[i][0] !== 0 && SPEEDS[i][0] === state.speed));
        });
      });
      speeds.appendChild(b);
    });

    range.addEventListener("input", function(){
      // Scrubbing is a deliberate jump, so it takes over from playback rather
      // than fighting it for the playhead.
      stop();
      state.t = span * (Number(range.value) / 1000);
      paint(state.t, false);
    });

    if(STILL){
      // Reduced motion: no self-driving playhead. The final state is shown at
      // once and the scrub bar is still there for anyone who wants a moment.
      controls.hidden = true;
      state.t = span;
      paint(state.t, true);
      box.appendChild(el("p", "hint-row",
        "동작 최소화 설정이 켜져 있어 재생 애니메이션 없이 마지막 상태를 보여준다. " +
        "슬라이더로 원하는 시점을 직접 볼 수 있다."));
    } else {
      paint(0, true);
    }
    return box;
  }

  function benchPanel(){
    var card = el("div", "card");
    card.appendChild(el("div", "label", "부하 테스트"));
    card.appendChild(el("p", "prose",
      "가상시간으로 즉시 완주시킨 뒤 프레임으로 접어 보낸다. 실제 모델이나 벤치마크를 " +
      "돌리지 않는다 — 재생은 브라우저가 하므로 배속을 올리거나 스톰 순간으로 바로 " +
      "건너뛸 수 있다. 조립과 달리 이 계산은 버튼으로만 돈다."));

    if(lab.benchBusy){
      card.appendChild(el("p", "hint-row", "부하를 만들고 큐를 돌리는 중…"));
      return card;
    }
    if(lab.benchError){
      card.appendChild(el("div", "note", "부하 테스트 실패: " + lab.benchError));
    }
    var runs = activeKeys()
      .map(function(k){ return {key: k, d: lab.bench[k]}; })
      .filter(function(r){ return r.d && r.d.frames && r.d.frames.length; });
    if(!runs.length){
      card.appendChild(el("p", "hint-row",
        "왼쪽에서 프로파일을 고르고 실행하면 여기에 재생 가능한 결과가 나온다."));
      return card;
    }
    if(lab.benchStale){
      card.appendChild(el("div", "note",
        "입력이 바뀌었다. 아래 결과는 지금 조립과 맞지 않으므로 다시 실행해야 한다."));
    }
    var primary = runs[0].d;
    card.appendChild(el("p", "cpu", primary.profile.label));
    card.appendChild(el("p", "hint",
      "총 " + num(primary.profile.total_alarms).toLocaleString() + "건 · " +
      fmtClock(primary.profile.span_s) + " 구간 · " + primary.frames.length + "프레임 · 시드 " +
      primary.seed));
    card.appendChild(breachLine(primary));
    card.appendChild(statsGrid(primary));
    card.appendChild(playbackPanel(runs));

    if(runs.length > 1){
      var b = runs[1].d;
      card.appendChild(el("p", "hint-row",
        "머신 B: " + b.machine.headline + " · p95 " +
        (b.stats ? num(b.stats.p95_s).toFixed(1) : "-") + "초 · " +
        (b.breach ? fmtClock(b.breach.t_s) + "에 붕괴" : "끝까지 통과")));
    }
    if(primary.worst && primary.worst.length){
      card.appendChild(el("p", "hint-row", "가장 오래 걸린 " +
        primary.worst.length + "건"));
      card.appendChild(worstTable(primary.worst));
    }
    var notes = (primary.profile.notes || []).concat(primary.notes || []);
    if(notes.length){
      var ul = el("ul", "notes");
      notes.forEach(function(n){ ul.appendChild(el("li", "", n)); });
      card.appendChild(ul);
    }
    return card;
  }

  function stopBenchPlayback(){
    if(bench){
      if(bench.raf) cancelAnimationFrame(bench.raf);
      bench = null;
    }
  }

  function renderLab(){
    var host = byId("lab-results");
    if(!host) return;
    stopBenchPlayback();
    host.textContent = "";

    var keys = activeKeys();
    var pair = el("div", "asm-pair");
    keys.forEach(function(k){
      if(lab.asmError[k]){
        var bad = el("div", "card");
        bad.appendChild(el("div", "label", "머신 " + k));
        bad.appendChild(el("div", "note", "조립 실패: " + lab.asmError[k]));
        pair.appendChild(bad);
      } else if(lab.asm[k]){
        pair.appendChild(assemblyCard(k, lab.asm[k]));
      }
    });
    host.appendChild(section("가상 서버 조립",
      "값을 바꾸면 즉시 다시 계산한다", pair));
    host.appendChild(section("부하 테스트",
      lab.asm.A ? lab.asm.A.cpu.label : "", benchPanel()));

    var ready = keys.every(function(k){ return lab.asm[k] && lab.asm[k].ok; });
    var runBtn = byId("lab-run");
    runBtn.disabled = lab.benchBusy || !ready;
    runBtn.textContent = lab.benchBusy ? "실행 중…"
      : (lab.bench.A ? "▶ 다시 실행" : "▶ 부하 테스트 실행");
    byId("lab-run-hint").textContent = !ready
      ? "구성에 오류가 있다 — 위의 오류를 고쳐야 실행할 수 있다."
      : "가상시간으로 즉시 완주한 뒤, 재생은 브라우저가 한다.";
    var active = lab.asm[lab.active];
    byId("lab-build-hint").textContent = active
      ? active.channels.populated + "/" + active.channels.total + " 채널 · " +
        active.ram.total_gb + " GB · " + active.bandwidth.gbs + " GB/s"
      : "";
  }

  // ---- lab wiring ----
  function setActiveMachine(key){
    lab.active = key;
    ["A", "B"].forEach(function(k){
      byId("mach-" + k.toLowerCase()).setAttribute("aria-pressed", String(k === key));
    });
    writeMachine();
    runLab();
  }

  function onLabInput(ev){
    var id = (ev && ev.target && ev.target.id) || "";
    if(id === "lab-compare"){
      lab.compare = byId("lab-compare").checked;
      byId("mach-b").disabled = !lab.compare;
      if(!lab.compare && lab.active === "B"){ setActiveMachine("A"); return; }
    }
    if(id === "lab-profile") showProfileGroup();
    if(id === "lab-cpu") fillSockets();
    readMachine();
    markBenchStale();
    // Profile knobs change the load, not the board, so they must not pay for a
    // reassembly round trip.
    if(id.indexOf("bp-") === 0 || id === "lab-profile"){ renderLab(); return; }
    runLab();
  }

  // Three screens now, one nav. "모델 성능" is the default because it answers the
  // question the other two build on: how fast is this model on this machine at
  // all. The third screen assembles a machine and drives load at it -- it was
  // briefly labelled "적용 사례: 관제 알람", which described one of its four load
  // profiles as though it were the whole screen. It is a load test.
  // Screen containers are prefixed because "model" was already taken by the
  // LLM <select>. Two elements shared that id, so getElementById("model")
  // returned the <main> and the boot code poured 32 <optgroup>s into it --
  // the dropdown stayed empty and the option groups rendered as loose text
  // across the screen. Duplicate ids fail silently and look like everything
  // broke at once, which is exactly how it was reported.
  var VIEWS = ["model", "size", "lab"];
  function screenOf(v){ return byId("screen-" + v); }
  function setView(which){
    if(VIEWS.indexOf(which) < 0) which = "model";
    VIEWS.forEach(function(v){
      screenOf(v).hidden = (v !== which);
      byId("view-" + v).setAttribute("aria-pressed", String(v === which));
    });
    // Each screen owns an animation; leaving it must stop it, or a hidden
    // canvas keeps burning frames for a view nobody is looking at.
    if(which !== "size") stopPlayback();
    if(which !== "lab") stopBenchPlayback();
    if(which === "lab" && !lab.asm.A && !lab.asmError.A) runLab();
  }

  function initLab(){
    fillSelect(byId("lab-cpu"), (catalog.cpus || []).map(function(c){
      return [c.id, c.label + " · " + c.cores + "C " + c.mem_channels + "ch " + c.ddr_gen];
    }));
    fillSelect(byId("lab-model"), catalog.models.map(function(m){
      return [m.id, m.name + " (" + m.params_b + "B)"];
    }));
    fillSelect(byId("lab-quant"), catalog.quants.map(function(q){
      return [q.id, q.id + " (" + q.bpw + " bpw)"];
    }));

    MACHINES.A = defaultMachine();
    MACHINES.B = defaultMachine();
    // B opens on the cautionary build: the same capacity as A, on a quarter of
    // the channels. Same gigabytes, a fraction of the bandwidth -- which is the
    // one comparison this screen exists to make.
    MACHINES.B.dimm_gb = MACHINES.A.dimm_gb * 2;
    MACHINES.B.dimm_count = Math.max(1, Math.round(MACHINES.A.dimm_count / 2));

    showProfileGroup();
    setActiveMachine("A");

    var railForm = byId("lab-rail");
    railForm.addEventListener("input", onLabInput);
    railForm.addEventListener("change", onLabInput);
    railForm.addEventListener("submit", function(ev){ ev.preventDefault(); });
    byId("lab-run").addEventListener("click", runBench);
    byId("mach-a").addEventListener("click", function(){ setActiveMachine("A"); });
    byId("mach-b").addEventListener("click", function(){
      if(lab.compare) setActiveMachine("B");
    });
    byId("view-size").addEventListener("click", function(){ setView("size"); });
    byId("view-lab").addEventListener("click", function(){ setView("lab"); });
  }

  // ---- model performance -------------------------------------------
  // The default screen. Four axes of one model on one machine, and none of them
  // mention an alarm: this is the question underneath the other two screens.
  //
  // Button-driven, same discipline and same reason as the capacity search and
  // the load bench: dozens of predictions plus a training verdict is not a
  // keystroke's worth of work, and the answer does not change usefully while
  // somebody is still typing a DIMM count. Inputs move -> the result goes stale
  // and says so; nothing re-runs by itself.
  // Korean and the industry term together, everywhere. Two kinds of reader open
  // this screen -- one who says "TTFT" and one who says "첫 토큰까지" -- and the
  // engine's own field names (`decode_tps_single`) are never shown to either.
  var MB_GEN = "생성 속도(Generation tok/s)";
  var MB_TOTAL = "전체 합계(Total tok/s)";
  var MB_TTFT = "첫 토큰까지(TTFT)";
  var MB_METRICS = [
    ["decode_tps_single", MB_GEN, "tok/s"],
    ["decode_tps_total", MB_TOTAL, "tok/s"],
    ["ttft_s", MB_TTFT, "초"],
    ["prefill_tps", "프롬프트 처리(Prefill tok/s)", "tok/s"],
    // RAM belongs in this list because it is the axis that decides whether a
    // cell can run at all. llama.cpp reserves the whole context per slot, so
    // the bottom-right of this grid can want ten times the top-left, and a
    // reader picking an operating point off the speed columns needs to see
    // which ones the machine cannot actually load.
    ["ram_gb", "필요 메모리(RAM)", "GB"]
  ];
  var MB_UNIT = "tok/s";
  var mbState = {key: null, data: null, error: null, busy: false,
                 metric: "decode_tps_single"};
  var NO_MB_BRIDGE =
    "이 데스크톱 빌드에는 모델 성능 측정이 연결되어 있지 않다. 서버 모드(svrspec gui)에서 실행해라.";

  function askModelBench(p){
    if(DESKTOP){
      // The bridge is a fixed surface in the packaged app. An installer that
      // predates this screen must say so, not throw at the first click.
      if(!(window.pywebview.api && window.pywebview.api.modelbench))
        return Promise.resolve({error: NO_MB_BRIDGE});
      return window.pywebview.api.modelbench(p);
    }
    return fetch("/api/modelbench", {method:"POST",
                                     headers:{"Content-Type":"application/json"},
                                     body: JSON.stringify(p)})
      .then(function(r){ return r.json(); });
  }

  // The catalogue's size classes are keys, not captions. "sub-2B" sitting in a
  // Korean dropdown reads as an untranslated string, so name the buckets.
  var SIZE_CLASS_LABEL = {
    "sub-2B": "2B 미만",
    "2-5B": "2–5B",
    "7-9B": "7–9B",
    "13-15B": "13–15B",
    "30-35B": "30–35B",
    "70B+": "70B 이상"
  };

  var CONF_LABEL = {measured: "실측", derived: "실측유도", estimate: "추정"};
  var SOURCE_LABEL = {
    third_party_db: "공개 벤치마크",
    vendor_datasheet: "벤더 데이터시트",
    model_card: "모델 카드",
    unverified: "출처 없음"
  };

  // Every coefficient the engine uses, with its confidence and a link to where
  // it came from. The point is not decoration: a tool that predicts for
  // hardware nobody can touch has to be auditable in place, because the reader
  // has no way to check it against the machine.
  function renderEvidence(){
    var host = byId("mb-evidence");
    if(!host) return;
    host.textContent = "";
    var rows = (catalog && catalog.evidence) || [];
    if(!rows.length){ host.appendChild(el("p", "hint", "계수 정보 없음.")); return; }

    var weakest = rows.filter(function(r){ return r.confidence === "estimate"; }).length;
    host.appendChild(el("p", "hint",
      rows.length + "개 계수 · 추정 " + weakest + "개. 추정이 섞인 만큼이 오차 범위로 나간다."));

    var ul = el("ul", "notes");
    rows.forEach(function(r){
      var li = el("li", "");
      li.appendChild(el("strong", "", r.kind + (r.key && r.key !== "*" ? " · " + r.key : "")));
      li.appendChild(document.createTextNode(
        " = " + r.value + "  [" + (CONF_LABEL[r.confidence] || r.confidence) + " / " +
        (SOURCE_LABEL[r.source] || r.source) + "]"));
      if(r.source_url){
        li.appendChild(document.createTextNode(" "));
        var a = el("a", "", "출처");
        a.href = r.source_url;
        a.target = "_blank";
        a.rel = "noopener noreferrer";
        li.appendChild(a);
      }
      if(r.notes) li.appendChild(el("div", "detail", r.notes));
      ul.appendChild(li);
    });
    host.appendChild(ul);
  }

  function mbParams(){
    // The prompt token profile comes from the 자원 screen, the way the lab's
    // does: one description of a prompt, not three that can disagree.
    var p = params();
    p.model = labVal("mb-model") || p.model;
    p.quant = labVal("mb-quant") || p.quant;
    p.cpu = labVal("mb-cpu");
    p.sockets = labNum("mb-sockets", 1);
    p.slots = labNum("mb-slots", 4);
    p.dimm_gb = labNum("mb-dimm-gb", 32);
    p.dimm_count = labNum("mb-dimm-count", 8);
    // One generation length: it times the concurrency curve and it sizes the KV
    // cache the assembly reports. Two would drift.
    p.output_tokens = labNum("mb-output-tokens", 256);
    p.train_samples = labNum("mb-train-samples", 10000);
    p.os_name = labVal("mb-os");
    p.name = "M";
    delete p.volumes; delete p.only_pass;
    return p;
  }
  function mbKey(p){
    return JSON.stringify([p.model, p.quant, p.cpu, p.sockets, p.slots,
                           p.dimm_gb, p.dimm_count, p.output_tokens, p.train_samples,
                           p.system_tokens, p.fewshot_tokens, p.alarm_tokens,
                           p.prompt_cache]);
  }

  function runModelBench(){
    if(mbState.busy) return;
    var p = mbParams(), key = mbKey(p);
    mbState.busy = true; mbState.error = null;
    renderModel();
    askModelBench(p).then(function(d){
      mbState.busy = false;
      if(!d || d.error){
        mbState.error = (d && d.error) || "알 수 없는 오류";
        mbState.data = null;
      } else {
        mbState.key = key; mbState.data = d; mbState.error = null;
      }
      renderModel();
    }).catch(function(err){
      mbState.busy = false;
      mbState.error = String(err);
      renderModel();
    });
  }

  function mbAt(d, batch, ctx){
    var out = null;
    (d.throughput || []).forEach(function(r){
      if(r.batch === batch && r.ctx_tokens === ctx) out = r;
    });
    return out;
  }
  function mbCell(row, metric){
    if(!row) return "-";
    var v = row[metric];
    return (v === null || v === undefined) ? "-" : fmtNum(num(v));
  }
  function mbMetricUnit(metric){
    var unit = MB_UNIT;
    MB_METRICS.forEach(function(m){ if(m[0] === metric) unit = m[2]; });
    return unit;
  }
  function mbMetricLabel(metric){
    var label = metric;
    MB_METRICS.forEach(function(m){ if(m[0] === metric) label = m[1]; });
    return label;
  }

  // The headline. Generation speed and TTFT are what somebody feels; the server
  // total is the number those two get mistaken for, so it is shown next to them
  // with the batch it belongs to spelled out.
  function mbSummaryTiles(d){
    var s = d.summary || {}, box = el("div", "mb-head");
    function tile(name, value, unit, sub, cls){
      var t = el("div", "h-tile" + (cls ? " " + cls : ""));
      t.appendChild(el("div", "h-name", name));
      var v = el("div", "h-val",
                 (value === null || value === undefined) ? "-" : fmtNum(num(value)));
      v.appendChild(el("span", "h-unit", unit));
      t.appendChild(v);
      if(sub) t.appendChild(el("div", "h-sub", sub));
      box.appendChild(t);
    }
    tile(MB_GEN, s.gen_tps, MB_UNIT,
         s.condition + " · " + (s.readable ? "쓸 만함" : "답답함"));
    tile(MB_TTFT, s.ttft_s, "초",
         "프롬프트 " + num(s.ctx_tokens).toLocaleString() + " 토큰을 처리하는 동안 기다리는 시간");
    tile(MB_TOTAL, s.total_tps, MB_UNIT,
         (s.busy_batch
           ? "배치 " + s.busy_batch + "에서는 " + fmtNum(num(s.busy_total_tps)) + " " +
             MB_UNIT + " — 대신 사용자당 " + fmtNum(num(s.busy_gen_tps)) + " " + MB_UNIT
           : "서버 전체 합계 — 한 사용자가 보는 속도가 아니다"),
         "total");
    return box;
  }

  // Batch x context, one metric at a time. Three metrics on one grid would be
  // three numbers per cell, and the two things this table has to make obvious --
  // that batching buys total throughput with per-sequence speed, and that a long
  // context slows decode down -- would be lost in the density.
  function mbThroughput(d){
    var wrap = el("div");
    var picker = el("div", "mb-metrics");
    MB_METRICS.forEach(function(pair){
      var b = el("button", "", pair[1]);
      b.type = "button";
      b.setAttribute("aria-pressed", String(mbState.metric === pair[0]));
      b.addEventListener("click", function(){
        mbState.metric = pair[0]; renderModel();
      });
      picker.appendChild(b);
    });
    wrap.appendChild(picker);

    var box = el("div", "scroll");
    var tbl = el("table", "mb-grid"), thead = el("thead"), htr = el("tr");
    htr.appendChild(el("th", "", "배치"));
    (d.contexts || []).forEach(function(c){
      htr.appendChild(el("th", "num", "ctx " + c.toLocaleString()));
    });
    htr.appendChild(el("th", "", "병목"));
    thead.appendChild(htr); tbl.appendChild(thead);

    var tbody = el("tbody");
    (d.batches || []).forEach(function(b){
      var tr = el("tr"), bound = "";
      tr.appendChild(el("td", "", b + " 시퀀스"));
      (d.contexts || []).forEach(function(c, i){
        var row = mbAt(d, b, c);
        // A cell the machine cannot load is marked wherever the reader is
        // looking, not only on the RAM view: somebody picking an operating
        // point off the speed columns is exactly who needs to be told.
        var td = el("td", "num" + (row && row.fits === false ? " mb-nofit" : ""),
                    mbCell(row, mbState.metric));
        if(row && row.fits === false){
          td.title = "이 구성은 장착 메모리에 들어가지 않는다 (필요 " +
                     fmtNum(num(row.ram_gb)) + " GB)";
          td.appendChild(el("span", "mb-nofit-mark", " ✕"));
        }
        tr.appendChild(td);
        if(!i && row){
          // TTFT is a prefill number, so it reads the prefill ceiling.
          bound = (mbState.metric === "prefill_tps" || mbState.metric === "ttft_s")
            ? row.prefill_bound_label : row.decode_bound_label;
        }
      });
      tr.appendChild(el("td", "", bound));
      tbody.appendChild(tr);
    });
    tbl.appendChild(tbody); box.appendChild(tbl);
    wrap.appendChild(box);
    wrap.appendChild(el("p", "hint-row",
      mbMetricLabel(mbState.metric) + " · 단위 " + mbMetricUnit(mbState.metric) +
      " · 병목 열은 가장 짧은 컨텍스트에서 이 위상이 걸린 천장이다."));
    wrap.appendChild(mbTradeoff(d));
    return wrap;
  }

  // The two sentences the grid exists to support, computed from its own corners
  // so they cannot claim a shape the numbers do not have.
  function mbTradeoff(d){
    var ul = el("ul", "notes");
    var cs = d.contexts || [], bs = d.batches || [];
    if(!cs.length || !bs.length) return ul;
    var c0 = cs[0], cN = cs[cs.length - 1], b0 = bs[0], bN = bs[bs.length - 1];
    var lo = mbAt(d, b0, c0), hi = mbAt(d, bN, c0), long = mbAt(d, b0, cN);
    if(lo && hi){
      ul.appendChild(el("li", "",
        "배치를 " + b0 + " → " + bN + "로 올리면 " + MB_TOTAL + "은 " +
        fmtNum(num(lo.decode_tps_total)) + " → " + fmtNum(num(hi.decode_tps_total)) +
        " " + MB_UNIT + "로 오르지만, 한 사용자가 보는 " + MB_GEN + "는 " +
        fmtNum(num(lo.decode_tps_single)) + " → " + fmtNum(num(hi.decode_tps_single)) +
        " " + MB_UNIT + "로 떨어진다. 배치는 처리량을 사고 응답 속도를 판다."));
    }
    if(lo && long){
      ul.appendChild(el("li", "",
        "컨텍스트를 " + c0.toLocaleString() + " → " + cN.toLocaleString() +
        " 토큰으로 늘리면 같은 배치에서 " + MB_GEN + "가 " +
        fmtNum(num(lo.decode_tps_single)) + " → " + fmtNum(num(long.decode_tps_single)) +
        " " + MB_UNIT + "로 떨어지고, " + MB_TTFT + "는 " +
        fmtNum(num(lo.ttft_s)) + " → " + fmtNum(num(long.ttft_s)) +
        "초로 늘어난다. 토큰마다 그만큼 커진 KV 캐시를 다시 읽어야 하기 때문이다."));
    }
    return ul;
  }

  // "몇 명까지 쓸 만한가" is a threshold question, so the threshold is drawn.
  function mbConcurrencyChart(d){
    var W = 600, H = 190, PAD = 34;
    var rows = d.concurrency || [];
    var vals = rows.map(function(c){ return num(c.decode_tps_each); });
    var line = num(d.readable_tps);
    var max = Math.max.apply(null, vals.concat([line * 1.3, 1]));
    var last = vals.length ? vals[vals.length - 1] : 0;
    var svg = svgEl("svg", {viewBox: "0 0 " + W + " " + H,
                            preserveAspectRatio: "none", role: "img",
                            "aria-label":
                              "동시 사용자 " + (rows.length ? rows[0].users : 0) + "명에서 " +
                              (rows.length ? rows[rows.length - 1].users : 0) +
                              "명까지, 사용자당 체감 생성 속도가 " + fmtNum(vals[0] || 0) +
                              "에서 " + fmtNum(last) + " " + MB_UNIT + "로 변한다. " +
                              "읽기 편한 기준선은 " + fmtNum(line) + " " + MB_UNIT + "다."});
    var iw = W - PAD, ih = H - 14;
    for(var i = 0; i <= 4; i++){
      var y = (ih / 4) * i;
      svg.appendChild(svgEl("line", {x1: PAD, y1: y.toFixed(1), x2: W, y2: y.toFixed(1),
                                     stroke: "var(--border)", "stroke-width": "1"}));
      var t = svgEl("text", {x: 0, y: (y + 4).toFixed(1), class: "axis"});
      t.textContent = fmtNum(max * (1 - i / 4));
      svg.appendChild(t);
    }
    var g = svgEl("g", {transform: "translate(" + PAD + ",0)"});
    g.appendChild(svgEl("path", {d: areaPath(vals, max, iw, ih),
                                 fill: "var(--accent)", opacity: "0.20"}));
    g.appendChild(svgEl("path", {d: linePath(vals, max, iw, ih), fill: "none",
                                 stroke: "var(--accent)", "stroke-width": "1.5"}));
    g.appendChild(svgEl("path", {d: flatLine(line, max, iw, ih), fill: "none",
                                 stroke: "var(--warning)", "stroke-width": "1.5",
                                 "stroke-dasharray": "5 4"}));
    svg.appendChild(g);
    return svg;
  }

  function mbConcurrency(d){
    var wrap = el("div"), rows = d.concurrency || [];
    var chart = el("div", "mb-chart");
    var chead = el("div", "graph-head");
    chead.appendChild(el("span", "g-title", "사용자당 " + MB_GEN));
    chead.appendChild(el("span", "spacer"));
    chead.appendChild(el("span", "g-scale", rows.length + "개 지점"));
    chart.appendChild(chead);
    chart.appendChild(mbConcurrencyChart(d));
    var xs = el("div", "axis-x");
    rows.forEach(function(c){ xs.appendChild(el("span", "", c.users + "명")); });
    chart.appendChild(xs);
    chart.appendChild(chartLegend([
      {token: "accent", name: "사용자당 " + MB_GEN},
      {token: "warning", name: "읽기 편한 하한 " + fmtNum(num(d.readable_tps)) + " " + MB_UNIT,
       kind: "dash"}
    ]));
    wrap.appendChild(chart);

    var box = el("div", "scroll");
    var tbl = el("table"), thead = el("thead"), htr = el("tr");
    ["동시 사용자", MB_TTFT, MB_GEN, "응답 완료", MB_TOTAL, "판정"]
      .forEach(function(name, i){ htr.appendChild(el("th", i ? "num" : "", name)); });
    thead.appendChild(htr); tbl.appendChild(thead);
    var tbody = el("tbody");
    rows.forEach(function(c){
      var tr = el("tr", c.readable ? "" : "weak");
      tr.appendChild(el("td", "", c.users + "명"));
      tr.appendChild(el("td", "num", fmtNum(num(c.ttft_s)) + "초"));
      tr.appendChild(el("td", "num", fmtNum(num(c.decode_tps_each)) + " " + MB_UNIT));
      tr.appendChild(el("td", "num", fmtNum(num(c.response_s)) + "초"));
      tr.appendChild(el("td", "num", fmtNum(num(c.total_tps)) + " " + MB_UNIT));
      // Colour plus the word, like every other verdict in this tool.
      tr.appendChild(el("td", c.readable ? "v-pass" : "v-fail",
                        c.readable ? "쓸 만함" : "답답함"));
      tbody.appendChild(tr);
    });
    tbl.appendChild(tbody); box.appendChild(tbl);
    wrap.appendChild(box);

    var first = null;
    rows.forEach(function(c){ if(first === null && !c.readable) first = c.users; });
    // The headline tile and this table are both TTFT, and they can differ:
    // different prompt length, different number. Saying so is cheaper than
    // letting somebody find the discrepancy and stop trusting both.
    var s = d.summary || {};
    if(s.ctx_tokens){
      wrap.appendChild(el("p", "hint-row",
        "이 표의 " + MB_TTFT + "는 이 축이 쓰는 프롬프트 길이 기준이고, 상단 요약 타일은 배치 " +
        s.batch + " · 컨텍스트 " + num(s.ctx_tokens).toLocaleString() +
        " 토큰 조건이다 — 조건이 다르면 숫자도 다르다. 정확한 조건은 아래 “가정과 한계”에 있다."));
    }
    wrap.appendChild(el("p", "hint-row",
      first === null
        ? "탐색한 " + (rows.length ? rows[rows.length - 1].users : 0) +
          "명까지는 사용자당 체감 속도가 " + fmtNum(num(d.readable_tps)) + " " + MB_UNIT +
          " 아래로 내려가지 않았다."
        : "사용자 " + first + "명에서 체감 속도가 " + fmtNum(num(d.readable_tps)) + " " +
          MB_UNIT + " 아래로 내려간다 — 그 지점부터는 글자가 사람이 읽는 속도보다 늦게 나온다. " +
          "응답 완료는 " + d.output_tokens + " 토큰 생성 기준이다."));
    return wrap;
  }

  function mbGauge(name, pct, cls){
    var box = el("div", "gauge");
    box.appendChild(el("div", "g-name", name));
    box.appendChild(el("div", "g-val", fmtNum(num(pct)) + "%"));
    var bar = el("div", "bar " + cls + (num(pct) >= 90 ? " hot" : ""));
    var span = el("span");
    span.style.width = Math.max(0, Math.min(100, num(pct))) + "%";
    bar.appendChild(span);
    box.appendChild(bar);
    return box;
  }

  // The screen that answers "코어를 살지 메모리 채널을 살지". Prefill and decode
  // side by side because the answer is that they use opposite halves of the box,
  // and one phase on its own cannot show that.
  function mbResources(d){
    var wrap = el("div"), rows = d.resources || [];
    var pair = el("div", "mb-pair");
    rows.forEach(function(s){
      var card = el("div", "card");
      card.appendChild(el("div", "label", s.phase_label));
      card.appendChild(el("p", "cpu", "병목 · " + s.bound_label));
      var gauges = el("div", "gauges");
      gauges.appendChild(mbGauge("메모리 대역폭", s.bandwidth_pct, "g-bw"));
      gauges.appendChild(mbGauge("연산", s.compute_pct, ""));
      card.appendChild(gauges);
      card.appendChild(el("p", "hint",
        "토큰당 " + num(s.bytes_per_token).toLocaleString() + " B 읽기 · " +
        num(s.flops_per_token).toLocaleString() + " FLOP"));
      if(s.advice) card.appendChild(el("p", "hint-row", s.advice));
      pair.appendChild(card);
    });
    wrap.appendChild(pair);

    var byPhase = {};
    rows.forEach(function(s){ byPhase[s.phase] = s; });
    var pf = byPhase.prefill, de = byPhase.decode;
    if(pf && de){
      wrap.appendChild(el("p", "hint-row",
        pf.bound_by === de.bound_by
          ? "이 조합에서는 두 위상이 같은 천장(" + pf.bound_label + ")에 걸렸다. " +
            "그쪽을 늘리는 것이 양쪽 모두에 듣는다는 뜻이다."
          : "프롬프트 처리는 " + pf.bound_label + "에, 토큰 생성은 " + de.bound_label +
            "에 걸린다 — 기계의 반대쪽이다. 프롬프트가 긴 부하라면 코어와 벡터 ISA를, " +
            "생성이 긴 부하라면 메모리 채널과 DDR 등급을 사야 한다."));
    }
    return wrap;
  }

  // Usually the answer is "안 된다", and that is the honest result. It is not
  // shrunk into a footnote: the reasons and the GPU comparison are what make a
  // refusal actionable.
  function mbTraining(d){
    var wrap = el("div");
    var pair = el("div", "mb-pair");
    (d.training || []).forEach(function(t){
      var card = el("div", "card");
      card.appendChild(el("div", "label", t.kind_label));
      var verdict = el("div", "mb-verdict");
      verdict.appendChild(el("strong", t.feasible ? "v-pass" : "v-fail", t.verdict));
      verdict.appendChild(el("span", "num",
        "필요 " + fmtNum(num(t.memory_needed_gb)) + " GB / 장착 " +
        fmtNum(num(t.memory_available_gb)) + " GB"));
      card.appendChild(verdict);
      var bits = [];
      if(t.step_seconds !== null && t.step_seconds !== undefined)
        bits.push("1 step " + fmtNum(num(t.step_seconds)) + "초");
      if(t.epoch_hours !== null && t.epoch_hours !== undefined)
        bits.push("1 epoch " + fmtNum(num(t.epoch_hours)) + "시간");
      if(bits.length) card.appendChild(el("p", "hint", bits.join(" · ")));
      if(t.reasons && t.reasons.length){
        var ul = el("ul", "notes");
        t.reasons.forEach(function(r){ ul.appendChild(el("li", "", r)); });
        card.appendChild(ul);
      }
      card.appendChild(el("p", "hint-row",
        "GPU 비교 · " + (t.gpu_comparison || "엔진이 비교값을 내지 않았다")));
      pair.appendChild(card);
    });
    wrap.appendChild(pair);
    wrap.appendChild(el("p", "hint-row",
      "학습 축의 계수는 카탈로그에 근거가 없다 — 공개 자료로 환산한 추정이고, 위 판정은 그 " +
      "가정 위에 서 있다. 샘플 " + num(d.train_samples).toLocaleString() +
      "건 기준이며, 안 된다는 판정도 결과다: 근거를 읽고 GPU 쪽과 비교해라."));
    return wrap;
  }

  function mbList(items){
    var ul = el("ul", "notes");
    items.forEach(function(x){ ul.appendChild(el("li", "", x)); });
    return ul;
  }

  function renderModel(){
    var host = byId("mb-results");
    if(!host) return;
    host.textContent = "";
    var fresh = !!mbState.data && mbState.key === mbKey(mbParams());
    var d = mbState.data;

    var card = el("div", "card");
    card.appendChild(el("div", "label", "모델 성능"));
    card.appendChild(el("p", "prose",
      "이 서버에 이 모델을 올리면 실제로 어떤 성능이 나오는지를 낸다 — 추론 처리량, " +
      "동시 사용자, 연산 자원 분해, 학습 가능 여부. 실제 모델이나 벤치마크를 " +
      "돌리지 않는다: 카탈로그의 물리와 계수로 예측한다."));
    if(d && fresh && !d.blocked){
      card.appendChild(el("p", "cpu", d.model_name + " · " + d.quant_id));
      card.appendChild(el("p", "detail", d.hardware));
      // The two felt numbers first, before any table: this is what somebody
      // opened the screen to read.
      card.appendChild(mbSummaryTiles(d));
      card.appendChild(el("p", "hint-row",
        "사람은 " + fmtNum(num(d.readable_tps)) + " " + MB_UNIT +
        " 아래로 내려가면 글자가 읽는 속도보다 늦게 나와 기다린다는 느낌을 받는다 — " +
        "아래 동시 사용자 표에 그 지점을 표시했다."));
      card.appendChild(el("p", "hint",
        "추론 시 실사용 " + fmtNum(num(d.memory_gb)) + " GB · 예측 오차 ±" +
        fmtNum(num(d.uncertainty)) + "% · 생성 " + d.output_tokens + " 토큰 기준"));
      // Which OS the memory figures assumed, and what happens if they are
      // exceeded. A memory number that hides its operating system is not a
      // number, and "swaps" and "is killed" are not the same warning.
      if(d.os){
        var os = el("p", "hint-row",
          "메모리 기준 OS: " + d.os.label + " (상주 " + fmtNum(num(d.os.runtime_gb)) +
          " GB · 여유 " + fmtNum(num(d.os.headroom)) + "배)" +
          (d.os.chosen ? "" : " — 기본값이다"));
        card.appendChild(os);
        card.appendChild(el("p", "hint-row" + (d.os.hard_limit ? " v-fail" : ""),
          d.os.overrun));
        card.appendChild(el("p", "hint-row", d.os.note));
      }
    }
    if(mbState.error){
      card.appendChild(el("div", "note", "모델 성능 측정 실패: " + mbState.error));
    }
    if(mbState.busy){
      card.appendChild(el("p", "hint-row",
        "배치·컨텍스트 격자와 동시 사용자 곡선, 학습 판정을 계산하는 중…"));
    } else if(!d){
      card.appendChild(el("p", "hint-row",
        "왼쪽에서 모델과 서버를 고르고 “모델 성능 측정”을 누르면 여기에 네 축이 나온다. " +
        "입력을 바꿔도 자동으로 다시 돌지 않는다."));
    } else if(!fresh){
      card.appendChild(el("div", "note",
        "입력이 바뀌었다. 아래 결과는 지금 조건과 맞지 않으므로 다시 측정해야 한다."));
    }
    host.appendChild(card);

    var run = byId("mb-run");
    if(run){
      run.disabled = mbState.busy;
      run.textContent = mbState.busy ? "측정 중…"
        : (d ? "▶ 다시 측정" : "▶ 모델 성능 측정");
    }
    var hint = byId("mb-run-hint");
    if(hint){
      hint.textContent = mbState.busy
        ? "계산하는 동안 다른 화면은 계속 쓸 수 있다."
        : "실제 모델을 돌리지 않는다 — 카탈로그의 물리와 계수로 예측한다.";
    }

    if(!d) return;
    if(d.blocked){
      var bad = el("div", "card");
      bad.appendChild(el("div", "label", "조립 오류"));
      bad.appendChild(el("p", "headline", d.blocked));
      if(d.findings && d.findings.length) bad.appendChild(findingList(d.findings));
      host.appendChild(bad);
      return;
    }

    host.appendChild(section("추론 처리량", "배치 × 컨텍스트", mbThroughput(d)));
    host.appendChild(section("동시 사용자",
      "출력 " + d.output_tokens + " 토큰 기준", mbConcurrency(d)));
    host.appendChild(section("연산 자원 분해",
      "prefill과 decode는 기계의 반대쪽을 쓴다", mbResources(d)));
    host.appendChild(section("학습·파인튜닝", "full · LoRA · QLoRA", mbTraining(d)));
    if(d.findings && d.findings.length){
      host.appendChild(section("조립 지적", "성능 이전에 하드웨어 쪽 문제",
                               findingList(d.findings)));
    }
    if(d.warnings && d.warnings.length){
      host.appendChild(section("주의사항", "예측 엔진이 낸 경고", mbList(d.warnings)));
    }
    if(d.notes && d.notes.length){
      host.appendChild(section("가정과 한계", "이 숫자가 무엇을 가정했는지", mbList(d.notes)));
    }
  }

  function mbFillSockets(){
    var cpu = cpuById(labVal("mb-cpu"));
    var most = Math.max(1, (cpu && cpu.sockets_max) || 1);
    var want = labNum("mb-sockets", 1);
    var list = [];
    for(var i = 1; i <= most; i++) list.push([i, i + "소켓"]);
    fillSelect(byId("mb-sockets"), list);
    byId("mb-sockets").value = String(Math.max(1, Math.min(most, want)));
  }

  function mbHints(){
    var id = labVal("mb-model");
    var m = (catalog.models || []).filter(function(x){ return x.id === id; })[0];
    var mh = byId("mb-model-hint");
    if(mh){
      mh.textContent = m
        ? [m.params_b + "B", "KV " + m.kv_kib + " KiB/토큰",
           "학습 컨텍스트 " + num(m.ctx_train).toLocaleString()].join(" · ")
        : "";
    }
    var cpu = cpuById(labVal("mb-cpu")), hh = byId("mb-hw-hint");
    if(!hh) return;
    if(!cpu){ hh.textContent = ""; return; }
    var sockets = labNum("mb-sockets", 1);
    var count = labNum("mb-dimm-count", 0), gb = labNum("mb-dimm-gb", 0);
    var channels = num(cpu.mem_channels) * sockets;
    hh.textContent =
      count * gb + " GB · " + Math.min(count, channels) + "/" + channels + " 채널 · " +
      cpu.ddr_gen + " · 최대 " + num(cpu.max_mem_gb) * sockets + " GB";
  }

  function onModelInput(ev){
    var id = (ev && ev.target && ev.target.id) || "";
    if(id === "mb-cpu") mbFillSockets();
    mbHints();
    // Never re-runs the measurement: it only redraws, which is what marks the
    // previous answer stale.
    renderModel();
  }

  function initModel(){
    fillSelect(byId("mb-model"), (catalog.models || []).map(function(m){
      return [m.id, m.name + " (" + m.params_b + "B)"];
    }));
    fillSelect(byId("mb-quant"), (catalog.quants || []).map(function(q){
      return [q.id, q.id + " (" + q.bpw + " bpw)"];
    }));
    fillSelect(byId("mb-cpu"), (catalog.cpus || []).map(function(c){
      return [c.id, c.label + " · " + c.cores + "C " + c.mem_channels + "ch " + c.ddr_gen];
    }));
    renderEvidence();
    fillSelect(byId("mb-os"), (catalog.os_profiles || []).map(function(o){
      return [o.id, o.label + " · 상주 " + o.runtime_gb + "GB" +
                    (o.hard_limit ? " · 초과 시 OOM" : "")];
    }));
    if(catalog.os_default) byId("mb-os").value = catalog.os_default;
    byId("mb-model").value = byId("model").value;
    byId("mb-quant").value = byId("quant").value;
    // Open on the widest memory bus, same reasoning as the lab: it is the part
    // where the choice of DIMM layout matters most.
    var cpus = catalog.cpus || [], cpu = cpus[0];
    cpus.forEach(function(c){ if(cpu && c.mem_channels > cpu.mem_channels) cpu = c; });
    if(cpu){
      byId("mb-cpu").value = cpu.id;
      byId("mb-dimm-count").value = String(Math.max(1, num(cpu.mem_channels) || 8));
    }
    mbFillSockets();
    mbHints();

    var form = byId("mb-rail");
    form.addEventListener("input", onModelInput);
    form.addEventListener("change", onModelInput);
    form.addEventListener("submit", function(ev){ ev.preventDefault(); });
    byId("mb-run").addEventListener("click", runModelBench);
    byId("view-model").addEventListener("click", function(){ setView("model"); });
    renderModel();
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
        groups[m.size_class].label = SIZE_CLASS_LABEL[m.size_class] || m.size_class;
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
    initLab();
    initModel();
    setView("model");     // the default screen is the model, not the alarm
    modelHint(); tokenHint(); run();
    checkForUpdate();
  }).catch(function(err){
    // Whatever went wrong, the window must stop saying "불러오는 중" and say
    // what happened. The bridge failure carries its own remedy; anything else
    // gets the generic one.
    var msg = (err && err.message) ? err.message : String(err);
    var text = msg.indexOf("브리지") >= 0 ? msg : ("카탈로그를 불러오지 못했다: " + msg);
    // Every screen, not just the sizing one: the default view is the model
    // screen now, and a window that only reported the failure on a hidden tab
    // would still be sitting there claiming it was loading.
    ["mb-results", "results", "lab-results"].forEach(function(id){
      var host = document.getElementById(id);
      if(!host) return;
      host.textContent = "";
      var box = el("div", "note");
      box.appendChild(el("p", "", text));
      host.appendChild(box);
    });
    var empty = document.getElementById("empty");
    if(empty) empty.remove();
  });
})();
</script>
</body>
</html>
"""
    )
