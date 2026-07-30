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
        if route not in ("/api/size", "/api/resources", "/api/capacity"):
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
  #rail{position:static; max-height:none; overflow:visible}
}
#rail{
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
