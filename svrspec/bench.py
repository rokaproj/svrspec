"""One load profile run against one assembled build, folded into replay frames.

Why there is no streaming here, and why that is the whole design
---------------------------------------------------------------
The obvious shape for "watch a load test run" is a stream: start the run in the
background, push events to the browser, let the page draw them as they land.
That was measured before it was built, and it is the wrong shape for this tool.

A virtual-time run of `run_pipeline` finishes in **tens of milliseconds** --
8.5 ms for a 359-alarm day, 35.5 ms for 2,000 alarms. There is no long-running
job to stream *from*. Adding a background worker, an event channel and a
reconnect story would buy nothing except three new ways for the page to
disagree with the server about what happened.

So the server runs the whole thing to completion, immediately, and hands back a
finished picture. The browser owns playback: it holds the frames and steps
through them at whatever speed the operator picked, and pausing or scrubbing
backwards costs nothing because the future is already known. Replay speed
becomes a client-side concern, which is where it belongs -- the model of the
server being sized has no opinion about how fast a human wants to watch it.

Why frames rather than events
-----------------------------
The raw `Delivery` list is 172 KB for a measured day and 957 KB for a heavy
ramp. That is a lot to push at a page that is going to draw at most a few
hundred pixels of chart. Folding it to a fixed 600 frames costs ~30 KB and
loses nothing a chart could have shown. The resolution is fixed rather than
derived from the span on purpose: a 72-hour soak and a 24-hour replay then
animate over the same number of steps, so one playback control fits both.

How a frame is reconstructed
----------------------------
`pipeline.py` is not modified to emit any of this. Everything comes from the
`Delivery` timestamps it already records, swept into intervals of constant
occupancy:

    queued      arrived_s <= t < started_s
    prefilling  started_s <= t < first_token_s
    decoding    first_token_s <= t < generated_s

Between two consecutive breakpoints nothing changes, so each interval is
exactly a `simulate.SimSegment` -- the same record the 15-minute resource
timeline and the host telemetry are built from. Its resource draw comes from
`timeline.segment_rates`, imported rather than copied: that function is this
project's single definition of how work becomes bytes and flops, and a private
copy here would be the third one and the first to drift.

The one thing the timestamps cannot say is how a request's own tokens were
distributed *inside* its phase -- the pipeline splits its rate across whoever
else is in flight, moment to moment. Tokens are therefore spread uniformly
across each request's own prefill and decode spans. Occupancy, queue depth and
latency are exact; the token rate inside one frame is a reconstruction, and it
is the only approximation in this module.

Nothing here loads a model, starts a program, or opens a connection, and no
part of the run waits on the clock -- `speed=0.0` is virtual time. The tool has
to stay usable on the laptop it is sizing a server from.
"""

from __future__ import annotations

import bisect
import csv
import io
from dataclasses import dataclass, fields
from typing import Any

from .loadgen import rate_at
from .memory import GB
from .perf import Efficiency, peak_flops
from .pipeline import Delivery, RunStats, ServiceModel, run_pipeline
from .simulate import SimSegment
from .timeline import Ceilings, ceilings_for, segment_rates
from .types import ThroughputPrediction, Workload

DAY_SECONDS = 24 * 3600.0

#: Frames a run is folded into. 600 at 24 hours is 2.4 minutes per frame, which
#: is finer than the eye can follow at any watchable playback speed, and ~30 KB
#: on the wire.
DEFAULT_FRAMES = 600

#: Slowest requests reported alongside the frames.
DEFAULT_WORST = 10

#: A bandwidth ceiling this large is a unit mistake, not a machine. Guards the
#: `Assembly.bandwidth_gbs` boundary: bytes/s handed in where GB/s was meant
#: would draw every chart at 0.0% and look like an idle box.
_IMPLAUSIBLE_GBS = 1e6


@dataclass(frozen=True)
class Frame:
    """One playback step. Everything a chart needs for that instant."""

    t_s: float
    queued: int
    active: int
    #: Share of the frame with at least one request in flight. llama.cpp pins
    #: every thread while it works, so this is the CPU line.
    cpu_pct: float
    bw_gbs: float
    bw_pct: float
    compute_pct: float
    kv_gb: float
    ram_gb: float
    arrived: int
    delivered: int
    #: Offered load in alarms/day. Analytic where the profile declares a rate
    #: curve, counted from the frame's own arrivals where it does not.
    offered_rate: float
    #: p95 end-to-end latency over everything delivered up to the end of this
    #: frame. The running figure, not the frame's own -- a single frame holds
    #: too few completions for a percentile to mean anything.
    p95_so_far_s: float


@dataclass(frozen=True)
class BenchResult:
    profile: Any
    #: Serialisable summary of the build. Not the `Assembly` itself: this
    #: crosses to a browser, and dataclasses full of catalogue objects do not.
    machine: dict
    stats: RunStats
    frames: list[Frame]
    worst: list[dict]
    findings: list
    #: Where the running p95 first crossed the SLA: `{t_s, offered_rate, p95_s}`.
    #: On a ramp this is the answer the whole profile exists to produce. None
    #: when the build held for the entire run.
    breach: dict | None
    notes: list[str]


def run_bench(
    cat,
    asm,
    alarms,
    profile,
    *,
    workload: Workload,
    frames: int = DEFAULT_FRAMES,
    queue_limit: int | None = None,
    worst_n: int = DEFAULT_WORST,
    service: ServiceModel | None = None,
    ceilings: Ceilings | None = None,
) -> BenchResult:
    """Run `alarms` through the build `asm` describes and fold the result.

    `service` and `ceilings` default to being derived from `asm` and are exposed
    so a caller that already has them does not pay for them twice, and so this
    can be tested against a stub build without a catalogue behind it.

    The run is always virtual time. Replay speed is the browser's business --
    see the module docstring for why.
    """
    if frames <= 0:
        raise ValueError("frames는 1 이상이어야 한다 — 0프레임은 재생할 것이 없다")
    if worst_n < 0:
        raise ValueError("worst_n은 0 이상이어야 한다")

    if service is None:
        service = _service_for(cat, asm, workload)
    if ceilings is None:
        ceilings = _ceilings_for(cat, asm, workload)

    stats, deliveries = run_pipeline(
        alarms, service, workload, speed=0.0, queue_limit=queue_limit
    )

    span_s = _window(profile, deliveries)
    built = _build_frames(deliveries, profile, ceilings, span_s, frames)
    breach = _breach(built, workload.sla_seconds)

    return BenchResult(
        profile=profile,
        machine=_machine(asm),
        stats=stats,
        frames=built,
        worst=_worst(deliveries, worst_n),
        findings=list(getattr(asm, "findings", []) or []),
        breach=breach,
        notes=_notes(profile, asm, stats, ceilings, span_s, breach, workload),
    )


def frames_to_csv(result: BenchResult) -> str:
    """Header row plus one row per frame, in field order.

    Columns are read off the dataclass rather than listed here, so a field added
    to `Frame` cannot silently go missing from the export.
    """
    columns = [f.name for f in fields(Frame)]
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(columns)
    for frame in result.frames:
        writer.writerow([getattr(frame, name) for name in columns])
    return buffer.getvalue()


# --------------------------------------------------------------------------
# Deriving the build's service rates and ceilings
# --------------------------------------------------------------------------


def _service_for(cat, asm, workload: Workload) -> ServiceModel:
    """The one route from an assembled build to service rates.

    Imported inside the function so this module can be used with an injected
    `service` on a tree where `lab` is not present -- and so the import error,
    if it happens, names the thing that is actually missing.
    """
    from .lab import to_service

    return to_service(cat, asm, workload)


def _ceilings_for(cat, asm, workload: Workload) -> Ceilings:
    """Denominators for this build, from the Assembly's own numbers.

    The bandwidth ceiling is taken from `asm.bandwidth_gbs` rather than
    recomputed, so the bench chart can never quote a different bandwidth from
    the assembly panel that produced it. `ceilings_for` reads only that field
    off the prediction; the rest of the record is filled from the same
    Assembly so nothing invented can leak out of this adapter.
    """
    bandwidth_gbs = float(getattr(asm, "bandwidth_gbs", 0.0))
    if bandwidth_gbs >= _IMPLAUSIBLE_GBS:
        raise ValueError(
            f"대역폭이 GB/s 값으로 보이지 않는다: {bandwidth_gbs:,.0f} "
            f"— Assembly.bandwidth_gbs는 GB/s 단위여야 한다"
        )

    eff = Efficiency.from_catalog(cat.coefficients)
    sockets = max(int(getattr(asm.vm, "sockets", 1)), 1)
    prediction = ThroughputPrediction(
        prefill_tps=float(getattr(asm, "prefill_tps", 0.0)),
        decode_tps_single=float(getattr(asm, "decode_tps_single", 0.0)),
        decode_tps_aggregate=float(getattr(asm, "decode_tps_single", 0.0)),
        prefill_bound_by="assembly",
        decode_bound_by="assembly",
        effective_bandwidth_gbs=bandwidth_gbs,
        peak_flops_tflops=peak_flops(asm.cpu, sockets)[0] / 1e12,
        uncertainty=float(getattr(asm, "uncertainty", 0.0)),
    )
    return ceilings_for(
        asm.model, asm.quant, asm.cpu, asm.memory, eff, workload,
        prediction, sockets, int(getattr(asm, "ram_total_gb", 0)) or None,
    )


# --------------------------------------------------------------------------
# Frames
# --------------------------------------------------------------------------


def _window(profile, deliveries: list[Delivery]) -> float:
    """Seconds the frames must cover.

    The profile's span, extended if the run outlived it. A build that is still
    draining backlog after the load stopped is the case worth seeing, and
    charging the overrun to the last frame -- which is what a fixed window would
    do -- flattens exactly that into one tall bar at the right-hand edge.
    """
    span = float(getattr(profile, "span_s", 0.0) or 0.0)
    for d in deliveries:
        span = max(span, d.delivered_s, d.arrived_s)
    return max(span, 1e-9)


def _build_frames(
    deliveries: list[Delivery],
    profile,
    ceilings: Ceilings,
    span_s: float,
    count: int,
) -> list[Frame]:
    width = span_s / count
    acc = [_Accumulator() for _ in range(count)]

    served = [d for d in deliveries if not d.dropped]

    for d in deliveries:
        acc[_index(d.arrived_s, width, count)].arrived += 1
    for d in served:
        acc[_index(d.delivered_s, width, count)].delivered += 1

    for seg in _segments(served, width, count, span_s):
        acc[_index(seg.t_s, width, count)].add(seg, segment_rates(seg, ceilings))

    running = _RunningP95(served)
    seen = 0
    out = []
    for index, bucket in enumerate(acc):
        t_s = index * width
        seen += bucket.arrived
        out.append(
            bucket.finish(
                t_s=t_s,
                width=width,
                ceilings=ceilings,
                offered_rate=_offered_rate(profile, t_s, t_s + width, seen),
                p95_so_far_s=running.at(t_s + width),
            )
        )
    return out


def _offered_rate(profile, t_s: float, end_s: float, seen: int) -> float:
    """Offered load in alarms/day at this frame.

    Where the profile declares a rate curve the instantaneous value off that
    curve is exact, and it is the x-axis of the breach a ramp exists to find.

    `replay` declares none -- a measured day has no offered rate other than what
    arrived -- so there the figure is the load offered *so far*: arrivals to
    date, annualised. Counting one frame's own arrivals instead would be
    Poisson noise at this resolution (2.4 minutes holds zero or one alarm on a
    measured day), and a breach labelled "0건/일" because that one frame
    happened to be empty is worse than no label at all.
    """
    declared = rate_at(profile, t_s) if profile is not None else None
    if declared is not None:
        return float(declared)
    return (seen / end_s) * DAY_SECONDS if end_s > 0 else 0.0


def _index(at: float, width: float, count: int) -> int:
    return min(count - 1, max(0, int(at / width)))


# --------------------------------------------------------------------------
# The occupancy sweep
# --------------------------------------------------------------------------


class _State:
    """Running occupancy and token rates as the sweep passes each breakpoint.

    Kept as counters rather than a set of in-flight requests: the state only
    ever changes by the deltas an event carries, so an O(events) sweep gives an
    exact reconstruction where re-scanning every request per interval would be
    quadratic in the alarm count.
    """

    __slots__ = ("queued", "prefilling", "decoding", "prefill_rate",
                 "decode_rate", "kv", "kv_slope")

    def __init__(self) -> None:
        self.queued = 0
        self.prefilling = 0
        self.decoding = 0
        self.prefill_rate = 0.0
        self.decode_rate = 0.0
        self.kv = 0.0
        self.kv_slope = 0.0

    def advance(self, dt: float) -> None:
        self.kv += self.kv_slope * dt

    def apply(self, delta: "_Delta") -> None:
        self.queued += delta.queued
        self.prefilling += delta.prefilling
        self.decoding += delta.decoding
        self.prefill_rate += delta.prefill_rate
        self.decode_rate += delta.decode_rate
        self.kv += delta.kv
        self.kv_slope += delta.kv_slope

    def segment(self, t_s: float, span_s: float) -> SimSegment:
        """This interval as the record every resource view in the project reads.

        `kv_tokens` is the interval's time-average: residency is linear inside
        an interval by construction, so the midpoint value is exact rather than
        a sample.
        """
        return SimSegment(
            t_s=t_s,
            span_s=span_s,
            active=self.prefilling + self.decoding,
            queued=self.queued,
            prefilling=self.prefilling,
            decoding=self.decoding,
            prefill_tokens=self.prefill_rate * span_s,
            decode_tokens=self.decode_rate * span_s,
            kv_tokens=max(self.kv + self.kv_slope * span_s / 2.0, 0.0),
        )


@dataclass
class _Delta:
    """What changes at one instant. Summed when several requests share it."""

    queued: int = 0
    prefilling: int = 0
    decoding: int = 0
    prefill_rate: float = 0.0
    decode_rate: float = 0.0
    #: Step change in resident KV tokens -- a request leaving takes its context
    #: with it, and a zero-length prefill deposits its prompt all at once.
    kv: float = 0.0
    kv_slope: float = 0.0


def _segments(served: list[Delivery], width: float, count: int, span_s: float):
    """Yield one `SimSegment` per interval of unchanging occupancy.

    Frame edges are breakpoints too, so no interval ever straddles two frames
    and each can be charged whole. Without that the caller would have to split a
    segment and re-derive its KV level mid-interval -- the same work, done in a
    worse place.
    """
    events: dict[float, _Delta] = {}

    def at(t: float) -> _Delta:
        return events.setdefault(t, _Delta())

    for d in served:
        prefill_span = d.first_token_s - d.started_s
        decode_span = d.generated_s - d.first_token_s
        prompt = float(d.prompt_tokens)
        output = float(d.output_tokens)

        at(d.arrived_s).queued += 1
        start = at(d.started_s)
        start.queued -= 1
        start.prefilling += 1
        first = at(d.first_token_s)
        first.prefilling -= 1
        first.decoding += 1
        done = at(d.generated_s)
        done.decoding -= 1
        done.kv -= prompt + output

        if prefill_span > 0:
            rate = prompt / prefill_span
            start.prefill_rate += rate
            start.kv_slope += rate
            first.prefill_rate -= rate
            first.kv_slope -= rate
        else:
            # Prefill finished within float resolution of starting. The prompt
            # still lands in KV; there is just no interval to spread it over.
            start.kv += prompt

        if decode_span > 0:
            rate = output / decode_span
            first.decode_rate += rate
            first.kv_slope += rate
            done.decode_rate -= rate
            done.kv_slope -= rate
        else:
            first.kv += output

    for index in range(count + 1):
        at(min(index * width, span_s))

    state = _State()
    times = sorted(events)
    for position, t in enumerate(times):
        state.apply(events[t])
        if position + 1 >= len(times):
            break
        nxt = times[position + 1]
        if nxt > t:
            yield state.segment(t, nxt - t)
            state.advance(nxt - t)


class _Accumulator:
    """Time-weighted sums for one frame, plus the peaks inside it."""

    def __init__(self) -> None:
        self.busy_s = 0.0
        self.bytes = 0.0
        self.flops = 0.0
        self.peak_kv_bytes = 0.0
        self.max_queued = 0
        self.max_active = 0
        self.arrived = 0
        self.delivered = 0

    def add(self, seg: SimSegment, rates) -> None:
        if seg.active:
            self.busy_s += seg.span_s
        self.bytes += rates.bandwidth_bytes_s * seg.span_s
        self.flops += rates.flops * seg.span_s
        self.peak_kv_bytes = max(self.peak_kv_bytes, rates.kv_bytes)
        self.max_queued = max(self.max_queued, seg.queued)
        self.max_active = max(self.max_active, seg.active)

    def finish(
        self,
        t_s: float,
        width: float,
        ceilings: Ceilings,
        offered_rate: float,
        p95_so_far_s: float,
    ) -> Frame:
        bandwidth = self.bytes / width if width else 0.0
        flops = self.flops / width if width else 0.0
        ram_bytes = ceilings.static_bytes + self.peak_kv_bytes
        # Rounded on the way out. These are chart values -- nothing downstream
        # can use the 15th significant digit, and full float repr roughly
        # doubles the payload the browser has to hold for playback. Coarse
        # enough to matter, fine enough that no visible line moves.
        return Frame(
            t_s=round(t_s, 3),
            queued=self.max_queued,
            active=self.max_active,
            cpu_pct=round(_pct(min(self.busy_s / width, 1.0) if width else 0.0), 2),
            bw_gbs=round(bandwidth / 1e9, 4),
            bw_pct=round(_pct(
                bandwidth / ceilings.bandwidth_bytes_s if ceilings.bandwidth_bytes_s else 0.0
            ), 2),
            compute_pct=round(
                _pct(flops / ceilings.compute_flops if ceilings.compute_flops else 0.0), 2
            ),
            kv_gb=round(self.peak_kv_bytes / GB, 4),
            ram_gb=round(ram_bytes / GB, 4),
            arrived=self.arrived,
            delivered=self.delivered,
            offered_rate=round(offered_rate, 2),
            p95_so_far_s=round(p95_so_far_s, 3),
        )


class _RunningP95:
    """p95 over everything delivered so far, walked forward once.

    Dropped alarms are excluded, the same population `RunStats.p95_s` is
    computed over -- a shed alarm has a total of zero and would drag the running
    figure *down* exactly when the build is in trouble.
    """

    def __init__(self, served: list[Delivery]) -> None:
        self._order = sorted(
            ((d.delivered_s, d.total_s) for d in served), key=lambda pair: pair[0]
        )
        self._cursor = 0
        self._totals: list[float] = []

    def at(self, t_s: float) -> float:
        while self._cursor < len(self._order) and self._order[self._cursor][0] <= t_s:
            bisect.insort(self._totals, self._order[self._cursor][1])
            self._cursor += 1
        return _percentile(self._totals, 0.95)


def _percentile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolated percentile, defined exactly as `pipeline` defines it.

    Copied rather than imported for the same reason `pipeline` copies it from
    `simulate`: the running p95 is compared against `RunStats.p95_s` in the
    tests and by anyone reading the two side by side, and that comparison only
    means something if both use one definition. Reaching into another module's
    private helper to get it would be worse.
    """
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    low = int(pos)
    high = min(low + 1, len(sorted_values) - 1)
    frac = pos - low
    return sorted_values[low] * (1 - frac) + sorted_values[high] * frac


# --------------------------------------------------------------------------
# The findings a run produces
# --------------------------------------------------------------------------


def _breach(built: list[Frame], sla_s: float) -> dict | None:
    """First frame whose running p95 crossed the SLA.

    On a ramp this *is* the overload point: the offered rate recorded next to it
    is the load the build stopped holding at. Reported as the first crossing
    rather than the worst frame, because the operator is buying headroom up to
    that rate, not past it.
    """
    for frame in built:
        if frame.p95_so_far_s > sla_s:
            return {
                "t_s": frame.t_s,
                "offered_rate": frame.offered_rate,
                "p95_s": frame.p95_so_far_s,
            }
    return None


def _worst(deliveries: list[Delivery], worst_n: int) -> list[dict]:
    """The slowest served alarms, as plain dicts a page can render."""
    served = [d for d in deliveries if not d.dropped]
    served.sort(key=lambda d: (-d.total_s, d.alarm_id))
    return [
        {
            "alarm_id": d.alarm_id,
            "severity": d.severity,
            "storm_id": d.storm_id,
            "arrived_s": d.arrived_s,
            "queue_wait_s": d.queue_wait_s,
            "ttft_s": d.ttft_s,
            "generate_s": d.generate_s,
            "deliver_s": d.deliver_s,
            "total_s": d.total_s,
            "slot": d.slot,
        }
        for d in served[:worst_n]
    ]


def _machine(asm) -> dict:
    """Assembly summary, flattened to things that survive JSON."""
    vm = getattr(asm, "vm", None)
    cpu = getattr(asm, "cpu", None)
    model = getattr(asm, "model", None)
    quant = getattr(asm, "quant", None)
    sockets = int(getattr(vm, "sockets", 1) or 1)
    return {
        "name": str(getattr(vm, "name", "") or ""),
        "cpu": f"{getattr(cpu, 'vendor', '')} {getattr(cpu, 'model', '')}".strip(),
        "sockets": sockets,
        "cores": int(getattr(cpu, "cores", 0) or 0) * sockets,
        "model": str(getattr(model, "name", "") or getattr(model, "id", "")),
        "quant": str(getattr(quant, "id", "") or ""),
        "slots": int(getattr(vm, "slots", 0) or 0),
        "ram_total_gb": getattr(asm, "ram_total_gb", None),
        "ram_used_gb": getattr(asm, "ram_used_gb", None),
        "channels_total": getattr(asm, "channels_total", None),
        "channels_populated": getattr(asm, "channels_populated", None),
        "bandwidth_gbs": getattr(asm, "bandwidth_gbs", None),
        "bandwidth_full_gbs": getattr(asm, "bandwidth_full_gbs", None),
        "prefill_tps": getattr(asm, "prefill_tps", None),
        "decode_tps_single": getattr(asm, "decode_tps_single", None),
        "uncertainty": getattr(asm, "uncertainty", None),
    }


def _notes(
    profile,
    asm,
    stats: RunStats,
    ceilings: Ceilings,
    span_s: float,
    breach: dict | None,
    workload: Workload,
) -> list[str]:
    """What a reader has to know before trusting the numbers above."""
    notes = list(getattr(profile, "notes", []) or [])

    uncertainty = float(getattr(asm, "uncertainty", 0.0) or 0.0)
    if uncertainty > 0:
        notes.append(
            f"모든 소요 시간은 계산값이다 — 측정이 아니다. 처리량 예측의 오차는 "
            f"±{uncertainty:.0%}이고, 그만큼 이 결과 전체가 움직인다"
        )
    if ceilings.bandwidth_confidence == "estimate":
        notes.append("대역폭 천장이 추정 계수 위에 서 있다 — 사용률 %의 절대값은 신뢰하지 말 것")
    if ceilings.compute_confidence == "estimate":
        notes.append("연산 천장이 추정 계수 위에 서 있다 — 사용률 %의 절대값은 신뢰하지 말 것")

    notes.append(
        "프레임 안의 토큰 처리율은 각 요청의 prefill·decode 구간에 균등 분배해 복원한 값이다 "
        "— 점유·큐·지연은 정확하고, 프레임 단위 토큰율만 근사다"
    )

    declared = float(getattr(profile, "span_s", 0.0) or 0.0)
    if span_s > declared + 1.0:
        notes.append(
            f"부하가 끝난 뒤에도 {(span_s - declared) / 60:.1f}분 더 일했다 "
            f"— 백로그가 부하 구간을 넘겼다"
        )
    if stats.dropped:
        notes.append(
            f"{stats.dropped:,}건이 큐 상한에 걸려 버려졌다 — p95는 살아남은 건들만의 값이다"
        )
    if breach is None:
        notes.append(
            f"이 부하 전 구간에서 p95가 SLA {workload.sla_seconds:g}초를 넘지 않았다 "
            f"— 무너지는 지점은 이 부하 범위 밖에 있다"
        )
    else:
        notes.append(
            f"부하율 {breach['offered_rate']:,.0f}건/일 지점에서 p95가 "
            f"SLA {workload.sla_seconds:g}초를 넘었다 (p95 {breach['p95_s']:.1f}초)"
        )
    return notes


def _pct(fraction: float) -> float:
    return max(0.0, fraction) * 100.0
