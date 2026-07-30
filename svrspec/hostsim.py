"""Synthetic host telemetry: what a monitoring agent would have logged.

`timeline.py` answers "which ceiling binds this build" at 15-minute resolution.
This module sits one layer below it and answers a different question: "what
would Zabbix have drawn". Same run, same physics, but fixed-period samples of
the things an operator actually watches -- CPU, load average, RSS, DRAM read
bandwidth, queue depth.

Why it can exist without running anything: `SimSegment` is an interval over
which *every* rate in the system was constant, because the event loop advances
to the next state change rather than stepping time. Folding segments into fixed
periods is therefore a reconstruction of the day, not a sampling of it. Nothing
happened between two segments that a sample could have missed.

Nothing here reads the real machine: no external commands, no kernel counters,
no wall clock, no chance element. Every number is derived from the simulation
output, so the same input always produces the same bytes. That is a hard
constraint, not a preference -- this tool has to be runnable on the operator's
laptop while it is sizing a server, without loading that laptop, and a chart
that moves between two identical runs is a chart nobody can reason about.
`tests/test_hostsim.py` enforces it against the source text, which is also why
the forbidden module names are spelled nowhere in this file.
"""

from __future__ import annotations

import csv
import io
import math
from dataclasses import dataclass, field, fields

from .memory import GB, RUNTIME_OS_GB
from .simulate import SimSegment, SimTrace
from .timeline import Ceilings
from .types import CpuSpec

#: Time constant of the reported load average, in seconds. Linux reports 1/5/15
#: minute figures; one minute is the one an operator watches during an incident,
#: and it is short enough that a 30-second storm still registers.
LOAD_TAU_S = 60.0

#: Per-core spread as a share of the headroom to the nearer rail (0% or 100%).
#: Scaling by headroom rather than using a fixed percentage means the deviation
#: can never push a core past a rail, so no clamping is needed and the mean of
#: `per_core_pct` stays exactly equal to `cpu_pct`.
CORE_SPREAD = 0.02


@dataclass(frozen=True)
class HostSample:
    """One fixed-period observation of the host, as an agent would have logged it."""

    #: Start of the sampling interval, seconds from the start of the day.
    t_s: float
    #: Share of the interval with at least one request in flight, 0..100.
    cpu_pct: float
    #: Same load, split per core. Length is cores x sockets.
    per_core_pct: tuple[float, ...]
    #: One-minute EWMA of (active + queued). Lags by construction.
    load_avg_1m: float
    #: Resident set of the llama.cpp process: weights + buffers + touched KV.
    rss_gb: float
    #: The whole box: the process plus the OS underneath it.
    host_mem_used_gb: float
    host_mem_total_gb: float
    #: Interval-average effective DRAM read bandwidth.
    bandwidth_gbs: float
    bandwidth_pct: float
    #: Worst instant inside the interval -- an average queue depth is useless.
    queue_depth: int
    active_slots: int
    #: Interval-average token rate, both phases summed.
    tokens_per_s: float
    #: Raw busy seconds charged to this interval, *not* clamped to the period.
    #: Backlog that outran the window is charged to the last sample, so this can
    #: exceed `period_s` there while `cpu_pct` stays at 100. Kept so the series
    #: can be checked for conservation of work against the segments it came from.
    busy_s: float = 0.0


@dataclass
class HostTrace:
    samples: list[HostSample]
    period_s: float
    cpu_label: str
    #: Total logical cores across all sockets -- the length of `per_core_pct`.
    cores: int
    installed_gb: int
    peak_rss_gb: float
    notes: list[str] = field(default_factory=list)


def sample_host(
    trace: SimTrace,
    ceilings: Ceilings,
    cpu: CpuSpec,
    *,
    installed_gb: int,
    sockets: int = 1,
    period_s: float = 10.0,
    span_s: float = 86400.0,
) -> HostTrace:
    """Fold a simulated run into fixed-period host samples.

    The window is rounded up to a whole number of periods, so the last sample is
    never a partial one. Work that ran past the end of the window is charged to
    that last sample rather than dropped -- see `_spread`.
    """
    if period_s <= 0 or span_s <= 0:
        raise ValueError("period_s and span_s must be positive")

    count = max(1, math.ceil(span_s / period_s - 1e-9))
    window_s = count * period_s
    cores = max(cpu.cores * max(sockets, 1), 1)

    acc = [_Accumulator() for _ in range(count)]
    for seg in trace.segments:
        rates = _rates(seg, ceilings)
        for index, slice_s in _spread(seg.t_s, seg.span_s, period_s, count):
            acc[index].add(seg, rates, slice_s)

    offsets = _core_offsets(cores)
    os_bytes = RUNTIME_OS_GB * GB
    # Weights + compute buffers, i.e. the resident part of the process that does
    # not move. `static_bytes` bundles the OS in, so it comes back out here:
    # RSS is the process, `host_mem_used_gb` is the box.
    process_static = max(ceilings.static_bytes - os_bytes, 0.0)

    samples: list[HostSample] = []
    load = 0.0
    # A sampled EWMA with the period folded into the coefficient, so changing
    # the sampling period changes the resolution of the curve but not its shape.
    alpha = 1.0 - math.exp(-period_s / LOAD_TAU_S)

    for i, a in enumerate(acc):
        cpu_pct = min(a.busy_s / period_s, 1.0) * 100.0
        # Offered load, time-weighted over the interval: idle time contributes
        # zero, so a half-busy interval reads as half the load it carried.
        offered = a.load_seconds / period_s
        load += alpha * (offered - load)

        kv_bytes = a.peak_kv_tokens * ceilings.kv_bytes_token
        rss = process_static + kv_bytes
        used = ceilings.static_bytes + kv_bytes
        bytes_s = a.bytes / period_s

        samples.append(
            HostSample(
                t_s=i * period_s,
                cpu_pct=cpu_pct,
                per_core_pct=_per_core(cpu_pct, offsets),
                load_avg_1m=load,
                rss_gb=rss / GB,
                host_mem_used_gb=used / GB,
                host_mem_total_gb=float(installed_gb),
                bandwidth_gbs=bytes_s / 1e9,
                bandwidth_pct=(
                    max(bytes_s / ceilings.bandwidth_bytes_s, 0.0) * 100.0
                    if ceilings.bandwidth_bytes_s
                    else 0.0
                ),
                queue_depth=a.max_queued,
                active_slots=a.max_active,
                tokens_per_s=(a.prefill_tokens + a.decode_tokens) / period_s,
                busy_s=a.busy_s,
            )
        )

    notes: list[str] = []
    if ceilings.bandwidth_confidence == "estimate":
        notes.append(
            "대역폭 천장이 추정 계수 위에 서 있다 — 사용률 %의 절대값은 신뢰하지 말 것"
        )
    last = trace.segments[-1] if trace.segments else None
    overran = max((last.t_s + last.span_s) - window_s, 0.0) if last else 0.0
    if overran > 0:
        notes.append(
            f"창이 끝난 뒤에도 {overran / 3600:.1f}시간 더 일했다 — "
            f"그 몫은 마지막 샘플에 실려 있어 마지막 샘플의 대역폭·토큰 수치는 "
            f"구간 평균이 아니다"
        )

    return HostTrace(
        samples=samples,
        period_s=period_s,
        cpu_label=f"{cpu.vendor} {cpu.model}",
        cores=cores,
        installed_gb=installed_gb,
        peak_rss_gb=max((s.rss_gb for s in samples), default=0.0),
        notes=notes,
    )


def to_csv(host: HostTrace) -> str:
    """The samples as a CSV table, one row per sample plus a header.

    No BOM: the caller decides the encoding when it writes the file, and a BOM
    embedded here would end up in the middle of anything that concatenates.
    `per_core_pct` is a single space-separated column, so the header stays fixed
    regardless of core count.
    """
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    names = [f.name for f in fields(HostSample)]
    writer.writerow(names)
    for s in host.samples:
        row = []
        for name in names:
            value = getattr(s, name)
            if name == "per_core_pct":
                row.append(" ".join(f"{v:.4f}" for v in value))
            elif isinstance(value, bool) or isinstance(value, int):
                row.append(str(value))
            else:
                row.append(f"{value:.4f}")
        writer.writerow(row)
    return out.getvalue()


# --------------------------------------------------------------------------
# Per-segment physics
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Rates:
    bandwidth_bytes_s: float
    prefill_tps: float
    decode_tps: float


def _rates(seg: SimSegment, c: Ceilings) -> _Rates:
    """Instantaneous draw during one constant-rate segment.

    Deliberately a copy of `timeline._rates`, not an import of it: that function
    is private and returns fields this module has no use for. The formulas below
    must stay identical to it -- two modules describing the same run may not
    disagree about how many bytes it moved.

        decode   one sweep of the weight set per batched token round, plus each
                 request re-reading its own KV. `decoding` slots generating
                 together get one token each per sweep, which is the whole
                 reason continuous batching helps a bandwidth-bound box.
        prefill  the weights stream once per micro-batch, not once per token.
    """
    if seg.span_s <= 0:
        return _Rates(0.0, 0.0, 0.0)

    prefill_tps = seg.prefill_tokens / seg.span_s
    decode_tps = seg.decode_tokens / seg.span_s

    mean_ctx = seg.kv_tokens / seg.active if seg.active else 0.0
    sweeps_s = decode_tps / seg.decoding if seg.decoding else 0.0
    decode_bytes_s = sweeps_s * c.weight_bytes + decode_tps * c.kv_bytes_token * mean_ctx
    prefill_bytes_s = (prefill_tps / c.ubatch) * c.weight_bytes if c.ubatch else 0.0

    return _Rates(
        bandwidth_bytes_s=decode_bytes_s + prefill_bytes_s,
        prefill_tps=prefill_tps,
        decode_tps=decode_tps,
    )


class _Accumulator:
    """Time-weighted sums for one sampling interval, plus the peaks inside it."""

    def __init__(self) -> None:
        self.busy_s = 0.0
        self.bytes = 0.0
        self.prefill_tokens = 0.0
        self.decode_tokens = 0.0
        #: KV is reported at its peak, not its mean: memory is a high-water mark
        #: problem. A box that fit the average and not the peak is a box that
        #: OOMs. `peak` here is the largest residency of any segment touching
        #: this interval, which is the closest thing to what an agent polling
        #: RSS would have caught.
        self.peak_kv_tokens = 0.0
        self.load_seconds = 0.0
        self.max_queued = 0
        self.max_active = 0

    def add(self, seg: SimSegment, r: _Rates, slice_s: float) -> None:
        self.busy_s += slice_s
        self.bytes += r.bandwidth_bytes_s * slice_s
        self.prefill_tokens += r.prefill_tps * slice_s
        self.decode_tokens += r.decode_tps * slice_s
        self.peak_kv_tokens = max(self.peak_kv_tokens, seg.kv_tokens)
        self.load_seconds += (seg.active + seg.queued) * slice_s
        self.max_queued = max(self.max_queued, seg.queued)
        self.max_active = max(self.max_active, seg.active)


def _spread(start: float, span: float, width: float, count: int):
    """Split a segment across the samples it straddles, charging each its share.

    Same contract as `timeline._spread`, and the same reason for the trailing
    clause: an overloaded run does not finish inside its window, and everything
    past the last sample is charged to that sample rather than dropped. Losing
    it would make the series claim the machine went idle exactly when it was
    most behind. The loop terminates on `remaining`, never on the cursor
    reaching the window -- a backlog that ends hours late used to spin here.
    """
    end_of_window = width * count
    remaining, cursor = span, start
    while remaining > 1e-9:
        if cursor >= end_of_window - 1e-9:
            yield count - 1, remaining
            return
        index = _index(cursor, width, count)
        edge = (index + 1) * width
        slice_s = min(remaining, edge - cursor)
        if slice_s <= 0:
            yield count - 1, remaining
            return
        yield index, slice_s
        cursor += slice_s
        remaining -= slice_s


def _index(at: float, width: float, count: int) -> int:
    return min(count - 1, max(0, int(at / width)))


def _core_offsets(cores: int) -> tuple[float, ...]:
    """Fixed per-core bias, summing to zero.

    Real per-core graphs are never a flat set of identical lines, so a chart of
    identical lines reads as fake. But the deviation has to be *deterministic*:
    a drawn-at-runtime jitter would make two runs of the same sizing disagree,
    and an operator comparing two builds could not tell a real difference from
    the jitter.

    A fixed function of the core index also happens to be the honest model. The
    bias on a real box is not noise either -- core 0 takes the interrupts, the
    scheduler has its favourites, and the same cores run hot run after run.

    Summing to zero is what keeps `mean(per_core_pct) == cpu_pct`, so the
    detail costs the aggregate nothing.
    """
    raw = [math.sin(i * 1.7 + 0.5) for i in range(cores)]
    mean = sum(raw) / cores
    return tuple(r - mean for r in raw)


def _per_core(cpu_pct: float, offsets: tuple[float, ...]) -> tuple[float, ...]:
    # Scaled by the distance to the nearer rail so an idle or pegged host shows
    # no spread at all -- which is also what a real one shows.
    amplitude = CORE_SPREAD * min(cpu_pct, 100.0 - cpu_pct)
    return tuple(cpu_pct + amplitude * o for o in offsets)
