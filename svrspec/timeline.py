"""Resource usage over the simulated day, phase-resolved.

The point of this module is that "CPU usage" is not one number. A task manager
that reports a single busy percentage and then reuses it for memory and for
bandwidth is not showing three resources, it is showing one resource three
times -- which is exactly what the earlier resource view did.

The two phases of an LLM request load completely different parts of the box:

    prefill   a GEMM over the prompt. Saturates the vector units. Touches DRAM
              only once per micro-batch, so bandwidth sits near idle.
    decode    one full sweep of the weight set per emitted token. Saturates
              DRAM. The vector units are mostly waiting.

So during a storm the bandwidth graph and the compute graph move in opposite
directions, and the resource that runs out first is visible rather than
inferred. That is the whole question "where does it overload" reduces to.

Everything here is derived from `SimTrace.segments`, which the event loop emits
at every state change. Because rates are constant inside a segment, the series
is an exact reconstruction of the run -- not a sampling of it. Feed it segments
from a *measured* bench run instead and the same code draws the real machine.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .memory import (
    DEFAULT_UBATCH,
    GB,
    compute_buffer_bytes,
    decode_weight_bytes,
    kv_bytes_per_token,
    weight_bytes,
)
from .perf import Efficiency, peak_flops, widest_isa
from .simulate import SimSegment, SimTrace
from .types import (
    CpuSpec,
    MemoryOption,
    ModelSpec,
    QuantSpec,
    ThroughputPrediction,
    TokenProfile,
    Workload,
)

DAY_SECONDS = 24 * 3600.0

#: 15-minute resolution over a day. Coarse enough to draw, fine enough that a
#: thirty-second storm still reads as a spike instead of being averaged away.
DEFAULT_BUCKETS = 96

#: A resource is called saturated above this share of its ceiling.
SATURATED = 0.85


@dataclass(frozen=True)
class Ceilings:
    """Denominators. Every percentage in a bucket is against one of these.

    Split out from the prediction because a measured run needs the same
    denominators to be comparable, and because writing them down makes it
    obvious which ones are hardware facts (weight bytes, KV bytes) and which
    carry an efficiency coefficient (bandwidth, flops).
    """

    #: Achievable DRAM read bandwidth, bytes/s. Already derated by eta_bw.
    bandwidth_bytes_s: float
    #: Achievable vector throughput, FLOP/s. Already derated by eta_compute.
    compute_flops: float
    #: Bytes pulled from DRAM for one full sweep of the active weight set.
    weight_bytes: float
    #: KV bytes for one token at one context position.
    kv_bytes_token: float
    #: FLOPs to push one token through the network, either phase.
    flops_per_token: float
    #: Resident bytes that never move: weights + compute buffers + OS.
    static_bytes: float
    #: KV that llama.cpp reserves up front (full context x slots), whether or
    #: not it is touched. Reported alongside the working set so a memory graph
    #: cannot imply the box needs less RAM than it must actually be given.
    kv_reserved_bytes: float
    installed_bytes: float
    ubatch: int = DEFAULT_UBATCH
    #: Provenance for the two derated ceilings, so a chart can say whether the
    #: line it is drawing rests on a measurement or on a guess.
    bandwidth_confidence: str = "estimate"
    compute_confidence: str = "estimate"

    @property
    def allocated_bytes(self) -> float:
        return self.static_bytes + self.kv_reserved_bytes


@dataclass(frozen=True)
class ResourceBucket:
    """One time slice of the day, as a monitoring agent would have logged it."""

    t_s: float
    span_s: float
    #: Share of the slice with at least one request in flight. llama.cpp pins
    #: every thread while it works, so this is the CPU graph.
    cpu_pct: float
    #: Averaged over the whole slice -- the honest daily-load figure.
    bandwidth_avg_gbs: float
    bandwidth_avg_pct: float
    compute_avg_tflops: float
    compute_avg_pct: float
    #: The worst instant inside the slice. This is the number that decides
    #: whether the box coped, and averaging it away is how a sizing tool ends
    #: up claiming an idle server that misses its SLA every storm.
    bandwidth_peak_pct: float
    compute_peak_pct: float
    prefill_tps: float
    decode_tps: float
    #: KV actually touched, peak within the slice.
    kv_used_gb: float
    ram_used_gb: float
    ram_pct: float
    queued: int
    active: int
    arrived: int
    completed: int

    @property
    def saturated(self) -> str | None:
        """Which ceiling the slice pressed hardest against, if either.

        Both are usually touched inside one slice -- any prefill instant sits
        on the compute ceiling and any decode instant on a bandwidth or compute
        one -- so checking them in a fixed order would report whichever came
        first in the code rather than whichever mattered. Report the one that
        got closer.
        """
        top = max(self.bandwidth_peak_pct, self.compute_peak_pct)
        if top < SATURATED * 100:
            return None
        return "bandwidth" if self.bandwidth_peak_pct >= self.compute_peak_pct else "compute"


@dataclass(frozen=True)
class Timeline:
    buckets: list[ResourceBucket]
    ceilings: Ceilings
    #: Share of *busy* time each phase held the machine. Answers "is this box
    #: prompt-bound or generation-bound for this workload", which decides
    #: whether more cores or more memory channels is the right money.
    prefill_share: float
    decode_share: float
    busy_seconds: float
    peak_bandwidth_pct: float
    peak_compute_pct: float
    peak_kv_used_gb: float
    peak_queue: int
    #: Seconds of the day spent with a resource above SATURATED.
    seconds_bandwidth_bound: float
    seconds_compute_bound: float
    #: How far past the window the run was still working. Non-zero means the
    #: day's alarms did not fit in the day -- the backlog spilled into the next
    #: one, which is overload in its plainest form.
    overran_s: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def binding_resource(self) -> str:
        """Which ceiling held the machine for more of its working time.

        Not "was anything saturated" -- something almost always is. A decoding
        request sits at the bandwidth ceiling by definition and a prefilling
        one sits at the compute ceiling, so the useful question is which of the
        two owned more of the day. That is the answer that decides whether the
        next server should buy memory channels or cores.
        """
        if self.seconds_bandwidth_bound == 0 and self.seconds_compute_bound == 0:
            return "none"
        return (
            "bandwidth"
            if self.seconds_bandwidth_bound >= self.seconds_compute_bound
            else "compute"
        )


def ceilings_for(
    model: ModelSpec,
    quant: QuantSpec,
    cpu: CpuSpec,
    memory: MemoryOption,
    eff: Efficiency,
    workload: Workload,
    throughput: ThroughputPrediction,
    sockets: int = 1,
    installed_gb: int | None = None,
    ctx_tokens: int | None = None,
    measured_bpw: float | None = None,
) -> Ceilings:
    """Resolve the denominators for one build.

    `throughput` supplies the bandwidth ceiling rather than recomputing it, so
    the timeline can never disagree with the prediction the report quotes.
    """
    tokens: TokenProfile = workload.tokens
    isa = widest_isa(cpu)
    raw_flops, _ = peak_flops(cpu, sockets)
    eta_c = eff.eta_compute(isa)
    eta_b = eff.eta_bw(memory.ddr_gen)

    ctx = ctx_tokens if ctx_tokens else tokens.peak_ctx_tokens
    static = (
        weight_bytes(model, quant, measured_bpw)
        + compute_buffer_bytes(model, workload.slots)
        + _os_bytes()
    )
    reserved = kv_bytes_per_token(model) * ctx * max(workload.slots, 1)
    installed = installed_gb if installed_gb else _round_up_gb((static + reserved) / GB * 1.5)

    return Ceilings(
        bandwidth_bytes_s=throughput.effective_bandwidth_gbs * 1e9,
        compute_flops=raw_flops * eta_c.value,
        weight_bytes=decode_weight_bytes(model, quant, measured_bpw),
        kv_bytes_token=kv_bytes_per_token(model),
        flops_per_token=2.0 * model.decode_params_b * 1e9,
        static_bytes=static,
        kv_reserved_bytes=reserved,
        installed_bytes=installed * GB,
        bandwidth_confidence=eta_b.confidence,
        compute_confidence=eta_c.confidence,
    )


def build_timeline(
    trace: SimTrace,
    ceilings: Ceilings,
    span_s: float = DAY_SECONDS,
    buckets: int = DEFAULT_BUCKETS,
) -> Timeline:
    """Fold a run's segments into a fixed-width resource series."""
    if buckets <= 0 or span_s <= 0:
        raise ValueError("span_s and buckets must be positive")

    width = span_s / buckets
    acc = [_Accumulator() for _ in range(buckets)]

    total_prefill_s = 0.0
    total_decode_s = 0.0
    for seg in trace.segments:
        rates = segment_rates(seg, ceilings)
        # Phase occupancy: charge the segment to whichever phase held the
        # machine. A segment with both running counts for both, weighted.
        if seg.active:
            total_prefill_s += seg.span_s * seg.prefilling / seg.active
            total_decode_s += seg.span_s * seg.decoding / seg.active
        for index, slice_s in _spread(seg.t_s, seg.span_s, width, buckets):
            acc[index].add(seg, rates, slice_s, ceilings)

    for at in trace.arrivals_at:
        acc[_index(at, width, buckets)].arrived += 1
    for at in trace.completions_at:
        acc[_index(at, width, buckets)].completed += 1

    out = [a.finish(i * width, width, ceilings) for i, a in enumerate(acc)]

    busy = sum(a.busy_s for a in acc)
    bw_bound = sum(a.saturated_bandwidth_s for a in acc)
    cp_bound = sum(a.saturated_compute_s for a in acc)

    notes: list[str] = []
    if ceilings.bandwidth_confidence == "estimate":
        notes.append("대역폭 천장이 추정 계수 위에 서 있다 — 사용률 %의 절대값은 신뢰하지 말 것")
    if ceilings.compute_confidence == "estimate":
        notes.append("연산 천장이 추정 계수 위에 서 있다 — 사용률 %의 절대값은 신뢰하지 말 것")
    last = trace.segments[-1] if trace.segments else None
    overran = max((last.t_s + last.span_s) - span_s, 0.0) if last else 0.0
    if overran > 0:
        notes.append(
            f"하루가 끝난 뒤에도 {overran / 3600:.1f}시간 더 일했다 — "
            f"백로그가 다음 날로 넘어간다"
        )

    peak_queue = max((b.queued for b in out), default=0)
    if peak_queue == 0:
        # Worth saying plainly: an instantaneous ceiling is not overload. A
        # decoding request sits at ~100% of the bandwidth ceiling by
        # definition -- that is what "bandwidth bound" means, not a problem.
        # Overload is when work arrives faster than it drains, and that shows
        # up as a queue.
        notes.append("큐가 한 번도 쌓이지 않았다 — 이 부하에서 과부하는 발생하지 않는다")

    return Timeline(
        buckets=out,
        ceilings=ceilings,
        prefill_share=total_prefill_s / busy if busy else 0.0,
        decode_share=total_decode_s / busy if busy else 0.0,
        busy_seconds=busy,
        peak_bandwidth_pct=max((b.bandwidth_peak_pct for b in out), default=0.0),
        peak_compute_pct=max((b.compute_peak_pct for b in out), default=0.0),
        peak_kv_used_gb=max((b.kv_used_gb for b in out), default=0.0),
        peak_queue=peak_queue,
        seconds_bandwidth_bound=bw_bound,
        seconds_compute_bound=cp_bound,
        overran_s=overran,
        notes=notes,
    )


# --------------------------------------------------------------------------
# Per-segment physics
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SegmentRates:
    """Instantaneous resource draw during one constant-rate segment."""

    bandwidth_bytes_s: float
    flops: float
    prefill_tps: float
    decode_tps: float
    kv_bytes: float


def segment_rates(seg: SimSegment, c: Ceilings) -> SegmentRates:
    """Convert one segment into the resources it draws while it runs.

    Public because it is the single definition of how this project turns work
    into bytes and flops, and more than one view needs it: the 15-minute
    resource timeline, the fixed-period host samples in `hostsim`, and the
    bench frames in `bench`. Each used to carry its own copy, and copies of a
    formula drift until two screens quote different bandwidths for the same
    second.
    """
    if seg.span_s <= 0:
        return SegmentRates(0.0, 0.0, 0.0, 0.0, 0.0)

    prefill_tps = seg.prefill_tokens / seg.span_s
    decode_tps = seg.decode_tokens / seg.span_s

    # Decode: one sweep of the weights per batched token round, plus each
    # request re-reading its own KV. With `decoding` slots generating together
    # a single sweep yields `decoding` tokens -- that amortisation is the whole
    # reason continuous batching helps on a bandwidth-bound box.
    mean_ctx = seg.kv_tokens / seg.active if seg.active else 0.0
    sweeps_s = decode_tps / seg.decoding if seg.decoding else 0.0
    decode_bytes_s = sweeps_s * c.weight_bytes + decode_tps * c.kv_bytes_token * mean_ctx

    # Prefill: the weights stream once per micro-batch, not once per token.
    prefill_bytes_s = (prefill_tps / c.ubatch) * c.weight_bytes if c.ubatch else 0.0

    return SegmentRates(
        bandwidth_bytes_s=decode_bytes_s + prefill_bytes_s,
        flops=(prefill_tps + decode_tps) * c.flops_per_token,
        prefill_tps=prefill_tps,
        decode_tps=decode_tps,
        kv_bytes=seg.kv_tokens * c.kv_bytes_token,
    )


class _Accumulator:
    """Time-weighted sums for one bucket, plus the peaks inside it."""

    def __init__(self) -> None:
        self.busy_s = 0.0
        self.bytes = 0.0
        self.flops = 0.0
        self.prefill_tokens = 0.0
        self.decode_tokens = 0.0
        self.peak_bandwidth = 0.0
        self.peak_flops = 0.0
        self.peak_kv_bytes = 0.0
        self.max_queued = 0
        self.max_active = 0
        self.arrived = 0
        self.completed = 0
        self.saturated_bandwidth_s = 0.0
        self.saturated_compute_s = 0.0

    def add(self, seg: SimSegment, r: SegmentRates, slice_s: float, c: Ceilings) -> None:
        self.busy_s += slice_s
        self.bytes += r.bandwidth_bytes_s * slice_s
        self.flops += r.flops * slice_s
        self.prefill_tokens += r.prefill_tps * slice_s
        self.decode_tokens += r.decode_tps * slice_s
        self.peak_bandwidth = max(self.peak_bandwidth, r.bandwidth_bytes_s)
        self.peak_flops = max(self.peak_flops, r.flops)
        self.peak_kv_bytes = max(self.peak_kv_bytes, r.kv_bytes)
        self.max_queued = max(self.max_queued, seg.queued)
        self.max_active = max(self.max_active, seg.active)
        if c.bandwidth_bytes_s and r.bandwidth_bytes_s / c.bandwidth_bytes_s >= SATURATED:
            self.saturated_bandwidth_s += slice_s
        if c.compute_flops and r.flops / c.compute_flops >= SATURATED:
            self.saturated_compute_s += slice_s

    def finish(self, t_s: float, width: float, c: Ceilings) -> ResourceBucket:
        used_kv = self.peak_kv_bytes
        ram_used = c.static_bytes + max(used_kv, 0.0)
        return ResourceBucket(
            t_s=t_s,
            span_s=width,
            cpu_pct=_pct(self.busy_s / width if width else 0.0),
            bandwidth_avg_gbs=(self.bytes / width) / 1e9 if width else 0.0,
            bandwidth_avg_pct=_pct(
                (self.bytes / width) / c.bandwidth_bytes_s if width and c.bandwidth_bytes_s else 0.0
            ),
            compute_avg_tflops=(self.flops / width) / 1e12 if width else 0.0,
            compute_avg_pct=_pct(
                (self.flops / width) / c.compute_flops if width and c.compute_flops else 0.0
            ),
            bandwidth_peak_pct=_pct(
                self.peak_bandwidth / c.bandwidth_bytes_s if c.bandwidth_bytes_s else 0.0
            ),
            compute_peak_pct=_pct(self.peak_flops / c.compute_flops if c.compute_flops else 0.0),
            prefill_tps=self.prefill_tokens / width if width else 0.0,
            decode_tps=self.decode_tokens / width if width else 0.0,
            kv_used_gb=used_kv / GB,
            ram_used_gb=ram_used / GB,
            ram_pct=_pct(ram_used / c.installed_bytes if c.installed_bytes else 0.0),
            queued=self.max_queued,
            active=self.max_active,
            arrived=self.arrived,
            completed=self.completed,
        )


def _spread(start: float, span: float, width: float, count: int):
    """Split a segment across the buckets it straddles, charging each its share.

    A run that overloads does not finish inside its window: a backlog drains
    into the next day. Everything past the last bucket is charged to it rather
    than dropped, so the series still conserves the work -- and so the caller
    can see the pile-up instead of losing it off the end of the chart.
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


def _pct(fraction: float) -> float:
    return max(0.0, fraction) * 100.0


def _os_bytes() -> float:
    from .memory import RUNTIME_OS_GB

    return RUNTIME_OS_GB * GB


def _round_up_gb(gb: float) -> int:
    from .memory import provisionable

    return provisionable(gb)
