"""Where a build breaks, and what breaks first.

`recommend` answers "which server for this load". This module answers the
question after it: given *this* server and *this* model, how much load can it
actually take before the SLA goes, and which resource ran out.

Both halves matter. A knee on its own ("breaks at 900 alarms/day") is a number
the customer cannot act on. Paired with the limiter ("storm drain went first;
the box was compute bound 80% of its working time") it tells them whether to
buy cores, buy memory channels, shorten the prompt, or put deduplication in
front of the LLM.

Four load axes, because they fail differently and the fix differs with them:

    alarms   daily volume. Usually the *least* interesting: a box idle 97% of
             the day does not care whether it is idle 94% instead.
    storm    alarms inside the burst window. This is what actually sizes the
             server, and the axis that breaks first almost every time.
    prompt   per-alarm prompt tokens. What RAG does to you: retrieved runbook
             chunks differ per alarm, so they cannot be prefix-cached and every
             one of them is billed prefill.
    output   generated tokens per alarm. Pure decode, so it walks straight into
             the memory bandwidth wall.

The search brackets the break geometrically and then bisects it, so the cost is
logarithmic in the answer rather than linear in the axis. Every point evaluated
is kept, because the curve on the way to the knee is what makes the result
legible as an experiment rather than an assertion.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from .catalog import Catalog
from .perf import Efficiency
from .simulate import simulate
from .sizing import MARGINAL, PASS, decode_table, evaluate
from .timeline import build_timeline, ceilings_for
from .types import CpuSpec, MemoryOption, ModelSpec, QuantSpec, Workload

#: Axis names accepted by `find_knee`.
AXES = ("alarms", "storm", "prompt", "output")

#: Ramp factor while bracketing. 2x doubles until it breaks, which finds the
#: bracket in log2 steps even when the starting guess is off by 100x.
RAMP = 2.0

#: Stop bisecting once the bracket is this tight, relatively. 2% is finer than
#: the input data justifies -- nobody knows their alarm volume to 2%.
TOLERANCE = 0.02

#: Hard stop per axis. Past these the simulated day carries so many events that
#: the cost stops buying anything: 50k alarms/day is 35 a minute sustained,
#: which is two orders of magnitude past any monitoring feed this tool is aimed
#: at, and a knee found out there is not a planning input.
CEILING = {"alarms": 50_000, "storm": 5_000, "prompt": 131_072, "output": 32_768}

#: Below this nothing can be evaluated meaningfully. The alarms floor is
#: computed per workload -- see `_floor`.
FLOOR = {"alarms": 1, "storm": 0, "prompt": 1, "output": 1}


def _floor(workload: Workload, axis: str) -> int:
    """Lowest value at which this axis still measures what it claims to.

    Storms draw their alarms *from* the daily budget rather than adding to it
    (see `workload.generate_arrivals`). So once the daily volume drops below
    the storm total, the generator truncates the storms and every alarm in the
    day becomes a storm alarm -- there is no quiet-hours traffic left, the
    steady-state p95 falls back to the whole population, and the axis stops
    measuring daily volume and starts measuring a degenerate storm.

    Walking below that point produced knees like "4 alarms/day" for a build
    that comfortably handles 359, which is not a finding, it is an artefact.
    """
    if axis != "alarms":
        return FLOOR[axis]
    storm_total = max(workload.storm_size, 0) * max(workload.storms_per_day, 0)
    return max(1, storm_total)


@dataclass(frozen=True)
class LoadPoint:
    """One evaluated load level."""

    value: int
    verdict: str
    p95_steady_s: float
    storm_drain_s: float
    max_queue: int
    busy_fraction: float
    #: Which ceiling owned most of the working time at this load.
    binding_resource: str
    prefill_share: float
    ram_needed_gb: int
    #: Why it failed, empty if it did not.
    reasons: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.verdict in (PASS, MARGINAL)


@dataclass(frozen=True)
class CapacityCurve:
    """The ramp, the knee, and the diagnosis."""

    axis: str
    #: Every level evaluated, ascending. The curve, for graphing.
    points: list[LoadPoint]
    #: Highest level that still met the SLA. None if even the floor failed.
    knee: LoadPoint | None
    #: Lowest level that failed. None if nothing failed below the ceiling.
    breaks_at: LoadPoint | None
    #: The clause that gave way first at `breaks_at`.
    limiter: str
    #: Load the workload actually runs at, for context against the knee.
    baseline: int
    hit_ceiling: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def headroom(self) -> float:
        """Knee divided by the load being planned for. <1 means already over."""
        if self.knee is None:
            return 0.0
        return self.knee.value / self.baseline if self.baseline else float("inf")


def apply_axis(workload: Workload, axis: str, value: int) -> Workload:
    """Set one load axis on a workload, leaving the others alone."""
    if axis == "alarms":
        return replace(workload, alarms_per_day=max(int(value), 1))
    if axis == "storm":
        return replace(workload, storm_size=max(int(value), 0))
    if axis == "prompt":
        return replace(workload, tokens=replace(workload.tokens, alarm_tokens=max(int(value), 1)))
    if axis == "output":
        return replace(workload, tokens=replace(workload.tokens, output_tokens=max(int(value), 1)))
    raise ValueError(f"unknown axis {axis!r}; expected one of {AXES}")


def axis_value(workload: Workload, axis: str) -> int:
    if axis == "alarms":
        return workload.alarms_per_day
    if axis == "storm":
        return workload.storm_size
    if axis == "prompt":
        return workload.tokens.alarm_tokens
    if axis == "output":
        return workload.tokens.output_tokens
    raise ValueError(f"unknown axis {axis!r}; expected one of {AXES}")


def find_knee(
    model: ModelSpec,
    quant: QuantSpec,
    cpu: CpuSpec,
    memory: MemoryOption,
    eff: Efficiency,
    workload: Workload,
    axis: str = "storm",
    sockets: int = 1,
    tolerance: float = TOLERANCE,
    max_evaluations: int = 40,
) -> CapacityCurve:
    """Ramp `axis` until the SLA breaks, then bisect to the knee.

    The verdict at each level is the same pessimistic one `recommend` uses, so
    a knee found here and a pass/fail printed there can never disagree.
    """
    if axis not in AXES:
        raise ValueError(f"unknown axis {axis!r}; expected one of {AXES}")

    baseline = axis_value(workload, axis)
    ceiling = CEILING[axis]
    floor = _floor(workload, axis)
    seen: dict[int, LoadPoint] = {}
    budget = [max_evaluations]
    probe_args = (model, quant, cpu, memory, eff, workload, sockets, axis)

    def probe(value: int) -> LoadPoint:
        value = int(max(floor, min(value, ceiling)))
        if value in seen:
            return seen[value]
        if budget[0] <= 0:
            return seen[min(seen, key=lambda v: abs(v - value))]
        budget[0] -= 1
        point = _evaluate_point(
            model, quant, cpu, memory, eff,
            apply_axis(workload, axis, value), sockets, value,
        )
        seen[value] = point
        return point

    notes: list[str] = []
    start = max(baseline, floor + 1)
    first = probe(start)

    if not first.ok:
        # Already over at the load being planned for. Walk *down* to find where
        # it would have held -- a customer whose current volume already misses
        # needs to know by how much, not just that it misses.
        good, bad = None, first
        low = floor
        while budget[0] > 0 and start > low + 1:
            start = max(low + 1, int(start / RAMP))
            point = probe(start)
            if point.ok:
                good = point
                break
            bad = point
        if good is not None:
            good, bad = _bisect(probe, good, bad, tolerance, budget)
        else:
            notes.append(
                f"현재 부하에서 이미 미달이고, 탐색 하한 {floor:,}까지 내려도 통과하는 "
                f"지점이 없다 — 이 축을 낮추는 것으로는 해결되지 않는다"
            )
        if axis == "alarms" and floor > FLOOR["alarms"]:
            notes.append(
                f"일 알람 하한을 스톰 총량({floor:,}건)에 맞춰 막았다 — 그 아래로는 "
                f"스톰이 하루를 통째로 차지해 이 축이 일량을 측정하지 못한다"
            )
        _diagnose(probe_args, seen, (good, bad))
        good = seen.get(good.value, good) if good else None
        bad = seen.get(bad.value, bad) if bad else None
        return _curve(axis, seen, good, bad, baseline, False, notes)

    # Holds at baseline: ramp up until it does not.
    good, bad = first, None
    value = max(start, floor + 1)
    while budget[0] > 0 and value < ceiling:
        value = min(ceiling, max(int(value * RAMP), value + 1))
        point = probe(value)
        if not point.ok:
            bad = point
            break
        good = point

    hit_ceiling = bad is None
    if hit_ceiling:
        notes.append(
            f"탐색 상한 {ceiling:,}까지 올려도 SLA가 깨지지 않았다 — "
            f"이 축에서는 이 빌드의 한계를 찾지 못했다"
        )
    else:
        good, bad = _bisect(probe, good, bad, tolerance, budget)

    if budget[0] <= 0:
        notes.append(f"평가 횟수 상한({max_evaluations}회)에 걸려 탐색을 중단했다")

    _diagnose(probe_args, seen, (good, bad))
    good = seen.get(good.value, good) if good else None
    bad = seen.get(bad.value, bad) if bad else None
    return _curve(axis, seen, good, bad, baseline, hit_ceiling, notes)


def _bisect(probe, good: LoadPoint, bad: LoadPoint, tolerance: float, budget):
    """Narrow the pass/fail bracket. Returns the tightest (good, bad)."""
    low, high = good.value, bad.value
    while budget[0] > 0 and high - low > 1 and (high - low) / max(high, 1) > tolerance:
        mid = (low + high) // 2
        if mid == low or mid == high:
            break
        point = probe(mid)
        if point.ok:
            good, low = point, mid
        else:
            bad, high = point, mid
    return good, bad


#: Placeholder for the resource diagnosis on probes that did not get one.
UNDIAGNOSED = "?"


def _evaluate_point(
    model: ModelSpec,
    quant: QuantSpec,
    cpu: CpuSpec,
    memory: MemoryOption,
    eff: Efficiency,
    workload: Workload,
    sockets: int,
    value: int,
    diagnose: bool = False,
) -> LoadPoint:
    """Evaluate one load level.

    `diagnose` adds the resource diagnosis, and costs a third simulated day on
    top of the two `evaluate` already runs. Most probes are stepping stones on
    the way to the knee and nobody reads their bottleneck, so it is off by
    default and switched on for the two points that get quoted. On the alarms
    axis, where a probe can be a 40,000-alarm day, that is a third of the
    search's wall clock.
    """
    candidate = evaluate(model, quant, cpu, memory, eff, workload, sockets)
    sim = candidate.sim_pessimistic or candidate.sim

    binding, prefill_share = UNDIAGNOSED, 0.0
    if diagnose:
        _, trace = simulate(
            workload,
            prefill_tps=candidate.throughput.prefill_tps,
            decode_by_active=decode_table(
                model, quant, cpu, memory, workload, sockets, eff),
        )
        ceilings = ceilings_for(
            model, quant, cpu, memory, eff, workload,
            candidate.throughput, sockets, candidate.memory_gb,
        )
        timeline = build_timeline(trace, ceilings)
        binding, prefill_share = timeline.binding_resource, timeline.prefill_share

    return LoadPoint(
        value=value,
        verdict=candidate.verdict,
        p95_steady_s=sim.p95_steady_s,
        storm_drain_s=sim.storm_drain_s,
        max_queue=sim.max_queue_depth,
        busy_fraction=sim.busy_fraction,
        binding_resource=binding,
        prefill_share=prefill_share,
        ram_needed_gb=candidate.memory_gb,
        reasons=list(candidate.reasons),
    )


def _diagnose(probe_args, seen, points):
    """Fill in the resource diagnosis for the points that get quoted.

    Only the knee and the first failure are read closely enough for the
    bottleneck to matter, so only those two pay for the extra simulated day.
    """
    model, quant, cpu, memory, eff, workload, sockets, axis = probe_args
    for point in points:
        if point is None or point.binding_resource != UNDIAGNOSED:
            continue
        full = _evaluate_point(
            model, quant, cpu, memory, eff,
            apply_axis(workload, axis, point.value), sockets, point.value,
            diagnose=True,
        )
        seen[point.value] = full


def _curve(axis, seen, good, bad, baseline, hit_ceiling, notes) -> CapacityCurve:
    points = [seen[v] for v in sorted(seen)]
    return CapacityCurve(
        axis=axis,
        points=points,
        knee=good,
        breaks_at=bad,
        limiter=_limiter(bad),
        baseline=baseline,
        hit_ceiling=hit_ceiling,
        notes=notes,
    )


def _limiter(bad: LoadPoint | None) -> str:
    """Classify the clause that gave way, from the judge's own reasons."""
    if bad is None:
        return "none"
    for reason in bad.reasons:
        if "RAM" in reason:
            return "ram"
    for reason in bad.reasons:
        if "스톰 소진" in reason:
            return "storm-drain"
    for reason in bad.reasons:
        if "p95" in reason:
            return "latency"
    return "headroom"


LIMITER_LABEL = {
    "latency": "평상시 p95 지연이 SLA를 넘음",
    "storm-drain": "스톰 소진 시간이 목표를 넘음",
    "ram": "장착 가능한 RAM을 초과",
    "headroom": "SLA 여유가 기준 미만으로 떨어짐",
    "none": "한계를 찾지 못함",
}

AXIS_LABEL = {
    "alarms": ("일 알람 건수", "건/일"),
    "storm": ("스톰 크기(30초 창)", "건"),
    "prompt": ("알람당 프롬프트 토큰", "tok"),
    "output": ("알람당 생성 토큰", "tok"),
}


def sweep_axes(
    catalog: Catalog,
    model: ModelSpec,
    quant: QuantSpec,
    cpu: CpuSpec,
    memory: MemoryOption,
    workload: Workload,
    axes: tuple[str, ...] = AXES,
    sockets: int = 1,
) -> dict[str, CapacityCurve]:
    """Knee on every axis for one build.

    Run together they rank the fixes: whichever axis has the least headroom is
    the one that will break first in production, and therefore the one worth
    spending on.
    """
    eff = Efficiency.from_catalog(catalog.coefficients)
    return {
        axis: find_knee(model, quant, cpu, memory, eff, workload, axis, sockets)
        for axis in axes
    }


def weakest_axis(curves: dict[str, CapacityCurve]) -> str | None:
    """The axis with the least room before it breaks."""
    scored = [(c.headroom, a) for a, c in curves.items() if c.knee is not None and not c.hit_ceiling]
    if not scored:
        return None
    return min(scored)[1]
