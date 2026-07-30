"""Discrete-event simulation of the alarm -> LLM -> Teams pipeline.

Why simulate at all instead of dividing throughput by load: because the answer
the delivery document needs is a tail latency during a burst, and that is a
queueing property, not a throughput property. A server with ample average
capacity can still blow a 30-second SLA if forty alarms arrive at once and it
serves two at a time.

The server is modelled as a station with `slots` concurrent places, fed FIFO.
Rates are piecewise constant between events, so the loop advances to the next
arrival or the next phase completion, whichever comes first -- no time
stepping, no accumulated drift.

Resource sharing follows llama.cpp's continuous batching: prefill is compute
bound so the aggregate prefill rate is fixed and split across whoever is
prefilling; decode is bandwidth bound and its aggregate rate *rises* with the
number of active slots, because one sweep over the weights emits a token for
every slot. Both aggregates are split evenly among active requests.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .types import SimResult, TokenProfile, Workload
from .workload import Arrival, generate_arrivals, teams_rtt

PREFILL = "prefill"
DECODE = "decode"
DONE = "done"


@dataclass
class _Request:
    index: int
    arrived_s: float
    storm_id: int | None
    remaining_prefill: float
    remaining_decode: float
    rtt_s: float
    phase: str = PREFILL
    started_s: float | None = None
    finished_s: float | None = None

    @property
    def latency_s(self) -> float:
        assert self.finished_s is not None
        return self.finished_s - self.arrived_s


@dataclass
class SimTrace:
    """Per-request detail, kept so reports can show the worst offenders."""

    latencies: list[float] = field(default_factory=list)
    storm_latencies: list[float] = field(default_factory=list)
    steady_latencies: list[float] = field(default_factory=list)
    queue_samples: list[int] = field(default_factory=list)
    #: (time, active requests, queued requests) at every event boundary, plus
    #: the span each state held for. A task-manager graph needs the shape of the
    #: day, not just its summary statistics -- and because rates are piecewise
    #: constant between events, these samples describe it exactly.
    events: list[tuple[float, int, int, float]] = field(default_factory=list)


def simulate(
    workload: Workload,
    prefill_tps: float,
    decode_by_active: dict[int, float],
    arrivals: list[Arrival] | None = None,
) -> tuple[SimResult, SimTrace]:
    """Run one day.

    `decode_by_active` maps concurrent-slot count -> aggregate decode tok/s.
    `prefill_tps` is the aggregate prompt-processing rate, shared when several
    requests are prefilling at once.
    """
    if prefill_tps <= 0 or not decode_by_active:
        raise ValueError("prefill_tps must be positive and decode_by_active non-empty")

    rng = random.Random(workload.seed + 1)
    tokens: TokenProfile = workload.tokens
    incoming = list(arrivals if arrivals is not None else generate_arrivals(workload))

    pending: list[_Request] = [
        _Request(
            index=i,
            arrived_s=a.at_s,
            storm_id=a.storm_id,
            remaining_prefill=float(tokens.billed_prefill_tokens),
            remaining_decode=float(tokens.output_tokens),
            rtt_s=teams_rtt(workload, rng),
        )
        for i, a in enumerate(incoming)
    ]

    queue: list[_Request] = []
    active: list[_Request] = []
    finished: list[_Request] = []
    trace = SimTrace()

    now = pending[0].arrived_s if pending else 0.0
    next_arrival = 0
    busy_slot_seconds = 0.0
    any_busy_seconds = 0.0
    max_queue_depth = 0
    first_event_s = now

    while next_arrival < len(pending) or queue or active:
        # Admit everything that has arrived by now.
        while next_arrival < len(pending) and pending[next_arrival].arrived_s <= now + 1e-12:
            queue.append(pending[next_arrival])
            next_arrival += 1

        while len(active) < workload.slots and queue:
            req = queue.pop(0)
            req.started_s = now
            active.append(req)

        max_queue_depth = max(max_queue_depth, len(queue))
        trace.queue_samples.append(len(queue))

        if not active:
            # Idle: jump to the next arrival rather than stepping through time.
            if next_arrival < len(pending):
                now = pending[next_arrival].arrived_s
                continue
            break

        share = len(active)
        decode_aggregate = _decode_rate(decode_by_active, share)
        prefill_rate = prefill_tps / share
        decode_rate = decode_aggregate / share

        # Time for each active request to finish its current phase.
        horizons: list[float] = []
        for req in active:
            if req.phase == PREFILL:
                rate = prefill_rate
                remaining = req.remaining_prefill
            else:
                rate = decode_rate
                remaining = req.remaining_decode
            horizons.append(remaining / rate if rate > 0 else float("inf"))

        step = min(horizons)
        if next_arrival < len(pending):
            step = min(step, max(pending[next_arrival].arrived_s - now, 0.0))
        if step <= 0 or step == float("inf"):
            # Nothing can progress and nothing can arrive: guard against a
            # zero-rate configuration rather than spinning.
            break

        for req in active:
            if req.phase == PREFILL:
                req.remaining_prefill -= prefill_rate * step
                if req.remaining_prefill <= 1e-9:
                    req.phase = DECODE
                    req.remaining_prefill = 0.0
            else:
                req.remaining_decode -= decode_rate * step

        busy_slot_seconds += share * step
        any_busy_seconds += step
        trace.events.append((now, share, len(queue), step))
        now += step

        still_active: list[_Request] = []
        for req in active:
            if req.phase == DECODE and req.remaining_decode <= 1e-9:
                # The slot frees as soon as generation ends; the Teams POST is
                # network wait, not server work.
                req.finished_s = now + req.rtt_s
                finished.append(req)
            else:
                still_active.append(req)
        active = still_active

    for req in finished:
        trace.latencies.append(req.latency_s)
        if req.storm_id is not None:
            trace.storm_latencies.append(req.latency_s)
        else:
            trace.steady_latencies.append(req.latency_s)

    span = max(now - first_event_s, 1e-9)
    result = _summarise(
        workload=workload,
        finished=finished,
        trace=trace,
        max_queue_depth=max_queue_depth,
        slot_utilisation=busy_slot_seconds / (span * workload.slots),
        busy_fraction=any_busy_seconds / span,
    )
    return result, trace


def _decode_rate(table: dict[int, float], active: int) -> float:
    """Aggregate decode rate at `active` concurrent requests.

    Clamps to the measured/predicted range rather than extrapolating past it.
    """
    if active in table:
        return table[active]
    keys = sorted(table)
    if active < keys[0]:
        return table[keys[0]]
    return table[keys[-1]]


def _summarise(
    workload: Workload,
    finished: list[_Request],
    trace: SimTrace,
    max_queue_depth: int,
    slot_utilisation: float,
    busy_fraction: float = 0.0,
) -> SimResult:
    latencies = sorted(trace.latencies)
    if not latencies:
        return SimResult(
            p50_s=0.0,
            p95_s=0.0,
            p99_s=0.0,
            max_s=0.0,
            max_queue_depth=0,
            storm_drain_s=0.0,
            slot_utilisation=0.0,
            sla_met=True,
            storm_sla_met=True,
            completed=0,
        )

    drain = _storm_drain(finished)
    steady = sorted(trace.steady_latencies)
    # With no quiet-hours traffic at all, fall back to the whole population so
    # the per-alarm SLA is still judged on something.
    p95_steady = _percentile(steady or latencies, 0.95)

    return SimResult(
        p50_s=_percentile(latencies, 0.50),
        p95_s=_percentile(latencies, 0.95),
        p99_s=_percentile(latencies, 0.99),
        max_s=latencies[-1],
        max_queue_depth=max_queue_depth,
        storm_drain_s=drain,
        slot_utilisation=min(slot_utilisation, 1.0),
        busy_fraction=min(busy_fraction, 1.0),
        sla_met=p95_steady <= workload.sla_seconds,
        storm_sla_met=drain <= workload.storm_drain_sla_s,
        completed=len(latencies),
        p95_steady_s=p95_steady,
        steady_completed=len(steady),
    )


def _storm_drain(finished: list[_Request]) -> float:
    """Longest time from a storm's first alarm arriving to its last one sent."""
    by_storm: dict[int, list[_Request]] = {}
    for req in finished:
        if req.storm_id is not None:
            by_storm.setdefault(req.storm_id, []).append(req)

    worst = 0.0
    for reqs in by_storm.values():
        begin = min(r.arrived_s for r in reqs)
        end = max(r.finished_s or 0.0 for r in reqs)
        worst = max(worst, end - begin)
    return worst


def _percentile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolated percentile of an already-sorted list."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    low = int(pos)
    high = min(low + 1, len(sorted_values) - 1)
    frac = pos - low
    return sorted_values[low] * (1 - frac) + sorted_values[high] * frac
