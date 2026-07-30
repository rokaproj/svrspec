"""Alarm arrival generation.

The headline number -- 150 alarms per day -- is not what sizes the server.
Spread evenly that is one alarm every ten minutes, which any CPU handles. What
actually sizes it is the storm: a switch goes down and forty alarms land inside
half a minute. So the generator produces a realistic day (business-hours
weighted) with storms embedded, and the simulator is judged on the storm.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .types import Workload

DAY_SECONDS = 24 * 3600


@dataclass(frozen=True)
class Arrival:
    at_s: float
    #: Storm alarms are tagged so the simulator can report drain time for the
    #: burst separately from the quiet-hours average.
    storm_id: int | None = None

    @property
    def in_storm(self) -> bool:
        return self.storm_id is not None


def generate_arrivals(workload: Workload) -> list[Arrival]:
    """One day of alarms. Deterministic for a given `workload.seed`.

    Storm alarms are drawn from the daily budget rather than added on top, so
    the total stays at `alarms_per_day` -- the customer said 100-150 per day
    including the bad days, not plus.
    """
    rng = random.Random(workload.seed)

    storms = _storm_plan(workload, rng)
    storm_alarms = sum(len(s) for s in storms)
    background = max(0, workload.alarms_per_day - storm_alarms)

    arrivals: list[Arrival] = []
    for storm_id, times in enumerate(storms):
        arrivals.extend(Arrival(at_s=t, storm_id=storm_id) for t in times)
    arrivals.extend(Arrival(at_s=t) for t in _background_times(workload, background, rng))

    arrivals.sort(key=lambda a: a.at_s)
    return arrivals


def _storm_plan(workload: Workload, rng: random.Random) -> list[list[float]]:
    """Storm start times inside business hours, each a burst of arrivals."""
    if workload.storms_per_day <= 0 or workload.storm_size <= 0:
        return []

    start_h, end_h = workload.business_hours
    # Keep the whole burst inside business hours.
    latest = end_h * 3600 - workload.storm_window_s
    earliest = start_h * 3600
    if latest <= earliest:
        earliest, latest = 0.0, DAY_SECONDS - workload.storm_window_s

    budget = workload.alarms_per_day
    storms: list[list[float]] = []
    for _ in range(workload.storms_per_day):
        size = min(workload.storm_size, budget - sum(len(s) for s in storms))
        if size <= 0:
            break
        begin = rng.uniform(earliest, latest)
        # Alarms in a storm are not evenly spaced; a cascading failure fires
        # thickest at the start.
        times = sorted(begin + workload.storm_window_s * rng.random() ** 1.6 for _ in range(size))
        storms.append(times)
    return storms


def _background_times(workload: Workload, count: int, rng: random.Random) -> list[float]:
    """Non-storm alarms, concentrated in business hours.

    Conditional on the count in a window, a homogeneous Poisson process places
    events uniformly at random inside it -- so uniform draws per window give a
    correct piecewise-constant-rate arrival process.
    """
    if count <= 0:
        return []

    start_h, end_h = workload.business_hours
    day_start, day_end = start_h * 3600.0, end_h * 3600.0

    in_hours = int(round(count * workload.business_share))
    out_hours = count - in_hours

    times = [rng.uniform(day_start, day_end) for _ in range(in_hours)]
    for _ in range(out_hours):
        # Off-hours is the day minus the business window; pick either side.
        night_len = DAY_SECONDS - (day_end - day_start)
        offset = rng.uniform(0.0, night_len)
        times.append(offset if offset < day_start else offset + (day_end - day_start))
    return times


def teams_rtt(workload: Workload, rng: random.Random) -> float:
    """A Teams webhook round trip, in seconds."""
    low, high = workload.teams_rtt_ms
    return rng.uniform(low, high) / 1000.0


def describe(workload: Workload) -> str:
    t = workload.tokens
    cache = "on" if t.prompt_cache else "off"
    return (
        f"{workload.alarms_per_day} alarms/day, "
        f"storm {workload.storm_size} in {workload.storm_window_s:.0f}s x{workload.storms_per_day}, "
        f"{workload.slots} slot(s), "
        f"prefill {t.prefill_tokens} tok (billed {t.billed_prefill_tokens}, cache {cache}), "
        f"output {t.output_tokens} tok, SLA {workload.sla_seconds:.0f}s"
    )
