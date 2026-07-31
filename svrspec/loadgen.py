"""Load shapes to run a build against, beyond "replay one measured day".

Why more than one shape
-----------------------
`mockdata.generate_day` answers exactly one question: "does this box survive the
day the customer actually had". That is the question the sizing report has to
answer, and it is the reason that module reproduces the measured counts rather
than inventing a load. But it is not the only question an operator has before
signing a purchase order:

    ramp    at what offered rate does this build stop meeting the SLA? A day
            that fits tells you nothing about the margin above it.
    spike   the day fits on average; does the box survive the ten minutes it
            does not? Averages hide exactly the interval that hurts.
    soak    does anything drift over three days that a 24-hour run cannot show
            -- backlog that never fully drains, KV that never comes back down?

So this module keeps `replay` as the measured baseline and adds three synthetic
shapes next to it. All four produce the same thing: a list of
`mockdata.Alarm` and a `LoadProfile` describing what was asked for.

What is reused and what is rewritten
------------------------------------
Only the *arrival times* are synthetic. Alarm content -- device, code, severity,
message, token count -- and the correlation structure come from
`mockdata.generate_day` unchanged, because that content is what makes the
pipeline's prompt sizes and the correlation stage realistic, and re-deriving it
here would be a second generator to keep in agreement with the first.

Times are rewritten *per correlated group*, not per alarm. A storm is one
physical failure whose derived alarms follow within thirty seconds; scattering
its members independently across the day would leave the records pointing at
each other through `parent_id` while describing something that never happened,
and `RunStats.storm_drain_s` would become noise.

Why ramp and soak carry no storms
---------------------------------
`ramp` and `soak` exist to move exactly one variable -- the offered rate -- and
read off what breaks. A storm is a second, much larger variable riding on top:
forty alarms inside thirty seconds is a rate spike of its own, and when the SLA
finally breaks there is no way to tell whether the ramp reached the build's
limit or a storm happened to land at that moment. Two runs of the same profile
would break at different rates for reasons that have nothing to do with the
rate. So the rate axis is kept clean and storms stay in `replay` (where they are
part of the measured day being reproduced) and in `spike` (where the burst *is*
the thing being tested). This decision is repeated in Korean in
`LoadProfile.notes`, because the operator reading the chart is the one who has
to know why the storm markers vanished.

Arrival placement
-----------------
Each shape is a piecewise-linear rate curve in alarms-per-day. Conditional on
the number of arrivals in a window, a non-homogeneous Poisson process places
them independently with density proportional to the rate -- so the placement is
an inverse-CDF draw per unit, sorted. That is the correct process, not merely a
plausible one, and it is the same argument `mockdata._background_times` makes
for its uniform draws.

Nothing here runs a model, starts a program or opens a connection. The whole
point is to size a machine nobody here can log into, from a laptop that must
stay usable while it happens.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field, replace
from datetime import date as _Date
from datetime import timedelta

from .mockdata import DAY_SECONDS, Alarm, generate_day

#: The four shapes. `replay` is the measured day; the rest are synthetic.
KINDS = ("replay", "ramp", "spike", "soak")

#: Where alarm *content* is drawn from. Walking forward through the measured
#: window means a long profile gets real per-day counts and, more importantly,
#: distinct dates -- ids are stamped with the date, so days never collide.
CONTENT_START_DATE = "2025-02-01"
CONTENT_LAST_DATE = "2026-06-30"

#: Wall-clock origin stamped into the rewritten `raw["event_time"]`. Deliberately
#: outside the measured window: these arrival times are synthetic, and dating
#: them inside the export would invite someone to read them back as measured.
SYNTHETIC_START_DATE = "2026-07-01"

#: Defaults per shape. Also the allow-list: anything else passed to `build_load`
#: is rejected rather than silently ignored, because a typo in a GUI field that
#: quietly does nothing is worse than an error message.
DEFAULTS: dict[str, dict] = {
    "replay": {
        "date": "2026-06-01",
        "count": None,
        "storm_size": 40,
        "storms_per_day": 2,
        "storm_window_s": 30.0,
    },
    "ramp": {
        "start_rate": 100.0,
        "end_rate": 2000.0,
        "hours": 24.0,
    },
    "spike": {
        "base_rate": 165.0,
        "peak_rate": 800.0,
        "spike_at_h": 12.0,
        "spike_minutes": 30.0,
        "storm_size": 40,
        "storms_per_day": 2,
        "storm_window_s": 30.0,
    },
    "soak": {
        "rate": 300.0,
        "hours": 72.0,
    },
}

#: Notes from `mockdata` that stop being true once the arrival times are
#: rewritten. Matched as substrings: if the generator rephrases one of them the
#: worst case is a redundant line in the report, not a crash.
_STALE_NOTE_MARKERS = ("알람 수", "시간대 분포")
_STORM_NOTE_MARKER = "스톰"


@dataclass(frozen=True)
class LoadProfile:
    """What was asked for, in a form a chart and a report can both read."""

    kind: str
    #: One Korean line for the operator. This is what labels the run.
    label: str
    #: Seconds this load covers. The run itself may overrun it -- that is the
    #: interesting case, and `bench` reports it.
    span_s: float
    total_alarms: int
    #: Reproduction and charting. Always JSON-serialisable. Carries
    #: `rate_segments` for every shape that has a declared offered rate, which
    #: is what lets a frame say "this broke at 1,340 alarms/day".
    params: dict
    #: Assumptions, in Korean, carried through from `mockdata` plus the ones
    #: this module adds. Named in the output rather than left to be discovered.
    notes: list[str] = field(default_factory=list)


def build_load(
    kind: str, *, seed: int = 20260730, **params
) -> tuple[list[Alarm], LoadProfile]:
    """Alarms and the profile that describes them. Deterministic in `seed`.

    `replay` returns `mockdata.generate_day` untouched, so the measured day
    stays byte-identical to what the rest of the tool already produces. The
    other three reuse that generator's alarm content and rewrite only `at_s`.
    """
    if kind not in DEFAULTS:
        raise ValueError(f"알 수 없는 부하 종류: {kind!r} — {', '.join(KINDS)} 중 하나여야 한다")

    settings = dict(DEFAULTS[kind])
    unknown = sorted(set(params) - set(settings))
    if unknown:
        raise ValueError(
            f"{kind} 프로파일이 모르는 파라미터다: {', '.join(unknown)} "
            f"— 가능한 것: {', '.join(sorted(settings))}"
        )
    settings.update(params)

    if kind == "replay":
        return _replay(seed, settings)
    return _synthetic(kind, seed, settings)


def rate_at(profile: LoadProfile, t_s: float) -> float | None:
    """Offered rate in alarms/day at `t_s`, or None if the shape declares none.

    `replay` has no declared rate -- it is a measured day, and the honest answer
    to "what rate was offered at 14:03" is whatever actually arrived, which the
    caller can count. Returning None rather than a fabricated curve is what
    stops `bench` drawing a smooth line over measured data.
    """
    segments = profile.params.get("rate_segments")
    if not segments:
        return None
    for t0, t1, r0, r1 in segments:
        if t0 <= t_s < t1:
            width = t1 - t0
            fraction = (t_s - t0) / width if width > 0 else 0.0
            return r0 + (r1 - r0) * fraction
    return 0.0


# --------------------------------------------------------------------------
# replay: the measured day, untouched
# --------------------------------------------------------------------------


def _replay(seed: int, settings: dict) -> tuple[list[Alarm], LoadProfile]:
    day = generate_day(
        settings["date"],
        settings["count"],
        seed=seed,
        storm_size=settings["storm_size"],
        storms_per_day=settings["storms_per_day"],
        storm_window_s=settings["storm_window_s"],
    )
    params = {k: v for k, v in settings.items()}
    params["source_date"] = day.date
    params["storms"] = day.storms

    notes = list(day.notes)
    notes.append("실측 하루를 그대로 재생한다 — 도착 시각도 알람 내용도 손대지 않았다")
    profile = LoadProfile(
        kind="replay",
        label=f"실측 재생 {day.date} · {len(day.alarms):,}건 · 스톰 {day.storms}회",
        span_s=float(DAY_SECONDS),
        total_alarms=len(day.alarms),
        params=params,
        notes=notes,
    )
    return list(day.alarms), profile


# --------------------------------------------------------------------------
# ramp / spike / soak: measured content, synthetic arrival curve
# --------------------------------------------------------------------------


def _synthetic(kind: str, seed: int, settings: dict) -> tuple[list[Alarm], LoadProfile]:
    span_s, segments, label = _shape(kind, settings)
    target = _expected_alarms(segments)
    with_storms = kind == "spike"

    days = _content_days(
        target,
        seed=seed,
        storm_size=settings["storm_size"] if with_storms else 0,
        storms_per_day=settings["storms_per_day"] if with_storms else 0,
        storm_window_s=settings.get("storm_window_s", 30.0),
    )
    units = _take_units(_to_units([d.alarms for d in days]), target)

    rng = random.Random(_derive_seed(seed, kind))
    placements = sorted(_place(rng.random(), segments) for _ in units)

    base_date = _parse(SYNTHETIC_START_DATE)
    alarms: list[Alarm] = []
    storm_number = 0
    for unit, at in zip(units, placements):
        storm_id = None
        if len(unit) > 1 or unit[0].storm_id is not None:
            storm_id = storm_number
            storm_number += 1
        alarms.extend(_retime(unit, at, span_s, storm_id, base_date))
    alarms.sort(key=lambda a: (a.at_s, 0 if a.is_representative else 1, a.id))

    params = {k: v for k, v in settings.items()}
    params["span_s"] = span_s
    params["rate_segments"] = [list(s) for s in segments]
    params["storms"] = storm_number
    params["content_dates"] = [d.date for d in days]

    profile = LoadProfile(
        kind=kind,
        label=label,
        span_s=span_s,
        total_alarms=len(alarms),
        params=params,
        notes=_notes(kind, days[0].notes, with_storms, span_s, len(alarms)),
    )
    return alarms, profile


def _shape(kind: str, s: dict) -> tuple[float, list[tuple[float, float, float, float]], str]:
    """Span, the piecewise-linear rate curve, and the Korean label.

    A segment is `(t0, t1, r0, r1)` with the rate in alarms *per day*, linearly
    interpolated. Every shape reduces to this one form so there is a single
    integration and a single inverse, rather than three of each.
    """
    if kind == "ramp":
        start, end, hours = float(s["start_rate"]), float(s["end_rate"]), float(s["hours"])
        _positive_rate(start, "start_rate")
        _positive_rate(end, "end_rate")
        span = _positive_hours(hours)
        return (
            span,
            [(0.0, span, start, end)],
            f"램프 {start:,.0f} → {end:,.0f}건/일 · {hours:g}시간",
        )

    if kind == "soak":
        rate, hours = float(s["rate"]), float(s["hours"])
        _positive_rate(rate, "rate")
        span = _positive_hours(hours)
        return (
            span,
            [(0.0, span, rate, rate)],
            f"장시간 균일 {rate:,.0f}건/일 · {hours:g}시간",
        )

    base, peak = float(s["base_rate"]), float(s["peak_rate"])
    _positive_rate(base, "base_rate")
    _positive_rate(peak, "peak_rate")
    at_h, minutes = float(s["spike_at_h"]), float(s["spike_minutes"])
    if minutes <= 0:
        raise ValueError("spike_minutes는 0보다 커야 한다")
    span = float(DAY_SECONDS)
    begin = at_h * 3600.0
    end = begin + minutes * 60.0
    if not (0.0 <= begin < span) or end > span:
        raise ValueError(
            f"급증 구간이 하루 밖으로 나간다: {at_h:g}시부터 {minutes:g}분"
        )
    segments = [
        (0.0, begin, base, base),
        (begin, end, peak, peak),
        (end, span, base, base),
    ]
    return (
        span,
        [seg for seg in segments if seg[1] > seg[0]],
        f"급증 {base:,.0f} → {peak:,.0f}건/일 · {at_h:g}시부터 {minutes:g}분",
    )


def _positive_rate(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name}는 0보다 커야 한다 — 도착이 없는 부하는 부하가 아니다")


def _positive_hours(hours: float) -> float:
    if hours <= 0:
        raise ValueError("hours는 0보다 커야 한다")
    return hours * 3600.0


def _expected_alarms(segments) -> int:
    """Integral of the rate curve, in alarms.

    Rates are quoted per day because that is the unit the customer's export used
    and the unit the report quotes, so every segment is divided back down to
    per-second here rather than at four call sites.
    """
    total = 0.0
    for t0, t1, r0, r1 in segments:
        total += (r0 + r1) / 2.0 * (t1 - t0) / DAY_SECONDS
    return max(int(round(total)), 0)


def _place(u: float, segments) -> float:
    """Inverse CDF of the rate curve at quantile `u`.

    Within a segment the rate is linear, so the expected count is quadratic in
    the offset and the inverse is the positive root. The degenerate case (a flat
    segment) is linear and is handled separately rather than by letting the
    quadratic formula divide by zero.
    """
    weights = [
        (r0 + r1) / 2.0 * (t1 - t0) / DAY_SECONDS for t0, t1, r0, r1 in segments
    ]
    total = sum(weights)
    if total <= 0:
        return 0.0

    goal = u * total
    for (t0, t1, r0, r1), weight in zip(segments, weights):
        if weight <= 0:
            continue
        if goal > weight:
            goal -= weight
            continue
        length = t1 - t0
        if length <= 0:
            return t0
        target = goal * DAY_SECONDS
        slope = (r1 - r0) / length
        if abs(slope) < 1e-12:
            offset = target / r0 if r0 > 0 else 0.0
        else:
            # a*x^2 + b*x - target = 0 with a = slope/2, b = r0.
            a = slope / 2.0
            disc = r0 * r0 + 4.0 * a * target
            offset = (-r0 + math.sqrt(max(disc, 0.0))) / (2.0 * a)
        return t0 + min(max(offset, 0.0), length)
    return segments[-1][1]


# --------------------------------------------------------------------------
# Alarm content: borrowed whole from the measured-day generator
# --------------------------------------------------------------------------


def _content_days(
    need: int,
    *,
    seed: int,
    storm_size: int,
    storms_per_day: int,
    storm_window_s: float,
):
    """Consecutive measured days until they hold at least `need` alarms.

    Days rather than one oversized `generate_day(count=N)` call because the
    per-day count is the one thing in that generator that *is* measured, and a
    single 900-alarm "day" would silently rewrite it. Distinct dates also keep
    ids unique, since an id carries its date.
    """
    if need <= 0:
        return [
            generate_day(
                CONTENT_START_DATE, 1, seed=seed, storm_size=0, storms_per_day=0,
                storm_window_s=storm_window_s,
            )
        ]

    last = _parse(CONTENT_LAST_DATE)
    cursor = _parse(CONTENT_START_DATE)
    days = []
    have = 0
    while have < need:
        if cursor > last:
            raise ValueError(
                f"요청한 {need:,}건을 만들 실측 날짜가 부족하다 "
                f"({CONTENT_START_DATE}~{CONTENT_LAST_DATE}) — 부하율이나 기간을 줄여라"
            )
        day = generate_day(
            cursor.isoformat(),
            seed=seed,
            storm_size=storm_size,
            storms_per_day=storms_per_day,
            storm_window_s=storm_window_s,
        )
        days.append(day)
        have += len(day.alarms)
        cursor += timedelta(days=1)
    return days


def _to_units(days: list[list[Alarm]]) -> list[list[Alarm]]:
    """Group each day's alarms into placeable units.

    A correlated group is one unit, because its members describe one failure and
    have to stay together. Everything else is a unit of one. Units come out in
    arrival order within a day so truncating the list to a target count keeps a
    representative sample of both kinds rather than all the storms or none.
    """
    units: list[list[Alarm]] = []
    for alarms in days:
        groups: dict[int, list[Alarm]] = {}
        singles: list[tuple[float, str, list[Alarm]]] = []
        for alarm in alarms:
            if alarm.storm_id is None:
                singles.append((alarm.at_s, alarm.id, [alarm]))
            else:
                groups.setdefault(alarm.storm_id, []).append(alarm)
        ordered = list(singles)
        for members in groups.values():
            members = sorted(members, key=lambda a: (a.at_s, 0 if a.parent_id is None else 1, a.id))
            ordered.append((members[0].at_s, members[0].id, members))
        ordered.sort(key=lambda item: (item[0], item[1]))
        units.extend(unit for _, _, unit in ordered)
    return units


def _take_units(units: list[list[Alarm]], target: int) -> list[list[Alarm]]:
    """Units in order until they hold `target` alarms.

    A storm can overshoot the target by up to its own size. Accepted rather than
    split: half a storm is not a storm, and the overshoot is at most a few dozen
    alarms against a load measured in hundreds.
    """
    if target <= 0:
        return []
    taken: list[list[Alarm]] = []
    count = 0
    for unit in units:
        taken.append(unit)
        count += len(unit)
        if count >= target:
            break
    return taken


def _retime(
    unit: list[Alarm], at: float, span_s: float, storm_id: int | None, base_date: _Date
) -> list[Alarm]:
    """Move a unit to `at`, keeping the offsets between its members.

    `raw` is rewritten in step with `at_s`. A record whose audit copy contradicts
    its typed field is a trap for whoever reads the export next, and `raw` is
    the field this project keeps precisely so a reader can check the typed ones.
    """
    origin = min(alarm.at_s for alarm in unit)
    out = []
    for alarm in unit:
        t = _inside(at + (alarm.at_s - origin), span_s)
        raw = dict(alarm.raw)
        raw["event_offset_s"] = t
        raw["event_time"] = _stamp(base_date, t)
        out.append(replace(alarm, at_s=t, storm_id=storm_id, raw=raw))
    return out


def _notes(
    kind: str, source_notes: list[str], with_storms: bool, span_s: float, total: int
) -> list[str]:
    """Carry the generator's assumptions forward, minus the ones now untrue."""
    notes = [
        f"도착 시각은 {kind} 곡선에서 뽑았다 — 실측 시간대 분포가 아니다",
        f"{span_s / 3600:.0f}시간 {total:,}건 — 부하율 곡선의 적분값이다",
    ]
    for note in source_notes:
        if any(marker in note for marker in _STALE_NOTE_MARKERS):
            continue
        if not with_storms and _STORM_NOTE_MARKER in note:
            continue
        notes.append(note)

    # Why the storm decision is stated at all: it changes what the result means.
    # Kept to one line each -- the reasoning belongs in the module docstring,
    # not in front of every reader of every run.
    if with_storms:
        notes.append("스톰을 넣는다 — 급증에 상관 알람이 겹칠 때 버티는지를 본다")
    else:
        notes.append(f"{kind}에는 스톰을 넣지 않는다 — 부하율만 움직여야 원인이 분명해진다")
    return notes


# --------------------------------------------------------------------------
# Plumbing
# --------------------------------------------------------------------------


def _inside(t: float, span_s: float) -> float:
    """Clamp into `[0, span_s)`. A storm placed at the end must not spill out."""
    return min(max(t, 0.0), span_s - 1e-3)


def _stamp(base_date: _Date, at_s: float) -> str:
    seconds = int(at_s)
    day = base_date + timedelta(days=seconds // int(DAY_SECONDS))
    rest = seconds % int(DAY_SECONDS)
    return (
        f"{day.isoformat()}T"
        f"{rest // 3600:02d}:{rest % 3600 // 60:02d}:{rest % 60:02d}"
    )


def _parse(date: str) -> _Date:
    return _Date.fromisoformat(date)


def _derive_seed(seed: int, kind: str) -> int:
    """A per-shape seed, so two shapes at one seed do not share a draw sequence.

    SHA-256 rather than `hash()`, for the same reason `mockdata` does it: string
    hashing is salted per process, and a generator that changes between runs is
    the one property this module promises not to have.
    """
    digest = hashlib.sha256(f"{seed}:{kind}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")
