"""Synthetic NOC alarm records that reproduce the customer's measured volume.

Why this module exists
----------------------
`workload.py` generates arrival *times* -- `Arrival(at_s, storm_id)` -- because
that is all a queueing simulation needs. But the request the tool is now being
asked to answer is different: "can this server actually serve the alarms we
get?" Serving means a prompt, and a prompt needs an alarm *record*: which box,
which fault code, how severe, what text. There was no such record anywhere in
the codebase, so the pipeline had nothing to process and no way to be exercised
end to end. This module is that missing record.

It is also the test data for the correlation stage. A feed of independent,
unrelated alarms can never exercise deduplication or root-cause grouping -- the
three-stage pipeline would look correct on data that has no correlation in it to
find. So a storm here is not a burst of arrivals, it is *one physical failure*:
a site loses power, the representative alarm fires, and the derived alarms from
the same site follow inside thirty seconds, each pointing at the representative
through `parent_id`.

What is measured and what is assumed
------------------------------------
Measured -- 17 months of the customer's own alarm counts, 81,002 records:

    2025 (Feb-Dec)   51,049   152.8/day
    2026 (Jan-Jun)   29,953   165.5/day
    peak  2026-04     5,600   186.7/day
    trough 2025-12     3,767   121.5/day
    2026-06 daily     mean 147.0, median 131.5, min 26, max 359

That is *all* that was measured. The export was a daily total per day; there was
no hourly breakdown, no severity histogram, no device inventory, and nothing at
all about correlation. Everything below the daily count is therefore an
assumption, and every assumption is named in `AlarmDay.notes` rather than left
for a reader to discover:

    hourly shape   `business_share` of the day inside `business_hours`, the same
                   weighting `workload.py` uses. Not measured.
    severity mix   SEVERITY_WEIGHTS. Not measured -- shaped only by the rule
                   that critical is rare and warnings are common.
    storms         size, count and correlation structure. Not measured.
    inventory      sites, device types and names. Not measured.

Sizing a server on the daily count is defensible; sizing it on the hourly peak
this module produces is defensible only as long as everyone reading the number
knows the peak came from an assumption. Hence the notes.

Nothing here executes or connects to anything -- the tool runs on air-gapped
machines, and the whole point of a generator is to stand in for a live feed
rather than go looking for one. A test asserts this file contains no way to
spawn a process and no way to reach the network.
"""

from __future__ import annotations

import calendar
import csv
import hashlib
import io
import json
import random
from dataclasses import dataclass, field
from datetime import date as _Date
from datetime import datetime

DAY_SECONDS = 24 * 3600

# --------------------------------------------------------------------------
# The measured data. These are the customer's numbers, not the generator's.
# --------------------------------------------------------------------------

#: Alarms per day for 2026-06, in date order from the 1st. The only month the
#: export broke down by day, which makes it the only month where the generator
#: can be checked against reality rather than against its own average.
DAILY_2026_06: tuple[int, ...] = (
    359, 164, 86, 129, 187, 149, 88, 212, 250, 134,
    298, 281, 102, 129, 108, 87, 95, 114, 90, 46,
    26, 216, 142, 138, 249, 193, 67, 40, 72, 160,
)

#: Monthly totals. 2025 starts in February -- the export begins there.
MONTHLY_2025: dict[int, int] = {
    2: 3564, 3: 3878, 4: 5252, 5: 4715, 6: 5569, 7: 4558,
    8: 4444, 9: 5340, 10: 5092, 11: 4870, 12: 3767,
}
MONTHLY_2026: dict[int, int] = {
    1: 4540, 2: 4519, 3: 5706, 4: 5600, 5: 5177, 6: 4411,
}

MONTHLY: dict[int, dict[int, int]] = {2025: MONTHLY_2025, 2026: MONTHLY_2026}

# --------------------------------------------------------------------------
# Assumptions, all in one place so a reviewer can argue with them
# --------------------------------------------------------------------------

SEVERITIES = ("critical", "major", "minor", "warning")

#: Assumed severity mix. Not measured -- the export had no severity column.
#: The shape comes from how an NOC queue actually looks: a genuine
#: service-affecting fault is rare and is what the operator is paid to notice,
#: while most of the volume is threshold crossings that resolve themselves.
#: Inverting that ratio would make the LLM's job look harder than it is, since
#: a critical alarm is the one that needs a real answer.
SEVERITY_WEIGHTS: dict[str, float] = {
    "critical": 0.04,
    "major": 0.16,
    "minor": 0.42,
    "warning": 0.38,
}

#: Severity mix *inside* a storm, excluding the representative. A cascading
#: failure's derived alarms skew more severe than background traffic -- when a
#: site loses power the transport links really are down, not merely degraded --
#: but they are still mostly consequences rather than new faults, so `major` is
#: doubled against the background mix rather than made dominant. Overdoing the
#: skew drags the whole-day distribution away from SEVERITY_WEIGHTS, which is
#: the ratio the report quotes; at 0.32 the pooled month stays within ~0.05 of
#: the declared mix even when storms are a third of the volume.
STORM_DERIVED_WEIGHTS: dict[str, float] = {
    "critical": 0.02,
    "major": 0.32,
    "minor": 0.44,
    "warning": 0.22,
}

DEVICE_TYPES = ("기지국", "전송", "코어", "전원", "회선")

#: Assumed share of alarms by device type. Radio access dominates an operator's
#: alarm volume because there are simply far more base stations than core nodes.
TYPE_WEIGHTS: tuple[float, ...] = (0.46, 0.20, 0.10, 0.12, 0.12)

#: Device types a *derived* storm alarm lands on. A dead site takes its
#: transport and its circuits with it; the core keeps running.
DERIVED_TYPE_WEIGHTS: tuple[float, ...] = (0.40, 0.28, 0.02, 0.10, 0.20)

#: Codes and one-line Korean texts by device type. The third element is the
#: fault tier, and it is the reason this is a table rather than two independent
#: draws: severity and content have to agree. `CELL-DOWN` cannot be a warning
#: and `CPU-HIGH` cannot be critical. So a severity is drawn first and the code
#: is then chosen from the entries that tier matches -- which keeps the severity
#: distribution exactly SEVERITY_WEIGHTS while the text stays plausible.
#:
#:     hard  a fault. Service is affected now. -> critical / major
#:     soft  a threshold crossing or a warning. -> minor / warning
ALARM_CATALOG: dict[str, tuple[tuple[str, str, str], ...]] = {
    "기지국": (
        ("CELL-DOWN", "기지국 셀 다운 — 서비스 중단", "hard"),
        ("PWR-LOSS", "기지국 전원 상실 — 셀 전체 정지", "hard"),
        ("RRU-FAIL", "RRU 모듈 응답 없음 — 원격 재기동 실패", "hard"),
        ("VSWR-HIGH", "안테나 VSWR 임계 초과 — 급전선 점검 필요", "soft"),
        ("PRACH-FAIL-HIGH", "랜덤액세스 실패율 상승 — 접속 품질 저하", "soft"),
        ("CELL-TEMP-WARN", "함체 내부 온도 경고 — 냉각 상태 확인", "soft"),
    ),
    "전송": (
        ("LNK-DOWN", "광 링크 단절 감지", "hard"),
        ("LOS", "광 신호 손실 — 상위 구간 확인 필요", "hard"),
        ("CARD-FAIL", "전송 카드 장애 — 보호 절체 발생", "hard"),
        ("BER-HIGH", "비트오류율 임계 초과", "soft"),
        ("OPT-PWR-LOW", "수신 광파워 저하 — 커넥터 오염 의심", "soft"),
        ("PROT-SWITCH", "보호 경로로 절체됨 — 이중화 여유 없음", "soft"),
    ),
    "코어": (
        ("SESS-FULL", "세션 테이블 임계 초과 — 신규 호 거절 중", "hard"),
        ("PROC-DOWN", "코어 프로세스 비정상 종료", "hard"),
        ("DB-REPL-FAIL", "가입자 DB 복제 중단", "hard"),
        ("CPU-HIGH", "코어 노드 CPU 사용률 임계 초과", "soft"),
        ("MEM-HIGH", "코어 노드 메모리 사용률 임계 초과", "soft"),
        ("LIC-WARN", "라이선스 용량 임계 접근", "soft"),
    ),
    "전원": (
        ("AC-FAIL", "상용전원 정전 — 배터리 전환", "hard"),
        ("BATT-LOW", "배터리 잔량 임계 이하 — 정지 임박", "hard"),
        ("RECT-FAIL", "정류기 출력 이상 — 이중화 상실", "hard"),
        ("TEMP-HIGH", "함체 온도 상승 — 냉각 확인 필요", "soft"),
        ("DOOR-OPEN", "함체 문 열림 감지", "soft"),
        ("FAN-WARN", "냉각 팬 회전수 저하", "soft"),
    ),
    "회선": (
        ("E2E-LOSS", "구간 종단 패킷 손실 — 회선 절단 의심", "hard"),
        ("CKT-DOWN", "임대회선 다운 — 사업자 통보 필요", "hard"),
        ("LATENCY-HIGH", "구간 지연 임계 초과", "soft"),
        ("JITTER-HIGH", "구간 지터 임계 초과", "soft"),
        ("UTIL-HIGH", "회선 사용률 임계 초과", "soft"),
    ),
}

#: Which tier a severity draws its content from.
_TIER_FOR_SEVERITY = {
    "critical": "hard",
    "major": "hard",
    "minor": "soft",
    "warning": "soft",
}

#: Coarse `probable_cause`, the field an NOC system fills in from the code. Two
#: values rather than a per-code table: the tier is genuinely all the original
#: export would have supported, and inventing 26 distinct causes would be
#: dressing up a guess as detail.
_TIER_CAUSE = {
    "hard": "장비 장애 또는 구간 절단",
    "soft": "임계값 초과",
}

#: (label, code, region). Fourteen sites, five device types, four units each --
#: 280 physical boxes. Deliberately fewer boxes than a busy day has alarms
#: (359), so devices necessarily repeat within a day. A generator that invented
#: a fresh device per alarm would produce a feed where deduplication can never
#: fire, which is the opposite of what this data is for.
SITES: tuple[tuple[str, str, str], ...] = (
    ("서울 강남 12", "SEL", "수도권"),
    ("서울 마포 04", "SEL", "수도권"),
    ("서울 종로 21", "SEL", "수도권"),
    ("인천 연수 07", "INC", "수도권"),
    ("경기 성남 33", "GGI", "수도권"),
    ("경기 수원 18", "GGI", "수도권"),
    ("경기 고양 09", "GGI", "수도권"),
    ("대전 유성 05", "DJN", "중부권"),
    ("세종 나성 02", "SJG", "중부권"),
    ("강원 원주 03", "GWN", "중부권"),
    ("부산 해운대 11", "PUS", "영남권"),
    ("대구 수성 06", "TAE", "영남권"),
    ("광주 서구 08", "KWJ", "호남권"),
    ("제주 제주 01", "CJU", "제주권"),
)

_TYPE_CODE = {"기지국": "BS", "전송": "TX", "코어": "CR", "전원": "PW", "회선": "LN"}

#: Units of each type per site. Small on purpose -- see SITES.
UNITS_PER_TYPE = 4

VENDORS = ("벤더A", "벤더B", "벤더C")

#: A storm may take at most this share of a day's alarms. Without the cap a
#: 26-alarm day (the measured trough) would be nothing but one storm, and the
#: busiest structure in the data would be an artefact of the generator rather
#: than anything the customer reported.
MAX_STORM_SHARE = 0.35

#: A storm needs a representative and at least one derived alarm, or it is not
#: a correlated group and there is nothing for the pipeline to correlate.
MIN_STORM_SIZE = 2

#: Korean runs roughly 1.5 characters per token on a BPE tokeniser -- a
#: syllable block is usually one token, punctuation and ASCII field values less.
#: So `prompt_tokens = round(len(body) / 1.5)`. It is an estimate and it is only
#: used for sizing; the exact count depends on the tokeniser the model ships
#: with, and `TokenProfile` is where a caller overrides it with a real one.
KOREAN_CHARS_PER_TOKEN = 1.5

CSV_COLUMNS = (
    "id", "at_s", "device", "device_type", "code", "severity",
    "message", "site", "storm_id", "parent_id", "prompt_tokens", "raw",
)


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Alarm:
    """One alarm as the NOC system would have handed it over."""

    id: str
    #: Seconds since 00:00 of `AlarmDay.date`. Same clock as
    #: `workload.Arrival.at_s`, so a trace can be replayed against either.
    at_s: float
    device: str
    device_type: str
    code: str
    severity: str
    message: str
    site: str
    #: Set when this alarm belongs to a correlated group.
    storm_id: int | None = None
    #: The group's representative. `None` means this alarm *is* the
    #: representative, or that it stands alone.
    parent_id: str | None = None
    #: Estimated tokens for `raw["body"]` -- see KOREAN_CHARS_PER_TOKEN.
    prompt_tokens: int = 0
    #: The full upstream record, kept verbatim for audit. Everything the
    #: generator decided is visible here even if the typed fields drop it.
    raw: dict = field(default_factory=dict)

    @property
    def in_storm(self) -> bool:
        return self.storm_id is not None

    @property
    def is_representative(self) -> bool:
        return self.storm_id is not None and self.parent_id is None


@dataclass(frozen=True)
class AlarmDay:
    date: str
    alarms: list[Alarm]
    #: Number of correlated groups actually generated. Can be lower than the
    #: requested `storms_per_day` when the day's budget could not hold them.
    storms: int
    #: The effective seed for *this* day, derived from the base seed and the
    #: date. Recorded so one day can be reproduced from its own file.
    seed: int
    #: What in this day is measured and what is assumed. Korean, because it is
    #: read by the operator, not by a maintainer.
    notes: list[str] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.alarms)


# --------------------------------------------------------------------------
# How many alarms a day had
# --------------------------------------------------------------------------


def observed_count(date: str) -> int:
    """The customer's measured alarm count for `date`.

    2026-06 is returned per day because that month was exported per day.
    Everywhere else only a monthly total exists, so the day gets the month's
    mean -- which keeps the measured seasonality (2025-12 is the trough,
    2026-04 the peak) instead of flattening it into a global average.

    Raises for a date outside the measured window rather than extrapolating.
    A generated number that looks measured but is not is worse than an error.
    """
    day = _parse_date(date)
    if day.year == 2026 and day.month == 6:
        return DAILY_2026_06[day.day - 1]
    months = MONTHLY.get(day.year)
    if not months or day.month not in months:
        raise ValueError(
            f"{date}는 실측 구간(2025-02 ~ 2026-06) 밖이다 — count를 직접 지정해 주세요"
        )
    days_in_month = calendar.monthrange(day.year, day.month)[1]
    return round(months[day.month] / days_in_month)


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------


def generate_day(
    date: str = "2026-06-01",
    count: int | None = None,
    *,
    seed: int = 20260730,
    storm_size: int = 40,
    storms_per_day: int = 2,
    storm_window_s: float = 30.0,
    business_hours: tuple[int, int] = (8, 20),
    business_share: float = 0.8,
) -> AlarmDay:
    """One day of alarm records. Deterministic for `(seed, date)`.

    `count=None` uses the measured count for that date. Storm alarms come out
    of that budget rather than on top of it -- the customer's figure was alarms
    per day *including* the bad days, so adding a storm to it would size the
    server for a day nobody has.
    """
    day_date = _parse_date(date)
    notes: list[str] = []

    if count is None:
        total = observed_count(date)
        if day_date.year == 2026 and day_date.month == 6:
            notes.append(f"알람 수 {total}건은 {date} 고객사 실측 일별 값이다")
        else:
            months = MONTHLY[day_date.year]
            days_in_month = calendar.monthrange(day_date.year, day_date.month)[1]
            notes.append(
                f"알람 수 {total}건은 {day_date.year}-{day_date.month:02d} 실측 월합계 "
                f"{months[day_date.month]}건을 {days_in_month}일로 나눈 월평균이다 "
                f"— 이 달은 일별 실측이 없다"
            )
    else:
        if count < 0:
            raise ValueError("count는 0 이상이어야 한다")
        total = count
        notes.append(f"알람 수 {total}건은 호출자가 지정한 값이다 — 실측이 아니다")

    _validate(storm_size, storms_per_day, storm_window_s, business_hours, business_share)

    day_seed = _day_seed(seed, date)
    rng = random.Random(day_seed)

    storm_times, per_storm = _storm_plan(
        total, storm_size, storms_per_day, storm_window_s, business_hours, rng
    )
    storm_sites = _pick_storm_sites(len(storm_times), rng)
    storm_alarms = sum(len(t) for t in storm_times)
    background = total - storm_alarms

    background_times = _background_times(
        background, storm_alarms, total, business_hours, business_share, rng
    )

    alarms = _fill(
        day_date, storm_times, storm_sites, background_times, rng
    )

    notes.extend(
        _assumption_notes(
            business_hours, business_share, storm_size, storm_window_s,
            storms_per_day, len(storm_times), per_storm,
        )
    )

    return AlarmDay(
        date=date,
        alarms=alarms,
        storms=len(storm_times),
        seed=day_seed,
        notes=notes,
    )


def generate_range(start: str, days: int, **kw) -> list[AlarmDay]:
    """Consecutive days from `start`. Each day gets its own measured count."""
    if days <= 0:
        raise ValueError("days는 1 이상이어야 한다")
    first = _parse_date(start)
    return [
        generate_day(_Date.fromordinal(first.toordinal() + offset).isoformat(), **kw)
        for offset in range(days)
    ]


# --------------------------------------------------------------------------
# Placement: when the alarms land
# --------------------------------------------------------------------------


def _storm_plan(
    total: int,
    storm_size: int,
    storms_per_day: int,
    storm_window_s: float,
    business_hours: tuple[int, int],
    rng: random.Random,
) -> tuple[list[list[float]], int]:
    """Arrival times per correlated group, and the per-group size used.

    Sizes are capped by MAX_STORM_SHARE so a small day keeps a background
    population. Below MIN_STORM_SIZE there is no group at all -- a "storm" of
    one alarm is just an alarm.
    """
    if storms_per_day <= 0 or storm_size <= 0 or total <= 0:
        return [], 0

    affordable = int(total * MAX_STORM_SHARE) // storms_per_day
    per_storm = min(storm_size, affordable)
    if per_storm < MIN_STORM_SIZE:
        return [], 0

    start_h, end_h = business_hours
    # Keep the whole burst inside business hours -- same rule as workload.py.
    earliest = start_h * 3600.0
    latest = end_h * 3600.0 - storm_window_s
    if latest <= earliest:
        earliest, latest = 0.0, DAY_SECONDS - storm_window_s

    groups: list[list[float]] = []
    for _ in range(storms_per_day):
        begin = rng.uniform(earliest, latest)
        # A cascading failure fires thickest at the start: the representative
        # trips, then the consequences arrive over the following seconds.
        times = sorted(
            begin + storm_window_s * rng.random() ** 1.6 for _ in range(per_storm)
        )
        groups.append([_inside_day(t) for t in times])
    return groups, per_storm


def _pick_storm_sites(count: int, rng: random.Random) -> list[tuple[str, str, str]]:
    """One site per storm -- a storm is one physical failure, in one place."""
    if count <= 0:
        return []
    if count <= len(SITES):
        return rng.sample(SITES, count)
    return [rng.choice(SITES) for _ in range(count)]


def _background_times(
    count: int,
    storm_alarms: int,
    total: int,
    business_hours: tuple[int, int],
    business_share: float,
    rng: random.Random,
) -> list[float]:
    """Non-storm arrivals, placed so the *whole day* honours `business_share`.

    Storm alarms already sit inside business hours, so they are counted toward
    the in-hours quota instead of being added to it. `workload.py` applies the
    share to the background population alone, which is fine when storms are a
    rounding error but drifts badly once they are a fifth of the day: 40x2
    storms in a 359-alarm day would put 84% of arrivals in business hours while
    reporting 80%.

    Within a window the times are uniform. Conditional on the count in a
    window, a homogeneous Poisson process places its events uniformly at
    random -- so uniform draws per window give a correct
    piecewise-constant-rate arrival process, not merely a plausible one.
    """
    if count <= 0:
        return []

    start_h, end_h = business_hours
    day_start, day_end = start_h * 3600.0, end_h * 3600.0
    window = day_end - day_start
    night_len = DAY_SECONDS - window

    quota = round(total * business_share)
    in_hours = max(0, min(count, quota - storm_alarms))
    out_hours = count - in_hours
    if night_len <= 0:
        in_hours, out_hours = count, 0

    times = [rng.uniform(day_start, day_end) for _ in range(in_hours)]
    for _ in range(out_hours):
        # Off-hours is the day minus the business window; fold the single draw
        # onto whichever side of the window it falls.
        offset = rng.uniform(0.0, night_len)
        times.append(offset if offset < day_start else offset + window)
    return [_inside_day(t) for t in times]


def _inside_day(t: float) -> float:
    """Clamp to [0, DAY_SECONDS). `uniform` can return its upper bound."""
    return min(max(t, 0.0), float(DAY_SECONDS) - 1e-3)


# --------------------------------------------------------------------------
# Content: what each alarm says
# --------------------------------------------------------------------------


def _fill(
    day_date: _Date,
    storm_times: list[list[float]],
    storm_sites: list[tuple[str, str, str]],
    background_times: list[float],
    rng: random.Random,
) -> list[Alarm]:
    """Sort the arrivals, name them, then give each one its content.

    Ids are assigned after sorting so they run in arrival order, and content is
    drawn afterwards so a group's representative already has its id when its
    derived alarms need to reference it. The tie-break keeps a representative
    ahead of its own derivatives even if two draws land on the same instant.
    """
    slots: list[dict] = []
    for storm_id, times in enumerate(storm_times):
        for index, at_s in enumerate(times):
            slots.append(
                {
                    "at_s": at_s,
                    "storm_id": storm_id,
                    "parent": index == 0,
                    "site": storm_sites[storm_id],
                }
            )
    for at_s in background_times:
        slots.append({"at_s": at_s, "storm_id": None, "parent": False, "site": None})

    slots.sort(key=lambda s: (s["at_s"], 0 if s["parent"] else 1))

    stamp = day_date.strftime("%Y%m%d")
    for sequence, slot in enumerate(slots, start=1):
        slot["id"] = f"ALM-{stamp}-{sequence:06d}"

    representative = {
        slot["storm_id"]: slot["id"] for slot in slots if slot["parent"]
    }

    out: list[Alarm] = []
    for slot in slots:
        storm_id = slot["storm_id"]
        if storm_id is None:
            severity = _draw_severity(SEVERITY_WEIGHTS, rng)
            device_type = rng.choices(DEVICE_TYPES, weights=TYPE_WEIGHTS, k=1)[0]
            site = rng.choice(SITES)
            parent_id = None
        elif slot["parent"]:
            # The representative is the fault itself, so it is always the
            # severe one -- that is what makes it the root of the group.
            severity = "critical"
            device_type = rng.choices(DEVICE_TYPES, weights=TYPE_WEIGHTS, k=1)[0]
            site = slot["site"]
            parent_id = None
        else:
            severity = _draw_severity(STORM_DERIVED_WEIGHTS, rng)
            device_type = rng.choices(DEVICE_TYPES, weights=DERIVED_TYPE_WEIGHTS, k=1)[0]
            site = slot["site"]
            parent_id = representative[storm_id]

        code, message, tier = _draw_content(device_type, severity, rng)
        site_label, site_code, region = site
        device = _device_name(site, device_type, rng)
        body = _body(severity, device, device_type, code, site_label, message)

        out.append(
            Alarm(
                id=slot["id"],
                at_s=slot["at_s"],
                device=device,
                device_type=device_type,
                code=code,
                severity=severity,
                message=message,
                site=site_label,
                storm_id=storm_id,
                parent_id=parent_id,
                prompt_tokens=_estimate_tokens(body),
                raw={
                    "ne_id": device,
                    "ne_type": device_type,
                    "vendor": _vendor(site_code, device_type),
                    "alarm_code": code,
                    "perceived_severity": severity,
                    "event_time": _event_time(day_date, slot["at_s"]),
                    "event_offset_s": slot["at_s"],
                    "specific_problem": message,
                    "probable_cause": _TIER_CAUSE[tier],
                    "site": site_label,
                    "site_code": site_code,
                    "region": region,
                    "correlation_id": parent_id or slot["id"],
                    "body": body,
                },
            )
        )
    return out


def _draw_severity(weights: dict[str, float], rng: random.Random) -> str:
    return rng.choices(SEVERITIES, weights=[weights[s] for s in SEVERITIES], k=1)[0]


def _draw_content(
    device_type: str, severity: str, rng: random.Random
) -> tuple[str, str, str]:
    """Pick a code whose tier matches the severity already drawn.

    Drawing the two independently is what produces a `CELL-DOWN` at severity
    `warning` -- a record no NOC system would emit, and one that would teach a
    correlation stage the wrong thing.
    """
    tier = _TIER_FOR_SEVERITY[severity]
    entries = [e for e in ALARM_CATALOG[device_type] if e[2] == tier]
    return rng.choice(entries)


def _device_name(
    site: tuple[str, str, str], device_type: str, rng: random.Random
) -> str:
    """A name that identifies a *physical* box, so the same box can re-fire.

    The number is derived from (site, type, unit) rather than drawn freely:
    inventory is finite, and a feed where every alarm names a new device makes
    deduplication structurally impossible.
    """
    site_index = SITES.index(site)
    type_index = DEVICE_TYPES.index(device_type)
    unit = rng.randrange(UNITS_PER_TYPE)
    serial = 1000 + site_index * (len(DEVICE_TYPES) * UNITS_PER_TYPE) + type_index * UNITS_PER_TYPE + unit
    return f"{site[1]}-{_TYPE_CODE[device_type]}-{serial:04d}"


def _vendor(site_code: str, device_type: str) -> str:
    """Stable per (site, type): one box does not change manufacturer."""
    index = (len(site_code) + DEVICE_TYPES.index(device_type)) % len(VENDORS)
    return VENDORS[index]


def _body(
    severity: str, device: str, device_type: str, code: str, site: str, message: str
) -> str:
    """The one line that would actually be put in front of the model."""
    return f"[{severity.upper()}] {site} {device}({device_type}) {code} — {message}"


def _estimate_tokens(body: str) -> int:
    return max(1, round(len(body) / KOREAN_CHARS_PER_TOKEN))


def _event_time(day_date: _Date, at_s: float) -> str:
    seconds = int(at_s)
    return (
        f"{day_date.isoformat()}T"
        f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"
    )


# --------------------------------------------------------------------------
# Notes: naming the assumptions in the output itself
# --------------------------------------------------------------------------


def _assumption_notes(
    business_hours: tuple[int, int],
    business_share: float,
    storm_size: int,
    storm_window_s: float,
    storms_per_day: int,
    storms_made: int,
    per_storm: int,
) -> list[str]:
    start_h, end_h = business_hours
    # One line each. These travel with every run and get read every time, so
    # the reason belongs in the module docstring and only the claim belongs here.
    notes = [
        f"시간대 분포는 가정이다 — {start_h}~{end_h}시에 {business_share:.0%} "
        f"(원본은 일별 합계뿐이다)",
        f"스톰 규모({storm_size}건/{storm_window_s:.0f}초)와 상관 구조는 가정이다",
        "심각도 분포·장비·사이트 목록은 가정이다",
    ]
    if storms_per_day > 0 and storms_made == 0:
        notes.append(
            "일일 알람 수가 적어 스톰을 만들지 않았다 — 스톰이 하루를 다 차지하면 "
            "배경 트래픽이 사라진다"
        )
    elif storms_made > 0 and per_storm < storm_size:
        notes.append(
            f"일일 예산 상한(하루의 {MAX_STORM_SHARE:.0%})에 걸려 스톰 규모를 "
            f"{storm_size}건에서 {per_storm}건으로 줄였다"
        )
    return notes


# --------------------------------------------------------------------------
# Serialisation. JSONL is the round-trippable form; CSV is for humans.
# --------------------------------------------------------------------------


def to_jsonl(day: AlarmDay) -> str:
    """One JSON object per line: a day header, then the alarms in order.

    The header carries the day-level fields, because a feed that serialises
    only the alarms loses the notes -- and the notes are what stop a reader
    treating the hourly peak as measured. `alarm_count` is there so a truncated
    file is detected on read instead of silently becoming a quieter day.
    """
    lines = [
        _dumps(
            {
                "record": "day",
                "date": day.date,
                "storms": day.storms,
                "seed": day.seed,
                "notes": list(day.notes),
                "alarm_count": len(day.alarms),
            }
        )
    ]
    for alarm in day.alarms:
        lines.append(
            _dumps(
                {
                    "record": "alarm",
                    "id": alarm.id,
                    "at_s": alarm.at_s,
                    "device": alarm.device,
                    "device_type": alarm.device_type,
                    "code": alarm.code,
                    "severity": alarm.severity,
                    "message": alarm.message,
                    "site": alarm.site,
                    "storm_id": alarm.storm_id,
                    "parent_id": alarm.parent_id,
                    "prompt_tokens": alarm.prompt_tokens,
                    "raw": alarm.raw,
                }
            )
        )
    return "\n".join(lines) + "\n"


_ALARM_FIELDS = frozenset({
    "id", "at_s", "device", "device_type", "code", "severity",
    "message", "site", "storm_id", "parent_id", "prompt_tokens", "raw",
})


def from_jsonl(text: str) -> AlarmDay:
    """Inverse of `to_jsonl`. Refuses a feed it cannot fully reconstruct."""
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("빈 입력이다 — day 헤더 레코드가 필요하다")

    try:
        header = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise ValueError(f"첫 줄이 JSON이 아니다: {exc}") from exc
    if not isinstance(header, dict) or header.get("record") != "day":
        raise ValueError('첫 줄은 {"record": "day"} 헤더여야 한다')

    alarms: list[Alarm] = []
    for number, line in enumerate(lines[1:], start=2):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{number}번째 줄이 JSON이 아니다: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"{number}번째 줄이 JSON 객체가 아니다")
        if record.get("record") != "alarm":
            raise ValueError(f"{number}번째 줄의 record가 alarm이 아니다")
        missing = sorted(_ALARM_FIELDS - set(record))
        if missing:
            raise ValueError(f"{number}번째 줄에 {', '.join(missing)} 가 없다")
        alarms.append(
            Alarm(
                id=record["id"],
                at_s=record["at_s"],
                device=record["device"],
                device_type=record["device_type"],
                code=record["code"],
                severity=record["severity"],
                message=record["message"],
                site=record["site"],
                storm_id=record["storm_id"],
                parent_id=record["parent_id"],
                prompt_tokens=record["prompt_tokens"],
                raw=record["raw"],
            )
        )

    expected = header.get("alarm_count")
    if expected is not None and expected != len(alarms):
        raise ValueError(
            f"알람 수가 헤더와 다르다 — 헤더 {expected}건, 실제 {len(alarms)}건 (잘린 파일)"
        )

    return AlarmDay(
        date=header["date"],
        alarms=alarms,
        storms=header["storms"],
        seed=header["seed"],
        notes=list(header.get("notes", [])),
    )


def to_csv(day: AlarmDay) -> str:
    """Header plus one row per alarm. `raw` travels as an embedded JSON string.

    Not round-trippable by design -- CSV has no types and the day-level notes
    have nowhere to go. Use JSONL for anything that gets read back.
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(CSV_COLUMNS)
    for alarm in day.alarms:
        writer.writerow(
            [
                alarm.id,
                repr(alarm.at_s),
                alarm.device,
                alarm.device_type,
                alarm.code,
                alarm.severity,
                alarm.message,
                alarm.site,
                "" if alarm.storm_id is None else alarm.storm_id,
                alarm.parent_id or "",
                alarm.prompt_tokens,
                _dumps(alarm.raw),
            ]
        )
    return buffer.getvalue()


def describe(day: AlarmDay) -> str:
    """One line for a report or a CLI, in the tone of `workload.describe`."""
    storm_alarms = sum(1 for a in day.alarms if a.storm_id is not None)
    tokens = sum(a.prompt_tokens for a in day.alarms)
    return (
        f"{day.date}: {len(day.alarms)} alarms "
        f"({storm_alarms} in {day.storms} storm(s)), "
        f"{tokens} prompt tokens, seed {day.seed}"
    )


# --------------------------------------------------------------------------
# Plumbing
# --------------------------------------------------------------------------


def _dumps(obj: object) -> str:
    """Stable bytes for a given object: sorted keys, no incidental spacing."""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _parse_date(date: str) -> _Date:
    try:
        return datetime.strptime(date, "%Y-%m-%d").date()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"날짜 형식이 YYYY-MM-DD가 아니다: {date!r}") from exc


def _day_seed(seed: int, date: str) -> int:
    """A per-day seed derived from the base seed.

    Without this every day of a range would be the same day repeated, since the
    generator is a pure function of its seed. Derived with SHA-256 rather than
    `hash()` because `hash()` of a string is salted per process -- the output
    would change between runs, which is exactly the property this module
    promises not to have.
    """
    digest = hashlib.sha256(f"{seed}:{date}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _validate(
    storm_size: int,
    storms_per_day: int,
    storm_window_s: float,
    business_hours: tuple[int, int],
    business_share: float,
) -> None:
    if storm_size < 0 or storms_per_day < 0:
        raise ValueError("storm_size와 storms_per_day는 0 이상이어야 한다")
    if storm_window_s <= 0:
        raise ValueError("storm_window_s는 0보다 커야 한다")
    start_h, end_h = business_hours
    if not (0 <= start_h < end_h <= 24):
        raise ValueError(f"business_hours가 하루 안에 있지 않다: {business_hours}")
    if not (0.0 <= business_share <= 1.0):
        raise ValueError(f"business_share는 0~1 사이여야 한다: {business_share}")
