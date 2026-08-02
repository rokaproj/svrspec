"""The mock alarm feed must reproduce the customer's measured volume.

One test per acceptance criterion in the brief, in order. The through-line: a
generator whose output nobody checked against the real numbers is a generator
that quietly sizes the server for a day the customer never has.
"""

import csv
import io
import json
from collections import Counter
from datetime import date as _date
from datetime import timedelta
from pathlib import Path

import pytest

from svrspec.mockdata import (
    ALARM_CATALOG,
    DAILY_2026_06,
    DAY_SECONDS,
    DEVICE_TYPES,
    MONTHLY_2025,
    MONTHLY_2026,
    SEVERITIES,
    SEVERITY_WEIGHTS,
    Alarm,
    AlarmDay,
    from_jsonl,
    generate_day,
    generate_range,
    observed_count,
    to_csv,
    to_jsonl,
)


def _dates(start: str, days: int) -> list[str]:
    first = _date.fromisoformat(start)
    return [(first + timedelta(days=i)).isoformat() for i in range(days)]


# --------------------------------------------------------------------------
# The measured data itself. If these drift, every test below is measuring
# the generator against the wrong reality.
# --------------------------------------------------------------------------


def test_the_measured_constants_add_up():
    assert len(DAILY_2026_06) == 30
    assert sum(DAILY_2026_06) == 4411
    assert sum(MONTHLY_2025.values()) == 51049
    assert sum(MONTHLY_2026.values()) == 29953
    assert sum(MONTHLY_2025.values()) + sum(MONTHLY_2026.values()) == 81002
    assert MONTHLY_2026[6] == sum(DAILY_2026_06)
    assert max(DAILY_2026_06) == 359
    assert min(DAILY_2026_06) == 26


# --------------------------------------------------------------------------
# AC#1 -- the count is the measured count
# --------------------------------------------------------------------------


def test_the_first_of_june_has_the_measured_359_alarms():
    day = generate_day("2026-06-01")
    assert len(day.alarms) == 359
    assert day.date == "2026-06-01"


def test_every_day_of_june_2026_matches_the_measured_series():
    for offset, expected in enumerate(DAILY_2026_06):
        target = _dates("2026-06-01", 30)[offset]
        assert len(generate_day(target).alarms) == expected, target


def test_a_month_without_daily_data_falls_back_to_its_measured_average():
    """2025 has monthly totals only, so a day there is the month's mean.

    Worth pinning: the fallback must come from the customer's own monthly
    figure, not from a global average that would flatten the seasonality the
    monthly table actually records.
    """
    assert observed_count("2025-12-05") == round(MONTHLY_2025[12] / 31)
    assert observed_count("2026-04-10") == round(MONTHLY_2026[4] / 30)
    assert len(generate_day("2025-12-05").alarms) == observed_count("2025-12-05")
    # 2025-12 is the measured trough and 2026-04 the measured peak; the
    # fallback has to preserve that ordering.
    assert observed_count("2025-12-05") < observed_count("2026-04-10")


def test_an_explicit_count_overrides_the_measured_value():
    assert len(generate_day("2026-06-01", count=50).alarms) == 50
    assert len(generate_day("2026-06-01", count=0).alarms) == 0


def test_a_date_outside_the_measured_window_is_refused_rather_than_invented():
    """No measurement, no number. Guessing one would launder a fiction."""
    with pytest.raises(ValueError):
        generate_day("2024-01-01")
    with pytest.raises(ValueError):
        generate_day("not-a-date")
    # ...but an explicit count is the caller taking responsibility, so allow it.
    assert len(generate_day("2024-01-01", count=17).alarms) == 17


# --------------------------------------------------------------------------
# AC#2, AC#3 -- reproducible from the seed, and only from the seed
# --------------------------------------------------------------------------


def test_the_same_seed_produces_the_same_bytes():
    a = to_jsonl(generate_day("2026-06-01", seed=20260730))
    b = to_jsonl(generate_day("2026-06-01", seed=20260730))
    assert a == b
    assert to_csv(generate_day("2026-06-01", seed=1)) == to_csv(
        generate_day("2026-06-01", seed=1)
    )


def test_a_different_seed_produces_a_different_day():
    a = to_jsonl(generate_day("2026-06-01", seed=1))
    b = to_jsonl(generate_day("2026-06-01", seed=2))
    assert a != b
    # Same volume though -- the seed moves the content, not the measured count.
    assert len(generate_day("2026-06-01", seed=1).alarms) == 359
    assert len(generate_day("2026-06-01", seed=2).alarms) == 359


def test_each_date_gets_its_own_stream_from_one_seed():
    """Two days of one run must not be the same day twice."""
    first, second = generate_range("2026-06-02", 2, count=120, seed=7)
    assert [a.at_s for a in first.alarms] != [a.at_s for a in second.alarms]
    assert first.seed != second.seed


# --------------------------------------------------------------------------
# AC#4 -- arrival times are sorted and inside the day
# --------------------------------------------------------------------------


def test_arrivals_are_sorted_and_inside_the_day():
    for day in generate_range("2026-06-01", 30):
        times = [a.at_s for a in day.alarms]
        assert times == sorted(times)
        assert all(0.0 <= t < DAY_SECONDS for t in times)


def test_ids_run_in_arrival_order():
    day = generate_day("2026-06-01")
    assert [a.id for a in day.alarms] == sorted(a.id for a in day.alarms)
    assert day.alarms[0].id == "ALM-20260601-000001"
    assert day.alarms[-1].id == "ALM-20260601-000359"
    assert len({a.id for a in day.alarms}) == len(day.alarms)


# --------------------------------------------------------------------------
# AC#5 -- the business-hours weighting
# --------------------------------------------------------------------------


@pytest.mark.parametrize("share", [0.5, 0.8, 0.95])
def test_the_business_hours_share_is_honoured(share):
    day = generate_day("2026-06-01", business_share=share)
    lo, hi = 8 * 3600, 20 * 3600
    inside = sum(1 for a in day.alarms if lo <= a.at_s < hi)
    assert inside / len(day.alarms) == pytest.approx(share, abs=0.05)


def test_a_shifted_business_window_moves_the_mass():
    day = generate_day("2026-06-01", business_hours=(0, 6), business_share=0.9)
    inside = sum(1 for a in day.alarms if a.at_s < 6 * 3600)
    assert inside / len(day.alarms) == pytest.approx(0.9, abs=0.05)


# --------------------------------------------------------------------------
# AC#6 -- storms are correlated events, not just bursts of arrivals
# --------------------------------------------------------------------------


def test_storms_are_correlated_groups_with_one_representative():
    day = generate_day("2026-06-01", storms_per_day=2, storm_size=40, storm_window_s=30.0)
    assert day.storms == 2

    groups: dict[int, list[Alarm]] = {}
    for alarm in day.alarms:
        if alarm.storm_id is not None:
            groups.setdefault(alarm.storm_id, []).append(alarm)
    assert len(groups) == 2

    for storm_id, members in groups.items():
        assert len(members) == 40
        assert {a.storm_id for a in members} == {storm_id}
        # One site, one physical failure.
        assert len({a.site for a in members}) == 1
        # Inside the window.
        span = members[-1].at_s - members[0].at_s
        assert 0.0 <= span <= 30.0
        # Exactly one representative; everything else points at it.
        parents = [a for a in members if a.parent_id is None]
        assert len(parents) == 1
        parent = parents[0]
        assert parent.at_s == min(a.at_s for a in members)
        assert all(a.parent_id == parent.id for a in members if a is not parent)
        # The representative must already be in the stream when its
        # derivatives reference it.
        ids = [a.id for a in day.alarms]
        assert all(ids.index(parent.id) < ids.index(a.id) for a in members if a is not parent)


def test_background_alarms_are_not_tagged_as_storm_or_derived():
    day = generate_day("2026-06-01")
    background = [a for a in day.alarms if a.storm_id is None]
    assert background
    assert all(a.parent_id is None for a in background)


def test_storms_can_be_switched_off():
    day = generate_day("2026-06-01", storms_per_day=0)
    assert day.storms == 0
    assert all(a.storm_id is None for a in day.alarms)


def test_a_day_too_small_for_a_storm_says_so_instead_of_faking_one():
    """26 alarms cannot contain two forty-alarm storms.

    The measured trough day is 26 alarms. Letting a storm swallow the whole day
    would make the busiest structure in the data an artefact of the generator.
    """
    day = generate_day("2026-06-21")  # measured: 26
    assert len(day.alarms) == 26
    storm_alarms = [a for a in day.alarms if a.storm_id is not None]
    assert len(storm_alarms) <= len(day.alarms) // 2
    assert any("스톰" in n for n in day.notes)


def test_a_single_alarm_day_has_no_storm_at_all():
    day = generate_day("2026-06-01", count=1)
    assert len(day.alarms) == 1
    assert day.storms == 0


# --------------------------------------------------------------------------
# AC#7 -- storms come out of the budget, not on top of it
# --------------------------------------------------------------------------


@pytest.mark.parametrize("count", [1, 5, 26, 100, 359, 1000])
def test_storm_alarms_are_deducted_from_the_daily_budget(count):
    day = generate_day("2026-06-01", count=count, storm_size=40, storms_per_day=2)
    assert len(day.alarms) == count


# --------------------------------------------------------------------------
# AC#8 -- JSONL round trip
# --------------------------------------------------------------------------


def test_jsonl_round_trips_without_loss():
    day = generate_day("2026-06-01")
    back = from_jsonl(to_jsonl(day))
    assert back == day
    assert to_jsonl(back) == to_jsonl(day)
    # Field by field, so a dataclass __eq__ that forgets a field cannot hide.
    for original, restored in zip(day.alarms, back.alarms):
        assert original.id == restored.id
        assert original.at_s == restored.at_s
        assert original.device == restored.device
        assert original.device_type == restored.device_type
        assert original.code == restored.code
        assert original.severity == restored.severity
        assert original.message == restored.message
        assert original.site == restored.site
        assert original.storm_id == restored.storm_id
        assert original.parent_id == restored.parent_id
        assert original.prompt_tokens == restored.prompt_tokens
        assert original.raw == restored.raw


def test_jsonl_carries_the_day_level_metadata():
    day = generate_day("2026-06-03", seed=99)
    back = from_jsonl(to_jsonl(day))
    assert back.date == day.date == "2026-06-03"
    assert back.seed == day.seed
    assert back.storms == day.storms
    assert back.notes == day.notes


def test_jsonl_is_one_json_object_per_line():
    day = generate_day("2026-06-01", count=20)
    lines = to_jsonl(day).splitlines()
    assert len(lines) == 21  # one header record plus the alarms
    records = [json.loads(line) for line in lines]
    assert records[0]["record"] == "day"
    assert all(r["record"] == "alarm" for r in records[1:])


def test_an_empty_day_round_trips_too():
    day = generate_day("2026-06-01", count=0)
    assert from_jsonl(to_jsonl(day)) == day


def test_a_truncated_feed_is_refused_rather_than_half_read():
    with pytest.raises(ValueError):
        from_jsonl("")
    with pytest.raises(ValueError):
        from_jsonl('{"record": "alarm", "id": "ALM-1"}')  # no day header
    with pytest.raises(ValueError, match="객체"):
        from_jsonl('{"record": "day"}\n[]')


# --------------------------------------------------------------------------
# AC#9 -- CSV shape
# --------------------------------------------------------------------------


def test_csv_has_a_header_and_one_row_per_alarm():
    day = generate_day("2026-06-01")
    rows = list(csv.reader(io.StringIO(to_csv(day))))
    assert len(rows) == len(day.alarms) + 1

    header = rows[0]
    for field in (
        "id", "at_s", "device", "device_type", "code",
        "severity", "message", "site", "storm_id", "parent_id",
        "prompt_tokens", "raw",
    ):
        assert field in header, field

    first = dict(zip(header, rows[1]))
    assert first["id"] == day.alarms[0].id
    assert first["severity"] == day.alarms[0].severity
    assert json.loads(first["raw"]) == day.alarms[0].raw


def test_csv_of_an_empty_day_is_the_header_alone():
    rows = list(csv.reader(io.StringIO(to_csv(generate_day("2026-06-01", count=0)))))
    assert len(rows) == 1


# --------------------------------------------------------------------------
# AC#10 -- severities are realistic and bounded
# --------------------------------------------------------------------------


def test_severities_stay_inside_the_declared_set():
    day = generate_day("2026-06-01")
    assert {a.severity for a in day.alarms} <= set(SEVERITIES)


def test_the_severity_mix_follows_the_declared_weights():
    """Pooled over the measured month, so one quiet day cannot skew it."""
    counts: Counter[str] = Counter()
    for day in generate_range("2026-06-01", 30):
        counts.update(a.severity for a in day.alarms)
    total = sum(counts.values())
    assert total == 4411
    for severity, weight in SEVERITY_WEIGHTS.items():
        assert counts[severity] / total == pytest.approx(weight, abs=0.1), severity
    # Critical must be the minority and warning/minor the bulk -- the ordering
    # matters more than the exact ratio.
    assert counts["critical"] < counts["major"] < counts["minor"] + counts["warning"]


def test_the_weights_are_a_distribution():
    assert set(SEVERITY_WEIGHTS) == set(SEVERITIES)
    assert sum(SEVERITY_WEIGHTS.values()) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# AC#11 -- the month reproduces the measured total
# --------------------------------------------------------------------------


def test_june_2026_reproduces_the_measured_4411():
    days = generate_range("2026-06-01", 30)
    assert len(days) == 30
    assert sum(len(d.alarms) for d in days) == 4411
    assert [len(d.alarms) for d in days] == list(DAILY_2026_06)
    assert [d.date for d in days] == _dates("2026-06-01", 30)


def test_a_range_rejects_a_nonsensical_length():
    with pytest.raises(ValueError):
        generate_range("2026-06-01", 0)
    with pytest.raises(ValueError):
        generate_range("2026-06-01", -3)


# --------------------------------------------------------------------------
# AC#12 -- what is measured and what is assumed
# --------------------------------------------------------------------------


def test_the_notes_admit_that_the_hourly_shape_is_an_assumption():
    day = generate_day("2026-06-01")
    assert any("가정" in n for n in day.notes)
    assert any("일별 합계" in n for n in day.notes)
    joined = " ".join(day.notes)
    assert "시간대" in joined


def test_the_notes_say_where_the_daily_count_came_from():
    measured = generate_day("2026-06-01")
    assert any("실측" in n for n in measured.notes)
    derived = generate_day("2025-12-05")
    assert any("월평균" in n for n in derived.notes)
    explicit = generate_day("2026-06-01", count=42)
    assert any("호출자" in n for n in explicit.notes)


# --------------------------------------------------------------------------
# AC#13 -- this module makes data, it does not run or reach anything
# --------------------------------------------------------------------------


def test_the_generator_cannot_execute_or_reach_anything():
    """Same guard as the log importer: the string, not the import.

    This tool runs on air-gapped machines and the point of the generator is to
    replace a live feed, not to go looking for one.
    """
    source = (
        Path(__file__).parent.parent / "svrspec" / "mockdata.py"
    ).read_text(encoding="utf-8")
    assert "subprocess" not in source
    assert "urllib" not in source
    assert "socket" not in source
    assert "os.system" not in source


# --------------------------------------------------------------------------
# Content quality -- the records have to be usable by the pipeline
# --------------------------------------------------------------------------


def test_every_alarm_is_fully_populated():
    day = generate_day("2026-06-01")
    for alarm in day.alarms:
        assert alarm.id
        assert alarm.device
        assert alarm.device_type in DEVICE_TYPES
        assert alarm.code
        assert alarm.message
        assert alarm.site
        assert alarm.prompt_tokens > 0
        assert isinstance(alarm.raw, dict) and alarm.raw


def test_the_code_and_message_tables_cover_every_device_type():
    """No type may fall back to a generic message, and no severity may be
    unreachable for a type -- otherwise the code chosen would be decided by
    which table happened to have an entry."""
    assert set(ALARM_CATALOG) == set(DEVICE_TYPES)
    for device_type, entries in ALARM_CATALOG.items():
        assert len(entries) >= 4, device_type
        tiers = {tier for _, _, tier in entries}
        assert tiers == {"hard", "soft"}, device_type
        for code, message, _ in entries:
            assert code.upper() == code
            assert message.strip() == message and "\n" not in message


def test_the_code_matches_the_severity_it_was_drawn_for():
    """A CELL-DOWN cannot be a warning, and a threshold crossing cannot be
    critical -- content and severity are drawn together, not independently."""
    hard = {
        code
        for entries in ALARM_CATALOG.values()
        for code, _, tier in entries
        if tier == "hard"
    }
    soft = {
        code
        for entries in ALARM_CATALOG.values()
        for code, _, tier in entries
        if tier == "soft"
    }
    assert not (hard & soft)
    for alarm in generate_day("2026-06-01").alarms:
        if alarm.severity in ("critical", "major"):
            assert alarm.code in hard, alarm
        else:
            assert alarm.code in soft, alarm


def test_a_storm_representative_is_a_hard_failure():
    day = generate_day("2026-06-01")
    for alarm in day.alarms:
        if alarm.storm_id is not None and alarm.parent_id is None:
            assert alarm.severity == "critical"


def test_prompt_tokens_track_the_body_that_would_be_sent():
    day = generate_day("2026-06-01", count=60)
    for alarm in day.alarms:
        body = alarm.raw["body"]
        assert alarm.message in body
        assert alarm.code in body
        # The documented estimator: Korean text is ~1.5 characters per token.
        assert alarm.prompt_tokens == max(1, round(len(body) / 1.5))
    # Longer bodies must cost more tokens; a constant would be a bug.
    assert len({a.prompt_tokens for a in day.alarms}) > 1


def test_the_raw_record_survives_json_and_names_its_source():
    day = generate_day("2026-06-01", count=10)
    for alarm in day.alarms:
        assert json.loads(json.dumps(alarm.raw, ensure_ascii=False)) == alarm.raw
        assert alarm.raw["alarm_code"] == alarm.code
        assert alarm.raw["perceived_severity"] == alarm.severity
        assert alarm.raw["event_time"].startswith("2026-06-01T")


def test_devices_repeat_within_a_site_the_way_real_estate_does():
    """A generator that invents a fresh device per alarm produces a feed where
    deduplication and correlation can never fire -- which is exactly the data
    gap this module exists to close."""
    day = generate_day("2026-06-01")
    devices = Counter(a.device for a in day.alarms)
    assert max(devices.values()) > 1
    assert len(devices) < len(day.alarms)


def test_the_dataclasses_are_frozen():
    day = generate_day("2026-06-01", count=3)
    with pytest.raises(Exception):
        day.alarms[0].severity = "critical"  # type: ignore[misc]
    with pytest.raises(Exception):
        day.date = "2026-06-02"  # type: ignore[misc]
    assert isinstance(day, AlarmDay)
