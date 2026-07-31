"""A load profile has to be the shape it claims, and the same shape twice.

Three properties carry this module. The first is that each shape actually does
what its name says -- a ramp that is not denser at the end is not a ramp, and a
spike that spreads its burst over the day is just a slightly busier day. The
second is determinism: a sizing argument that cannot be re-run byte for byte is
an anecdote. The third is that only the arrival *times* are synthetic --
content and correlation come from the measured-day generator and have to
survive being re-timed, because a `parent_id` pointing at an alarm that is no
longer in the list is a corrupt feed dressed up as a load test.
"""

import ast
import dataclasses
import json
import statistics
from pathlib import Path

import pytest

from svrspec.loadgen import DEFAULTS, KINDS, LoadProfile, build_load, rate_at
from svrspec.mockdata import DAY_SECONDS, generate_day

LOADGEN_SRC = Path(__file__).resolve().parents[1] / "svrspec" / "loadgen.py"


def _fingerprint(alarms) -> str:
    """Every field of every alarm, in a stable encoding."""
    return json.dumps(
        [dataclasses.asdict(a) for a in alarms],
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )


def _histogram(alarms, span_s: float, buckets: int) -> list[int]:
    width = span_s / buckets
    counts = [0] * buckets
    for alarm in alarms:
        counts[min(buckets - 1, int(alarm.at_s / width))] += 1
    return counts


# --------------------------------------------------------------------------
# AC1 each shape is the shape it says it is
# --------------------------------------------------------------------------


def test_replay_is_the_measured_day_untouched():
    """`replay` must stay byte-identical to what the rest of the tool produces.

    It is the baseline every other profile is compared against, so if this
    module quietly re-times it too, the comparison loses its anchor.
    """
    alarms, profile = build_load("replay", date="2026-06-01")
    day = generate_day("2026-06-01", seed=20260730, storm_size=40, storms_per_day=2)

    assert profile.kind == "replay"
    assert profile.span_s == float(DAY_SECONDS)
    assert [a.at_s for a in alarms] == [a.at_s for a in day.alarms]
    assert _fingerprint(alarms) == _fingerprint(day.alarms)


def test_ramp_gets_denser_as_it_goes():
    alarms, profile = build_load("ramp", start_rate=100, end_rate=2000, hours=24)
    quarters = _histogram(alarms, profile.span_s, 4)

    assert quarters == sorted(quarters), quarters
    assert sum(quarters[2:]) > 3 * sum(quarters[:2]), quarters
    # The integral of the rate curve, in alarms: (100+2000)/2 * 24/24.
    assert profile.total_alarms == pytest.approx(1050, rel=0.05)


def test_spike_puts_its_burst_only_in_its_own_window():
    alarms, profile = build_load(
        "spike", base_rate=165, peak_rate=800, spike_at_h=12, spike_minutes=30
    )
    begin, end = 12 * 3600.0, 12 * 3600.0 + 30 * 60.0
    inside = [a for a in alarms if begin <= a.at_s < end]
    outside = [a for a in alarms if not (begin <= a.at_s < end)]

    inside_per_hour = len(inside) / 0.5
    outside_per_hour = len(outside) / 23.5
    assert inside_per_hour > 4 * outside_per_hour, (len(inside), len(outside))


def test_soak_is_flat_for_its_whole_length():
    alarms, profile = build_load("soak", rate=300, hours=72)
    counts = _histogram(alarms, profile.span_s, 12)

    assert profile.span_s == 72 * 3600.0
    cov = statistics.pstdev(counts) / statistics.mean(counts)
    assert cov < 0.30, (cov, counts)
    # And flat over the long haul, not just on average: no six-hour block may be
    # half or double any other.
    assert min(counts) > 0.5 * max(counts), counts


# --------------------------------------------------------------------------
# AC2 determinism
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_same_seed_gives_the_same_bytes(kind):
    first, profile_a = build_load(kind, seed=99)
    second, profile_b = build_load(kind, seed=99)
    assert _fingerprint(first) == _fingerprint(second)
    assert profile_a == profile_b


@pytest.mark.parametrize("kind", KINDS)
def test_a_different_seed_gives_a_different_day(kind):
    first, _ = build_load(kind, seed=1)
    second, _ = build_load(kind, seed=2)
    assert _fingerprint(first) != _fingerprint(second)


def test_two_shapes_at_one_seed_do_not_share_a_draw():
    """Otherwise a ramp and a soak would place their units identically."""
    ramp, _ = build_load("ramp", seed=7, start_rate=300, end_rate=300, hours=72)
    soak, _ = build_load("soak", seed=7, rate=300, hours=72)
    assert [a.at_s for a in ramp] != [a.at_s for a in soak]


# --------------------------------------------------------------------------
# AC3 ordering and bounds
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_arrivals_are_ordered_and_inside_the_span(kind):
    alarms, profile = build_load(kind)
    times = [a.at_s for a in alarms]

    assert times == sorted(times)
    assert all(0.0 <= t < profile.span_s for t in times)
    assert profile.total_alarms == len(alarms)


def test_a_storm_at_the_very_end_is_not_pushed_out_of_the_span():
    """The clamp has to hold, or the last unit lands past the window."""
    alarms, profile = build_load("spike", spike_at_h=23, spike_minutes=59)
    assert max(a.at_s for a in alarms) < profile.span_s


# --------------------------------------------------------------------------
# AC4 content and correlation survive the re-timing
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_ids_are_unique_and_every_parent_is_present(kind):
    alarms, _ = build_load(kind)
    ids = [a.id for a in alarms]
    assert len(set(ids)) == len(ids)

    known = set(ids)
    for alarm in alarms:
        if alarm.parent_id is not None:
            assert alarm.parent_id in known, alarm.id


@pytest.mark.parametrize("kind", KINDS)
def test_alarm_content_is_never_invented_here(kind):
    """Every record still carries the generator's fields, filled in."""
    alarms, _ = build_load(kind)
    for alarm in alarms:
        assert alarm.device and alarm.code and alarm.message and alarm.site
        assert alarm.prompt_tokens > 0
        assert alarm.raw["body"]
        # The audit copy must agree with the typed field it duplicates.
        assert alarm.raw["event_offset_s"] == pytest.approx(alarm.at_s)


def test_a_storm_stays_together_after_being_moved():
    """A correlated group is one physical failure; scattering it is a lie."""
    alarms, _ = build_load("spike")
    groups: dict[int, list] = {}
    for alarm in alarms:
        if alarm.storm_id is not None:
            groups.setdefault(alarm.storm_id, []).append(alarm)

    assert groups, "spike must carry storms"
    for storm_id, members in groups.items():
        span = max(m.at_s for m in members) - min(m.at_s for m in members)
        assert span <= 31.0, (storm_id, span)
        representatives = [m for m in members if m.parent_id is None]
        assert len(representatives) == 1
        assert all(
            m.parent_id == representatives[0].id for m in members if m is not representatives[0]
        )


def test_storm_ids_are_renumbered_so_days_cannot_collide():
    """Content comes from several measured days, each numbering from zero."""
    alarms, profile = build_load("spike")
    per_storm: dict[int, set] = {}
    for alarm in alarms:
        if alarm.storm_id is not None:
            per_storm.setdefault(alarm.storm_id, set()).add(alarm.parent_id or alarm.id)

    assert len(per_storm) == profile.params["storms"]
    for storm_id, roots in per_storm.items():
        assert len(roots) == 1, (storm_id, roots)


@pytest.mark.parametrize("kind", ["ramp", "soak"])
def test_ramp_and_soak_carry_no_storms_and_say_why(kind):
    alarms, profile = build_load(kind)
    assert all(a.storm_id is None for a in alarms)
    assert profile.params["storms"] == 0
    assert any("스톰을 넣지 않는다" in note for note in profile.notes), profile.notes


def test_spike_says_why_it_does_carry_storms():
    _, profile = build_load("spike")
    assert profile.params["storms"] > 0
    assert any("스톰을 넣는다" in note for note in profile.notes), profile.notes


# --------------------------------------------------------------------------
# The rate curve a chart reads back
# --------------------------------------------------------------------------


def test_params_carry_enough_to_rebuild_the_offered_rate():
    _, profile = build_load("ramp", start_rate=100, end_rate=2000, hours=24)
    assert profile.params["start_rate"] == 100
    assert profile.params["end_rate"] == 2000
    assert profile.params["span_s"] == 24 * 3600.0

    assert rate_at(profile, 0.0) == pytest.approx(100.0)
    assert rate_at(profile, 12 * 3600.0) == pytest.approx(1050.0)
    assert rate_at(profile, 24 * 3600.0 - 1) == pytest.approx(2000.0, rel=1e-3)
    assert rate_at(profile, 25 * 3600.0) == 0.0


def test_the_spike_rate_curve_is_a_step():
    _, profile = build_load(
        "spike", base_rate=165, peak_rate=800, spike_at_h=12, spike_minutes=30
    )
    assert rate_at(profile, 11 * 3600.0) == pytest.approx(165.0)
    assert rate_at(profile, 12 * 3600.0 + 60) == pytest.approx(800.0)
    assert rate_at(profile, 13 * 3600.0) == pytest.approx(165.0)


def test_replay_declares_no_offered_rate():
    """A measured day has no rate other than what arrived. Say None, not a curve."""
    _, profile = build_load("replay")
    assert "rate_segments" not in profile.params
    assert rate_at(profile, 3600.0) is None


def test_params_survive_json():
    for kind in KINDS:
        _, profile = build_load(kind)
        assert json.loads(json.dumps(profile.params)) == profile.params


# --------------------------------------------------------------------------
# Assumptions have to travel with the data
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_the_generators_assumptions_are_carried_forward(kind):
    _, profile = build_load(kind)
    joined = " | ".join(profile.notes)
    assert "심각도 분포는 가정" in joined, profile.notes
    assert "장비·사이트 목록은 가정" in joined, profile.notes


@pytest.mark.parametrize("kind", ["ramp", "spike", "soak"])
def test_a_note_that_stopped_being_true_is_dropped(kind):
    """`mockdata`'s business-hours note describes placement this module replaced."""
    _, profile = build_load(kind)
    joined = " | ".join(profile.notes)
    assert "하루 안의 시간대 분포는 실측이 아니다" not in joined, profile.notes
    assert "부하율 곡선에서 다시 뽑았다" in joined, profile.notes


def test_replay_keeps_the_business_hours_note():
    _, profile = build_load("replay")
    joined = " | ".join(profile.notes)
    assert "하루 안의 시간대 분포는 실측이 아니다" in joined


# --------------------------------------------------------------------------
# Bad input is refused, not absorbed
# --------------------------------------------------------------------------


def test_an_unknown_kind_is_refused():
    with pytest.raises(ValueError, match="알 수 없는 부하 종류"):
        build_load("stress")


def test_a_misspelled_parameter_is_refused_rather_than_ignored():
    """A GUI field that silently does nothing is worse than an error."""
    with pytest.raises(ValueError, match="모르는 파라미터"):
        build_load("ramp", end_rates=2000)


@pytest.mark.parametrize(
    "kind,params",
    [
        ("ramp", {"start_rate": 0}),
        ("ramp", {"end_rate": -1}),
        ("ramp", {"hours": 0}),
        ("soak", {"rate": 0}),
        ("soak", {"hours": -3}),
        ("spike", {"spike_minutes": 0}),
        ("spike", {"spike_at_h": 23.9, "spike_minutes": 30}),
        ("spike", {"base_rate": 0}),
    ],
)
def test_impossible_parameters_are_refused(kind, params):
    with pytest.raises(ValueError):
        build_load(kind, **params)


def test_a_load_bigger_than_the_measured_window_says_so():
    """There are only ~485 measured days to borrow alarm content from."""
    with pytest.raises(ValueError, match="실측 날짜가 부족하다"):
        build_load("soak", rate=5000, hours=24 * 400)


def test_every_kind_is_reachable_and_documented():
    assert tuple(DEFAULTS) == KINDS
    for kind in KINDS:
        alarms, profile = build_load(kind)
        assert isinstance(profile, LoadProfile)
        assert profile.kind == kind
        assert profile.label.strip()
        assert profile.notes


# --------------------------------------------------------------------------
# No model, no process, no network
# --------------------------------------------------------------------------


def test_the_module_cannot_reach_a_process_or_the_network():
    """Static check: generating a load must never put load on this machine."""
    source = LOADGEN_SRC.read_text(encoding="utf-8")
    banned = {"subprocess", "socket", "urllib", "http", "ssl", "asyncio",
              "multiprocessing", "ctypes", "requests", "httpx", "torch",
              "llama_cpp", "time"}

    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module.split(".")[0])
    assert not (imported & banned), sorted(imported & banned)

    for text in ("subprocess", "urllib", "socket", "Popen", "urlopen", "sleep"):
        assert text not in source, text
