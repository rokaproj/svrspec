"""Knee finding: where a build breaks, and what gave way."""

import pytest

from svrspec.capacity import (
    AXES,
    CEILING,
    apply_axis,
    axis_value,
    find_knee,
    sweep_axes,
    weakest_axis,
)
from svrspec.perf import Efficiency
from svrspec.sizing import evaluate
from svrspec.types import TokenProfile, Workload


@pytest.fixture
def build(catalog):
    return (
        catalog.model("test-8b-gqa"),
        catalog.quant("Q4_K_M"),
        catalog.cpu("test-amx-8ch"),
        catalog.memory_for(catalog.cpu("test-amx-8ch"), 1),
        Efficiency.from_catalog(catalog.coefficients),
    )


def _knee(build, workload, axis, **kw):
    model, quant, cpu, memory, eff = build
    return find_knee(model, quant, cpu, memory, eff, workload, axis, **kw)


@pytest.mark.parametrize("axis", AXES)
def test_every_axis_round_trips(axis):
    workload = Workload()
    changed = apply_axis(workload, axis, 777)
    assert axis_value(changed, axis) == 777
    # And only that axis moved.
    for other in AXES:
        if other != axis:
            assert axis_value(changed, other) == axis_value(workload, other)


def test_unknown_axis_is_rejected():
    with pytest.raises(ValueError):
        apply_axis(Workload(), "cpu_temperature", 1)
    with pytest.raises(ValueError):
        axis_value(Workload(), "cpu_temperature")


def test_the_knee_passes_and_the_break_fails(build):
    """The two sides of the bracket must actually be the two sides."""
    workload = Workload(alarms_per_day=120)
    curve = _knee(build, workload, "storm")
    assert curve.knee is not None and curve.knee.ok
    if curve.breaks_at is not None:
        assert not curve.breaks_at.ok
        assert curve.breaks_at.value > curve.knee.value


def test_the_knee_agrees_with_the_recommender(build):
    """A knee found here and a verdict printed by `recommend` cannot disagree.

    Both must come from the same pessimistic evaluation, or the tool will tell
    the customer two different things about the same build.
    """
    model, quant, cpu, memory, eff = build
    workload = Workload(alarms_per_day=120)
    curve = _knee(build, workload, "storm")
    assert curve.knee is not None

    at_knee = evaluate(
        model, quant, cpu, memory, eff, apply_axis(workload, "storm", curve.knee.value), 1
    )
    assert at_knee.verdict == curve.knee.verdict


def test_points_are_ordered_and_unique(build):
    curve = _knee(build, Workload(alarms_per_day=120), "prompt")
    values = [p.value for p in curve.points]
    assert values == sorted(values)
    assert len(values) == len(set(values))


def test_limiter_names_the_clause_that_broke(build):
    curve = _knee(build, Workload(alarms_per_day=120), "storm")
    if curve.breaks_at is None:
        pytest.skip("this build has no knee on this axis")
    assert curve.limiter in ("latency", "storm-drain", "ram", "headroom")
    # The limiter must be justified by a reason the judge actually gave.
    assert curve.breaks_at.reasons


def test_headroom_is_knee_over_planned_load(build):
    workload = Workload(alarms_per_day=120, storm_size=20)
    curve = _knee(build, workload, "storm")
    assert curve.baseline == 20
    if curve.knee:
        assert curve.headroom == pytest.approx(curve.knee.value / 20)


def test_the_alarms_axis_stops_at_the_storm_total(build):
    """Below the storm total the axis measures a degenerate storm, not volume.

    Storms draw from the daily budget rather than adding to it, so a day with
    fewer alarms than the storms contain has no quiet traffic at all. Walking
    down there once produced a knee of "4 alarms/day" for a build that handles
    359 comfortably -- an artefact of the generator, reported as a finding.
    """
    from svrspec.capacity import _floor

    workload = Workload(
        alarms_per_day=359,
        storm_size=60,
        storms_per_day=3,
        tokens=TokenProfile(alarm_tokens=3000, output_tokens=800),
    )
    assert _floor(workload, "alarms") == 180
    assert _floor(workload, "storm") == 0

    curve = _knee(build, workload, "alarms")
    for point in curve.points:
        assert point.value >= 180
    if curve.knee is None:
        assert any("탐색 하한" in n for n in curve.notes)
        assert any("스톰 총량" in n for n in curve.notes)


def test_the_storm_floor_does_not_apply_to_other_axes(build):
    workload = Workload(alarms_per_day=200, storm_size=40, storms_per_day=2)
    curve = _knee(build, workload, "prompt")
    assert not any("스톰 총량" in n for n in curve.notes)


def test_a_build_already_over_reports_a_lower_knee(build):
    """When the planned load already misses, say by how much.

    Telling a customer "it fails" is half an answer. They need the level it
    would have held, so they know whether to shave the prompt or buy a box.
    """
    crushing = Workload(
        alarms_per_day=400,
        storm_size=200,
        storm_window_s=30.0,
        tokens=TokenProfile(alarm_tokens=4000, output_tokens=1500),
    )
    curve = _knee(build, crushing, "storm")
    assert curve.breaks_at is not None
    assert not curve.breaks_at.ok
    if curve.knee is not None:
        assert curve.knee.value < curve.baseline


def test_search_stops_at_the_ceiling_and_says_so(build):
    """An unbreakable axis must be reported as unbreakable, not as a huge knee."""
    trivial = Workload(
        alarms_per_day=1,
        storm_size=0,
        storms_per_day=0,
        sla_seconds=100000.0,
        storm_drain_sla_s=100000.0,
        tokens=TokenProfile(alarm_tokens=1, output_tokens=1),
    )
    curve = _knee(build, trivial, "alarms", max_evaluations=25)
    if curve.hit_ceiling:
        assert curve.breaks_at is None
        assert any("탐색 상한" in n for n in curve.notes)
        assert curve.points[-1].value <= CEILING["alarms"]


def test_evaluation_budget_is_honoured(build):
    curve = _knee(build, Workload(alarms_per_day=120), "prompt", max_evaluations=4)
    assert len(curve.points) <= 4


def test_sweep_ranks_the_axes(build, catalog):
    model, quant, cpu, memory, _ = build
    workload = Workload(alarms_per_day=150)
    curves = sweep_axes(catalog, model, quant, cpu, memory, workload)
    assert set(curves) == set(AXES)

    weakest = weakest_axis(curves)
    if weakest is not None:
        breakable = {
            a: c.headroom for a, c in curves.items()
            if c.knee is not None and not c.hit_ceiling
        }
        assert curves[weakest].headroom == min(breakable.values())


def test_an_overloaded_run_is_still_bucketed(catalog):
    """A day that does not fit in a day must not hang or lose its work.

    The backlog from an overloaded run drains past midnight. Charging those
    segments to a bucket that no longer exists used to spin forever on a
    zero-width slice; it now lands in the last bucket and is counted.
    """
    from svrspec.simulate import simulate
    from svrspec.sizing import decode_table
    from svrspec.timeline import build_timeline, ceilings_for

    eff = Efficiency.from_catalog(catalog.coefficients)
    model = catalog.model("test-8b-gqa")
    quant = catalog.quant("Q4_K_M")
    cpu = catalog.cpu("test-desktop-2ch")
    memory = catalog.memory_for(cpu, 1)
    workload = Workload(
        alarms_per_day=3000,
        tokens=TokenProfile(alarm_tokens=4000, output_tokens=1000),
    )
    candidate = evaluate(model, quant, cpu, memory, eff, workload, 1)
    _, trace = simulate(
        workload,
        prefill_tps=candidate.throughput.prefill_tps,
        decode_by_active=decode_table(model, quant, cpu, memory, workload, 1, eff),
    )
    ceilings = ceilings_for(
        model, quant, cpu, memory, eff, workload,
        candidate.throughput, 1, candidate.memory_gb,
    )
    timeline = build_timeline(trace, ceilings)

    assert timeline.overran_s > 0
    assert any("다음 날로 넘어간다" in n for n in timeline.notes)
    # Work is conserved: no segment time was dropped off the end.
    assert timeline.busy_seconds == pytest.approx(
        sum(s.span_s for s in trace.segments), rel=1e-6
    )
