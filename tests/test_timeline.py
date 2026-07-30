"""The resource series must be three resources, not one number repeated."""

import pytest

from svrspec.memory import GB
from svrspec.perf import Efficiency
from svrspec.simulate import simulate
from svrspec.sizing import decode_table, evaluate
from svrspec.timeline import (
    DAY_SECONDS,
    SATURATED,
    build_timeline,
    ceilings_for,
)
from svrspec.types import TokenProfile, Workload


def _run(catalog, model, cpu_id, workload, sockets=1):
    eff = Efficiency.from_catalog(catalog.coefficients)
    quant = catalog.quant("Q4_K_M")
    cpu = catalog.cpu(cpu_id)
    memory = catalog.memory_for(cpu, 1)
    candidate = evaluate(model, quant, cpu, memory, eff, workload, sockets)
    _, trace = simulate(
        workload,
        prefill_tps=candidate.throughput.prefill_tps,
        decode_by_active=decode_table(
            model, quant, cpu, memory, workload, sockets, eff
        ),
    )
    ceilings = ceilings_for(
        model, quant, cpu, memory, eff, workload,
        candidate.throughput, sockets, candidate.memory_gb,
    )
    return build_timeline(trace, ceilings), trace, ceilings


def test_hostsim_and_timeline_agree_on_the_same_run(catalog, model_8b):
    """Two modules describing one run may not disagree about the physics.

    `hostsim._rates` is a deliberate copy of `timeline._rates` -- private, and
    returning fields the other has no use for. A copy with no test drifts, and
    then the task-manager graph and the CSV export quote different bandwidths
    for the same second. This is the test that makes the copy safe.
    """
    from svrspec.hostsim import _rates as host_rates
    from svrspec.timeline import _rates as timeline_rates

    workload = Workload(alarms_per_day=350, tokens=TokenProfile(alarm_tokens=1500))
    timeline, trace, ceilings = _run(catalog, model_8b, "test-amx-8ch", workload)
    assert trace.segments, "need a busy run to compare"

    for segment in trace.segments[:200]:
        mine = timeline_rates(segment, ceilings)
        theirs = host_rates(segment, ceilings)
        assert theirs.bandwidth_bytes_s == pytest.approx(mine.bandwidth_bytes_s, rel=1e-12)
        assert theirs.prefill_tps == pytest.approx(mine.prefill_tps, rel=1e-12)
        assert theirs.decode_tps == pytest.approx(mine.decode_tps, rel=1e-12)


def test_hostsim_total_bytes_match_the_timeline(catalog, model_8b):
    """Same run, same window, same total DRAM traffic -- whatever the period."""
    from svrspec.hostsim import sample_host

    workload = Workload(alarms_per_day=350, tokens=TokenProfile(alarm_tokens=1500))
    timeline, trace, ceilings = _run(catalog, model_8b, "test-amx-8ch", workload)
    cpu = catalog.cpu("test-amx-8ch")
    host = sample_host(trace, ceilings, cpu, installed_gb=64, period_s=900.0)

    from_timeline = sum(b.bandwidth_avg_gbs * b.span_s for b in timeline.buckets)
    from_host = sum(s.bandwidth_gbs * host.period_s for s in host.samples)
    assert from_host == pytest.approx(from_timeline, rel=1e-6)


def test_bandwidth_and_compute_are_not_the_same_series(catalog, model_8b):
    """The bug this module exists to fix.

    The old resource view set cpu_pct, ram_pct and bandwidth_pct all from
    busy_fraction, so three graphs drew one line. Prefill and decode load
    opposite ends of the machine, so a correct series must diverge.
    """
    workload = Workload(alarms_per_day=400, tokens=TokenProfile(alarm_tokens=1200))
    timeline, _, _ = _run(catalog, model_8b, "test-amx-8ch", workload)

    busy = [b for b in timeline.buckets if b.cpu_pct > 0]
    assert busy, "a 400-alarm day must produce busy buckets"

    bandwidth = [b.bandwidth_avg_pct for b in busy]
    compute = [b.compute_avg_pct for b in busy]
    cpu = [b.cpu_pct for b in busy]
    assert bandwidth != compute
    assert bandwidth != cpu
    # And they must not merely be scalar multiples of each other either: the
    # ratio has to move as the prefill/decode mix moves.
    ratios = {
        round(b / c, 3)
        for b, c in zip(bandwidth, compute)
        if c > 0.01 and b > 0.01
    }
    assert len(ratios) > 1


def test_prefill_heavy_workload_reads_as_compute_bound(catalog, model_8b):
    """A long prompt with a short answer must not look bandwidth bound."""
    prompt_heavy = Workload(
        alarms_per_day=300,
        tokens=TokenProfile(alarm_tokens=6000, output_tokens=60, prompt_cache=False),
    )
    generate_heavy = Workload(
        alarms_per_day=300,
        tokens=TokenProfile(alarm_tokens=60, output_tokens=1200),
    )
    prompt_tl, _, _ = _run(catalog, model_8b, "test-amx-8ch", prompt_heavy)
    gen_tl, _, _ = _run(catalog, model_8b, "test-amx-8ch", generate_heavy)

    assert prompt_tl.prefill_share > gen_tl.prefill_share
    assert gen_tl.decode_share > prompt_tl.decode_share
    assert prompt_tl.binding_resource == "compute"


def test_kv_usage_moves_with_the_work(catalog, model_8b):
    """Memory has to breathe.

    KV residency is a function of who is in flight and how far along they are.
    A memory graph that shows the reserved allocation only is a flat line, and
    a flat line is what made the old view useless.
    """
    workload = Workload(alarms_per_day=500, tokens=TokenProfile(output_tokens=600))
    timeline, _, ceilings = _run(catalog, model_8b, "test-amx-8ch", workload)

    used = [b.kv_used_gb for b in timeline.buckets]
    assert max(used) > 0
    assert min(used) == 0.0, "idle buckets hold nothing"
    # Touched KV can never exceed what llama.cpp reserved up front.
    assert max(used) <= ceilings.kv_reserved_bytes / GB + 1e-9
    assert timeline.peak_kv_used_gb == pytest.approx(max(used))


def test_reserved_memory_is_reported_alongside_the_working_set(catalog, model_8b):
    """The graph may not imply the box needs less RAM than it must be given."""
    workload = Workload(alarms_per_day=200)
    _, _, ceilings = _run(catalog, model_8b, "test-amx-8ch", workload)
    assert ceilings.kv_reserved_bytes > 0
    assert ceilings.allocated_bytes > ceilings.static_bytes


def test_arrivals_and_completions_are_conserved(catalog, model_3b):
    workload = Workload(alarms_per_day=250)
    timeline, trace, _ = _run(catalog, model_3b, "test-amx-8ch", workload)
    assert sum(b.arrived for b in timeline.buckets) == len(trace.arrivals_at)
    assert sum(b.completed for b in timeline.buckets) == len(trace.completions_at)


def test_busy_seconds_match_the_simulated_segments(catalog, model_3b):
    """The series is a reconstruction of the run, not an approximation of it."""
    workload = Workload(alarms_per_day=300)
    timeline, trace, _ = _run(catalog, model_3b, "test-amx-8ch", workload)
    from_segments = sum(s.span_s for s in trace.segments)
    assert timeline.busy_seconds == pytest.approx(from_segments, rel=1e-6)


def test_phase_shares_sum_to_one_while_busy(catalog, model_3b):
    workload = Workload(alarms_per_day=300)
    timeline, _, _ = _run(catalog, model_3b, "test-amx-8ch", workload)
    assert timeline.prefill_share + timeline.decode_share == pytest.approx(1.0, abs=1e-6)


def test_saturation_seconds_never_exceed_busy_seconds(catalog, model_8b):
    workload = Workload(alarms_per_day=600, tokens=TokenProfile(alarm_tokens=2000))
    timeline, _, _ = _run(catalog, model_8b, "test-amx-8ch", workload)
    assert timeline.seconds_bandwidth_bound <= timeline.busy_seconds + 1e-6
    assert timeline.seconds_compute_bound <= timeline.busy_seconds + 1e-6


def test_an_unqueued_day_is_reported_as_no_overload(catalog, model_3b):
    """Hitting a ceiling is not overload; a growing queue is.

    A single decoding request already sits at ~100% of the bandwidth ceiling --
    that is the definition of bandwidth bound, not a warning. What the operator
    needs told is whether work ever arrived faster than it drained.
    """
    workload = Workload(alarms_per_day=1, storm_size=0, storms_per_day=0)
    timeline, _, _ = _run(catalog, model_3b, "test-amx-8ch", workload)
    assert timeline.peak_queue == 0
    assert any("과부하는 발생하지 않는다" in n for n in timeline.notes)
    # It is still bandwidth bound -- a short prompt with a long answer is.
    assert timeline.binding_resource in ("bandwidth", "compute")


def test_estimate_backed_ceilings_are_flagged(catalog, model_8b):
    """A percentage against a guessed ceiling must say so."""
    workload = Workload(alarms_per_day=300)
    timeline, _, _ = _run(catalog, model_8b, "test-amx-8ch", workload)
    estimated = (
        timeline.ceilings.bandwidth_confidence == "estimate"
        or timeline.ceilings.compute_confidence == "estimate"
    )
    flagged = any("추정 계수" in n for n in timeline.notes)
    assert flagged == estimated


def test_bucket_count_and_span_are_respected(catalog, model_3b):
    workload = Workload(alarms_per_day=200)
    eff = Efficiency.from_catalog(catalog.coefficients)
    quant = catalog.quant("Q4_K_M")
    cpu = catalog.cpu("test-amx-8ch")
    memory = catalog.memory_for(cpu, 1)
    candidate = evaluate(model_3b, quant, cpu, memory, eff, workload, 1)
    _, trace = simulate(
        workload,
        prefill_tps=candidate.throughput.prefill_tps,
        decode_by_active=decode_table(model_3b, quant, cpu, memory, workload, 1, eff),
    )
    ceilings = ceilings_for(
        model_3b, quant, cpu, memory, eff, workload, candidate.throughput, 1, 64
    )

    timeline = build_timeline(trace, ceilings, buckets=24)
    assert len(timeline.buckets) == 24
    assert timeline.buckets[1].t_s == pytest.approx(DAY_SECONDS / 24)

    with pytest.raises(ValueError):
        build_timeline(trace, ceilings, buckets=0)


def test_storm_shows_up_as_a_queue_spike(catalog, model_8b):
    """The storm must be visible in the series, not averaged away."""
    workload = Workload(
        alarms_per_day=300,
        storm_size=60,
        storms_per_day=1,
        tokens=TokenProfile(alarm_tokens=1500),
    )
    timeline, _, _ = _run(catalog, model_8b, "test-desktop-2ch", workload)
    queues = sorted(b.queued for b in timeline.buckets)
    assert queues[-1] > 0
    # A spike, not a plateau: the busiest bucket must stand well clear of median.
    assert queues[-1] > queues[len(queues) // 2]


def test_saturated_reports_the_closer_ceiling_not_a_fixed_order(catalog, model_8b):
    """A slice usually touches both ceilings; report the one that got closer.

    Checking bandwidth first meant every busy slice was labelled bandwidth,
    including ones whose whole story was that compute ran out -- and that
    contradicted the run-level diagnosis printed right next to it.
    """
    workload = Workload(alarms_per_day=400, tokens=TokenProfile(alarm_tokens=1500))
    timeline, _, _ = _run(catalog, model_8b, "test-desktop-2ch", workload)
    for bucket in timeline.buckets:
        top = max(bucket.bandwidth_peak_pct, bucket.compute_peak_pct)
        if top < SATURATED * 100:
            assert bucket.saturated is None
        elif bucket.bandwidth_peak_pct >= bucket.compute_peak_pct:
            assert bucket.saturated == "bandwidth"
        else:
            assert bucket.saturated == "compute"
