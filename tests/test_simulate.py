"""Queue simulation.

Checked against closed-form expectations where they exist: an unloaded server
must return the bare service time, and a saturated one must drain a burst at the
service rate.
"""

from dataclasses import replace

import pytest

from svrspec.simulate import simulate
from svrspec.types import TokenProfile, Workload
from svrspec.workload import Arrival, generate_arrivals

FAST = {1: 200.0, 2: 320.0, 4: 480.0}
SLOW = {1: 4.0, 2: 6.0}


def _workload(**kw) -> Workload:
    base = Workload(
        alarms_per_day=150,
        storm_size=40,
        storm_window_s=30.0,
        storms_per_day=2,
        slots=2,
        sla_seconds=30.0,
        teams_rtt_ms=(200.0, 200.0),
    )
    return replace(base, **kw)


def test_is_deterministic_for_a_fixed_seed():
    w = _workload()
    a, _ = simulate(w, prefill_tps=300.0, decode_by_active=FAST)
    b, _ = simulate(w, prefill_tps=300.0, decode_by_active=FAST)
    assert a == b


def test_different_seeds_give_different_days():
    a, _ = simulate(_workload(seed=1), 300.0, FAST)
    b, _ = simulate(_workload(seed=2), 300.0, FAST)
    assert (a.p95_s, a.storm_drain_s) != (b.p95_s, b.storm_drain_s)


def test_single_isolated_alarm_takes_exactly_the_service_time():
    """With no contention, latency must equal prefill + decode + Teams RTT."""
    tokens = TokenProfile(system_tokens=0, fewshot_tokens=0, alarm_tokens=500, output_tokens=100)
    w = _workload(slots=1, storms_per_day=0, alarms_per_day=1, tokens=tokens,
                  teams_rtt_ms=(500.0, 500.0))
    result, _ = simulate(w, prefill_tps=100.0, decode_by_active={1: 10.0},
                         arrivals=[Arrival(at_s=0.0)])
    expected = 500 / 100.0 + 100 / 10.0 + 0.5  # 5 + 10 + 0.5
    assert result.completed == 1
    assert abs(result.p50_s - expected) < 1e-6


def test_queueing_appears_only_when_arrivals_outpace_service():
    tokens = TokenProfile(system_tokens=0, fewshot_tokens=0, alarm_tokens=100, output_tokens=100)
    # Two alarms at once, one slot: the second waits out the first in full.
    w = _workload(slots=1, storms_per_day=0, tokens=tokens, teams_rtt_ms=(0.0, 0.0))
    result, trace = simulate(
        w, prefill_tps=100.0, decode_by_active={1: 10.0},
        arrivals=[Arrival(at_s=0.0), Arrival(at_s=0.0)],
    )
    service = 100 / 100.0 + 100 / 10.0  # 11 s
    first, second = sorted(trace.latencies)
    assert abs(first - service) < 1e-6
    assert abs(second - 2 * service) < 1e-6
    assert result.max_queue_depth >= 1


def test_storm_drain_time_is_reported_and_bounded_by_capacity():
    tokens = TokenProfile(system_tokens=0, fewshot_tokens=0, alarm_tokens=100, output_tokens=100)
    w = _workload(slots=1, alarms_per_day=40, storm_size=40, storms_per_day=1,
                  storm_window_s=10.0, tokens=tokens, teams_rtt_ms=(0.0, 0.0))
    result, _ = simulate(w, prefill_tps=100.0, decode_by_active={1: 10.0})
    # 40 alarms x 11 s of work on one slot cannot finish faster than 440 s.
    assert result.storm_drain_s >= 430
    assert result.completed == 40


def test_more_slots_drain_a_storm_faster():
    tokens = TokenProfile(system_tokens=0, fewshot_tokens=0, alarm_tokens=100, output_tokens=100)
    base = _workload(alarms_per_day=40, storm_size=40, storms_per_day=1,
                     storm_window_s=10.0, tokens=tokens, teams_rtt_ms=(0.0, 0.0))
    one, _ = simulate(replace(base, slots=1), 100.0, {1: 10.0})
    four, _ = simulate(replace(base, slots=4), 100.0, {1: 10.0, 2: 16.0, 4: 24.0})
    assert four.storm_drain_s < one.storm_drain_s


def test_slot_utilisation_is_a_fraction():
    result, _ = simulate(_workload(), 300.0, FAST)
    assert 0.0 <= result.slot_utilisation <= 1.0


def test_a_slow_server_misses_the_sla_and_says_so():
    result, _ = simulate(_workload(), prefill_tps=8.0, decode_by_active=SLOW)
    assert not result.sla_met
    assert result.p95_s > 30.0


def test_a_fast_server_meets_the_sla():
    result, _ = simulate(_workload(), prefill_tps=400.0, decode_by_active=FAST)
    assert result.sla_met
    assert result.p95_s < 30.0


def test_percentiles_are_ordered():
    result, _ = simulate(_workload(), 60.0, {1: 20.0, 2: 30.0})
    assert result.p50_s <= result.p95_s <= result.p99_s <= result.max_s


def test_every_alarm_completes():
    w = _workload()
    arrivals = generate_arrivals(w)
    result, trace = simulate(w, 300.0, FAST, arrivals=arrivals)
    assert result.completed == len(arrivals)
    assert len(trace.latencies) == len(arrivals)


def test_prompt_cache_shortens_latency():
    cached = _workload(tokens=TokenProfile(prompt_cache=True))
    uncached = _workload(tokens=TokenProfile(prompt_cache=False))
    a, _ = simulate(cached, 40.0, {1: 12.0, 2: 18.0})
    b, _ = simulate(uncached, 40.0, {1: 12.0, 2: 18.0})
    assert a.p95_s < b.p95_s


def test_rejects_a_zero_rate_configuration():
    with pytest.raises(ValueError):
        simulate(_workload(), prefill_tps=0.0, decode_by_active=FAST)
    with pytest.raises(ValueError):
        simulate(_workload(), prefill_tps=100.0, decode_by_active={})
    with pytest.raises(ValueError, match="slots"):
        simulate(_workload(slots=0), prefill_tps=100.0, decode_by_active=FAST)


def test_decode_table_clamps_instead_of_extrapolating():
    # Table only knows 1 and 2 active; asking for 4 slots must not explode.
    w = _workload(slots=4)
    result, _ = simulate(w, 300.0, {1: 20.0, 2: 30.0})
    assert result.completed > 0
