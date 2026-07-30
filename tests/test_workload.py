from dataclasses import replace

from svrspec.types import TokenProfile, Workload
from svrspec.workload import DAY_SECONDS, generate_arrivals


def test_total_respects_the_daily_budget():
    """Storm alarms come out of the daily total, not on top of it.

    The customer said 100-150 alarms per day including bad days.
    """
    w = Workload(alarms_per_day=150, storm_size=40, storms_per_day=2)
    arrivals = generate_arrivals(w)
    assert len(arrivals) == 150


def test_storms_are_present_and_tagged():
    w = Workload(alarms_per_day=150, storm_size=40, storms_per_day=2)
    arrivals = generate_arrivals(w)
    storm_ids = {a.storm_id for a in arrivals if a.in_storm}
    assert storm_ids == {0, 1}
    assert sum(1 for a in arrivals if a.storm_id == 0) == 40


def test_a_storm_fits_inside_its_window():
    w = Workload(alarms_per_day=150, storm_size=30, storm_window_s=30.0, storms_per_day=1)
    times = [a.at_s for a in generate_arrivals(w) if a.in_storm]
    assert max(times) - min(times) <= 30.0


def test_arrivals_are_sorted_and_inside_the_day():
    arrivals = generate_arrivals(Workload())
    times = [a.at_s for a in arrivals]
    assert times == sorted(times)
    assert all(0 <= t <= DAY_SECONDS for t in times)


def test_business_hours_carry_most_of_the_volume():
    w = Workload(alarms_per_day=200, storms_per_day=0, business_hours=(8, 20),
                 business_share=0.8)
    arrivals = generate_arrivals(w)
    in_hours = sum(1 for a in arrivals if 8 * 3600 <= a.at_s <= 20 * 3600)
    assert in_hours / len(arrivals) > 0.7


def test_deterministic_for_a_seed():
    a = generate_arrivals(Workload(seed=7))
    b = generate_arrivals(Workload(seed=7))
    c = generate_arrivals(Workload(seed=8))
    assert [x.at_s for x in a] == [x.at_s for x in b]
    assert [x.at_s for x in a] != [x.at_s for x in c]


def test_storms_can_be_disabled():
    w = Workload(storms_per_day=0)
    assert not any(a.in_storm for a in generate_arrivals(w))


def test_storm_larger_than_the_daily_budget_is_clamped():
    w = Workload(alarms_per_day=20, storm_size=40, storms_per_day=2)
    arrivals = generate_arrivals(w)
    assert len(arrivals) == 20


def test_token_profile_accounting():
    t = TokenProfile(system_tokens=300, fewshot_tokens=400, alarm_tokens=250, output_tokens=250)
    assert t.prefill_tokens == 950
    assert t.peak_ctx_tokens == 1200
    # With the prefix cached, only the alarm text is prefilled per request.
    assert t.billed_prefill_tokens == 250
    assert replace(t, prompt_cache=False).billed_prefill_tokens == 950
