"""End-to-end sizing: evaluate, sweep, and pick tiers."""

from dataclasses import replace

from svrspec.sizing import evaluate, sweep_cpus, sweep_models, tiers
from svrspec.types import TokenProfile, Workload


def _workload(**kw) -> Workload:
    return replace(Workload(alarms_per_day=150, slots=2, sla_seconds=30.0), **kw)


def test_verdict_uses_the_pessimistic_run(catalog, eff, model_3b, q4):
    cpu = catalog.cpu("test-amx-8ch")
    c = evaluate(model_3b, q4, cpu, catalog.memory_for(cpu), eff, _workload())
    assert c.sim_pessimistic is not None
    # Derated throughput cannot be faster than nominal.
    assert c.sim_pessimistic.p95_s >= c.sim.p95_s


def test_a_capable_server_passes(catalog, eff, model_3b, q4):
    cpu = catalog.cpu("test-amx-8ch")
    c = evaluate(model_3b, q4, cpu, catalog.memory_for(cpu), eff, _workload())
    assert c.verdict == "pass", c.reasons
    assert c.headroom > 1.0


def test_ram_ceiling_fails_before_anything_else_is_considered(catalog, eff, moe):
    """A 64 GB box cannot host a deployment needing more, whatever its speed."""
    cpu = catalog.cpu("test-lowcore-8ch")  # 64 GB ceiling
    f16 = catalog.quant("F16")  # 30.5B unquantised: ~57 GiB of weights alone
    c = evaluate(moe, f16, cpu, catalog.memory_for(cpu), eff, _workload(slots=8))
    assert c.ram.provision_gb > cpu.max_mem_gb
    assert c.verdict == "fail"
    assert any("RAM" in r for r in c.reasons)


def test_an_impossible_sla_fails(catalog, eff, model_8b, q4):
    cpu = catalog.cpu("test-desktop-2ch")
    c = evaluate(model_8b, q4, cpu, catalog.memory_for(cpu), eff,
                 _workload(sla_seconds=1.0))
    assert c.verdict == "fail"
    assert any("p95" in r or "스톰" in r for r in c.reasons)


def test_marginal_sits_between_pass_and_fail(catalog, eff, model_8b, q4):
    """Loosening the SLA must walk the verdict through all three states.

    Storms are off here so the sweep isolates the per-alarm SLA; storm drain is
    judged separately and would otherwise pin every verdict to fail.
    """
    cpu = catalog.cpu("test-desktop-2ch")
    verdicts = []
    for sla in (5, 10, 20, 40, 80, 160, 320):
        c = evaluate(model_8b, q4, cpu, catalog.memory_for(cpu), eff,
                     _workload(sla_seconds=float(sla), storms_per_day=0))
        verdicts.append(c.verdict)
    assert {"fail", "marginal", "pass"} == set(verdicts)
    # And it must be monotonic: no flip-flopping as the budget grows.
    order = {"fail": 0, "marginal": 1, "pass": 2}
    assert [order[v] for v in verdicts] == sorted(order[v] for v in verdicts)


def test_storm_drain_is_judged_separately_from_per_alarm_latency(catalog, eff, model_3b, q4):
    """A burst queues by definition; that must not condemn the steady state.

    Forty alarms in thirty seconds cannot each answer inside thirty seconds, so
    the storm is held to its drain target instead, and the per-alarm SLA is
    measured on normal traffic. Here the server is quick enough that ordinary
    alarms are comfortable, yet the burst still cannot clear in one minute.
    """
    cpu = catalog.cpu("test-amx-8ch")
    c = evaluate(model_3b, q4, cpu, catalog.memory_for(cpu), eff,
                 _workload(sla_seconds=30.0, storm_drain_sla_s=60.0))
    sim = c.sim_pessimistic
    assert sim.p95_steady_s < sim.p95_s  # storm alarms are the slow tail
    assert sim.sla_met  # normal alarms are comfortable
    assert not sim.storm_sla_met  # but the burst cannot drain in a minute
    assert c.verdict == "fail"
    assert any("스톰" in r for r in c.reasons)


def test_sweep_orders_cheapest_first(catalog, eff, model_3b, q4):
    out = sweep_cpus(catalog, model_3b, q4, _workload())
    assert len(out) == 4
    cores = [c.cpu.cores for c in out]
    assert cores == sorted(cores)


def test_sweep_skips_cpus_that_cannot_take_the_socket_count(catalog, eff, model_3b, q4):
    out = sweep_cpus(catalog, model_3b, q4, _workload(), sockets=2)
    ids = {c.cpu.id for c in out}
    assert "test-desktop-2ch" not in ids  # sockets_max is 1
    assert "test-avx512-8ch" in ids


def test_tiers_are_progressively_more_capable(catalog, eff, model_3b, q4):
    out = sweep_cpus(catalog, model_3b, q4, _workload())
    t = tiers(out)
    assert t["minimum"] is not None
    if t["recommended"]:
        assert t["recommended"].verdict == "pass"
    if t["minimum"] and t["comfortable"]:
        from svrspec.sizing import cost_proxy

        assert cost_proxy(t["comfortable"]) >= cost_proxy(t["minimum"])


def test_tiers_return_none_when_nothing_passes(catalog, eff, model_8b, q4):
    out = sweep_cpus(catalog, model_8b, q4, _workload(sla_seconds=0.5))
    t = tiers(out)
    assert t["recommended"] is None


def test_fit_sweep_covers_every_model(catalog, eff, q4):
    cpu = catalog.cpu("test-amx-8ch")
    out = sweep_models(catalog, cpu, q4, _workload())
    assert len(out) == len(catalog.models)
    # Largest first, so a delivery table reads top-down from "too big".
    assert out[0].model.params_b >= out[-1].model.params_b


def test_smaller_model_is_never_slower(catalog, eff, model_8b, model_3b, q4):
    cpu = catalog.cpu("test-amx-8ch")
    memory = catalog.memory_for(cpu)
    big = evaluate(model_8b, q4, cpu, memory, eff, _workload())
    small = evaluate(model_3b, q4, cpu, memory, eff, _workload())
    assert small.latency.total_s < big.latency.total_s
    assert small.memory_gb <= big.memory_gb


def test_moe_needs_more_ram_but_runs_faster(catalog, eff, moe, model_8b, q4):
    cpu = catalog.cpu("test-avx512-8ch")
    memory = catalog.memory_for(cpu)
    m = evaluate(moe, q4, cpu, memory, eff, _workload())
    dense = evaluate(model_8b, q4, cpu, memory, eff, _workload())
    assert m.ram.weights_gb > dense.ram.weights_gb
    assert m.latency.generate_s < dense.latency.generate_s


def test_prompt_cache_changes_the_recommendation_surface(catalog, eff, model_8b, q4):
    cpu = catalog.cpu("test-desktop-2ch")
    memory = catalog.memory_for(cpu)
    cached = evaluate(model_8b, q4, cpu, memory, eff,
                      _workload(tokens=TokenProfile(prompt_cache=True)))
    uncached = evaluate(model_8b, q4, cpu, memory, eff,
                        _workload(tokens=TokenProfile(prompt_cache=False)))
    assert cached.latency.total_s < uncached.latency.total_s
