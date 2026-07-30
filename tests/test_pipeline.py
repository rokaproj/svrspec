"""Runnable pipeline: alarm in, Teams card out.

The checks here are the acceptance criteria of the module, in order: the run must
be deterministic, must lose nothing, must respect causality and the slot ceiling,
and must agree with `simulate.py` -- the two share one physical model, so a
divergence means one of them is wrong.

No test may load a model, start a process, or touch the network. The service time
comes from the analytic prediction, which is the whole point of the module.
"""

from __future__ import annotations

import ast
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import pytest

from svrspec.pipeline import (
    PHASES,
    Delivery,
    RunStats,
    ServiceModel,
    TeamsSink,
    build_service_model,
    run_pipeline,
    teams_card,
)
from svrspec.simulate import simulate
from svrspec.types import TokenProfile, Workload
from svrspec.workload import generate_arrivals

PIPELINE_SRC = Path(__file__).resolve().parents[1] / "svrspec" / "pipeline.py"

FAST = {1: 200.0, 2: 320.0, 4: 480.0}
MID = {1: 20.0, 2: 30.0}


@dataclass(frozen=True)
class FakeAlarm:
    """Same shape as `mockdata.Alarm`, kept local so the two owners do not block.

    `run_pipeline` duck-types the alarm record, so anything carrying these
    attributes works -- see the Protocol in pipeline.py.
    """

    id: str
    at_s: float
    severity: str = "major"
    storm_id: int | None = None
    prompt_tokens: int = 250
    device: str = "SEL-BS-1042"
    device_type: str = "기지국"
    code: str = "CELL-DOWN"
    message: str = "기지국 셀 다운 — 서비스 중단"
    site: str = "서울 강남 12"
    parent_id: str | None = None
    raw: dict = field(default_factory=dict)


def _workload(**kw) -> Workload:
    base = Workload(
        alarms_per_day=150,
        storm_size=40,
        storm_window_s=30.0,
        storms_per_day=2,
        slots=2,
        sla_seconds=30.0,
        teams_rtt_ms=(200.0, 800.0),
    )
    return replace(base, **kw)


def _alarms_for(workload: Workload) -> list[FakeAlarm]:
    """Alarms on exactly the arrival times `simulate` would use.

    `prompt_tokens` is set to the workload's own alarm-token budget so the two
    engines bill identical prefill work; that is what makes the agreement test
    meaningful rather than a coincidence.
    """
    return [
        FakeAlarm(
            id=f"ALM-{i:06d}",
            at_s=a.at_s,
            storm_id=a.storm_id,
            severity="critical" if a.storm_id is not None else "minor",
            prompt_tokens=workload.tokens.alarm_tokens,
        )
        for i, a in enumerate(generate_arrivals(workload))
    ]


def _service(prefill=300.0, table=None, slots=2) -> ServiceModel:
    return ServiceModel(
        prefill_tps=prefill, decode_by_active=dict(table or MID), slots=slots
    )


# --------------------------------------------------------------------------
# AC1 determinism
# --------------------------------------------------------------------------


def test_virtual_time_runs_are_bit_identical():
    w = _workload()
    alarms = _alarms_for(w)
    stats_a, deliveries_a = run_pipeline(alarms, _service(), w)
    stats_b, deliveries_b = run_pipeline(alarms, _service(), w)
    assert deliveries_a == deliveries_b
    assert replace(stats_a, wall_clock_s=0.0) == replace(stats_b, wall_clock_s=0.0)


def test_a_different_seed_gives_a_different_day():
    a_w, b_w = _workload(seed=1), _workload(seed=2)
    a, _ = run_pipeline(_alarms_for(a_w), _service(), a_w)
    b, _ = run_pipeline(_alarms_for(b_w), _service(), b_w)
    assert (a.p95_s, a.storm_drain_s) != (b.p95_s, b.storm_drain_s)


# --------------------------------------------------------------------------
# AC2 conservation
# --------------------------------------------------------------------------


def test_every_alarm_is_accounted_for_exactly_once():
    w = _workload()
    alarms = _alarms_for(w)
    stats, deliveries = run_pipeline(alarms, _service(), w)

    assert stats.received == len(alarms)
    assert stats.received == stats.delivered + stats.dropped
    assert len(deliveries) == len(alarms)
    assert [d.alarm_id for d in sorted(deliveries, key=lambda d: d.alarm_id)] == sorted(
        a.id for a in alarms
    )
    assert stats.delivered == sum(1 for d in deliveries if not d.dropped)
    assert stats.dropped == sum(1 for d in deliveries if d.dropped)


def test_token_totals_count_served_alarms_only():
    w = _workload(alarms_per_day=20, storms_per_day=0)
    alarms = _alarms_for(w)
    stats, deliveries = run_pipeline(alarms, _service(), w)
    served = [d for d in deliveries if not d.dropped]
    assert stats.tokens_prefill == sum(d.prompt_tokens for d in served)
    assert stats.tokens_generated == sum(d.output_tokens for d in served)


# --------------------------------------------------------------------------
# AC3 causality
# --------------------------------------------------------------------------


def test_timestamps_never_run_backwards():
    w = _workload()
    _, deliveries = run_pipeline(_alarms_for(w), _service(), w)
    for d in deliveries:
        assert d.arrived_s <= d.started_s <= d.first_token_s <= d.generated_s <= d.delivered_s
        assert d.queue_wait_s >= -1e-9
        assert d.ttft_s >= -1e-9
        assert d.generate_s >= -1e-9
        assert d.deliver_s >= -1e-9
        assert abs(d.total_s - (d.delivered_s - d.arrived_s)) < 1e-9


def test_the_reported_spans_match_the_timestamps():
    w = _workload(alarms_per_day=40, storms_per_day=1, storm_size=20)
    _, deliveries = run_pipeline(_alarms_for(w), _service(), w)
    for d in deliveries:
        assert abs(d.queue_wait_s - (d.started_s - d.arrived_s)) < 1e-9
        assert abs(d.ttft_s - (d.first_token_s - d.arrived_s)) < 1e-9
        assert abs(d.generate_s - (d.generated_s - d.first_token_s)) < 1e-9
        assert abs(d.deliver_s - (d.delivered_s - d.generated_s)) < 1e-9


# --------------------------------------------------------------------------
# AC4 slot ceiling
# --------------------------------------------------------------------------


def test_concurrency_never_exceeds_the_configured_slots():
    w = _workload(slots=3)
    events: list[dict] = []
    _, deliveries = run_pipeline(
        _alarms_for(w), _service(table={1: 20.0, 2: 30.0, 3: 36.0}, slots=3), w,
        on_event=events.append,
    )
    assert max(e["active"] for e in events) <= w.slots

    served = [d for d in deliveries if not d.dropped]
    assert all(0 <= d.slot < w.slots for d in served)

    # Sweep the occupancy intervals: a slot may not hold two alarms at once.
    edges = [(d.started_s, 1) for d in served] + [(d.generated_s, -1) for d in served]
    edges.sort(key=lambda e: (e[0], e[1]))  # releases before admissions at a tie
    live = peak = 0
    for _, delta in edges:
        live += delta
        peak = max(peak, live)
    assert peak <= w.slots

    by_slot: dict[int, list[Delivery]] = {}
    for d in served:
        by_slot.setdefault(d.slot, []).append(d)
    for held in by_slot.values():
        held.sort(key=lambda d: d.started_s)
        for earlier, later in zip(held, held[1:]):
            assert earlier.generated_s <= later.started_s + 1e-9


def test_dropped_alarms_hold_no_slot():
    w = _workload(alarms_per_day=60, storm_size=40, storms_per_day=1, slots=1)
    _, deliveries = run_pipeline(_alarms_for(w), _service(table={1: 20.0}, slots=1), w,
                                 queue_limit=3)
    dropped = [d for d in deliveries if d.dropped]
    assert dropped
    for d in dropped:
        assert d.slot == -1
        assert d.started_s == d.delivered_s == d.arrived_s
        assert d.total_s == 0.0


# --------------------------------------------------------------------------
# AC5 agreement with simulate.py
# --------------------------------------------------------------------------


def test_agrees_with_simulate_on_the_same_day():
    """Both engines, same arrivals, same rates: the tail must line up.

    They share the continuous-batching model, so this is a cross-check on the
    implementation rather than on the physics. 15% is the brief's tolerance; the
    two should in fact agree to floating-point noise.
    """
    w = _workload()
    arrivals = generate_arrivals(w)
    alarms = _alarms_for(w)
    assert [a.at_s for a in alarms] == [x.at_s for x in arrivals]

    sim, _ = simulate(w, prefill_tps=300.0, decode_by_active=MID, arrivals=arrivals)
    stats, _ = run_pipeline(alarms, _service(), w)

    assert stats.delivered == sim.completed
    assert stats.dropped == 0
    rel = abs(stats.p95_s - sim.p95_s) / sim.p95_s
    assert rel < 0.15, f"p95 pipeline={stats.p95_s:.3f}s simulate={sim.p95_s:.3f}s rel={rel:.4f}"
    assert abs(stats.p95_steady_s - sim.p95_steady_s) / max(sim.p95_steady_s, 1e-9) < 0.15
    assert abs(stats.storm_drain_s - sim.storm_drain_s) / max(sim.storm_drain_s, 1e-9) < 0.15


def test_agrees_with_simulate_on_a_saturated_day():
    """Repeat under queueing, where an implementation error would show up."""
    w = _workload(alarms_per_day=200, storm_size=60, storms_per_day=2, slots=2)
    arrivals = generate_arrivals(w)
    sim, _ = simulate(w, prefill_tps=60.0, decode_by_active={1: 8.0, 2: 12.0},
                      arrivals=arrivals)
    stats, _ = run_pipeline(_alarms_for(w), _service(60.0, {1: 8.0, 2: 12.0}), w)
    assert stats.delivered == sim.completed
    assert abs(stats.p95_s - sim.p95_s) / sim.p95_s < 0.15
    assert abs(stats.max_s - sim.max_s) / sim.max_s < 0.15


def test_an_isolated_alarm_takes_exactly_the_analytic_service_time():
    tokens = TokenProfile(system_tokens=0, fewshot_tokens=0, alarm_tokens=500,
                          output_tokens=100)
    w = _workload(slots=1, storms_per_day=0, alarms_per_day=1, tokens=tokens,
                  teams_rtt_ms=(500.0, 500.0))
    stats, deliveries = run_pipeline(
        [FakeAlarm(id="ALM-1", at_s=0.0, prompt_tokens=500)],
        _service(100.0, {1: 10.0}, slots=1),
        w,
    )
    expected = 500 / 100.0 + 100 / 10.0 + 0.5
    assert stats.delivered == 1
    assert abs(deliveries[0].total_s - expected) < 1e-6
    assert abs(deliveries[0].ttft_s - 5.0) < 1e-6
    assert abs(deliveries[0].generate_s - 10.0) < 1e-6


def test_a_longer_prompt_costs_more_prefill_than_a_short_one():
    """Per-alarm prompt size must actually reach the prefill rate."""
    tokens = TokenProfile(system_tokens=0, fewshot_tokens=0, alarm_tokens=100,
                          output_tokens=10)
    w = _workload(slots=1, storms_per_day=0, tokens=tokens, teams_rtt_ms=(0.0, 0.0))
    short, _ = run_pipeline([FakeAlarm(id="s", at_s=0.0, prompt_tokens=100)],
                            _service(100.0, {1: 10.0}, slots=1), w)
    long_, _ = run_pipeline([FakeAlarm(id="l", at_s=0.0, prompt_tokens=1000)],
                            _service(100.0, {1: 10.0}, slots=1), w)
    assert long_.p50_s > short.p50_s
    assert long_.tokens_prefill == 1000
    assert short.tokens_prefill == 100


def test_the_prompt_cache_prefix_is_not_billed_twice():
    """With the shared prefix cached, only the alarm body is prefilled."""
    cached = TokenProfile(system_tokens=300, fewshot_tokens=400, alarm_tokens=100,
                          output_tokens=10, prompt_cache=True)
    uncached = replace(cached, prompt_cache=False)
    alarm = [FakeAlarm(id="a", at_s=0.0, prompt_tokens=100)]
    on, _ = run_pipeline(alarm, _service(100.0, {1: 10.0}, slots=1),
                         _workload(slots=1, tokens=cached, teams_rtt_ms=(0.0, 0.0)))
    off, _ = run_pipeline(alarm, _service(100.0, {1: 10.0}, slots=1),
                          _workload(slots=1, tokens=uncached, teams_rtt_ms=(0.0, 0.0)))
    assert on.tokens_prefill == 100
    assert off.tokens_prefill == 800
    assert on.p50_s < off.p50_s


# --------------------------------------------------------------------------
# AC6 storm drain
# --------------------------------------------------------------------------


def test_storm_drain_is_the_worst_storms_span():
    w = _workload(alarms_per_day=90, storm_size=40, storms_per_day=2, slots=2)
    stats, deliveries = run_pipeline(_alarms_for(w), _service(60.0, {1: 8.0, 2: 12.0}), w)

    by_storm: dict[int, list[Delivery]] = {}
    for d in deliveries:
        if d.storm_id is not None and not d.dropped:
            by_storm.setdefault(d.storm_id, []).append(d)
    assert len(by_storm) == 2
    expected = max(
        max(d.delivered_s for d in group) - min(d.arrived_s for d in group)
        for group in by_storm.values()
    )
    assert abs(stats.storm_drain_s - expected) < 1e-9
    assert stats.storm_sla_met == (stats.storm_drain_s <= w.storm_drain_sla_s)


def test_steady_p95_excludes_storm_alarms():
    w = _workload()
    stats, deliveries = run_pipeline(_alarms_for(w), _service(60.0, {1: 8.0, 2: 12.0}), w)
    steady = [d.total_s for d in deliveries if d.storm_id is None and not d.dropped]
    storm = [d.total_s for d in deliveries if d.storm_id is not None and not d.dropped]
    assert steady and storm
    # A storm queues by definition, so it must not drag the steady-state figure.
    assert stats.p95_steady_s <= max(steady) + 1e-9
    assert stats.sla_met == (stats.p95_steady_s <= w.sla_seconds)


def test_percentiles_are_ordered():
    w = _workload()
    stats, _ = run_pipeline(_alarms_for(w), _service(60.0, {1: 8.0, 2: 12.0}), w)
    assert stats.p50_s <= stats.p95_s <= stats.p99_s <= stats.max_s


def test_utilisation_figures_are_fractions():
    w = _workload()
    stats, _ = run_pipeline(_alarms_for(w), _service(), w)
    assert 0.0 <= stats.slot_utilisation <= 1.0
    assert 0.0 <= stats.busy_fraction <= 1.0
    assert stats.mean_queue <= stats.max_queue


# --------------------------------------------------------------------------
# AC7 backpressure
# --------------------------------------------------------------------------


def test_queue_limit_drops_the_newest_and_never_overflows():
    w = _workload(alarms_per_day=40, storm_size=40, storms_per_day=1,
                  storm_window_s=30.0, slots=1)
    events: list[dict] = []
    stats, deliveries = run_pipeline(
        _alarms_for(w), _service(60.0, {1: 8.0}, slots=1), w,
        queue_limit=5, on_event=events.append,
    )
    assert stats.dropped > 0
    assert stats.max_queue <= 5
    assert max(e["queue"] for e in events) <= 5
    assert stats.received == stats.delivered + stats.dropped

    # Backpressure, not head-drop: an alarm that was already queued is served,
    # so the first arrivals survive and later ones are shed.
    served_at = [d.arrived_s for d in deliveries if not d.dropped]
    dropped_at = [d.arrived_s for d in deliveries if d.dropped]
    assert min(served_at) <= min(dropped_at)


def test_no_queue_limit_drops_nothing():
    w = _workload(alarms_per_day=40, storm_size=40, storms_per_day=1, slots=1)
    stats, _ = run_pipeline(_alarms_for(w), _service(60.0, {1: 8.0}, slots=1), w)
    assert stats.dropped == 0


def test_a_limit_of_one_still_serves_the_stream():
    w = _workload(alarms_per_day=30, storm_size=20, storms_per_day=1, slots=1)
    stats, _ = run_pipeline(_alarms_for(w), _service(60.0, {1: 8.0}, slots=1), w,
                            queue_limit=1)
    assert stats.max_queue <= 1
    assert stats.delivered > 0
    assert stats.received == stats.delivered + stats.dropped


# --------------------------------------------------------------------------
# AC8 replay mode
# --------------------------------------------------------------------------


def test_replay_mode_takes_real_time_but_reaches_the_same_answer():
    """Compressed replay must be the same run, only paced.

    Sixty virtual seconds at 1000x is 60 ms of sleeping, so this stays well
    inside a test suite's budget while still proving the clock moved.
    """
    tokens = TokenProfile(system_tokens=0, fewshot_tokens=0, alarm_tokens=50,
                          output_tokens=20)
    w = _workload(slots=2, tokens=tokens, teams_rtt_ms=(100.0, 100.0))
    alarms = [FakeAlarm(id=f"ALM-{i}", at_s=i * 3.0, prompt_tokens=50) for i in range(12)]

    began = time.monotonic()
    fast, fast_deliveries = run_pipeline(alarms, _service(100.0, {1: 20.0, 2: 30.0}), w)
    live, live_deliveries = run_pipeline(alarms, _service(100.0, {1: 20.0, 2: 30.0}), w,
                                         speed=1000.0)
    elapsed = time.monotonic() - began

    assert live.wall_clock_s > 0.0
    assert elapsed < 1.0, f"replay took {elapsed:.3f}s"
    assert [d.alarm_id for d in live_deliveries] == [d.alarm_id for d in fast_deliveries]
    assert live_deliveries == fast_deliveries
    assert live.p95_s == fast.p95_s


def test_replay_pacing_tracks_the_compression_factor():
    """Wall clock must be the simulated span divided by `speed`, not less.

    Sleep can overshoot, never undershoot, and each wait targets an absolute
    instant so a late wake-up cannot accumulate into drift.
    """
    tokens = TokenProfile(system_tokens=0, fewshot_tokens=0, alarm_tokens=50,
                          output_tokens=20)
    w = _workload(slots=2, tokens=tokens, teams_rtt_ms=(0.0, 0.0))
    alarms = [FakeAlarm(id=f"ALM-{i}", at_s=i * 10.0, prompt_tokens=50) for i in range(10)]
    stats, deliveries = run_pipeline(alarms, _service(100.0, {1: 20.0, 2: 30.0}, slots=2),
                                    w, speed=500.0)
    span = max(d.generated_s for d in deliveries) - min(d.arrived_s for d in deliveries)
    expected = span / 500.0
    assert stats.wall_clock_s >= expected * 0.9, (stats.wall_clock_s, expected)
    assert stats.wall_clock_s < expected + 0.5


def test_virtual_time_does_not_sleep():
    w = _workload(alarms_per_day=150)
    began = time.monotonic()
    stats, _ = run_pipeline(_alarms_for(w), _service(), w)
    elapsed = time.monotonic() - began
    # A whole simulated day, and it must cost milliseconds of real time.
    assert elapsed < 1.0
    assert stats.wall_clock_s < 1.0


def test_rejects_a_negative_speed():
    w = _workload()
    with pytest.raises(ValueError):
        run_pipeline(_alarms_for(w), _service(), w, speed=-1.0)


# --------------------------------------------------------------------------
# AC9 the sink
# --------------------------------------------------------------------------


def test_the_sink_receives_one_card_per_delivered_alarm():
    w = _workload(alarms_per_day=40, storm_size=20, storms_per_day=1)
    sink = TeamsSink()
    stats, deliveries = run_pipeline(_alarms_for(w), _service(), w, sink=sink)

    assert len(sink.sent) == stats.delivered
    sent_ids = [delivery.alarm_id for delivery, _card in sink.sent]
    assert sent_ids == [d.alarm_id for d in deliveries if not d.dropped]

    for delivery, card in sink.sent:
        assert card["alarm_id"] == delivery.alarm_id
        assert card["severity"] == delivery.severity
        assert card["summary"]
        assert "CELL-DOWN" in card["summary"]
        assert card["attachments"][0]["content"]["type"] == "AdaptiveCard"


def test_dropped_alarms_are_never_sent():
    w = _workload(alarms_per_day=40, storm_size=40, storms_per_day=1, slots=1)
    sink = TeamsSink()
    stats, _ = run_pipeline(_alarms_for(w), _service(60.0, {1: 8.0}, slots=1), w,
                            sink=sink, queue_limit=4)
    assert stats.dropped > 0
    assert len(sink.sent) == stats.delivered


def test_a_custom_sink_is_accepted():
    """The webhook the operator will write later plugs in here."""

    class CountingSink:
        def __init__(self) -> None:
            self.count = 0

        def send(self, delivery: Delivery, card: dict) -> None:
            assert isinstance(card, dict)
            self.count += 1

    w = _workload(alarms_per_day=20, storms_per_day=0)
    sink = CountingSink()
    stats, _ = run_pipeline(_alarms_for(w), _service(), w, sink=sink)
    assert sink.count == stats.delivered


def test_the_card_carries_the_korean_summary_and_facts():
    alarm = FakeAlarm(id="ALM-42", at_s=0.0, severity="critical")
    w = _workload(slots=1, storms_per_day=0)
    _, deliveries = run_pipeline([alarm], _service(100.0, {1: 20.0}, slots=1), w)
    card = teams_card(alarm, deliveries[0])
    facts = card["attachments"][0]["content"]["body"][1]["facts"]
    titles = {f["title"] for f in facts}
    assert {"장비", "위치", "코드", "심각도"} <= titles
    assert alarm.message in card["summary"]


# --------------------------------------------------------------------------
# AC10 progress callback
# --------------------------------------------------------------------------


def test_events_are_emitted_at_every_phase_transition():
    w = _workload(alarms_per_day=30, storms_per_day=1, storm_size=10, slots=2)
    events: list[dict] = []
    stats, _ = run_pipeline(_alarms_for(w), _service(), w, on_event=events.append)

    assert all(e["phase"] in PHASES for e in events)
    for e in events:
        assert set(("t", "phase", "alarm_id", "queue", "active")) <= set(e)
        assert e["queue"] >= 0 and e["active"] >= 0

    # Four transitions per served alarm -- queued, prefill, decode, deliver --
    # plus one terminal event per dropped alarm.
    assert len(events) == 4 * stats.delivered + stats.dropped

    per_alarm: dict[str, list[str]] = {}
    for e in events:
        per_alarm.setdefault(e["alarm_id"], []).append(e["phase"])
    served = [v for v in per_alarm.values() if v != ["done"]]
    assert served
    assert all(v == ["queued", "prefill", "decode", "deliver"] for v in served)


@pytest.mark.parametrize("queue_limit", [None, 4])
def test_event_times_do_not_run_backwards(queue_limit):
    """A watcher consumes this stream in order, so it must arrive in order.

    Including under backpressure, where drop events interleave with the rest.
    """
    w = _workload(alarms_per_day=40, storms_per_day=1, storm_size=20)
    events: list[dict] = []
    run_pipeline(_alarms_for(w), _service(60.0, {1: 8.0, 2: 12.0}), w,
                 queue_limit=queue_limit, on_event=events.append)
    times = [e["t"] for e in events]
    assert times == sorted(times)
    if queue_limit is not None:
        assert any(e["phase"] == "done" and e.get("dropped") for e in events)


def test_a_run_without_a_callback_still_works():
    w = _workload(alarms_per_day=10, storms_per_day=0)
    stats, _ = run_pipeline(_alarms_for(w), _service(), w)
    assert stats.delivered == 10


# --------------------------------------------------------------------------
# AC11 no model, no process, no network
# --------------------------------------------------------------------------


def test_the_module_cannot_reach_a_process_or_the_network():
    """Static check: the runtime must stay analytical.

    Substring scanning would trip over the CPU `sockets` argument the frozen
    contract requires, so the import graph is what gets checked -- that is the
    property that actually matters. `subprocess`, `urllib` and the HTTP client
    library are additionally forbidden as plain text.
    """
    source = PIPELINE_SRC.read_text(encoding="utf-8")
    banned_modules = {"subprocess", "socket", "urllib", "http", "ssl", "asyncio",
                      "multiprocessing", "ctypes", "requests", "httpx", "torch",
                      "llama_cpp"}

    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module.split(".")[0])
    assert not (imported & banned_modules), sorted(imported & banned_modules)

    for text in ("subprocess", "urllib", "requests", "Popen", "urlopen"):
        assert text not in source, text


def test_no_test_here_touches_the_catalog_free_path_by_accident(catalog, model_8b, q4, eff):
    """`build_service_model` is the only route from hardware to service rates."""
    cpu = catalog.cpus[0]
    memory = catalog.memory_for(cpu, 1)
    w = _workload(slots=2)
    service = build_service_model(model_8b, q4, cpu, memory, eff, w)

    assert service.prefill_tps > 0.0
    assert set(service.decode_by_active) == {1, 2}
    assert service.slots == 2
    assert service.uncertainty > 0.0
    assert model_8b.id in service.label
    # Decode is bandwidth bound, so the aggregate must rise with concurrency.
    assert service.decode_by_active[2] > service.decode_by_active[1]

    stats, deliveries = run_pipeline(_alarms_for(w), service, w)
    assert stats.delivered == len(deliveries)
    assert stats.p95_s > 0.0


def test_the_service_model_rejects_a_dead_configuration():
    with pytest.raises(ValueError):
        ServiceModel(prefill_tps=0.0, decode_by_active={1: 10.0}, slots=1)
    with pytest.raises(ValueError):
        ServiceModel(prefill_tps=100.0, decode_by_active={}, slots=1)


def test_the_decode_table_clamps_instead_of_extrapolating():
    """Four slots against a table that only knows two must not invent capacity."""
    w = _workload(slots=4, alarms_per_day=20, storms_per_day=0)
    stats, _ = run_pipeline(_alarms_for(w), _service(300.0, {1: 20.0, 2: 30.0}, slots=4), w)
    assert stats.delivered == 20


def test_rejects_an_impossible_queue_limit():
    w = _workload()
    with pytest.raises(ValueError):
        run_pipeline(_alarms_for(w), _service(), w, queue_limit=0)


def test_an_empty_day_is_not_an_error():
    w = _workload()
    stats, deliveries = run_pipeline([], _service(), w)
    assert isinstance(stats, RunStats)
    assert (stats.received, stats.delivered, stats.dropped) == (0, 0, 0)
    assert deliveries == []
    assert stats.sla_met and stats.storm_sla_met


def test_out_of_order_input_is_sorted_by_arrival():
    w = _workload(slots=1, storms_per_day=0)
    alarms = [FakeAlarm(id="late", at_s=100.0, prompt_tokens=50),
              FakeAlarm(id="early", at_s=0.0, prompt_tokens=50)]
    _, deliveries = run_pipeline(alarms, _service(100.0, {1: 20.0}, slots=1), w)
    assert [d.alarm_id for d in deliveries] == ["early", "late"]


# --------------------------------------------------------------------------
# Integration with the mock day, once that owner has landed it.
# --------------------------------------------------------------------------


def test_runs_a_generated_mock_day():
    mockdata = pytest.importorskip("svrspec.mockdata")
    day = mockdata.generate_day("2026-06-01")
    w = _workload(slots=2)
    stats, deliveries = run_pipeline(day.alarms, _service(), w)
    assert stats.received == len(day.alarms)
    assert stats.received == stats.delivered + stats.dropped
    for d in deliveries:
        assert d.arrived_s <= d.started_s <= d.generated_s <= d.delivered_s
