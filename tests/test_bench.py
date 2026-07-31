"""Frames have to be a faithful fold of the run, and the same fold twice.

The bench screen is the one place an operator watches a number move, which is
exactly where a plausible-looking reconstruction does the most damage. So the
tests here are mostly conservation laws: every arrival appears in some frame,
every delivery appears in some frame, no frame claims more requests in flight
than the build has slots, and the running p95 in the last frame is the same
number the pipeline reports for the whole run.

The other half is provenance. `bench` must not own a copy of the formula that
turns work into bytes -- `timeline.segment_rates` is the single definition, and
a test proves the call actually happens rather than trusting the import.

The build under test is a stub with the shape of `lab.Assembly`. That is
deliberate: `bench` reads a handful of fields off it and must not care which
module produced them, and these tests must not fail because a sibling owner is
mid-edit.
"""

import ast
import copy
import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from svrspec import bench
from svrspec.bench import Frame, frames_to_csv, run_bench
from svrspec.loadgen import build_load
from svrspec.perf import Efficiency, predict_throughput
from svrspec.pipeline import build_service_model
from svrspec.types import TokenProfile, Workload

BENCH_SRC = Path(__file__).resolve().parents[1] / "svrspec" / "bench.py"


# --------------------------------------------------------------------------
# A stand-in for lab.Assembly -- the fields `bench` actually reads
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class StubVm:
    name: str = "테스트 조립"
    sockets: int = 1
    slots: int = 2


@dataclass(frozen=True)
class StubAssembly:
    vm: StubVm
    cpu: object
    memory: object
    model: object
    quant: object
    channels_total: int
    channels_populated: int
    ram_total_gb: int
    ram_used_gb: float
    bandwidth_gbs: float
    bandwidth_full_gbs: float
    prefill_tps: float
    decode_tps_single: float
    uncertainty: float
    findings: list = field(default_factory=list)


def _build(catalog, cpu_id="test-amx-8ch", slots=2, sockets=1, findings=()):
    """(assembly, workload, service) for one build, from the test catalogue."""
    eff = Efficiency.from_catalog(catalog.coefficients)
    cpu = catalog.cpu(cpu_id)
    memory = catalog.memory_for(cpu, 1)
    model = catalog.model("test-8b-gqa")
    quant = catalog.quant("Q4_K_M")
    workload = Workload(slots=slots, tokens=TokenProfile())
    prediction = predict_throughput(
        model, quant, cpu, memory, workload.tokens, eff, slots=slots, sockets=sockets
    )
    assembly = StubAssembly(
        vm=StubVm(sockets=sockets, slots=slots),
        cpu=cpu,
        memory=memory,
        model=model,
        quant=quant,
        channels_total=cpu.mem_channels * sockets,
        channels_populated=cpu.mem_channels * sockets,
        ram_total_gb=128,
        ram_used_gb=8.0,
        bandwidth_gbs=prediction.effective_bandwidth_gbs,
        bandwidth_full_gbs=prediction.effective_bandwidth_gbs,
        prefill_tps=prediction.prefill_tps,
        decode_tps_single=prediction.decode_tps_single,
        uncertainty=prediction.uncertainty,
        findings=list(findings),
    )
    service = build_service_model(model, quant, cpu, memory, eff, workload, sockets)
    return assembly, workload, service


def _bench(catalog, kind="replay", *, cpu_id="test-amx-8ch", slots=2, frames=600,
           load=None, **kw):
    assembly, workload, service = _build(catalog, cpu_id, slots)
    alarms, profile = build_load(kind, **(load or {}))
    result = run_bench(
        None, assembly, alarms, profile,
        workload=workload, frames=frames, service=service,
        ceilings=_ceilings(catalog, assembly, workload), **kw,
    )
    return result, workload


def _ceilings(catalog, assembly, workload):
    """The same ceilings `run_bench` would derive, built through the public route."""
    return bench._ceilings_for(catalog, assembly, workload)


# --------------------------------------------------------------------------
# AC5 the frame series is the size that was asked for
# --------------------------------------------------------------------------


@pytest.mark.parametrize("count", [1, 7, 120, 600])
def test_frame_count_is_exactly_what_was_asked_for(catalog, count):
    result, _ = _bench(catalog, frames=count)
    assert len(result.frames) == count
    assert all(isinstance(f, Frame) for f in result.frames)


@pytest.mark.parametrize("count", [0, -1, -600])
def test_a_zero_or_negative_frame_count_is_an_error(catalog, count):
    with pytest.raises(ValueError, match="frames"):
        _bench(catalog, frames=count)


def test_a_negative_worst_n_is_an_error(catalog):
    with pytest.raises(ValueError, match="worst_n"):
        _bench(catalog, worst_n=-1)


# --------------------------------------------------------------------------
# AC6 the fold conserves the run
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["replay", "ramp", "spike", "soak"])
def test_frames_conserve_every_arrival_and_every_delivery(catalog, kind):
    result, _ = _bench(catalog, kind, slots=4)
    assert sum(f.arrived for f in result.frames) == result.stats.received
    assert sum(f.delivered for f in result.frames) == result.stats.delivered


def test_frames_conserve_arrivals_even_when_the_queue_sheds(catalog):
    """A dropped alarm still arrived. It must not vanish from the chart."""
    result, _ = _bench(catalog, "ramp", cpu_id="test-desktop-2ch", slots=1,
                       queue_limit=5)
    assert result.stats.dropped > 0
    assert sum(f.arrived for f in result.frames) == result.stats.received
    assert sum(f.delivered for f in result.frames) == result.stats.delivered
    assert any(str(result.stats.dropped) in note and "버" in note
               for note in result.notes), result.notes


def test_every_arrival_and_delivery_lands_in_the_frame_it_happened_in(catalog):
    """Conserving the totals is not enough -- a frame off by one still sums.

    The counts are recomputed straight from a second, identical `run_pipeline`
    (it is deterministic in virtual time) and compared frame by frame, so a
    shifted index shows up as a misplaced spike rather than hiding in the sum.
    """
    from svrspec.pipeline import run_pipeline

    assembly, workload, service = _build(catalog, slots=4)
    alarms, profile = build_load("replay")
    count = 48
    result = run_bench(
        None, assembly, alarms, profile, workload=workload, frames=count,
        service=service, ceilings=_ceilings(catalog, assembly, workload),
    )
    _, deliveries = run_pipeline(alarms, service, workload, speed=0.0)

    width = result.frames[1].t_s - result.frames[0].t_s
    arrived = [0] * count
    delivered = [0] * count
    for d in deliveries:
        arrived[min(count - 1, int(d.arrived_s / width))] += 1
        if not d.dropped:
            delivered[min(count - 1, int(d.delivered_s / width))] += 1

    assert [f.arrived for f in result.frames] == arrived
    assert [f.delivered for f in result.frames] == delivered


def test_nothing_is_in_flight_before_the_first_alarm_arrives(catalog):
    """Anchors the occupancy sweep to the clock, not just to its own totals."""
    assembly, workload, service = _build(catalog, slots=4)
    alarms, profile = build_load("replay")
    result = run_bench(
        None, assembly, alarms, profile, workload=workload, frames=600,
        service=service, ceilings=_ceilings(catalog, assembly, workload),
    )
    first_arrival = min(a.at_s for a in alarms)
    quiet = [f for f in result.frames if f.t_s + 144.0 <= first_arrival]

    assert quiet, "the measured day must start with an idle stretch"
    assert all(f.active == 0 and f.queued == 0 for f in quiet)
    assert all(f.cpu_pct == 0.0 and f.bw_gbs == 0.0 for f in quiet)
    assert all(f.delivered == 0 and f.arrived == 0 for f in quiet)


def test_the_last_frames_running_p95_is_the_runs_p95(catalog):
    """The reconstruction and the engine must agree on the headline number."""
    result, _ = _bench(catalog, "replay", slots=4)
    assert result.frames[-1].p95_so_far_s == pytest.approx(result.stats.p95_s, abs=1e-3)


def test_frames_cover_a_run_that_outlived_its_load(catalog):
    """Backlog draining past the load window is the case worth seeing."""
    result, _ = _bench(
        catalog, "ramp", cpu_id="test-desktop-2ch", slots=1,
        load={"start_rate": 100, "end_rate": 5000, "hours": 24},
    )
    last = result.frames[-1]
    assert last.t_s > result.profile.span_s, last.t_s
    assert any("부하가 끝난 뒤" in note for note in result.notes), result.notes
    assert sum(f.delivered for f in result.frames) == result.stats.delivered


# --------------------------------------------------------------------------
# AC7 physical impossibilities
# --------------------------------------------------------------------------


@pytest.mark.parametrize("slots", [1, 2, 4])
def test_no_frame_shows_more_in_flight_than_the_build_has_slots(catalog, slots):
    result, workload = _bench(catalog, "ramp", slots=slots)
    assert workload.slots == slots
    assert max(f.active for f in result.frames) <= slots
    assert max(f.active for f in result.frames) > 0


def test_nothing_in_a_frame_is_negative(catalog):
    result, _ = _bench(catalog, "spike", slots=4)
    for frame in result.frames:
        for f in dataclasses.fields(Frame):
            assert getattr(frame, f.name) >= 0, (f.name, frame)
        assert frame.cpu_pct <= 100.0 + 1e-9
        assert frame.ram_gb >= frame.kv_gb


# --------------------------------------------------------------------------
# AC8 determinism
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["replay", "ramp", "soak"])
def test_the_same_input_gives_the_same_frames_byte_for_byte(catalog, kind):
    first, _ = _bench(catalog, kind)
    second, _ = _bench(catalog, kind)
    assert frames_to_csv(first) == frames_to_csv(second)
    assert first.breach == second.breach
    assert first.worst == second.worst


def test_a_different_build_gives_different_frames(catalog):
    fast, _ = _bench(catalog, "ramp", cpu_id="test-amx-8ch", slots=4)
    slow, _ = _bench(catalog, "ramp", cpu_id="test-desktop-2ch", slots=4)
    assert frames_to_csv(fast) != frames_to_csv(slow)


# --------------------------------------------------------------------------
# AC9 one definition of the physics, and it is not in this module
# --------------------------------------------------------------------------


def test_bench_actually_calls_the_shared_segment_rates(catalog, monkeypatch):
    """Proof by explosion that the formula is not quietly reimplemented here.

    Importing `segment_rates` is not enough evidence -- an import can sit unused
    next to a private copy. So the shared function is replaced with one that
    raises, and the run has to fail.
    """
    class Detonated(RuntimeError):
        pass

    def boom(seg, ceilings):
        raise Detonated(seg.t_s)

    monkeypatch.setattr(bench, "segment_rates", boom)
    with pytest.raises(Detonated):
        _bench(catalog, "replay")


def test_bench_shares_the_definition_object_with_timeline():
    from svrspec import timeline

    assert bench.segment_rates is timeline.segment_rates


def test_resource_percentages_are_populated_and_bounded(catalog):
    """A chart of zeroes would pass every conservation test above."""
    result, _ = _bench(catalog, "ramp", cpu_id="test-desktop-2ch", slots=4)
    assert max(f.bw_pct for f in result.frames) > 1.0
    assert max(f.compute_pct for f in result.frames) > 0.0
    assert max(f.bw_gbs for f in result.frames) > 0.0
    assert max(f.kv_gb for f in result.frames) > 0.0
    # Bandwidth may edge slightly past its ceiling and that is not a defect:
    # prefill is modelled against the compute roofline and decode against the
    # bandwidth one, so a frame with both running sums two draws that were
    # never derated against each other. `timeline` has the same property. What
    # would be a defect is a reconstruction inventing work by a wide margin.
    assert 100.0 < max(f.bw_pct for f in result.frames) < 120.0


# --------------------------------------------------------------------------
# AC10 the breach -- the point of the ramp profile
# --------------------------------------------------------------------------


def test_a_ramp_finds_where_the_build_stops_holding_the_sla(catalog):
    result, workload = _bench(
        catalog, "ramp", cpu_id="test-lowcore-8ch", slots=1,
        load={"start_rate": 100, "end_rate": 5000, "hours": 24},
    )
    breach = result.breach
    assert breach is not None, "this build must break inside the ramp"
    assert breach["p95_s"] > workload.sla_seconds
    assert 0 < breach["t_s"] < result.profile.span_s
    assert breach["offered_rate"] > 100

    before = [f for f in result.frames if f.t_s < breach["t_s"]]
    assert before, "the breach must not be the very first frame"
    assert all(f.p95_so_far_s <= workload.sla_seconds for f in before)

    at = [f for f in result.frames if f.t_s == breach["t_s"]][0]
    assert at.offered_rate == breach["offered_rate"]
    assert at.p95_so_far_s == breach["p95_s"]
    assert any("SLA" in note for note in result.notes)


def test_the_offered_rate_at_the_breach_comes_off_the_declared_curve(catalog):
    """The breach is quoted as a load, so that load has to be the real one."""
    from svrspec.loadgen import rate_at

    result, _ = _bench(
        catalog, "ramp", cpu_id="test-lowcore-8ch", slots=1,
        load={"start_rate": 100, "end_rate": 5000, "hours": 24},
    )
    expected = rate_at(result.profile, result.breach["t_s"])
    assert result.breach["offered_rate"] == pytest.approx(expected, abs=0.01)


def test_a_build_with_headroom_reports_no_breach(catalog):
    result, workload = _bench(catalog, "soak", cpu_id="test-amx-8ch", slots=4)
    assert result.breach is None
    assert result.stats.p95_s <= workload.sla_seconds
    assert any("넘지 않았다" in note for note in result.notes)


def test_a_replay_breach_is_labelled_with_the_load_offered_so_far(catalog):
    """`replay` declares no rate curve, so the label must not read 0건/일."""
    result, _ = _bench(catalog, "replay", cpu_id="test-desktop-2ch", slots=1)
    assert result.breach is not None
    assert result.breach["offered_rate"] > 0


# --------------------------------------------------------------------------
# AC11 the CSV export
# --------------------------------------------------------------------------


def test_csv_has_a_header_row_and_one_row_per_frame(catalog):
    result, _ = _bench(catalog, "replay", frames=120)
    lines = frames_to_csv(result).splitlines()

    assert len(lines) == 120 + 1
    header = lines[0].split(",")
    assert header == [f.name for f in dataclasses.fields(Frame)]
    assert len(lines[1].split(",")) == len(header)


def test_csv_round_trips_the_first_frame(catalog):
    result, _ = _bench(catalog, "replay", frames=10)
    lines = frames_to_csv(result).splitlines()
    values = dict(zip(lines[0].split(","), lines[1].split(",")))
    first = result.frames[0]

    assert float(values["t_s"]) == first.t_s
    assert int(values["queued"]) == first.queued
    assert float(values["p95_so_far_s"]) == first.p95_so_far_s


# --------------------------------------------------------------------------
# The rest of the result
# --------------------------------------------------------------------------


def test_worst_is_the_slowest_alarms_capped_at_worst_n(catalog):
    result, _ = _bench(catalog, "replay", cpu_id="test-desktop-2ch", slots=1,
                       worst_n=5)
    assert len(result.worst) == 5
    totals = [row["total_s"] for row in result.worst]
    assert totals == sorted(totals, reverse=True)
    assert totals[0] == pytest.approx(result.stats.max_s)
    assert all("alarm_id" in row and "queue_wait_s" in row for row in result.worst)


def test_the_result_survives_json(catalog):
    """It has to cross to a browser, so nothing in it may be a catalogue object."""
    result, _ = _bench(catalog, "spike", frames=30)
    payload = {
        "machine": result.machine,
        "profile": {"kind": result.profile.kind, "params": result.profile.params},
        "frames": [dataclasses.asdict(f) for f in result.frames],
        "worst": result.worst,
        "breach": result.breach,
        "notes": result.notes,
    }
    assert json.loads(json.dumps(payload, ensure_ascii=False)) == payload
    assert result.machine["cpu"].strip()
    assert result.machine["slots"] == 2


def test_findings_are_passed_through_untouched(catalog):
    marker = object()
    assembly, workload, service = _build(catalog, findings=[marker])
    alarms, profile = build_load("replay")
    result = run_bench(
        None, assembly, alarms, profile, workload=workload, frames=20,
        service=service, ceilings=_ceilings(catalog, assembly, workload),
    )
    assert result.findings == [marker]


def test_the_result_does_not_repeat_the_profile_notes(catalog):
    """Both lists reach the page, so copying one into the other duplicates it.

    This test used to assert the opposite. The page renders `profile.notes` and
    `result.notes` one after the other, so carrying the profile's caveats into
    the result printed every one of them twice -- ten lines of small print where
    five would do, which is how small print stops being read. The profile owns
    the caveats about the load; the result owns the ones about the run.
    """
    result, _ = _bench(catalog, "ramp")
    assert result.profile.notes, "the profile must still carry its own caveats"
    assert result.notes, "and the run must still carry its own"
    assert not (set(result.profile.notes) & set(result.notes))


def test_ceilings_are_derived_from_the_assembly_when_not_supplied(catalog):
    """The default path must produce the same frames as the injected one."""
    assembly, workload, service = _build(catalog)
    alarms, profile = build_load("replay")
    injected = run_bench(
        None, assembly, alarms, profile, workload=workload, frames=60,
        service=service, ceilings=_ceilings(catalog, assembly, workload),
    )
    derived = run_bench(
        catalog, assembly, alarms, profile, workload=workload, frames=60,
        service=service,
    )
    assert frames_to_csv(injected) == frames_to_csv(derived)


def test_a_bandwidth_in_the_wrong_unit_is_refused(catalog):
    """`Assembly.bandwidth_gbs` is GB/s. Bytes/s would draw an idle machine."""
    assembly, workload, service = _build(catalog)
    wrong = dataclasses.replace(assembly, bandwidth_gbs=assembly.bandwidth_gbs * 1e9)
    alarms, profile = build_load("replay")
    with pytest.raises(ValueError, match="GB/s"):
        run_bench(catalog, wrong, alarms, profile, workload=workload, service=service)


def test_the_bench_does_not_mutate_the_alarms_it_was_given(catalog):
    assembly, workload, service = _build(catalog)
    alarms, profile = build_load("replay")
    before = copy.deepcopy(alarms)
    run_bench(
        None, assembly, alarms, profile, workload=workload, frames=30,
        service=service, ceilings=_ceilings(catalog, assembly, workload),
    )
    assert alarms == before


# --------------------------------------------------------------------------
# AC13 no model, no process, no network, no waiting on the clock
# --------------------------------------------------------------------------


def test_the_module_cannot_reach_a_process_or_the_network():
    """The bench runs in virtual time on the operator's laptop, or not at all.

    The import graph is the check that matters, because plain substring
    scanning trips over the CPU `sockets` count this module has to read off the
    assembly -- the same reason `test_pipeline` checks it this way. The names
    that cannot appear innocently are additionally banned as text.
    """
    source = BENCH_SRC.read_text(encoding="utf-8")
    banned = {"subprocess", "socket", "urllib", "http", "ssl", "asyncio",
              "multiprocessing", "ctypes", "requests", "httpx", "torch",
              "llama_cpp", "time", "threading", "concurrent"}

    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module.split(".")[0])
    assert not (imported & banned), sorted(imported & banned)

    # "socket" and "requests" are ordinary English here -- CPU sockets, and the
    # requests in flight. Both are already covered by the import graph above.
    for text in ("subprocess", "urllib", "Popen", "urlopen", "sleep"):
        assert text not in source, text


# --------------------------------------------------------------------------
# The seam to the real assembler
# --------------------------------------------------------------------------


def test_a_real_assembly_runs_end_to_end(catalog):
    """The stub above proves the shape; this proves the actual wiring.

    Everything else here injects `service` and `ceilings`, so a change to
    `lab.to_service` would go unnoticed until the GUI called it. Skipped rather
    than failed when `lab` is absent -- that module has its own owner, and this
    file must not go red because a sibling is mid-edit.
    """
    lab = pytest.importorskip("svrspec.lab")

    vm = lab.VirtualMachine(
        name="8ch 전채널", cpu_id="test-amx-8ch", sockets=1, dimm_gb=64,
        dimm_count=8, model_id="test-8b-gqa", quant_id="Q4_K_M", slots=2,
    )
    assembly = lab.assemble(catalog, vm)
    assert assembly.ok, [f.code for f in assembly.findings]

    alarms, profile = build_load("replay")
    result = run_bench(
        catalog, assembly, alarms, profile,
        workload=Workload(slots=2, tokens=TokenProfile()), frames=200,
    )
    assert len(result.frames) == 200
    assert sum(f.arrived for f in result.frames) == result.stats.received
    assert max(f.active for f in result.frames) <= 2
    assert result.machine["ram_total_gb"] == 512
    assert result.findings == list(assembly.findings)


def test_starving_the_memory_channels_moves_the_breach(catalog):
    """The whole reason `lab` exists, seen through the bench.

    Two DIMMs on an eight-channel board keep the capacity and lose three
    quarters of the bandwidth. Decode is bandwidth bound, so the build must
    give out at a markedly lower offered rate -- and if the bench cannot show
    that, it is not measuring the thing the operator is buying.
    """
    lab = pytest.importorskip("svrspec.lab")

    def ramp_breach(dimm_count):
        vm = lab.VirtualMachine(
            name=f"{dimm_count}×64GB", cpu_id="test-amx-8ch", sockets=1,
            dimm_gb=64, dimm_count=dimm_count, model_id="test-8b-gqa",
            quant_id="Q4_K_M", slots=2,
        )
        assembly = lab.assemble(catalog, vm)
        alarms, profile = build_load(
            "ramp", start_rate=100, end_rate=5000, hours=24
        )
        result = run_bench(
            catalog, assembly, alarms, profile,
            workload=Workload(slots=2, tokens=TokenProfile()),
        )
        return assembly, result

    full, full_run = ramp_breach(8)
    starved, starved_run = ramp_breach(2)

    assert starved.bandwidth_gbs == pytest.approx(full.bandwidth_gbs / 4, rel=1e-6)
    assert full_run.breach is None
    assert starved_run.breach is not None
    assert starved_run.breach["offered_rate"] < 5000
    assert starved_run.stats.p95_s > full_run.stats.p95_s


def test_a_build_that_cannot_exist_is_not_benched(catalog):
    """An error-level assembly has no service rates, so there is nothing to run."""
    lab = pytest.importorskip("svrspec.lab")

    vm = lab.VirtualMachine(
        name="소켓 초과", cpu_id="test-amx-8ch", sockets=99, dimm_gb=64,
        dimm_count=8, model_id="test-8b-gqa", quant_id="Q4_K_M", slots=2,
    )
    assembly = lab.assemble(catalog, vm)
    assert not assembly.ok

    alarms, profile = build_load("replay")
    with pytest.raises(ValueError):
        run_bench(
            catalog, assembly, alarms, profile,
            workload=Workload(slots=2, tokens=TokenProfile()),
        )


def test_the_run_is_always_virtual_time(catalog):
    """Replay speed belongs to the browser. Nothing here may pace itself."""
    result, _ = _bench(catalog, "soak", slots=4)
    assert result.stats.wall_clock_s < 1.0
