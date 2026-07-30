"""Host telemetry has to move, and it has to move the same way twice.

The complaint this module answers was "the resource usage never changes -- make
it feel like a simulator is actually running". So the tests here are mostly
about motion: RSS breathes, load average lags, per-core lines are not one line
drawn N times. The other half is the constraint that pays for that realism --
none of it may come from the machine running the tool, and none of it may come
from a draw, or the same sizing would produce two different charts.
"""

import math
import re
from pathlib import Path

import pytest

from svrspec.hostsim import HostSample, sample_host, to_csv
from svrspec.memory import GB, RUNTIME_OS_GB
from svrspec.perf import Efficiency
from svrspec.simulate import SimSegment, SimTrace, simulate
from svrspec.sizing import decode_table, evaluate
from svrspec.timeline import Ceilings, ceilings_for
from svrspec.types import TokenProfile, Workload

HOSTSIM_SOURCE = Path(__file__).resolve().parents[1] / "svrspec" / "hostsim.py"


def _run(catalog, model, cpu_id, workload, sockets=1):
    """Trace + ceilings for one build. Copied from test_timeline._run on purpose:
    the fixture belongs to whichever test file needs it, not to a shared helper
    that two owners would have to agree on."""
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
    return trace, ceilings, cpu, candidate.memory_gb


# --------------------------------------------------------------------------
# Synthetic traces, for the properties that need an exactly known input
# --------------------------------------------------------------------------


def _seg(t, span, active=1, queued=0, decoding=None, decode_tokens=0.0, kv_tokens=0.0):
    decoding = active if decoding is None else decoding
    return SimSegment(
        t_s=t,
        span_s=span,
        active=active,
        queued=queued,
        prefilling=active - decoding,
        decoding=decoding,
        prefill_tokens=0.0,
        decode_tokens=decode_tokens,
        kv_tokens=kv_tokens,
    )


def _trace(segments):
    trace = SimTrace()
    trace.segments = list(segments)
    return trace


def _ceilings():
    return Ceilings(
        bandwidth_bytes_s=200e9,
        compute_flops=5e12,
        weight_bytes=4.5e9,
        kv_bytes_token=131072.0,
        flops_per_token=16e9,
        static_bytes=8.0 * GB,
        kv_reserved_bytes=4.0 * GB,
        installed_bytes=64 * GB,
    )


# --------------------------------------------------------------------------


def test_two_runs_produce_identical_output(catalog, model_8b):
    """AC1. Nothing may be drawn at runtime: the same build must chart the same.

    An operator comparing two candidate servers cannot tell a real difference
    from a jitter, so there must be no jitter.
    """
    workload = Workload(alarms_per_day=400, tokens=TokenProfile(alarm_tokens=1200))
    trace, ceilings, cpu, gb = _run(catalog, model_8b, "test-amx-8ch", workload)

    first = sample_host(trace, ceilings, cpu, installed_gb=gb, period_s=60.0)
    second = sample_host(trace, ceilings, cpu, installed_gb=gb, period_s=60.0)

    assert first == second
    assert to_csv(first) == to_csv(second)


def test_rss_moves_over_the_day(catalog, model_8b):
    """AC2. The exact defect that started this: a flat memory graph.

    KV residency is a function of who is in flight and how far along they are,
    so a day with real traffic must show the process breathing.
    """
    workload = Workload(alarms_per_day=500, tokens=TokenProfile(output_tokens=600))
    trace, ceilings, cpu, gb = _run(catalog, model_8b, "test-amx-8ch", workload)
    host = sample_host(trace, ceilings, cpu, installed_gb=gb, period_s=60.0)

    rss = [s.rss_gb for s in host.samples]
    assert max(rss) > min(rss), "RSS must move -- a flat line is the bug"
    assert host.peak_rss_gb == pytest.approx(max(rss))
    # And the box always reports more than the process: the OS is underneath it.
    for s in host.samples:
        assert s.host_mem_used_gb > s.rss_gb
        assert s.host_mem_used_gb - s.rss_gb == pytest.approx(RUNTIME_OS_GB, abs=1e-6)


def test_bandwidth_is_not_a_copy_of_cpu(catalog, model_8b):
    """The other half of the same complaint: three graphs drawing one line."""
    workload = Workload(alarms_per_day=500, tokens=TokenProfile(alarm_tokens=1200))
    trace, ceilings, cpu, gb = _run(catalog, model_8b, "test-amx-8ch", workload)
    host = sample_host(trace, ceilings, cpu, installed_gb=gb, period_s=60.0)

    busy = [s for s in host.samples if s.cpu_pct > 0]
    assert busy
    ratios = {
        round(s.bandwidth_pct / s.cpu_pct, 3)
        for s in busy
        if s.cpu_pct > 0.01 and s.bandwidth_pct > 0.01
    }
    assert len(ratios) > 1, "bandwidth may not be a scalar multiple of CPU"


def test_touched_kv_never_exceeds_the_reservation(catalog, model_8b):
    """AC3. Touched KV cannot outgrow what llama.cpp reserved up front."""
    workload = Workload(alarms_per_day=500, tokens=TokenProfile(output_tokens=600))
    trace, ceilings, cpu, gb = _run(catalog, model_8b, "test-amx-8ch", workload)
    host = sample_host(trace, ceilings, cpu, installed_gb=gb, period_s=60.0)

    limit = ceilings.allocated_bytes / GB
    for s in host.samples:
        assert s.rss_gb <= limit + 1e-9
        assert s.host_mem_used_gb <= limit + 1e-9


def test_cpu_pct_stays_in_range_and_matches_its_cores(catalog, model_8b):
    """AC4 + AC5. Per-core lines differ, but they still average to the aggregate."""
    workload = Workload(
        alarms_per_day=600,
        storm_size=40,
        storms_per_day=2,
        tokens=TokenProfile(alarm_tokens=1500),
    )
    trace, ceilings, cpu, gb = _run(catalog, model_8b, "test-desktop-2ch", workload)
    host = sample_host(trace, ceilings, cpu, installed_gb=gb, period_s=60.0)

    assert host.cores == cpu.cores
    spread_seen = False
    for s in host.samples:
        assert 0.0 <= s.cpu_pct <= 100.0
        assert len(s.per_core_pct) == cpu.cores
        assert all(0.0 <= c <= 100.0 for c in s.per_core_pct)
        mean = sum(s.per_core_pct) / len(s.per_core_pct)
        assert mean == pytest.approx(s.cpu_pct, abs=0.5)
        if len(set(s.per_core_pct)) > 1:
            spread_seen = True
    assert spread_seen, "per-core lines must not all be identical"


def test_per_core_length_follows_the_socket_count(catalog, model_3b):
    workload = Workload(alarms_per_day=200)
    trace, ceilings, cpu, gb = _run(catalog, model_3b, "test-amx-8ch", workload, sockets=2)
    host = sample_host(
        trace, ceilings, cpu, installed_gb=gb, sockets=2, period_s=300.0
    )
    assert host.cores == cpu.cores * 2
    assert all(len(s.per_core_pct) == cpu.cores * 2 for s in host.samples)


def test_load_average_lags_the_offered_load(catalog):
    """AC6. The lag is the detail that makes it read as a real host.

    A ten-second burst of four busy slots with two queued behind them, then
    nothing. A one-minute EWMA must not reach the offered load inside a ten
    second burst, and must not fall to zero the instant the burst ends -- if it
    did, it would be an instantaneous gauge with a longer name.
    """
    cpu = catalog.cpu("test-lowcore-8ch")
    offered = 4 + 2
    storm = [_seg(float(i), 1.0, active=4, queued=2, decode_tokens=20.0) for i in range(10)]
    host = sample_host(
        _trace(storm), _ceilings(), cpu,
        installed_gb=64, period_s=10.0, span_s=200.0,
    )

    during = host.samples[0]
    after = host.samples[1]
    assert during.cpu_pct == pytest.approx(100.0)
    # Rises slowly: nowhere near the offered load after one time-constant/6.
    assert 0 < during.load_avg_1m < offered / 2
    # Falls slowly: still elevated after the storm, for more than one sample.
    assert after.load_avg_1m < during.load_avg_1m
    assert after.load_avg_1m < 4, "must sit below the peak slot count it decays from"
    assert host.samples[2].load_avg_1m > 0
    assert host.samples[3].load_avg_1m > 0
    # And it does eventually come down.
    assert host.samples[-1].load_avg_1m < host.samples[2].load_avg_1m
    # Decay is the EWMA's, not an arbitrary one.
    decay = math.exp(-10.0 / 60.0)
    assert host.samples[2].load_avg_1m == pytest.approx(after.load_avg_1m * decay)


def test_segment_time_is_conserved(catalog, model_3b):
    """AC7. The series is a reconstruction of the run, not an approximation."""
    workload = Workload(alarms_per_day=300)
    trace, ceilings, cpu, gb = _run(catalog, model_3b, "test-amx-8ch", workload)
    host = sample_host(trace, ceilings, cpu, installed_gb=gb, period_s=60.0)

    from_segments = sum(s.span_s for s in trace.segments)
    assert sum(s.busy_s for s in host.samples) == pytest.approx(from_segments, rel=1e-6)


def test_work_past_the_window_lands_on_the_last_sample(catalog):
    """AC7's hard half. An overloaded day does not end when the day does.

    The backlog has to be charged to the final sample rather than dropped off
    the end of the chart -- and the loop that does the charging has to
    terminate, which it once did not.
    """
    cpu = catalog.cpu("test-lowcore-8ch")
    trace = _trace([_seg(95.0, 20.0, active=2, decode_tokens=100.0, kv_tokens=1000.0)])
    host = sample_host(
        trace, _ceilings(), cpu, installed_gb=64, period_s=10.0, span_s=100.0
    )

    assert len(host.samples) == 10
    assert sum(s.busy_s for s in host.samples) == pytest.approx(20.0, rel=1e-9)
    assert host.samples[-1].busy_s == pytest.approx(20.0, rel=1e-9)
    # Charged, but the percentage still has to be a percentage.
    assert host.samples[-1].cpu_pct == pytest.approx(100.0)
    assert any("마지막 샘플" in n for n in host.notes)


def test_csv_has_a_header_a_row_per_sample_and_every_field(catalog, model_3b):
    """AC8."""
    workload = Workload(alarms_per_day=200)
    trace, ceilings, cpu, gb = _run(catalog, model_3b, "test-amx-8ch", workload)
    host = sample_host(trace, ceilings, cpu, installed_gb=gb, period_s=600.0)

    text = to_csv(host)
    assert not text.startswith("﻿"), "the caller owns the encoding, not this"
    lines = text.strip("\n").split("\n")
    assert len(lines) == len(host.samples) + 1

    header = lines[0].split(",")
    for f in HostSample.__dataclass_fields__:
        assert f in header
    assert len(lines[1].split(",")) == len(header)


def test_rejects_a_nonpositive_period_or_span(catalog, model_3b):
    """AC9."""
    workload = Workload(alarms_per_day=100)
    trace, ceilings, cpu, gb = _run(catalog, model_3b, "test-amx-8ch", workload)

    with pytest.raises(ValueError):
        sample_host(trace, ceilings, cpu, installed_gb=gb, period_s=0.0)
    with pytest.raises(ValueError):
        sample_host(trace, ceilings, cpu, installed_gb=gb, span_s=-1.0)


def test_an_empty_trace_still_produces_a_flat_idle_day(catalog):
    """A box that served nothing is idle, not undefined."""
    cpu = catalog.cpu("test-lowcore-8ch")
    host = sample_host(
        _trace([]), _ceilings(), cpu, installed_gb=64, period_s=60.0, span_s=600.0
    )
    assert len(host.samples) == 10
    assert all(s.cpu_pct == 0.0 for s in host.samples)
    assert all(s.load_avg_1m == 0.0 for s in host.samples)
    assert all(s.bandwidth_gbs == 0.0 for s in host.samples)
    # Idle still costs the weights: RSS is never zero on a loaded model.
    assert host.peak_rss_gb > 0


def test_hostsim_never_touches_the_real_host():
    """AC10. An explicit user constraint, so it gets a regression guard.

    Sizing a server may not load the machine doing the sizing: no benchmark
    runs, no external commands, no kernel counters, no wall clock, no draws.
    Anything that would make this module observe its own host would also make
    the same sizing chart differently on two laptops.
    """
    source = HOSTSIM_SOURCE.read_text(encoding="utf-8")
    for forbidden in ("subprocess", "psutil", "/proc", "random"):
        assert forbidden not in source, f"hostsim.py must not mention {forbidden!r}"
    for forbidden in ("os.getloadavg", "time.time", "time.monotonic", "open("):
        assert forbidden not in source, f"hostsim.py must not call {forbidden!r}"
    # No imports beyond the standard-library helpers and this package.
    imported = set(re.findall(r"^(?:from|import)\s+([\w.]+)", source, re.MULTILINE))
    assert imported <= {"__future__", "csv", "dataclasses", "io", "math",
                        ".memory", ".simulate", ".timeline", ".types"}, imported
