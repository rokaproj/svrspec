"""Tests for `svrspec.modelbench`: does the bench show the machine's trades?

These run against the **shipped** catalogue, like `test_lab`'s do, because the
module's output is a claim about parts somebody can buy -- a synthetic catalogue
would let the tests agree with themselves about a machine that does not exist.
Where a test depends on a catalogue fact (the Gold 6426Y is eight-channel), it
asserts that fact first, so a catalogue change is reported as a catalogue change
instead of as a mystery failure.

What is being tested here is not arithmetic -- `perf` owns that and has its own
tests. It is the *shape* of the answer: batching buys total throughput and spends
per-user latency, long contexts cost decode speed, prefill and decode bind on
opposite halves of the machine, and CPU training gets a "no" with a number
attached. Those are the four claims the module makes; if any of them inverts, the
screens built on it are lying.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from svrspec.catalog import Catalog
from svrspec.lab import VirtualMachine, assemble
from svrspec.modelbench import (
    SECTIONS,
    ConcurrencyPoint,
    ModelBench,
    ThroughputPoint,
    TrainingVerdict,
    bench_model,
    to_csv,
)

MODELBENCH_SRC = Path(__file__).resolve().parents[1] / "svrspec" / "modelbench.py"

#: Eight channels of DDR5-4800, 14 cores, AMX. The reference machine of the
#: brief: at 25 GB/s per core the DRAM roofline binds at both full and quarter
#: population, so a channel-count change shows up undiluted.
CPU_8CH = "xeon-gold-6426y"
#: Two channels, 128 GB ceiling. Used where a *small* machine is the point.
CPU_SMALL = "xeon-e-2488"

MODEL_7B = "qwen2.5-7b-instruct"
MODEL_TINY = "qwen2.5-0.5b-instruct"
MODEL_MOE = "qwen3-30b-a3b"
QUANT = "Q4_K_M"


@pytest.fixture(scope="module")
def cat() -> Catalog:
    return Catalog()


def vm(**over) -> VirtualMachine:
    """A sane, fully populated eight-channel 7B build, overridable field by field."""
    base = dict(
        name="bench",
        cpu_id=CPU_8CH,
        sockets=1,
        dimm_gb=64,
        dimm_count=8,
        model_id=MODEL_7B,
        quant_id=QUANT,
        slots=4,
    )
    base.update(over)
    return VirtualMachine(**base)


@pytest.fixture(scope="module")
def bench(cat) -> ModelBench:
    """The headline case: 7B / Q4_K_M on a fully populated Gold 6426Y."""
    return bench_model(assemble(cat, vm()))


def rows(bench: ModelBench, *, batch: int | None = None, ctx: int | None = None):
    return [
        p for p in bench.throughput
        if (batch is None or p.batch == batch) and (ctx is None or p.ctx_tokens == ctx)
    ]


def split(bench: ModelBench, phase: str):
    for r in bench.resources:
        if r.phase == phase:
            return r
    raise AssertionError(f"no {phase!r} resource split in {[r.phase for r in bench.resources]}")


def verdict(bench: ModelBench, kind: str) -> TrainingVerdict:
    for t in bench.training:
        if t.kind == kind:
            return t
    raise AssertionError(f"no {kind!r} training verdict in {[t.kind for t in bench.training]}")


# --------------------------------------------------------------------------
# AC1 it works on a duck-typed assembly and fills every axis
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FakeVm:
    sockets: int = 1


@dataclass(frozen=True)
class FakeAssembly:
    """Only the attributes the contract promises to read.

    Deliberately *without* `ram_total_gb` and `findings`: a caller holding a CPU
    and a DIMM should be able to ask what the tok/s is without building a lab
    assembly first, and this pins that the module does not quietly require more
    than it documents.
    """

    cpu: Any
    memory: Any
    model: Any
    quant: Any
    channels_populated: int
    uncertainty: float = 0.0
    vm: FakeVm = FakeVm()


def fake(cat: Catalog, cpu_id=CPU_8CH, model_id=MODEL_7B, populated=None,
         sockets=1, uncertainty=0.0) -> FakeAssembly:
    cpu = cat.cpu(cpu_id)
    return FakeAssembly(
        cpu=cpu,
        memory=cat.memory_for(cpu),
        model=cat.model(model_id),
        quant=cat.quant(QUANT),
        channels_populated=cpu.mem_channels * sockets if populated is None else populated,
        uncertainty=uncertainty,
        vm=FakeVm(sockets=sockets),
    )


def test_a_duck_typed_assembly_is_enough_and_every_axis_is_filled(cat):
    result = bench_model(
        fake(cat), batches=(1, 4), contexts=(512, 4096), users=(1, 8),
    )

    assert len(result.throughput) == 4      # 2 batches x 2 contexts
    assert len(result.concurrency) == 2
    assert [r.phase for r in result.resources] == ["prefill", "decode"]
    assert [t.kind for t in result.training] == ["full", "lora", "qlora"]
    assert result.model_name and result.quant_id == QUANT
    assert result.memory_gb > 0
    assert 0 < result.uncertainty <= 1.0
    assert result.notes, "assumptions must be stated"


def test_an_assemblys_own_uncertainty_is_never_narrowed(cat):
    """A build may carry error bars wider than the raw prediction's."""
    result = bench_model(fake(cat, uncertainty=0.9), batches=(1,), contexts=(2048,),
                         users=(1,))
    assert result.uncertainty == pytest.approx(0.9)


def test_an_emptied_axis_falls_back_instead_of_raising(cat):
    """A GUI that clears every checkbox gets a bench back, not a traceback."""
    result = bench_model(fake(cat), batches=(), contexts=(), users=())
    assert result.throughput and result.concurrency


# --------------------------------------------------------------------------
# AC2 batching buys total throughput and spends per-sequence throughput
# --------------------------------------------------------------------------


def test_batching_raises_the_total_and_never_raises_the_per_sequence_rate(bench):
    for ctx in sorted({p.ctx_tokens for p in bench.throughput}):
        points = sorted(rows(bench, ctx=ctx), key=lambda p: p.batch)
        totals = [p.decode_tps_total for p in points]
        singles = [p.decode_tps_single for p in points]

        assert totals == sorted(totals), f"ctx {ctx}: total fell with batch {totals}"
        assert singles == sorted(singles, reverse=True), (
            f"ctx {ctx}: per-sequence rate rose with batch {singles}"
        )
    # And the trade has to be visible, not merely non-inverted: this machine
    # saturates its compute ceiling somewhere in the grid, past which extra
    # sequences only divide what is already there.
    wide = rows(bench, ctx=2048)
    assert max(p.decode_tps_total for p in wide) > 2 * min(p.decode_tps_total for p in wide)
    assert min(p.decode_tps_single for p in wide) < 0.5 * max(p.decode_tps_single for p in wide)


def test_at_batch_one_the_total_is_what_one_sequence_gets(bench):
    for p in rows(bench, batch=1):
        assert p.decode_tps_single == pytest.approx(p.decode_tps_total)


# --------------------------------------------------------------------------
# AC3 a longer context costs decode speed, because the KV read grows
# --------------------------------------------------------------------------


def test_a_longer_context_slows_generation(bench):
    short = rows(bench, batch=1, ctx=512)[0]
    long = rows(bench, batch=1, ctx=16384)[0]
    assert long.decode_tps_single < short.decode_tps_single

    curve = [p.decode_tps_single for p in sorted(rows(bench, batch=1),
                                                 key=lambda p: p.ctx_tokens)]
    assert curve == sorted(curve, reverse=True), curve


# --------------------------------------------------------------------------
# AC4 prefill is a shared compute ceiling, so the batch does not move it
# --------------------------------------------------------------------------


def test_prefill_does_not_depend_on_the_batch(bench):
    for ctx in sorted({p.ctx_tokens for p in bench.throughput}):
        rates = {round(p.prefill_tps, 9) for p in rows(bench, ctx=ctx)}
        assert len(rates) == 1, f"ctx {ctx}: prefill moved with batch: {rates}"


# --------------------------------------------------------------------------
# AC5/AC6 concurrency: everyone waits longer, the server does more
# --------------------------------------------------------------------------


def test_more_users_costs_each_of_them_and_saturates_the_server(bench):
    points = sorted(bench.concurrency, key=lambda c: c.users)

    ttft = [c.ttft_s for c in points]
    response = [c.response_s for c in points]
    each = [c.decode_tps_each for c in points]
    total = [c.total_tps for c in points]

    assert all(b > a for a, b in zip(ttft, ttft[1:])), ttft
    assert all(b > a for a, b in zip(response, response[1:])), response
    assert all(b < a for a, b in zip(each, each[1:])), each
    assert total == sorted(total), total
    # Saturation: past the knee the server's total stops improving, so the last
    # two points are the same number. Without this the axis would suggest that
    # adding users is free.
    assert total[-1] == pytest.approx(total[-2])


def test_response_time_is_ttft_plus_generation(bench):
    output_tokens = 256  # the default this bench was built with
    for c in bench.concurrency:
        expected = c.ttft_s + output_tokens / c.decode_tps_each
        assert c.response_s == pytest.approx(expected, rel=1e-9)


# --------------------------------------------------------------------------
# AC7 the two phases use opposite halves of the machine
# --------------------------------------------------------------------------


def test_prefill_and_decode_bind_on_different_ceilings(bench):
    prefill, decode = split(bench, "prefill"), split(bench, "decode")

    assert prefill.bound_by != decode.bound_by, (
        f"both phases bound by {prefill.bound_by}; the opposite-halves claim is gone"
    )
    # Concretely, on this machine: prompt processing is vector work and token
    # generation is DRAM work.
    assert prefill.bound_by == "compute"
    assert decode.bound_by in ("bandwidth", "core-bandwidth")

    assert prefill.compute_pct > prefill.bandwidth_pct
    assert decode.bandwidth_pct > decode.compute_pct
    # The binding phase sits at its ceiling by construction; the other figure
    # says how much of the machine's other half is idle meanwhile.
    assert prefill.compute_pct == pytest.approx(100.0, abs=1e-6)
    assert decode.bandwidth_pct == pytest.approx(100.0, abs=1e-6)


def test_every_occupancy_is_a_percentage(bench):
    for r in bench.resources:
        for pct in (r.bandwidth_pct, r.compute_pct):
            assert -1e-6 <= pct <= 100.0 + 1e-6, (r.phase, pct)
        assert r.bytes_per_token > 0
        assert r.flops_per_token > 0


def test_decode_reads_more_bytes_per_token_than_prefill(bench):
    """Not a tautology -- it is *why* the bottlenecks differ.

    Prefill amortises one weight sweep over a whole micro-batch of tokens; decode
    pays for the sweep once per token and adds the KV history on top.
    """
    assert split(bench, "decode").bytes_per_token > split(bench, "prefill").bytes_per_token


# --------------------------------------------------------------------------
# AC8 training: the answer is usually no, with a reason
# --------------------------------------------------------------------------


def test_the_three_schemes_order_by_memory(bench):
    full, lora, qlora = (verdict(bench, k) for k in ("full", "lora", "qlora"))
    assert full.memory_needed_gb > lora.memory_needed_gb > qlora.memory_needed_gb


def test_cpu_training_is_refused_with_hours_attached(bench):
    """The 7B case: memory fits comfortably and it is still not a capability."""
    full = verdict(bench, "full")
    assert full.memory_needed_gb < full.memory_available_gb
    assert full.feasible is False
    assert full.epoch_hours is not None and full.epoch_hours > 24
    assert any("epoch" in r for r in full.reasons)
    assert any("추정" in r for r in full.reasons), "the coefficient basis must be disclosed"


def test_a_scheme_that_does_not_fit_says_what_is_missing(cat):
    """A 128 GB two-channel box asked to full fine-tune a 7B."""
    small = cat.cpu(CPU_SMALL)
    assert small.mem_channels == 2, "catalogue changed: CPU_SMALL is no longer 2ch"

    result = bench_model(
        assemble(cat, vm(cpu_id=CPU_SMALL, dimm_gb=32, dimm_count=2, slots=1)),
        batches=(1,), contexts=(2048,), users=(1,),
    )
    full = verdict(result, "full")
    assert full.memory_needed_gb > full.memory_available_gb
    assert full.feasible is False
    assert any("모자라" in r for r in full.reasons), full.reasons
    # ... and the cheap scheme is the one that fits, which is the actionable half.
    assert verdict(result, "qlora").memory_needed_gb < full.memory_needed_gb


def test_a_small_model_on_a_big_box_is_allowed_to_be_feasible(cat):
    """The axis has to be able to say yes, or its "no" means nothing."""
    result = bench_model(
        assemble(cat, vm(model_id=MODEL_TINY, slots=1)),
        batches=(1,), contexts=(2048,), users=(1,), train_samples=1_000,
    )
    assert verdict(result, "qlora").feasible is True


def test_every_verdict_carries_a_gpu_comparison(bench):
    for t in bench.training:
        assert t.gpu_comparison.strip(), t.kind
        assert "A100" in t.gpu_comparison
        assert "추정" in t.gpu_comparison, "the GPU line must admit it is arithmetic"


def test_a_moe_checkpoint_charges_memory_on_all_experts_and_compute_on_the_active_ones(cat):
    result = bench_model(
        fake(cat, model_id=MODEL_MOE), batches=(1,), contexts=(2048,), users=(1,),
    )
    dense = bench_model(
        fake(cat, model_id=MODEL_7B), batches=(1,), contexts=(2048,), users=(1,),
    )
    moe_full, dense_full = verdict(result, "full"), verdict(dense, "full")
    # 30B of weights to hold...
    assert moe_full.memory_needed_gb > dense_full.memory_needed_gb
    # ...but only ~3.35B of them run per token, so the step is cheaper than the
    # 7B dense model's despite the far bigger checkpoint.
    assert moe_full.step_seconds < dense_full.step_seconds
    assert any("MoE" in n for n in result.notes)


def test_an_empty_board_cannot_train_and_does_not_pretend_otherwise(cat):
    """`ram_total_gb == 0` must not fall through to the CPU's addressable max."""
    result = bench_model(
        assemble(cat, vm(dimm_count=0)), batches=(1,), contexts=(2048,), users=(1,),
    )
    for t in result.training:
        assert t.memory_available_gb == 0.0
        assert t.feasible is False
    # And the inference axes degrade instead of raising: no bandwidth is no
    # tokens, and the honest time to first token is forever.
    assert result.throughput[0].decode_tps_total == 0.0
    assert result.concurrency[0].response_s == float("inf")


# --------------------------------------------------------------------------
# AC9 the lab's channel population reaches the bench
# --------------------------------------------------------------------------


def test_under_populating_the_channels_slows_generation(cat):
    cpu = cat.cpu(CPU_8CH)
    assert cpu.mem_channels == 8, "catalogue changed: CPU_8CH is no longer 8ch"

    full = assemble(cat, vm(dimm_gb=64, dimm_count=8))
    quarter = assemble(cat, vm(dimm_gb=64, dimm_count=2))
    assert full.channels_populated == 8 and quarter.channels_populated == 2

    axes = dict(batches=(1,), contexts=(2048,), users=(1,))
    slow = bench_model(quarter, **axes).throughput[0]
    fast = bench_model(full, **axes).throughput[0]

    assert slow.decode_tps_total < fast.decode_tps_total
    # Bandwidth is linear in populated channels and decode is bandwidth bound,
    # so a quarter of the channels is close to a quarter of the tokens. Loose
    # bounds: the KV term does not scale with the weight sweep.
    ratio = slow.decode_tps_total / fast.decode_tps_total
    assert 0.2 < ratio < 0.45, ratio
    assert any("채널" in n for n in bench_model(quarter, **axes).notes)


def test_a_second_socket_reaches_the_bench_too(cat):
    """Not an AC, but the same wiring: `vm.sockets` must not be ignored."""
    axes = dict(batches=(1,), contexts=(2048,), users=(1,))
    one = bench_model(fake(cat, sockets=1), **axes).throughput[0]
    two = bench_model(fake(cat, sockets=2), **axes).throughput[0]
    assert two.decode_tps_total > one.decode_tps_total
    assert two.prefill_tps > one.prefill_tps


# --------------------------------------------------------------------------
# AC10 the physics is perf's, not a copy
# --------------------------------------------------------------------------


def test_the_bench_cannot_answer_without_perf(cat, monkeypatch):
    """Substitute the roofline and the bench has nothing to fall back on."""
    sentinel = RuntimeError("predict_throughput was called")

    def boom(*args, **kwargs):
        raise sentinel

    monkeypatch.setattr("svrspec.perf.predict_throughput", boom)
    with pytest.raises(RuntimeError) as caught:
        bench_model(fake(cat), batches=(1,), contexts=(2048,), users=(1,))
    assert caught.value is sentinel


def test_the_grid_costs_one_prediction_per_point(cat, monkeypatch):
    """One call per cell, plus the concurrency sweep and the reference point.

    Pinned because a bench that re-predicts inside a helper would multiply the
    catalogue work silently, and because it proves every published cell came from
    its own call rather than from a scaled neighbour.
    """
    from svrspec import perf as perf_module

    calls: list[dict] = []
    real = perf_module.predict_throughput

    def counting(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr("svrspec.perf.predict_throughput", counting)
    bench_model(fake(cat), batches=(1, 4), contexts=(512, 4096), users=(1, 2, 4))

    # 4 grid cells + 3 concurrency points + 1 reference point for the split.
    assert len(calls) == 8
    assert {c["slots"] for c in calls} == {1, 2, 4}


def test_a_full_board_asks_perf_for_its_own_default_path(cat, monkeypatch):
    """`channels_populated=None` when every channel is filled.

    Same convention as `lab.assemble`: a fully populated build must be
    bit-for-bit the prediction a caller gets from `predict_throughput` directly,
    and passing the channel count explicitly is a different code path in `perf`.
    """
    from svrspec import perf as perf_module

    seen: list[Any] = []
    real = perf_module.predict_throughput

    def watching(*args, **kwargs):
        seen.append(kwargs.get("channels_populated"))
        return real(*args, **kwargs)

    monkeypatch.setattr("svrspec.perf.predict_throughput", watching)
    bench_model(fake(cat), batches=(1,), contexts=(2048,), users=(1,))
    assert set(seen) == {None}

    seen.clear()
    bench_model(fake(cat, populated=2), batches=(1,), contexts=(2048,), users=(1,))
    assert set(seen) == {2}


def test_perfs_warnings_are_passed_through_unedited(cat):
    """The estimate warning belongs to the reader, not to this module."""
    result = bench_model(fake(cat, populated=2), batches=(1,), contexts=(2048,),
                         users=(1,))
    assert result.warnings
    assert any("채널" in w for w in result.warnings)
    # Deduplicated: the same warning comes back from every grid point and a
    # thirty-times-repeated caveat is one nobody reads.
    assert len(result.warnings) == len(set(result.warnings))


# --------------------------------------------------------------------------
# AC11 export
# --------------------------------------------------------------------------


EXPECTED_COLUMNS = {
    "throughput": [f.name for f in ThroughputPoint.__dataclass_fields__.values()],
    "concurrency": [f.name for f in ConcurrencyPoint.__dataclass_fields__.values()],
    "training": [f.name for f in TrainingVerdict.__dataclass_fields__.values()],
}


@pytest.mark.parametrize("section", SECTIONS)
def test_each_section_exports_every_field_and_every_row(bench, section):
    text = to_csv(bench, section)
    lines = text.strip().split("\n")

    header = lines[0].split(",")
    assert header == EXPECTED_COLUMNS[section]

    expected_rows = len(getattr(bench, section))
    # Quoted cells may contain no newline, so a line count is a row count.
    assert len(lines) - 1 == expected_rows


def test_a_list_valued_field_stays_one_cell(bench):
    text = to_csv(bench, "training")
    import csv
    import io

    table = list(csv.reader(io.StringIO(text)))
    assert len(table[0]) == len(table[1]) == len(EXPECTED_COLUMNS["training"])
    reasons = table[1][EXPECTED_COLUMNS["training"].index("reasons")]
    assert "|" in reasons, "several reasons should survive the join"


def test_an_unknown_section_is_an_error(bench):
    with pytest.raises(ValueError) as exc:
        to_csv(bench, "training-data")
    assert "training-data" in str(exc.value)


# --------------------------------------------------------------------------
# AC12 nothing here loads the machine it runs on
# --------------------------------------------------------------------------


def test_the_module_cannot_reach_a_process_or_the_network():
    """Static check, same shape as `test_lab`'s.

    The AC asks for the literal string "socket" to be absent, which is impossible
    in a tool that sizes dual-socket servers -- `vm.sockets` is part of the frozen
    contract. So the import graph is what gets checked, which is the property that
    actually matters, and the process/network module names that *can* be banned as
    plain text are.
    """
    source = MODELBENCH_SRC.read_text(encoding="utf-8")
    banned_modules = {"subprocess", "socket", "urllib", "http", "ssl", "asyncio",
                      "multiprocessing", "ctypes", "threading", "requests",
                      "httpx", "torch", "llama_cpp", "numpy", "time", "random"}

    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module.split(".")[0])
    assert not (imported & banned_modules), sorted(imported & banned_modules)

    # "requests" is not in this list: the word appears in prose about HTTP-free
    # concurrent requests, and the import check above already covers the library.
    for text in ("subprocess", "urllib", "Popen", "urlopen", "popen", "sleep"):
        assert text not in source, text


def test_the_module_only_imports_from_this_project_and_the_standard_library():
    """A bench that needed a wheel installed could not run on an air-gapped box."""
    source = MODELBENCH_SRC.read_text(encoding="utf-8")
    allowed_stdlib = {"csv", "io", "dataclasses", "typing", "__future__", "math"}

    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] in allowed_stdlib, alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                assert node.module.split(".")[0] in allowed_stdlib, node.module


def test_the_grid_carries_the_two_numbers_people_actually_feel(bench):
    """Generation tok/s and TTFT, under the names the industry uses.

    These are the two figures an engineer quotes: how fast does it type, and
    how long before it starts. Both were derivable from the grid but neither
    was named, so a screen had to translate `decode_tps_single` and recompute
    the prompt time itself.
    """
    for point in bench.throughput:
        assert point.gen_tps == pytest.approx(point.decode_tps_single, rel=1e-12)
        assert point.ttft_s == pytest.approx(
            point.ctx_tokens / point.prefill_tps, rel=1e-9
        )
        assert point.ttft_s > 0


def test_a_longer_prompt_takes_longer_before_the_first_token(bench):
    """TTFT is prompt processing, so it grows with the prompt."""
    single = sorted(
        (p for p in bench.throughput if p.batch == 1), key=lambda p: p.ctx_tokens
    )
    ttfts = [p.ttft_s for p in single]
    assert ttfts == sorted(ttfts)
    assert ttfts[-1] > ttfts[0] * 2, ttfts


def test_the_throughput_csv_exports_both(bench):
    from svrspec.modelbench import to_csv

    header = to_csv(bench, "throughput").splitlines()[0]
    assert "ttft_s" in header
    assert "gen_tps" in header


def test_ram_moves_across_the_grid_instead_of_being_one_number(bench):
    """A single memory figure for the whole grid hid a tenfold range.

    llama.cpp reserves the full context per slot up front, so batch 32 at 16k
    context wants an order of magnitude more RAM than batch 1 at 512. Reporting
    one number let a reader size the box from the summary and then find the
    bottom-right of the table would not load.
    """
    rams = [p.ram_gb for p in bench.throughput]
    assert min(rams) > 0
    assert max(rams) > min(rams) * 5, rams

    # Monotone in both axes: more sequences and longer contexts cost more.
    by_key = {(p.batch, p.ctx_tokens): p.ram_gb for p in bench.throughput}
    batches = sorted({p.batch for p in bench.throughput})
    contexts = sorted({p.ctx_tokens for p in bench.throughput})
    for ctx in contexts:
        series = [by_key[(b, ctx)] for b in batches]
        assert series == sorted(series), (ctx, series)
    for batch in batches:
        series = [by_key[(batch, c)] for c in contexts]
        assert series == sorted(series), (batch, series)


def test_a_point_that_does_not_fit_is_marked(catalog):
    """The grid must say which of its own cells the machine cannot run."""
    from svrspec.lab import VirtualMachine, assemble
    from svrspec.modelbench import bench_model

    # 8 x 8GB fills every channel but is only 64GB.
    asm = assemble(catalog, VirtualMachine(
        name="t", cpu_id="test-amx-8ch", sockets=1, dimm_gb=8, dimm_count=8,
        model_id="test-8b-gqa", quant_id="Q4_K_M", slots=4,
    ))
    bench = bench_model(asm, contexts=(512, 32768), batches=(1, 32))

    assert any(p.fits for p in bench.throughput), "something must fit"
    tight = max(bench.throughput, key=lambda p: p.ram_gb)
    assert tight.ram_gb > asm.ram_total_gb
    assert tight.fits is False


def test_the_os_changes_what_is_left_for_the_model(catalog):
    """Windows does not hand llama.cpp the same memory Linux does."""
    from svrspec.lab import VirtualMachine, assemble
    from svrspec.memory import OS_PROFILES
    from svrspec.modelbench import bench_model

    vm = VirtualMachine(
        name="t", cpu_id="test-amx-8ch", sockets=1, dimm_gb=64, dimm_count=8,
        model_id="test-8b-gqa", quant_id="Q4_K_M", slots=4,
    )
    asm = assemble(catalog, vm)
    lean = bench_model(asm, os_name="linux-container", contexts=(4096,), batches=(1,))
    fat = bench_model(asm, os_name="windows-desktop", contexts=(4096,), batches=(1,))

    assert lean.throughput[0].ram_gb < fat.throughput[0].ram_gb
    gap = fat.throughput[0].ram_gb - lean.throughput[0].ram_gb
    expected = (OS_PROFILES["windows-desktop"].runtime_gb
                - OS_PROFILES["linux-container"].runtime_gb)
    assert gap == pytest.approx(expected, rel=1e-6)

    # Throughput is a property of the silicon, not of the OS.
    assert lean.throughput[0].gen_tps == pytest.approx(fat.throughput[0].gen_tps)
