"""Tests for `svrspec.lab`: does assembling a virtual server catch bad builds?

These run against the **shipped** catalogue rather than `tests/data`, on purpose.
The module's whole job is to say which DIMM capacities exist and what happens
when the one you need does not -- a synthetic catalogue would let the tests agree
with themselves about parts nobody can buy. Where a test depends on a catalogue
fact (16 GB modules are not carried), it asserts that fact first, so a catalogue
change is reported as a catalogue change instead of as a mystery failure.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from svrspec.catalog import Catalog
from svrspec.lab import (
    MAX_DPC,
    Assembly,
    Finding,
    VirtualMachine,
    assemble,
    dimm_options,
    load,
    save,
    to_service,
)
from svrspec.perf import Efficiency, predict_throughput
from svrspec.types import TokenProfile, Workload

LAB_SRC = Path(__file__).resolve().parents[1] / "svrspec" / "lab.py"

#: Eight memory channels, DDR5-4800, 16 cores. The core count matters: at 25
#: GB/s per core the bandwidth roofline stays the DRAM one at both full and
#: quarter population, so a channel-count change shows up undiluted.
CPU_8CH = "xeon-gold-6426y"
#: Two channels, 128 GB ceiling, single socket. For the limit findings.
CPU_SMALL = "xeon-e-2488"
#: Six channels, single socket only.
CPU_1P = "epyc-8324p"

MODEL = "llama-3.1-8b-instruct"
TINY = "qwen2.5-0.5b-instruct"
QUANT = "Q4_K_M"


@pytest.fixture(scope="module")
def cat() -> Catalog:
    return Catalog()


def vm(**over) -> VirtualMachine:
    """A sane eight-channel build, overridable field by field."""
    base = dict(
        name="test",
        cpu_id=CPU_8CH,
        sockets=1,
        dimm_gb=32,
        dimm_count=8,
        model_id=MODEL,
        quant_id=QUANT,
        slots=2,
    )
    base.update(over)
    return VirtualMachine(**base)


def codes(asm: Assembly) -> list[str]:
    return [f.code for f in asm.findings]


def finding(asm: Assembly, code: str) -> Finding:
    for f in asm.findings:
        if f.code == code:
            return f
    raise AssertionError(f"expected a {code!r} finding, got {codes(asm)}")


# --------------------------------------------------------------------------
# AC1 the build this module exists for: right capacity, quarter bandwidth
# --------------------------------------------------------------------------


def test_two_64gb_dimms_in_an_eight_channel_board_quarter_the_bandwidth(cat):
    asm = assemble(cat, vm(dimm_gb=64, dimm_count=2))

    assert asm.channels_total == 8
    assert asm.channels_populated == 2
    assert asm.dimms_per_channel == 1
    assert asm.ram_total_gb == 128

    f = finding(asm, "channels-underfilled")
    assert f.level == "warn"
    # Capacity is fine, so this must not be an error -- the build boots.
    assert asm.ok is True

    assert asm.bandwidth_gbs == pytest.approx(asm.bandwidth_full_gbs / 4)
    assert asm.decode_tps_single == pytest.approx(asm.decode_tps_full / 4)
    # And the numbers the operator reads are in the message, not just the fields.
    assert "2/8" in f.message
    assert "75%" in f.message


def test_the_underfilled_finding_scales_with_how_underfilled_it_is(cat):
    half = assemble(cat, vm(dimm_gb=64, dimm_count=4))
    assert half.channels_populated == 4
    assert half.bandwidth_gbs == pytest.approx(half.bandwidth_full_gbs / 2)
    assert "50%" in finding(half, "channels-underfilled").message


# --------------------------------------------------------------------------
# AC2 the remedy has to be one the operator can actually follow
# --------------------------------------------------------------------------


def test_the_underfilled_remedy_says_the_module_size_needed_is_not_sold(cat):
    cpu = cat.cpu(CPU_8CH)
    available = {
        m.dimm_gb for m in cat.memory
        if m.ddr_gen == cpu.ddr_gen
        and m.dimms_per_channel == 1
        and m.rated_mts <= cpu.max_ddr_mts
    }
    # The premise: 128 GB over 8 channels wants 16 GB modules, which the
    # catalogue does not carry. If this ever changes, the remedy below changes
    # shape and this test is the place that should say so.
    assert 16 not in available, f"catalogue now carries 16GB DIMMs: {available}"
    assert 32 in available

    remedy = finding(assemble(cat, vm(dimm_gb=64, dimm_count=2)),
                     "channels-underfilled").remedy

    assert "8장" in remedy or "8×" in remedy      # fill every channel
    assert "16GB" in remedy                       # what that would take
    assert "카탈로그" in remedy                     # ...and that it is not sold
    assert "32GB" in remedy                       # the smallest one that is
    assert "256GB" in remedy                      # so this is the real option


def test_the_remedy_names_the_split_when_the_module_size_does_exist(cat):
    # 256 GB on 8 channels is 8x32GB, which is catalogued -- so the advice is to
    # redistribute, not to buy more.
    remedy = finding(assemble(cat, vm(dimm_gb=64, dimm_count=4)),
                     "channels-underfilled").remedy
    assert "8×32GB" in remedy
    assert "카탈로그" not in remedy


# --------------------------------------------------------------------------
# AC3 a fully populated board is clean
# --------------------------------------------------------------------------


def test_filling_every_channel_raises_nothing(cat):
    asm = assemble(cat, vm(dimm_gb=32, dimm_count=8))

    assert codes(asm) == []
    assert asm.ok is True
    assert asm.channels_populated == asm.channels_total == 8
    assert asm.bandwidth_gbs == asm.bandwidth_full_gbs
    assert asm.decode_tps_single == asm.decode_tps_full
    assert asm.memory.effective_mts == asm.memory.rated_mts


def test_a_two_socket_build_fills_both(cat):
    asm = assemble(cat, vm(sockets=2, dimm_gb=32, dimm_count=16))
    assert asm.channels_total == 16
    assert asm.channels_populated == 16
    assert asm.channels_per_socket == 8
    assert codes(asm) == []


# --------------------------------------------------------------------------
# AC4 two DIMMs per channel clocks the whole bus down
# --------------------------------------------------------------------------


def test_two_dimms_per_channel_derates_the_memory_speed(cat):
    asm = assemble(cat, vm(dimm_gb=32, dimm_count=16))

    assert asm.dimms_per_channel == 2
    assert asm.channels_populated == asm.channels_total == 8
    assert "channels-underfilled" not in codes(asm)   # every channel is occupied
    assert asm.ram_total_gb == 512

    f = finding(asm, "dpc-derate")
    assert f.level == "warn"
    assert asm.memory.dimms_per_channel == 2
    assert asm.memory.effective_mts < asm.memory.rated_mts
    assert str(asm.memory.rated_mts) in f.message
    assert str(asm.memory.effective_mts) in f.message
    # The derate is real throughput, not just a label.
    assert asm.decode_tps_single < assemble(cat, vm(dimm_gb=64, dimm_count=8)).decode_tps_single


def test_more_than_two_dimms_per_channel_is_an_error(cat):
    asm = assemble(cat, vm(dimm_gb=32, dimm_count=24))
    f = finding(asm, "too-many-dimms")
    assert f.level == "error"
    assert asm.ok is False
    assert "16" in f.remedy               # 8 channels x 2 DPC
    assert asm.dimms_per_channel == MAX_DPC


# --------------------------------------------------------------------------
# AC5 the errors: does not boot, does not fit, does not exist
# --------------------------------------------------------------------------


def test_ram_smaller_than_the_model_is_an_error(cat):
    asm = assemble(cat, vm(model_id="qwen3-32b", quant_id="Q8_0",
                           dimm_gb=32, dimm_count=1))
    f = finding(asm, "ram-too-small")
    assert f.level == "error"
    assert asm.ok is False
    assert asm.ram_total_gb == 32
    assert asm.ram_used_gb > asm.ram_total_gb
    assert "32GB" in f.message


def test_more_ram_than_the_cpu_can_address_is_an_error(cat):
    cpu = cat.cpu(CPU_SMALL)
    asm = assemble(cat, vm(cpu_id=CPU_SMALL, dimm_gb=64, dimm_count=4))
    f = finding(asm, "ram-exceeds-cpu")
    assert f.level == "error"
    assert asm.ok is False
    assert asm.ram_total_gb == 256 > cpu.max_mem_gb
    assert str(cpu.max_mem_gb) in f.message


def test_more_sockets_than_the_part_supports_is_an_error(cat):
    cpu = cat.cpu(CPU_1P)
    assert cpu.sockets_max == 1
    asm = assemble(cat, vm(cpu_id=CPU_1P, sockets=2, dimm_gb=32, dimm_count=12))
    f = finding(asm, "sockets-exceeded")
    assert f.level == "error"
    assert asm.ok is False
    assert "1" in f.message


def test_an_empty_board_is_an_error(cat):
    asm = assemble(cat, vm(dimm_count=0))
    assert finding(asm, "no-dimms").level == "error"
    assert asm.ok is False
    assert asm.ram_total_gb == 0
    # No capacity findings piled on top: one problem, one line.
    assert "channels-underfilled" not in codes(asm)
    assert "ram-too-small" not in codes(asm)


def test_errors_sort_ahead_of_warnings(cat):
    asm = assemble(cat, vm(model_id="qwen3-32b", quant_id="Q8_0",
                           dimm_gb=32, dimm_count=1))
    assert {"ram-too-small", "channels-underfilled"} <= set(codes(asm))
    levels = [f.level for f in asm.findings]
    assert levels == sorted(levels, key=("error", "warn", "info").index)
    assert levels[0] == "error"


# --------------------------------------------------------------------------
# The softer findings
# --------------------------------------------------------------------------


def test_dimms_that_cannot_be_split_evenly_are_flagged(cat):
    asm = assemble(cat, vm(sockets=2, dimm_gb=32, dimm_count=5))
    assert finding(asm, "dimms-not-divisible").level == "warn"


def test_an_awkward_channel_count_is_only_information(cat):
    asm = assemble(cat, vm(dimm_gb=64, dimm_count=3))
    f = finding(asm, "odd-channels")
    assert f.level == "info"
    assert asm.channels_populated == 3


def test_a_full_board_is_never_called_awkward(cat):
    # Twelve channels is not a power of two and is exactly what AMD ships.
    asm = assemble(cat, vm(cpu_id="epyc-9354", dimm_gb=32, dimm_count=12))
    assert asm.channels_total == 12
    assert "odd-channels" not in codes(asm)
    assert codes(asm) == []


def test_a_dimm_size_the_catalogue_lacks_is_flagged_not_silently_accepted(cat):
    asm = assemble(cat, vm(dimm_gb=16, dimm_count=8))
    f = finding(asm, "dimm-not-catalogued")
    assert f.level == "warn"
    assert "16GB" in f.message
    # Capacity still follows what was asked for; only the speed grade is borrowed.
    assert asm.ram_total_gb == 128
    assert asm.memory.dimm_gb != 16


# --------------------------------------------------------------------------
# AC6 a GUI calls this on every keystroke, so it may never raise
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "over",
    [
        {"dimm_count": 0},
        {"dimm_count": -5},
        {"dimm_count": 1000},
        {"dimm_gb": 0},
        {"dimm_gb": 1},
        {"dimm_gb": 4096, "dimm_count": 8},
        {"sockets": 0},
        {"sockets": -1},
        {"sockets": 99},
        {"slots": 0},
        {"slots": -3},
        {"slots": 512},
        {"cpu_id": CPU_SMALL, "dimm_count": 1},
        {"cpu_id": CPU_SMALL, "dimm_count": 1, "dimm_gb": 64},
        {"cpu_id": CPU_1P, "sockets": 4, "dimm_count": 3},
        {"model_id": "llama-3.3-70b-instruct", "quant_id": "F16"},
        {"name": ""},
    ],
)
def test_assemble_never_raises_on_a_bad_configuration(cat, over):
    asm = assemble(cat, vm(**over))
    assert isinstance(asm, Assembly)
    assert isinstance(asm.ok, bool)
    # Whatever the input, the derived facts stay self-consistent.
    assert asm.channels_populated <= asm.channels_total
    assert 1 <= asm.dimms_per_channel <= MAX_DPC
    assert asm.bandwidth_gbs >= 0
    assert asm.decode_tps_single >= 0
    assert all(f.level in ("error", "warn", "info") for f in asm.findings)
    assert all(f.message for f in asm.findings)


def test_an_unknown_catalogue_id_still_raises(cat):
    from svrspec.catalog import CatalogError

    with pytest.raises(CatalogError):
        assemble(cat, vm(cpu_id="no-such-cpu"))
    with pytest.raises(CatalogError):
        assemble(cat, vm(model_id="no-such-model"))


# --------------------------------------------------------------------------
# AC7 the loss survives into the runnable pipeline
# --------------------------------------------------------------------------


def test_to_service_carries_the_channel_loss_into_the_service_rates(cat):
    workload = Workload(slots=2)
    starved = to_service(cat, assemble(cat, vm(dimm_gb=64, dimm_count=2)), workload)
    full = to_service(cat, assemble(cat, vm(dimm_gb=64, dimm_count=8)), workload)

    assert set(starved.decode_by_active) == {1, 2}
    for k in (1, 2):
        assert starved.decode_by_active[k] < full.decode_by_active[k]
    assert sum(starved.decode_by_active.values()) < sum(full.decode_by_active.values())
    # Decode is bandwidth bound and prefill is not, so the cut lands on decode.
    assert starved.prefill_tps == pytest.approx(full.prefill_tps)
    assert starved.slots == 2
    assert starved.uncertainty > 0
    assert "2/8ch" in starved.label


def test_to_service_agrees_with_the_assembly_it_came_from(cat):
    workload = Workload(slots=2)
    asm = assemble(cat, vm(dimm_gb=64, dimm_count=2))
    service = to_service(cat, asm, workload)
    assert service.prefill_tps == pytest.approx(asm.prefill_tps)


def test_to_service_refuses_a_build_that_cannot_run(cat):
    asm = assemble(cat, vm(model_id="qwen3-32b", quant_id="Q8_0",
                           dimm_gb=32, dimm_count=1))
    assert asm.ok is False
    with pytest.raises(ValueError) as exc:
        to_service(cat, asm, Workload(slots=2))
    assert "ram-too-small" in str(exc.value)


def test_the_service_model_actually_drives_a_pipeline_run(cat):
    """End to end: an assembled build is something you can watch run."""
    from svrspec.mockdata import generate_day
    from svrspec.pipeline import run_pipeline

    workload = Workload(alarms_per_day=40, storms_per_day=0, slots=2)
    alarms = generate_day(count=40, storms_per_day=0).alarms

    starved, _ = run_pipeline(alarms, to_service(cat, assemble(
        cat, vm(dimm_gb=64, dimm_count=2)), workload), workload)
    full, _ = run_pipeline(alarms, to_service(cat, assemble(
        cat, vm(dimm_gb=64, dimm_count=8)), workload), workload)

    assert starved.delivered == full.delivered == 40
    assert starved.p95_s > full.p95_s


# --------------------------------------------------------------------------
# AC8 saved builds survive a round trip
# --------------------------------------------------------------------------


def test_save_load_round_trips_the_choice(cat, tmp_path):
    original = vm(name="현장 견적 A", dimm_gb=64, dimm_count=2, sockets=1, slots=3)
    path = tmp_path / "build.json"

    save(assemble(cat, original), path)
    reloaded = load(cat, path)

    assert reloaded.vm == original
    assert reloaded.channels_populated == 2
    assert "channels-underfilled" in codes(reloaded)


def test_save_accepts_a_bare_virtual_machine(cat, tmp_path):
    path = tmp_path / "vm.json"
    save(vm(dimm_count=4), path)
    assert load(cat, path).vm == vm(dimm_count=4)


def test_the_saved_file_holds_no_derived_number(cat, tmp_path):
    """A file that stored a prediction would outvote a corrected catalogue."""
    path = tmp_path / "vm.json"
    save(assemble(cat, vm()), path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert set(doc) == {"schema", "vm"}
    assert set(doc["vm"]) == {"name", "cpu_id", "sockets", "dimm_gb",
                             "dimm_count", "model_id", "quant_id", "slots"}


@pytest.mark.parametrize(
    "doc",
    [
        {"vm": {}},                                        # no schema
        {"schema": "svrspec-vm/v2", "vm": {}},             # wrong version
        {"schema": "svrspec-vm/v1"},                       # no vm
        {"schema": "svrspec-vm/v1", "vm": []},             # vm not an object
        {"schema": "svrspec-vm/v1", "vm": {"name": "x"}},  # missing fields
    ],
)
def test_load_rejects_a_malformed_file(cat, tmp_path, doc):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError):
        load(cat, path)


def test_load_rejects_a_wrongly_typed_field(cat, tmp_path):
    good = {"name": "x", "cpu_id": CPU_8CH, "sockets": 1, "dimm_gb": 32,
            "dimm_count": 8, "model_id": MODEL, "quant_id": QUANT, "slots": 2}
    for field, bad in (("sockets", "1"), ("dimm_gb", 32.5), ("dimm_count", True),
                       ("name", 3), ("slots", 0), ("extra", "x")):
        path = tmp_path / f"bad-{field}.json"
        payload = dict(good)
        payload[field] = bad
        path.write_text(json.dumps({"schema": "svrspec-vm/v1", "vm": payload}),
                        encoding="utf-8")
        with pytest.raises(ValueError, match=field):
            load(cat, path)


def test_load_rejects_broken_json(cat, tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        load(cat, path)


# --------------------------------------------------------------------------
# AC9 the picker only offers buildable combinations
# --------------------------------------------------------------------------


@pytest.mark.parametrize("cpu_id,sockets", [
    (CPU_8CH, 1), (CPU_8CH, 2), ("epyc-9354", 1), (CPU_SMALL, 1), (CPU_1P, 1),
])
def test_dimm_options_offers_only_buildable_combinations(cat, cpu_id, sockets):
    cpu = cat.cpu(cpu_id)
    rows = dimm_options(cat, cpu, sockets)
    assert rows

    catalogued = {m.dimm_gb for m in cat.memory
                  if m.ddr_gen == cpu.ddr_gen and m.rated_mts <= cpu.max_ddr_mts}

    for row in rows:
        assert row["dimm_gb"] in catalogued
        assert 1 <= row["dimms_per_channel"] <= MAX_DPC
        assert row["ram_total_gb"] == row["dimm_gb"] * row["dimm_count"]
        assert row["ram_total_gb"] <= cpu.max_mem_gb * sockets
        assert row["channels_total"] == cpu.mem_channels * sockets
        assert 1 <= row["channels_populated"] <= row["channels_total"]
        assert row["dimm_count"] % sockets == 0        # splits across sockets

        # The real check: assembling the row reproduces it, with no error.
        asm = assemble(cat, vm(cpu_id=cpu_id, sockets=sockets, model_id=TINY,
                               dimm_gb=row["dimm_gb"], dimm_count=row["dimm_count"]))
        assert asm.ok, codes(asm)
        assert asm.ram_total_gb == row["ram_total_gb"]
        assert asm.channels_populated == row["channels_populated"]
        assert asm.dimms_per_channel == row["dimms_per_channel"]
        assert asm.memory.effective_mts == row["effective_mts"]
        assert row["full_channels"] == (
            asm.channels_populated == asm.channels_total
        )
        assert "dimm-not-catalogued" not in codes(asm)


def test_dimm_options_includes_the_full_board_and_flags_which_rows_are(cat):
    rows = dimm_options(cat, cat.cpu(CPU_8CH), 1)
    full = [r for r in rows if r["full_channels"]]
    assert full
    assert any(r["dimm_count"] == 8 and r["dimm_gb"] == 32 for r in full)
    # And the notorious one is still offered -- you cannot warn about a build the
    # picker refuses to express.
    assert any(r["dimm_count"] == 2 and r["dimm_gb"] == 64
               and not r["full_channels"] for r in rows)


def test_dimm_options_respects_a_small_memory_ceiling(cat):
    cpu = cat.cpu(CPU_SMALL)
    rows = dimm_options(cat, cpu, 1)
    assert rows
    assert max(r["ram_total_gb"] for r in rows) <= cpu.max_mem_gb


def test_dimm_options_is_sorted_by_capacity(cat):
    rows = dimm_options(cat, cat.cpu(CPU_8CH), 1)
    assert [r["ram_total_gb"] for r in rows] == sorted(r["ram_total_gb"] for r in rows)


# --------------------------------------------------------------------------
# AC10 the physics is perf.py's, not a second derivation
# --------------------------------------------------------------------------


def test_a_full_board_matches_predict_throughput_exactly(cat):
    tokens = TokenProfile()
    machine = vm(dimm_gb=32, dimm_count=8, slots=2)
    asm = assemble(cat, machine, tokens)

    direct = predict_throughput(
        cat.model(MODEL), cat.quant(QUANT), cat.cpu(CPU_8CH), asm.memory,
        tokens, Efficiency.from_catalog(cat.coefficients),
        slots=2, sockets=1,
    )

    assert asm.prefill_tps == direct.prefill_tps
    assert asm.decode_tps_single == direct.decode_tps_single
    assert asm.bandwidth_gbs == direct.effective_bandwidth_gbs
    assert asm.uncertainty == direct.uncertainty


def test_an_underfilled_board_matches_predict_throughput_exactly(cat):
    tokens = TokenProfile()
    asm = assemble(cat, vm(dimm_gb=64, dimm_count=2, slots=2), tokens)

    direct = predict_throughput(
        cat.model(MODEL), cat.quant(QUANT), cat.cpu(CPU_8CH), asm.memory,
        tokens, Efficiency.from_catalog(cat.coefficients),
        slots=2, sockets=1, channels_populated=2,
    )

    assert asm.decode_tps_single == direct.decode_tps_single
    assert asm.bandwidth_gbs == direct.effective_bandwidth_gbs


def test_the_token_profile_reaches_the_prediction(cat):
    """A longer context costs decode speed, so the argument cannot be ignored."""
    short = assemble(cat, vm(), TokenProfile(output_tokens=64))
    long = assemble(cat, vm(), TokenProfile(output_tokens=4096))
    assert long.decode_tps_single < short.decode_tps_single


# --------------------------------------------------------------------------
# AC11 nothing here loads the machine it runs on
# --------------------------------------------------------------------------


def test_the_module_cannot_reach_a_process_or_the_network():
    """Static check, same shape as `test_pipeline`'s.

    Substring scanning for "socket" is impossible here -- the frozen contract has
    a `sockets` field -- so the import graph is what gets checked, which is the
    property that actually matters. `subprocess` and the HTTP client libraries are
    additionally forbidden as plain text.
    """
    source = LAB_SRC.read_text(encoding="utf-8")
    banned_modules = {"subprocess", "socket", "urllib", "http", "ssl", "asyncio",
                      "multiprocessing", "ctypes", "requests", "httpx", "torch",
                      "llama_cpp", "numpy"}

    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module.split(".")[0])
    assert not (imported & banned_modules), sorted(imported & banned_modules)

    for text in ("subprocess", "urllib", "requests", "Popen", "urlopen", "popen"):
        assert text not in source, text
