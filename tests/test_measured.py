"""Measured-log import.

The load-bearing test is `test_devbox_log_reproduces_the_catalog_ddr4_value`:
the coefficients in `catalog/coefficients.json` were derived by hand from a
llama-bench run on the development box, and this module automates exactly that
derivation. If it does not land on the hand-computed number, one of the two is
wrong.

Everything here reads text off disk. Nothing runs llama.cpp -- there is a test
asserting the module cannot, because the servers being sized are not here.
"""

import dataclasses
import json
from dataclasses import replace
from pathlib import Path

import pytest

from svrspec.catalog import Catalog, load_coefficients
from svrspec.measured import (
    Calibration,
    MeasuredPoint,
    compare_to_prediction,
    derive_eta_bw,
    derive_eta_compute,
    parse_llama_bench,
    parse_memory,
    parse_server_log,
    quant_from_label,
)
from svrspec.types import SOURCE_MEASUREMENT, CoefficientSpec, CpuSpec, MemoryOption, ThroughputPrediction

DATA = Path(__file__).parent / "data" / "measured"

DEVBOX_MD = DATA / "llama-bench-devbox.md"
DEVBOX_MD_REORDERED = DATA / "llama-bench-devbox-reordered.md"
EPYC_JSON = DATA / "llama-bench-epyc.json"
SERVER_LOG = DATA / "llama-server-devbox.log"
GARBAGE = DATA / "garbage.txt"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# The hardware the one real measurement came from.
#
# Deliberately built here rather than pulled from `svrspec/catalog/cpus.json`:
# that catalogue holds server SKUs a customer might buy, and a desktop
# development box does not belong in it. The model, quant and coefficients below
# do come from the real catalogue -- see `catalog` fixture usage.
# --------------------------------------------------------------------------

DEVBOX_CPU = CpuSpec(
    id="devbox-i7-12700",
    vendor="Intel",
    family="Core 12th Gen (Alder Lake)",
    model="i7-12700",
    cores=12,  # 8 P-cores + 4 E-cores; see the eta_compute test
    threads=20,
    base_ghz=2.1,
    all_core_turbo_ghz=3.6,  # assumed sustained clock, per the catalogue notes
    max_turbo_ghz=4.9,
    isa=["avx2"],
    mem_channels=2,
    ddr_gen="DDR4",
    max_ddr_mts=3200,
    max_mem_gb=128,
    sockets_max=1,
    l3_mb=25.0,
    tdp_w=65,
    notes="개발 박스. 기본 계수가 유도된 실측 하드웨어.",
)

DEVBOX_MEMORY = MemoryOption(
    id="devbox-ddr4-2667",
    ddr_gen="DDR4",
    rated_mts=2667,
    dimms_per_channel=2,
    effective_mts=2667,
    dimm_gb=16,
    ecc=False,
    kind="UDIMM",
)


@pytest.fixture
def real_catalog() -> Catalog:
    """The shipped catalogue, not the test fixtures.

    The point of this module is to reproduce the coefficients that actually
    ship, so it has to read the real models, quants and coefficients.
    """
    return Catalog()


@pytest.fixture
def devbox_points():
    return parse_llama_bench(read(DEVBOX_MD), str(DEVBOX_MD))


def pick(points, kind: str) -> MeasuredPoint:
    matching = [p for p in points if p.kind == kind]
    assert len(matching) == 1, f"expected exactly one {kind} point, got {len(matching)}"
    return matching[0]


# --------------------------------------------------------------------------
# AC#1 -- all three formats parse
# --------------------------------------------------------------------------


def test_markdown_table_parses_both_phases(devbox_points):
    assert [p.kind for p in devbox_points] == ["pp", "tg"]

    pp = pick(devbox_points, "pp")
    assert pp.tokens_per_s == 25.48
    assert pp.stddev == 0.31
    assert pp.n_batch == 512
    assert pp.n_threads == 10
    assert pp.params_b == 8.03
    assert pp.quant == "Q4_K_M"  # "Q4_K - Medium" spelled out in the model column
    assert pp.backend == "CPU"
    assert pp.model_label == "llama 8B Q4_K - Medium"
    assert pp.source == str(DEVBOX_MD)
    # The raw row survives for audit: a shipped coefficient must be re-checkable
    # against the text it came from.
    assert pp.raw["size"] == "4.58 GiB"

    tg = pick(devbox_points, "tg")
    assert (tg.tokens_per_s, tg.stddev) == (6.07, 0.03)
    assert tg.n_ctx == 128  # tg128 generated 128 tokens from an empty cache


def test_json_output_parses():
    points = parse_llama_bench(read(EPYC_JSON), str(EPYC_JSON))
    assert [p.kind for p in points] == ["pp", "tg"]

    pp = pick(points, "pp")
    assert pp.tokens_per_s == 210.35
    assert pp.stddev == 1.82
    assert pp.n_threads == 32
    assert pp.backend == "CPU"
    assert pp.quant == "Q4_K_M"  # from the .gguf filename
    assert abs(pp.params_b - 8.0303) < 0.001  # model_n_params / 1e9
    assert pp.n_ctx == 512

    tg = pick(points, "tg")
    assert tg.tokens_per_s == 30.12
    assert tg.n_ctx == 128


def test_server_log_parses_both_phases():
    points = parse_server_log(read(SERVER_LOG), "ticket-4471")
    pp = pick(points, "pp")
    tg = pick(points, "tg")

    assert pp.tokens_per_s == 41.47
    assert pp.n_batch == 512
    assert tg.tokens_per_s == 6.40
    # 512 prompt tokens were already resident when generation started, so the
    # cache held 640 by the end. Not the allocated n_ctx of 4096.
    assert tg.n_ctx == 640
    assert tg.stddev is None  # a single run reports no spread
    assert tg.quant == "Q4_K_M"
    assert tg.n_threads == 10
    assert tg.source == "ticket-4471"


def test_parse_memory_reads_the_lines_verify_compares():
    mem = parse_memory(read(SERVER_LOG), str(SERVER_LOG))
    assert mem.kv_self_mib == 512.0
    assert mem.compute_buffer_mib == 296.02
    assert abs(mem.model_size_mib - 4.58 * 1024) < 1.0  # 4.58 GiB
    assert mem.n_ctx == 4096  # the *allocated* context, unlike the point's n_ctx
    assert mem.n_slots == 2


# --------------------------------------------------------------------------
# AC#2 -- header matching, not column position
# --------------------------------------------------------------------------


def test_column_order_does_not_matter(devbox_points):
    reordered = parse_llama_bench(read(DEVBOX_MD_REORDERED), str(DEVBOX_MD))

    def comparable(points):
        return [
            (p.kind, p.tokens_per_s, p.stddev, p.model_label, p.params_b, p.quant,
             p.n_threads, p.n_batch, p.n_ctx, p.backend)
            for p in points
        ]

    assert comparable(reordered) == comparable(devbox_points)


# --------------------------------------------------------------------------
# AC#3 -- broken input is empty, never an exception
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   \n\n  ",
        "not a log at all",
        "[{oops",  # JSON that does not parse
        "{}",  # JSON with no benchmark rows
        "[]",
        "| a | b |\n| - | - |\n| 1 | 2 |",  # a table, but not a llama-bench one
        "| model | test | t/s |\n| - | - | - |\n| llama | pp512+tg128 | 9.9 |",  # combined row
        "| model | test | t/s |\n| - | - | - |\n| llama | pp512 |",  # ragged row
        "llama_perf_context_print: prompt eval time = broken",
    ],
)
def test_broken_input_yields_nothing(text):
    assert parse_llama_bench(text, "x") == []
    assert parse_server_log(text, "x") == []
    empty = parse_memory(text, "x")
    assert (empty.kv_self_mib, empty.compute_buffer_mib, empty.model_size_mib) == (None, None, None)


def test_a_non_log_attachment_is_rejected_quietly():
    """A customer replying with an e-mail instead of a log must not crash."""
    text = read(GARBAGE)
    assert parse_llama_bench(text, str(GARBAGE)) == []
    assert parse_server_log(text, str(GARBAGE)) == []


def test_unknown_quantisation_stays_unknown():
    """Never guess the quant: it moves bytes-per-weight, and so every result."""
    assert quant_from_label("some finetune of llama 8B") is None
    assert quant_from_label("") is None
    assert quant_from_label("llama 8B Q4_K - Small") == "Q4_K_S"
    assert quant_from_label("model-IQ4_XS.gguf") == "IQ4_XS"
    assert quant_from_label("Meta-Llama-3.1-8B-F16.gguf") == "F16"


# --------------------------------------------------------------------------
# AC#4 -- reproduce the derivation the catalogue was built from
# --------------------------------------------------------------------------


def test_devbox_log_reproduces_the_catalog_ddr4_value(real_catalog, devbox_points):
    """2ch DDR4-2667, Llama-3.1-8B Q4_K_M, 6.07 tok/s -> eta_bw ~ 0.70.

    The catalogue ships 0.70 for DDR4, hand-derived from this exact run.
    This module lands on 0.7020, i.e. +0.29% -- the difference is the KV cache
    term: the hand derivation used weights only (6.07 x 4.918 GB / 42.672 GB/s
    = 0.6996), while this one adds the 128 tokens of KV the tg128 test
    accumulated (+16.8 MiB per token-read). The KV term is 0.34% of the weight
    bytes at this context, which is why the two agree to a third decimal.
    """
    model = real_catalog.model("llama-3.1-8b-instruct")
    quant = real_catalog.quant("Q4_K_M")
    previous = real_catalog.coefficient("eta_bw", "DDR4")

    cal = derive_eta_bw(
        pick(devbox_points, "tg"), model, quant, DEVBOX_CPU, DEVBOX_MEMORY, previous=previous
    )

    assert abs(cal.coefficient.value - 0.7020) < 0.0005, cal.coefficient.value
    assert abs(cal.coefficient.value - previous.value) < 0.01  # catalogue ships 0.70
    assert cal.coefficient.kind == "eta_bw"
    assert cal.coefficient.key == "DDR4"
    assert cal.coefficient.id == "eta-bw-ddr4"  # same id as the row it replaces
    assert cal.measured_tps == 6.07
    # 6.07 tok/s x 4.935 GB/token = 29.96 GB/s achieved of a 42.67 GB/s ceiling.
    assert abs(cal.implied_ceiling / 1e9 - 29.96) < 0.05
    assert "6.07 tok/s" in cal.basis and "42.7 GB/s" in cal.basis


def test_devbox_log_derives_the_avx2_compute_value(real_catalog, devbox_points):
    """25.48 tok/s prefill -> eta_compute for avx2.

    The catalogue ships 0.355, and this returns 0.296 from the *same* tok/s.
    The gap is entirely in what counts as a core: 0.355 is 25.48 x 2 x 8.03B /
    (10 x 3.6 GHz x 32 flop), i.e. ten vector-issuing cores, while the i7-12700
    has twelve (8 P + 4 E) and llama.cpp was run with `-t 10`. On a hybrid part
    the E-cores do not contribute a P-core's worth of AVX2 throughput, so the
    peak this test divides by is optimistic and the coefficient correspondingly
    lower. That is exactly the case the docstring says to downgrade for.
    """
    previous = real_catalog.coefficient("eta_compute", "avx2")
    model = real_catalog.model("llama-3.1-8b-instruct")
    pp = pick(devbox_points, "pp")

    cal = derive_eta_compute(pp, model, DEVBOX_CPU, previous=previous, confidence="derived")
    assert abs(cal.coefficient.value - 0.2960) < 0.0005, cal.coefficient.value
    assert cal.coefficient.confidence == "derived"
    assert cal.coefficient.key == "avx2"  # widest_isa of an AVX2-only part
    assert cal.coefficient.id == "eta-compute-avx2"

    # Counting the ten threads llama.cpp actually ran as ten cores reproduces
    # the shipped 0.355 to three decimals.
    ten_cores = derive_eta_compute(pp, model, replace(DEVBOX_CPU, cores=10))
    assert abs(ten_cores.coefficient.value - previous.value) < 0.001


# --------------------------------------------------------------------------
# AC#5, #6 -- refuse rather than assume
# --------------------------------------------------------------------------


def test_unknown_context_refuses_to_derive(real_catalog, devbox_points):
    tg = replace(pick(devbox_points, "tg"), n_ctx=None)
    with pytest.raises(ValueError, match="컨텍스트"):
        derive_eta_bw(
            tg,
            real_catalog.model("llama-3.1-8b-instruct"),
            real_catalog.quant("Q4_K_M"),
            DEVBOX_CPU,
            DEVBOX_MEMORY,
        )


def test_a_value_above_one_is_a_data_error_not_a_discovery(real_catalog, devbox_points):
    """100 tok/s on 2 channels of DDR4 would need 494 GB/s. Refuse it."""
    impossible = replace(pick(devbox_points, "tg"), tokens_per_s=100.0)
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        derive_eta_bw(
            impossible,
            real_catalog.model("llama-3.1-8b-instruct"),
            real_catalog.quant("Q4_K_M"),
            DEVBOX_CPU,
            DEVBOX_MEMORY,
        )


def test_compute_value_above_one_is_refused(real_catalog, devbox_points):
    impossible = replace(pick(devbox_points, "pp"), tokens_per_s=5000.0)
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        derive_eta_compute(impossible, real_catalog.model("llama-3.1-8b-instruct"), DEVBOX_CPU)


def test_the_wrong_phase_cannot_calibrate_a_coefficient(real_catalog, devbox_points):
    """Prefill measures the compute roofline; decode measures the memory one."""
    model = real_catalog.model("llama-3.1-8b-instruct")
    with pytest.raises(ValueError, match="eta_bw"):
        derive_eta_bw(
            pick(devbox_points, "pp"),
            model,
            real_catalog.quant("Q4_K_M"),
            DEVBOX_CPU,
            DEVBOX_MEMORY,
        )
    with pytest.raises(ValueError, match="eta_compute"):
        derive_eta_compute(pick(devbox_points, "tg"), model, DEVBOX_CPU)


def test_an_invalid_confidence_is_refused(real_catalog, devbox_points):
    with pytest.raises(ValueError, match="confidence"):
        derive_eta_compute(
            pick(devbox_points, "pp"),
            real_catalog.model("llama-3.1-8b-instruct"),
            DEVBOX_CPU,
            confidence="probably-fine",
        )


# --------------------------------------------------------------------------
# AC#7 -- the result is something the catalogue loader accepts
# --------------------------------------------------------------------------


def test_provenance_is_the_log_itself(real_catalog, devbox_points):
    cal = derive_eta_bw(
        pick(devbox_points, "tg"),
        real_catalog.model("llama-3.1-8b-instruct"),
        real_catalog.quant("Q4_K_M"),
        DEVBOX_CPU,
        DEVBOX_MEMORY,
    )
    assert cal.coefficient.source == SOURCE_MEASUREMENT
    assert cal.coefficient.source_url == str(DEVBOX_MD)
    assert cal.coefficient.confidence == "measured"
    assert cal.coefficient.notes  # the derivation, for the report's caveats


def test_a_log_without_an_identity_cannot_be_promoted(real_catalog, devbox_points):
    """`source` becomes `source_url`, which the catalogue loader requires."""
    anonymous = replace(pick(devbox_points, "tg"), source="  ")
    with pytest.raises(ValueError, match="source"):
        derive_eta_bw(
            anonymous,
            real_catalog.model("llama-3.1-8b-instruct"),
            real_catalog.quant("Q4_K_M"),
            DEVBOX_CPU,
            DEVBOX_MEMORY,
        )


def test_derived_coefficient_loads_back_into_a_catalog(real_catalog, devbox_points, tmp_path):
    """Round-trip: the whole point is to write this row into coefficients.json."""
    cal = derive_eta_bw(
        pick(devbox_points, "tg"),
        real_catalog.model("llama-3.1-8b-instruct"),
        real_catalog.quant("Q4_K_M"),
        DEVBOX_CPU,
        DEVBOX_MEMORY,
    )
    globals_needed = [
        real_catalog.coefficient("per_core_bw_gbs"),
        real_catalog.coefficient("dual_socket_efficiency"),
    ]
    path = tmp_path / "coefficients.json"
    path.write_text(
        json.dumps(
            {
                "schema": "coefficients/v1",
                "entries": [dataclasses.asdict(c) for c in [cal.coefficient, *globals_needed]],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    loaded = load_coefficients(path)
    row = next(c for c in loaded if c.id == "eta-bw-ddr4")
    assert row.confidence == "measured"
    assert row.source == SOURCE_MEASUREMENT
    assert abs(row.value - cal.coefficient.value) < 1e-12


# --------------------------------------------------------------------------
# AC#8 -- scoring a prediction
# --------------------------------------------------------------------------


def prediction(prefill: float, decode: float, uncertainty: float = 0.15) -> ThroughputPrediction:
    return ThroughputPrediction(
        prefill_tps=prefill,
        decode_tps_single=decode,
        decode_tps_aggregate=decode * 1.8,
        prefill_bound_by="compute",
        decode_bound_by="bandwidth",
        effective_bandwidth_gbs=29.9,
        peak_flops_tflops=1.15,
        uncertainty=uncertainty,
    )


def test_over_and_under_prediction_get_the_right_sign(devbox_points):
    tg = pick(devbox_points, "tg")  # 6.07 tok/s measured

    over = compare_to_prediction(tg, prediction(25.0, 7.5))
    assert over["verdict"] == "과대예측"
    assert over["error_pct"] > 0
    assert abs(over["error_pct"] - 23.56) < 0.05
    assert over["within_uncertainty"] is False
    assert over["kind"] == "tg"
    assert over["measured_tps"] == 6.07
    assert over["predicted_tps"] == 7.5

    under = compare_to_prediction(tg, prediction(25.0, 5.0))
    assert under["verdict"] == "과소예측"
    assert under["error_pct"] < 0

    # Prefill is judged against prefill_tps, not the decode figure.
    pp = pick(devbox_points, "pp")  # 25.48 tok/s
    assert compare_to_prediction(pp, prediction(25.0, 5.0))["verdict"] == "확인"


def test_within_uncertainty_follows_the_predictions_own_band(devbox_points):
    """An 18% error is a miss at +/-15% and a hit at +/-25%."""
    tg = pick(devbox_points, "tg")
    tight = compare_to_prediction(tg, prediction(25.0, 6.07 * 1.18, uncertainty=0.15))
    wide = compare_to_prediction(tg, prediction(25.0, 6.07 * 1.18, uncertainty=0.25))

    assert tight["within_uncertainty"] is False and tight["verdict"] == "과대예측"
    assert wide["within_uncertainty"] is True and wide["verdict"] == "확인"
    assert abs(tight["error_pct"] - wide["error_pct"]) < 1e-9


def test_an_unusable_point_cannot_be_compared():
    with pytest.raises(ValueError, match="pp"):
        compare_to_prediction(
            MeasuredPoint(source="x", kind="prompt", tokens_per_s=1.0), prediction(1.0, 1.0)
        )


# --------------------------------------------------------------------------
# AC#9 -- what accepting the new value would change
# --------------------------------------------------------------------------


def test_previous_value_yields_the_change(real_catalog, devbox_points):
    previous = real_catalog.coefficient("eta_bw", "DDR4")
    cal = derive_eta_bw(
        pick(devbox_points, "tg"),
        real_catalog.model("llama-3.1-8b-instruct"),
        real_catalog.quant("Q4_K_M"),
        DEVBOX_CPU,
        DEVBOX_MEMORY,
        previous=previous,
    )
    assert cal.previous_value == previous.value == 0.70
    assert cal.change_pct is not None
    assert abs(cal.change_pct - 0.288) < 0.01  # +0.29% -- the KV term

    without = derive_eta_bw(
        pick(devbox_points, "tg"),
        real_catalog.model("llama-3.1-8b-instruct"),
        real_catalog.quant("Q4_K_M"),
        DEVBOX_CPU,
        DEVBOX_MEMORY,
    )
    assert (without.previous_value, without.change_pct) == (None, None)


def test_a_real_estimate_replacement_is_visible(real_catalog):
    """The case this module was written for: eta_compute[amx-bf16] is a guess.

    A single AMX log would promote it. Feed a hypothetical Xeon 6788P prefill
    figure and check the machinery reports how far the shipped estimate moves,
    so a reviewer sees the size of the change before accepting it.
    """
    cpu = real_catalog.cpu("xeon-6788p")
    model = real_catalog.model("llama-3.1-8b-instruct")
    previous = real_catalog.coefficient("eta_compute", "amx-bf16")
    assert previous.confidence == "estimate"  # the reason this module exists

    point = MeasuredPoint(
        source="ticket-4471 / customer AMX run",
        kind="pp",
        tokens_per_s=1500.0,
        model_label="llama 8B Q4_K - Medium",
        n_batch=512,
        n_ctx=512,
    )
    cal = derive_eta_compute(point, model, cpu, previous=previous)

    assert isinstance(cal, Calibration)
    assert cal.coefficient.key == "amx-bf16"
    assert cal.coefficient.confidence == "measured"  # a log from this very part
    assert cal.coefficient.source_url == "ticket-4471 / customer AMX run"
    assert 0 < cal.coefficient.value <= 1.0
    assert cal.change_pct is not None
    # Sanity on the arithmetic rather than on the invented tok/s: achieved
    # FLOP/s must be 2 x params x tok/s.
    assert abs(cal.implied_ceiling - 1500.0 * 2 * model.params_b * 1e9) < 1e6


def test_previous_with_a_zero_value_does_not_divide_by_zero(real_catalog, devbox_points):
    zero = CoefficientSpec(id="eta-bw-ddr4", kind="eta_bw", key="DDR4", value=0.0, confidence="estimate")
    cal = derive_eta_bw(
        pick(devbox_points, "tg"),
        real_catalog.model("llama-3.1-8b-instruct"),
        real_catalog.quant("Q4_K_M"),
        DEVBOX_CPU,
        DEVBOX_MEMORY,
        previous=zero,
    )
    assert cal.previous_value == 0.0
    assert cal.change_pct is None


# --------------------------------------------------------------------------
# AC#10 -- this module reads files, it does not run anything
# --------------------------------------------------------------------------


def test_the_importer_cannot_execute_anything():
    """The tool sizes servers it has no access to; benchmarking here is meaningless.

    Guarding the string rather than the import: `subprocess` appearing anywhere
    in this module -- even in a comment suggesting it -- is the thing to catch.
    """
    source = (Path(__file__).parent.parent / "svrspec" / "measured.py").read_text(encoding="utf-8")
    assert "subprocess" not in source
    assert "os.system" not in source
    assert "urllib" not in source  # no network either: the log arrives by hand


def test_a_multi_socket_run_can_never_be_measured():
    """2P logs fold NUMA loss into the coefficient, so they are never "measured".

    eta_bw and eta_compute both describe what one socket reaches against its
    own ceiling; the cross-socket loss is carried separately by
    `dual_socket_efficiency`. A 2P measurement divided by two sockets' worth of
    ceiling absorbs that loss, and every 2P prediction then subtracts it again.
    Leaving the downgrade to the caller made the mistake silent.
    """
    from svrspec.catalog import Catalog
    from svrspec.measured import derive_eta_bw, derive_eta_compute

    catalog = Catalog()
    cpu = catalog.cpu("epyc-9354")
    memory = catalog.memory_for(cpu, 1)
    model = catalog.model("llama-3.1-8b-instruct")
    quant = catalog.quant("Q4_K_M")

    tg = MeasuredPoint(source="2p-run.md", kind="tg", tokens_per_s=60.0,
                       stddev=None, model_label="llama 8B Q4_K - Medium",
                       params_b=8.03, quant="Q4_K_M", n_threads=64,
                       n_batch=None, n_ctx=128, backend="CPU", raw={})
    pp = MeasuredPoint(source="2p-run.md", kind="pp", tokens_per_s=300.0,
                       stddev=None, model_label="llama 8B Q4_K - Medium",
                       params_b=8.03, quant="Q4_K_M", n_threads=64,
                       n_batch=512, n_ctx=512, backend="CPU", raw={})

    for one, two in (
        (derive_eta_bw(tg, model, quant, cpu, memory, sockets=1),
         derive_eta_bw(tg, model, quant, cpu, memory, sockets=2)),
        (derive_eta_compute(pp, model, cpu, sockets=1),
         derive_eta_compute(pp, model, cpu, sockets=2)),
    ):
        assert one.coefficient.confidence == "measured"
        assert two.coefficient.confidence == "derived"
        assert "소켓" in two.basis
        # An explicit "measured" from the caller is exactly the mistake, so it
        # is overridden rather than honoured.
        assert two.coefficient.source == SOURCE_MEASUREMENT

    forced = derive_eta_bw(tg, model, quant, cpu, memory, sockets=2,
                           confidence="measured")
    assert forced.coefficient.confidence == "derived"
    # A caller who already said "derived" is left alone.
    assert derive_eta_bw(tg, model, quant, cpu, memory, sockets=2,
                         confidence="derived").coefficient.confidence == "derived"


def test_a_log_from_the_wrong_machine_is_not_measured():
    """Thread count is the one clue a log carries about which box ran it.

    Pointing a 10-thread desktop log at a 32-core EPYC derives eta_bw = 0.07
    and used to report it as "measured" without complaint. The derivation is
    still allowed -- pinning below core count is legitimate and we cannot tell
    the two cases apart -- but it may not be called a measurement.
    """
    from svrspec.catalog import Catalog
    from svrspec.measured import derive_eta_bw

    catalog = Catalog()
    cpu = catalog.cpu("epyc-9354")           # 32 cores
    memory = catalog.memory_for(cpu, 1)
    model = catalog.model("llama-3.1-8b-instruct")
    quant = catalog.quant("Q4_K_M")

    def point(threads):
        return MeasuredPoint(
            source="desktop.log", kind="tg", tokens_per_s=6.40, stddev=None,
            model_label="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", params_b=8.03,
            quant="Q4_K_M", n_threads=threads, n_batch=None, n_ctx=640,
            backend="CPU", raw={},
        )

    mismatched = derive_eta_bw(point(10), model, quant, cpu, memory)
    assert mismatched.coefficient.confidence == "derived"
    assert "10스레드" in mismatched.basis and "32코어" in mismatched.basis

    # 24 of 32 cores is ordinary pinning, not a mismatch.
    assert derive_eta_bw(point(24), model, quant, cpu, memory
                         ).coefficient.confidence == "measured"
    # No thread count in the log means no evidence either way.
    assert derive_eta_bw(point(None), model, quant, cpu, memory
                         ).coefficient.confidence == "measured"
