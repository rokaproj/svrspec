"""Throughput model.

The load-bearing test is `test_reproduces_the_one_real_measurement`: the same
roofline that extrapolates to server CPUs must first reproduce the measurement
its coefficients were derived from.

Nothing here runs a model or loads the CPU -- the whole point of the design is
that sizing a server does not require access to one.
"""

from dataclasses import replace

import pytest

from svrspec.perf import (
    CONFIDENCE_UNCERTAINTY,
    FLOP_AMX_BF16,
    FLOP_AVX2,
    FLOP_AVX512,
    FLOP_PER_CYCLE,
    Efficiency,
    effective_bandwidth,
    peak_flops,
    predict_latency,
    predict_throughput,
    widest_isa,
)
from svrspec.types import TokenProfile


def test_reproduces_the_one_real_measurement(catalog, eff, model_8b, q4):
    """2 channels of DDR4-2667 + 8B Q4_K_M measured 6.07 tok/s.

    eta_bw for DDR4 was derived from exactly this run, so the model must return
    it. If someone retunes that coefficient without re-deriving it, this fails.
    """
    cpu = catalog.cpu("test-desktop-2ch")
    memory = catalog.memory_option("ddr4-2666-2dpc")
    # Short generation, so little KV has accumulated -- matching the tg128
    # conditions of the original measurement.
    tokens = TokenProfile(system_tokens=0, fewshot_tokens=0, alarm_tokens=0, output_tokens=128)

    pred = predict_throughput(model_8b, q4, cpu, memory, tokens, eff, slots=1)
    assert abs(pred.decode_tps_single - 6.07) < 0.5, pred.decode_tps_single


def test_decode_is_bandwidth_bound_on_wide_memory(catalog, eff, model_8b, q4):
    cpu = catalog.cpu("test-avx512-8ch")
    memory = catalog.memory_option("ddr5-4800-1dpc")
    pred = predict_throughput(model_8b, q4, cpu, memory, TokenProfile(), eff, slots=1)
    assert pred.decode_bound_by == "bandwidth"


def test_few_cores_cannot_fill_the_channels(catalog, eff, model_8b, q4):
    """An 8-core part with 8 channels is limited by its cores, not its DIMMs.

    A real trap when picking a server: a low-core Xeon Silver advertises eight
    memory channels but cannot pull enough of them to matter.
    """
    cpu = catalog.cpu("test-lowcore-8ch")
    memory = catalog.memory_option("ddr5-4800-1dpc")

    achieved, limit = effective_bandwidth(cpu, memory, eff, sockets=1)
    assert limit == "core-bandwidth"
    # The DIMMs could deliver 8ch x 4800 MT/s; the cores cannot pull it.
    assert achieved < cpu.mem_channels * memory.effective_mts * 8 / 1000 * 1e9

    pred = predict_throughput(model_8b, q4, cpu, memory, TokenProfile(), eff, slots=1)
    assert any("코어" in w for w in pred.warnings)
    # Starved of both bandwidth and vector width, it lands far below a part
    # that can actually use its channels.
    fat = predict_throughput(
        model_8b, q4, catalog.cpu("test-avx512-8ch"), memory, TokenProfile(), eff, slots=1
    )
    assert pred.decode_tps_single < fat.decode_tps_single / 1.5


def test_more_channels_means_faster_decode(catalog, eff, model_8b, q4):
    narrow = predict_throughput(
        model_8b, q4, catalog.cpu("test-desktop-2ch"),
        catalog.memory_option("ddr4-3200-1dpc"), TokenProfile(), eff, slots=1,
    )
    wide = predict_throughput(
        model_8b, q4, catalog.cpu("test-avx512-8ch"),
        catalog.memory_option("ddr5-4800-1dpc"), TokenProfile(), eff, slots=1,
    )
    assert wide.decode_tps_single > narrow.decode_tps_single * 3


def test_two_dimms_per_channel_costs_throughput(catalog, eff, model_8b, q4):
    cpu = catalog.cpu("test-avx512-8ch")
    fast = predict_throughput(
        model_8b, q4, cpu, catalog.memory_option("ddr5-4800-1dpc"), TokenProfile(), eff, slots=1
    )
    derated = predict_throughput(
        model_8b, q4, cpu, catalog.memory_option("ddr5-4800-2dpc"), TokenProfile(), eff, slots=1
    )
    assert derated.decode_tps_single < fast.decode_tps_single


def test_isa_detection_picks_the_widest_unit(catalog):
    assert widest_isa(catalog.cpu("test-desktop-2ch")) == "avx2"
    assert widest_isa(catalog.cpu("test-avx512-8ch")) == "avx512"
    assert widest_isa(catalog.cpu("test-amx-8ch")) == "amx-bf16"
    assert FLOP_PER_CYCLE["avx2"] == FLOP_AVX2
    assert FLOP_PER_CYCLE["avx512"] == FLOP_AVX512
    assert FLOP_PER_CYCLE["amx-bf16"] == FLOP_AMX_BF16


def test_amx_beats_more_avx512_cores_on_prefill(catalog, eff, model_8b, q4):
    memory = catalog.memory_option("ddr5-4800-1dpc")
    amx = predict_throughput(
        model_8b, q4, catalog.cpu("test-amx-8ch"), memory, TokenProfile(), eff, slots=1
    )
    # 16 AMX cores at 2.7 GHz must out-prefill 32 AVX-512 cores at 3.1 GHz,
    # even after AMX's much lower achieved-efficiency coefficient.
    avx512 = predict_throughput(
        model_8b, q4, catalog.cpu("test-avx512-8ch"), memory, TokenProfile(), eff, slots=1
    )
    assert amx.prefill_tps > avx512.prefill_tps


def test_prefill_is_compute_bound(catalog, eff, model_8b, q4):
    pred = predict_throughput(
        model_8b, q4, catalog.cpu("test-amx-8ch"),
        catalog.memory_option("ddr5-4800-1dpc"), TokenProfile(), eff, slots=1,
    )
    assert pred.prefill_bound_by == "compute"


def test_batching_amortises_the_weight_read(catalog, eff, model_8b, q4):
    cpu, memory = catalog.cpu("test-avx512-8ch"), catalog.memory_option("ddr5-4800-1dpc")
    one = predict_throughput(model_8b, q4, cpu, memory, TokenProfile(), eff, slots=1)
    four = predict_throughput(model_8b, q4, cpu, memory, TokenProfile(), eff, slots=4)
    assert four.decode_tps_aggregate > one.decode_tps_aggregate
    # ...but sublinearly: KV reads are per-sequence and compute eventually binds.
    assert four.decode_tps_aggregate < 4 * one.decode_tps_aggregate


def test_moe_decodes_far_faster_than_its_size_suggests(catalog, eff, moe, model_8b, q4):
    cpu, memory = catalog.cpu("test-avx512-8ch"), catalog.memory_option("ddr5-4800-1dpc")
    moe_pred = predict_throughput(moe, q4, cpu, memory, TokenProfile(), eff, slots=1)
    dense = predict_throughput(model_8b, q4, cpu, memory, TokenProfile(), eff, slots=1)
    # 30B of weights but 3.3B read per token: beats a dense 8B despite being
    # four times the size. The reason MoE suits bandwidth-starved CPUs.
    assert moe_pred.decode_tps_single > dense.decode_tps_single


def test_estimated_coefficients_widen_the_error_bars(catalog, eff, model_8b, q4):
    """A platform class with no measurement behind it must say so."""
    amx = predict_throughput(
        model_8b, q4, catalog.cpu("test-amx-8ch"),
        catalog.memory_option("ddr5-4800-1dpc"), TokenProfile(), eff, slots=1,
    )
    measured_class = predict_throughput(
        model_8b, q4, catalog.cpu("test-desktop-2ch"),
        catalog.memory_option("ddr4-2666-2dpc"), TokenProfile(), eff, slots=1,
    )
    assert amx.uncertainty > measured_class.uncertainty
    assert any("추정" in w for w in amx.warnings)
    assert not any("추정" in w for w in measured_class.warnings)


def test_uncertainty_tracks_the_confidence_of_the_coefficients_used(catalog, eff, model_8b, q4):
    pred = predict_throughput(
        model_8b, q4, catalog.cpu("test-desktop-2ch"),
        catalog.memory_option("ddr4-2666-2dpc"), TokenProfile(), eff, slots=1,
    )
    # DDR4 eta_bw is measured, avx2 eta_compute is derived; the worse of the two
    # sets the floor, plus the unverified-CPU-spec adder.
    assert pred.uncertainty >= CONFIDENCE_UNCERTAINTY["derived"]
    assert pred.uncertainty < CONFIDENCE_UNCERTAINTY["estimate"]


def test_dual_socket_is_discounted_and_warned(catalog, eff, model_8b, q4):
    cpu, memory = catalog.cpu("test-avx512-8ch"), catalog.memory_option("ddr5-4800-1dpc")
    one, _ = effective_bandwidth(cpu, memory, eff, sockets=1)
    two, _ = effective_bandwidth(cpu, memory, eff, sockets=2)
    assert one < two < 2 * one  # some gain, not the full doubling

    pred = predict_throughput(model_8b, q4, cpu, memory, TokenProfile(), eff, slots=1, sockets=2)
    assert any("소켓" in w for w in pred.warnings)
    assert pred.uncertainty > predict_throughput(
        model_8b, q4, cpu, memory, TokenProfile(), eff, slots=1, sockets=1
    ).uncertainty


def test_peak_flops_scales_with_sockets(catalog):
    cpu = catalog.cpu("test-avx512-8ch")
    assert peak_flops(cpu, 2)[0] == 2 * peak_flops(cpu, 1)[0]


def test_prompt_cache_removes_the_fixed_prefix_from_ttft(catalog, eff, model_8b, q4):
    cpu, memory = catalog.cpu("test-amx-8ch"), catalog.memory_option("ddr5-4800-1dpc")
    pred = predict_throughput(model_8b, q4, cpu, memory, TokenProfile(), eff, slots=1)

    cached = predict_latency(pred, TokenProfile(prompt_cache=True))
    uncached = predict_latency(pred, TokenProfile(prompt_cache=False))
    assert cached.ttft_s < uncached.ttft_s
    assert cached.generate_s == uncached.generate_s


def test_a_better_coefficient_predicts_more_throughput(catalog, eff, model_8b, q4):
    cpu, memory = catalog.cpu("test-avx512-8ch"), catalog.memory_option("ddr5-4800-1dpc")
    original = eff.eta_bw("DDR5")
    optimistic = Efficiency(
        tuple(
            replace(c, value=min(c.value * 1.3, 1.0)) if c.id == original.id else c
            for c in eff.coefficients
        )
    )
    base = predict_throughput(model_8b, q4, cpu, memory, TokenProfile(), eff, slots=1)
    better = predict_throughput(model_8b, q4, cpu, memory, TokenProfile(), optimistic, slots=1)
    assert better.decode_tps_single > base.decode_tps_single


def test_efficiency_falls_back_to_the_global_entry(eff):
    # per_core_bw_gbs is stored with key "*" and must resolve for any key.
    assert eff.get("per_core_bw_gbs", "DDR9").value == eff.per_core_bw_gbs


def test_moe_gets_the_worse_dual_socket_scaling(catalog, eff, moe, model_8b, q4):
    """A second socket buys less for MoE than for a dense model.

    Measured llama.cpp scaling: dense Llama-70B generation went 1.8-1.9x across
    two sockets, Mixtral only 1.46-1.85x, because expert matrices are small
    enough that cross-socket synchronisation dominates. Sizing a MoE deployment
    with the dense figure would over-promise.
    """
    assert eff.dual_socket_efficiency(moe=True) < eff.dual_socket_efficiency(moe=False)

    cpu, memory = catalog.cpu("test-avx512-8ch"), catalog.memory_option("ddr5-4800-1dpc")
    dense_1, _ = effective_bandwidth(cpu, memory, eff, sockets=1, moe=False)
    dense_2, _ = effective_bandwidth(cpu, memory, eff, sockets=2, moe=False)
    moe_2, _ = effective_bandwidth(cpu, memory, eff, sockets=2, moe=True)
    assert moe_2 < dense_2
    assert moe_2 > dense_1  # still a gain, just a smaller one

    pred = predict_throughput(moe, q4, cpu, memory, TokenProfile(), eff, slots=1, sockets=2)
    assert any("MoE" in w for w in pred.warnings)


def test_coefficient_provenance_is_visible_on_every_row(eff):
    """Nothing may sit in the table without declaring how well it is known."""
    from svrspec.types import VALID_CONFIDENCE

    for c in eff.coefficients:
        assert c.confidence in VALID_CONFIDENCE
        assert c.notes, f"{c.id} has no note explaining where it came from"
        if c.source != "unverified":
            assert c.source_url, f"{c.id} claims a source but cites no URL"


def test_under_populated_channels_cut_bandwidth_proportionally(catalog, eff):
    """Two DIMMs in an eight-channel board is a quarter of the bandwidth.

    This is the expensive real-world mistake the parameter exists to express:
    the capacity looks fine, the bandwidth does not, and decode is bandwidth
    bound so token generation drops by exactly the same fraction.
    """
    from svrspec.perf import effective_bandwidth

    cpu = catalog.cpu("test-amx-8ch")
    memory = catalog.memory_for(cpu, 1)
    assert cpu.mem_channels == 8

    full, _ = effective_bandwidth(cpu, memory, eff)
    quarter, _ = effective_bandwidth(cpu, memory, eff, channels_populated=2)
    assert quarter == pytest.approx(full / 4)

    half, _ = effective_bandwidth(cpu, memory, eff, channels_populated=4)
    assert half == pytest.approx(full / 2)


def test_the_channel_parameter_defaults_to_no_change(catalog, eff):
    """Every existing caller must be untouched by the new parameter."""
    from svrspec.perf import effective_bandwidth

    cpu = catalog.cpu("test-amx-8ch")
    memory = catalog.memory_for(cpu, 1)
    assert effective_bandwidth(cpu, memory, eff) == effective_bandwidth(
        cpu, memory, eff, channels_populated=None
    )
    assert effective_bandwidth(cpu, memory, eff) == effective_bandwidth(
        cpu, memory, eff, channels_populated=cpu.mem_channels
    )
    # Over-filling is clamped rather than inventing bandwidth that cannot exist.
    assert effective_bandwidth(cpu, memory, eff, channels_populated=99) == \
        effective_bandwidth(cpu, memory, eff)


def test_under_population_slows_decode_and_says_so(catalog, eff, model_8b, q4):
    """The prediction must carry the warning, not just the smaller number."""
    from svrspec.perf import predict_throughput
    from svrspec.types import TokenProfile

    cpu = catalog.cpu("test-amx-8ch")
    memory = catalog.memory_for(cpu, 1)
    tokens = TokenProfile()

    full = predict_throughput(model_8b, q4, cpu, memory, tokens, eff, slots=1)
    thin = predict_throughput(
        model_8b, q4, cpu, memory, tokens, eff, slots=1, channels_populated=2
    )

    assert thin.decode_tps_single < full.decode_tps_single
    assert thin.effective_bandwidth_gbs == pytest.approx(
        full.effective_bandwidth_gbs / 4
    )
    assert any("2/8" in w for w in thin.warnings)
    assert not any("채널을" in w and "장착했다" in w for w in full.warnings)


def test_an_empty_board_does_not_divide_by_zero(catalog, eff, model_8b, q4):
    """Zero DIMMs is a real thing to ask about; it must answer, not crash.

    `lab` guards against calling here with an empty board, but a guard that
    lives only in the caller is one refactor away from being gone.
    """
    from svrspec.perf import effective_bandwidth, predict_throughput
    from svrspec.types import TokenProfile

    cpu = catalog.cpu("test-amx-8ch")
    memory = catalog.memory_for(cpu, 1)

    bandwidth, _ = effective_bandwidth(cpu, memory, eff, channels_populated=0)
    assert bandwidth == 0.0

    prediction = predict_throughput(
        model_8b, q4, cpu, memory, TokenProfile(), eff, slots=1, channels_populated=0
    )
    assert prediction.decode_tps_single == 0.0
    assert any("한 장도 장착되지" in w for w in prediction.warnings)
