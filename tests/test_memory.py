"""Memory model. These are the numbers that can be checked exactly."""

from svrspec.memory import (
    compute_buffer_bytes,
    kv_bytes_per_token,
    kv_cache_bytes,
    max_slots_within,
    provisionable,
    size_memory,
    weight_bytes,
)
from svrspec.types import TokenProfile

KIB = 1024
GB = 1024**3


def test_kv_per_token_matches_llama_cpp_for_8b_gqa(model_8b):
    # 2 (K+V) * 32 layers * 8 kv heads * 128 head dim * 2 bytes = 128 KiB.
    # This is the figure llama.cpp reports for Llama-3.1-8B, so it is a real
    # cross-check rather than a restatement of the formula.
    assert kv_bytes_per_token(model_8b) == 128 * KIB


def test_kv_scales_with_context_and_slots(model_8b):
    one_slot_8k = kv_cache_bytes(model_8b, 8192, slots=1)
    assert one_slot_8k == 1 * GB  # 128 KiB * 8192 = 1 GiB exactly
    assert kv_cache_bytes(model_8b, 8192, slots=4) == 4 * one_slot_8k


def test_gqa_is_the_dominant_kv_lever(model_8b, model_3b):
    # 3B has 2 kv heads against the 8B's 8, which is why a smaller model is
    # cheaper on context memory out of proportion to its parameter count.
    assert kv_bytes_per_token(model_3b) < kv_bytes_per_token(model_8b)


def test_explicit_head_dim_is_honoured(moe):
    # n_embd/n_head would be 2048/32 = 64, but the checkpoint declares 128.
    assert moe.kv_head_dim == 128
    assert kv_bytes_per_token(moe) == 2 * 48 * 4 * 128 * 2


def test_weight_bytes_tracks_bits_per_weight(model_8b, q4, catalog):
    q8 = catalog.quant("Q8_0")
    assert weight_bytes(model_8b, q8) > weight_bytes(model_8b, q4)
    ratio = weight_bytes(model_8b, q8) / weight_bytes(model_8b, q4)
    assert abs(ratio - q8.bits_per_weight / q4.bits_per_weight) < 1e-9


def test_weight_bytes_close_to_real_file(model_8b, q4):
    # The real Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf is 4.939e9 bytes.
    assert abs(weight_bytes(model_8b, q4) - 4.939e9) / 4.939e9 < 0.05


def test_moe_reads_only_active_weights(moe, q4):
    from svrspec.memory import decode_weight_bytes

    # Whole file resident, but a fraction of it read per token: exactly why MoE
    # suits a bandwidth-starved CPU.
    assert decode_weight_bytes(moe, q4) < weight_bytes(moe, q4) / 5


def test_compute_buffer_grows_without_flash_attention(model_8b):
    with_fa = compute_buffer_bytes(model_8b, flash_attention=True)
    without = compute_buffer_bytes(model_8b, flash_attention=False)
    assert without > with_fa


def test_size_memory_components_sum_to_subtotal(model_8b, q4):
    b = size_memory(model_8b, q4, TokenProfile(), slots=2)
    parts = b.weights_gb + b.kv_cache_gb + b.compute_buffer_gb + b.runtime_os_gb
    assert abs(parts - b.subtotal_gb) < 1e-6
    assert b.recommended_gb == b.subtotal_gb * b.headroom_factor
    assert b.provision_gb >= b.recommended_gb


def test_provision_rounds_up_to_populatable_capacity():
    assert provisionable(9.0) == 16
    assert provisionable(16.0) == 16
    assert provisionable(17.0) == 32
    assert provisionable(200.0) == 256


def test_more_slots_need_more_memory(model_8b, q4):
    tokens = TokenProfile()
    one = size_memory(model_8b, q4, tokens, slots=1)
    eight = size_memory(model_8b, q4, tokens, slots=8)
    assert eight.kv_cache_gb > one.kv_cache_gb
    assert eight.recommended_gb > one.recommended_gb


def test_max_slots_within_is_monotonic_in_budget(model_8b, q4):
    small = max_slots_within(model_8b, q4, 16, 4096)
    large = max_slots_within(model_8b, q4, 64, 4096)
    assert large > small
    assert max_slots_within(model_8b, q4, 4, 4096) == 0  # weights alone do not fit


def test_context_is_rounded_to_a_deployable_setting(model_8b, q4):
    # 950 prefill + 250 output = 1200 tokens, which nobody configures; the
    # sizing should assume the 2048 the operator will actually set.
    tokens = TokenProfile()
    assert tokens.peak_ctx_tokens == 1200
    b = size_memory(model_8b, q4, tokens, slots=1)
    assert b.kv_cache_gb == kv_cache_bytes(model_8b, 2048, 1) / GB
