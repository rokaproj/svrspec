"""Memory model. These are the numbers that can be checked exactly."""

import pytest

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


def test_os_profiles_change_the_sizing_and_say_they_are_estimates(model_8b, q4):
    """The same box does not offer the same memory on every OS.

    Sizing that ignores this under-provisions whichever machine carries the
    most operating system, and the range here is a whole DIMM's worth.
    """
    from svrspec.memory import OS_PROFILES, size_memory
    from svrspec.types import TokenProfile

    tokens = TokenProfile()
    sized = {
        name: size_memory(model_8b, q4, tokens, slots=4, os_name=name)
        for name in OS_PROFILES
    }

    assert sized["linux-container"].subtotal_gb < sized["linux-headless"].subtotal_gb
    assert sized["linux-headless"].subtotal_gb < sized["windows-server"].subtotal_gb
    assert sized["windows-server"].subtotal_gb < sized["windows-desktop"].subtotal_gb

    # Only the OS term moves; the model's own demand is identical.
    for name, breakdown in sized.items():
        assert breakdown.runtime_os_gb == OS_PROFILES[name].runtime_gb
        assert breakdown.weights_gb == pytest.approx(sized["windows-server"].weights_gb)
        assert breakdown.kv_cache_gb == pytest.approx(sized["windows-server"].kv_cache_gb)

    # Every profile has to admit what it is.
    for profile in OS_PROFILES.values():
        assert profile.note.strip()
        assert profile.label.strip()


def test_choosing_no_os_keeps_the_previous_behaviour_exactly(model_8b, q4):
    """The parameter is additive: every existing caller is untouched."""
    from svrspec.memory import DEFAULT_HEADROOM, RUNTIME_OS_GB, size_memory
    from svrspec.types import TokenProfile

    tokens = TokenProfile()
    before = size_memory(model_8b, q4, tokens, slots=4)
    assert before.runtime_os_gb == RUNTIME_OS_GB
    assert before.headroom_factor == DEFAULT_HEADROOM

    # An explicit headroom still wins over the profile's.
    forced = size_memory(model_8b, q4, tokens, slots=4,
                         os_name="linux-container", headroom=2.0)
    assert forced.headroom_factor == 2.0


def test_an_unknown_os_falls_back_rather_than_raising():
    """A typo in a dropdown must not take down a sizing run."""
    from svrspec.memory import DEFAULT_OS_PROFILE, os_profile

    assert os_profile("no-such-os").id == DEFAULT_OS_PROFILE
    assert os_profile(None).id == DEFAULT_OS_PROFILE


def test_a_hard_limit_is_sized_more_generously_than_a_soft_one(model_8b, q4):
    """Overrunning a cgroup kills the process; overrunning a host slows it.

    Those are not one kind of bad, and a single headroom number cannot say
    which is which. The container profile frees the most memory and punishes
    overrun the hardest, so it gets the largest headroom -- sizing it tightest
    because its baseline is lightest gets the trade backwards.
    """
    from svrspec.memory import OS_PROFILES, size_memory
    from svrspec.types import TokenProfile

    container = OS_PROFILES["linux-container"]
    headless = OS_PROFILES["linux-headless"]

    assert container.hard_limit is True
    assert headless.hard_limit is False
    assert container.runtime_gb < headless.runtime_gb      # lighter baseline
    assert container.headroom > headless.headroom          # larger margin

    tokens = TokenProfile()
    in_container = size_memory(model_8b, q4, tokens, slots=4, os_name="linux-container")
    on_host = size_memory(model_8b, q4, tokens, slots=4, os_name="linux-headless")

    # Less resident, but provisioned for more -- that is the point.
    assert in_container.subtotal_gb < on_host.subtotal_gb
    assert in_container.recommended_gb > on_host.recommended_gb


def test_every_profile_states_what_happens_when_memory_runs_out():
    """The consequence has to travel with the number, not live in a wiki."""
    from svrspec.memory import OS_PROFILES

    for profile in OS_PROFILES.values():
        consequence = profile.overrun_consequence
        assert consequence.strip()
        if profile.hard_limit:
            assert "OOM" in consequence and "죽는다" in consequence
        else:
            assert "스왑" in consequence


def test_a_measured_residency_can_replace_the_estimate(model_8b, q4):
    """The shipped OS figures are experience, not measurement — so allow the real one.

    Nothing in this project has watched a Windows Server idle. That is fine as
    a default and bad as a final answer, so a number read off the box that will
    run the model has to be able to win.
    """
    from svrspec.memory import measured_os_profile, size_memory
    from svrspec.types import TokenProfile

    tokens = TokenProfile()
    measured = measured_os_profile(
        1.8, "prod-llm-01, free -g idle 2026-07-31", base="linux-headless"
    )
    assert measured.runtime_gb == 1.8
    assert "실측" in measured.note and "prod-llm-01" in measured.note

    estimated = size_memory(model_8b, q4, tokens, slots=4, os_name="linux-headless")
    real = size_memory(model_8b, q4, tokens, slots=4, os_name=measured)

    assert real.runtime_os_gb == 1.8
    assert real.subtotal_gb > estimated.subtotal_gb
    # Only the OS term moved; the model's own demand is untouched.
    assert real.weights_gb == pytest.approx(estimated.weights_gb)
    assert real.kv_cache_gb == pytest.approx(estimated.kv_cache_gb)


def test_a_measured_profile_inherits_policy_it_cannot_measure(model_8b, q4):
    """A memory reading cannot tell you whether overrun kills the process."""
    from svrspec.memory import measured_os_profile

    in_container = measured_os_profile(0.4, "k8s node-7", base="linux-container")
    on_host = measured_os_profile(0.4, "bare-metal-3", base="linux-headless")

    assert in_container.hard_limit is True
    assert on_host.hard_limit is False
    assert in_container.headroom == pytest.approx(1.6)

    # Headroom is a policy choice and can still be overridden explicitly.
    assert measured_os_profile(0.4, "x", base="linux-container",
                               headroom=2.0).headroom == 2.0


def test_a_measurement_without_a_source_is_refused():
    """Same rule as a promoted coefficient: no identity, no promotion."""
    from svrspec.memory import measured_os_profile

    with pytest.raises(ValueError, match="source"):
        measured_os_profile(1.0, "   ")
    with pytest.raises(ValueError, match="음수"):
        measured_os_profile(-0.1, "somewhere")
