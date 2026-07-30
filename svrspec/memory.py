"""RAM sizing.

Unlike throughput, memory is almost pure arithmetic — no calibration constant
worth arguing about, and llama.cpp prints the two biggest terms (`KV self size`,
`compute buffer size`) in its startup log, so `svrspec verify` can check this
module against ground truth rather than trusting it.
"""

from __future__ import annotations

from .types import MemoryBreakdown, ModelSpec, QuantSpec, TokenProfile

GB = 1024**3

#: llama.cpp default micro-batch. Sets the width of the prefill activation
#: scratch, so it shows up directly in the compute buffer.
DEFAULT_UBATCH = 512

#: Activation scratch multiplier: how many n_ubatch x n_embd f32 tensors the
#: graph keeps live at once (residual, attention projections, the SwiGLU
#: intermediate at ~3.5x n_embd, norms). Empirical, and the single softest
#: number in this module -- `svrspec verify` reports the error against the real
#: `compute buffer size` line and this is the knob to turn.
ACTIVATION_TENSORS = 14.0

#: Base OS + llama.cpp runtime residency, before any model data.
RUNTIME_OS_GB = 2.5

#: Applied to the computed subtotal. Covers allocator slack, page cache the
#: kernel wants for the mmap'd model, log/metric agents, and the fact that a
#: server sized to exactly its working set starts swapping the day someone
#: raises the context window.
DEFAULT_HEADROOM = 1.5

#: Capacities you can actually populate with sane DIMM counts.
PROVISIONABLE_GB = (16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048)


def weight_bytes(model: ModelSpec, quant: QuantSpec, measured_bpw: float | None = None) -> float:
    """File-resident weight bytes. All of it is read per generated token."""
    bpw = measured_bpw if measured_bpw else quant.bits_per_weight
    return model.params_b * 1e9 * bpw / 8.0


def decode_weight_bytes(model: ModelSpec, quant: QuantSpec, measured_bpw: float | None = None) -> float:
    """Weight bytes touched per generated token.

    For a dense model that is the whole file. For MoE only the active experts
    are read, which is exactly why MoE is attractive on bandwidth-starved CPUs:
    Qwen3-30B-A3B holds 30B of weights but reads about 3B worth per token.
    """
    bpw = measured_bpw if measured_bpw else quant.bits_per_weight
    return model.decode_params_b * 1e9 * bpw / 8.0


def kv_bytes_per_token(model: ModelSpec, kv_bits: int = 16) -> float:
    """K and V for every layer, at GQA width.

    Llama-3.1-8B check: 2 * 32 layers * 8 kv heads * 128 dim * 2 bytes
    = 131072 B = 128 KiB per token, so 8k of context is 1 GiB. Matches the
    figure llama.cpp reports.
    """
    return 2.0 * model.n_layer * model.n_kv_head * model.kv_head_dim * (kv_bits / 8.0)


def kv_cache_bytes(model: ModelSpec, ctx_tokens: int, slots: int = 1, kv_bits: int = 16) -> float:
    """llama.cpp allocates the full context per slot up front, not on demand."""
    return kv_bytes_per_token(model, kv_bits) * ctx_tokens * slots


def compute_buffer_bytes(
    model: ModelSpec,
    slots: int = 1,
    n_ubatch: int = DEFAULT_UBATCH,
    flash_attention: bool = True,
) -> float:
    """Transient graph buffers: logits, activation scratch, attention scores."""
    # Logits are materialised only for tokens whose output is requested: the
    # last token of each sequence, not the whole prefill batch.
    logits = model.n_vocab * max(slots, 1) * 4.0
    activations = n_ubatch * model.n_embd * 4.0 * ACTIVATION_TENSORS
    # Without flash attention the n_ubatch x n_ubatch score matrix per head is
    # materialised, which at 512 ubatch and 32 heads is another ~33 MB.
    scores = 0.0 if flash_attention else model.n_head * n_ubatch * n_ubatch * 4.0
    return logits + activations + scores


def size_memory(
    model: ModelSpec,
    quant: QuantSpec,
    tokens: TokenProfile,
    slots: int = 1,
    ctx_tokens: int | None = None,
    kv_bits: int = 16,
    headroom: float = DEFAULT_HEADROOM,
    measured_bpw: float | None = None,
    flash_attention: bool = True,
) -> MemoryBreakdown:
    """Full RAM requirement for one (model, quant, concurrency) deployment."""
    ctx = ctx_tokens if ctx_tokens else _rounded_ctx(tokens.peak_ctx_tokens)

    weights = weight_bytes(model, quant, measured_bpw)
    kv = kv_cache_bytes(model, ctx, slots, kv_bits)
    compute = compute_buffer_bytes(model, slots, flash_attention=flash_attention)

    subtotal_gb = (weights + kv + compute) / GB + RUNTIME_OS_GB
    recommended = subtotal_gb * headroom

    return MemoryBreakdown(
        weights_gb=weights / GB,
        kv_cache_gb=kv / GB,
        compute_buffer_gb=compute / GB,
        runtime_os_gb=RUNTIME_OS_GB,
        subtotal_gb=subtotal_gb,
        headroom_factor=headroom,
        recommended_gb=recommended,
        provision_gb=provisionable(recommended),
    )


def provisionable(gb: float) -> int:
    """Smallest realistically populated capacity that covers `gb`."""
    for cap in PROVISIONABLE_GB:
        if cap >= gb:
            return cap
    return PROVISIONABLE_GB[-1]


def _rounded_ctx(needed: int) -> int:
    """Round the per-alarm context up to a power-of-two-ish server setting.

    Nobody deploys `-c 1450`; they set 2048 or 4096. Sizing against the round
    number is what the customer will actually run.
    """
    for size in (2048, 4096, 8192, 16384, 32768, 65536, 131072):
        if size >= needed:
            return size
    return 131072


def max_slots_within(
    model: ModelSpec,
    quant: QuantSpec,
    ram_gb: float,
    ctx_tokens: int,
    kv_bits: int = 16,
    headroom: float = DEFAULT_HEADROOM,
) -> int:
    """How many concurrent slots fit in a given RAM budget."""
    budget = ram_gb / headroom - RUNTIME_OS_GB
    fixed = weight_bytes(model, quant) / GB
    if budget <= fixed:
        return 0
    per_slot = (kv_cache_bytes(model, ctx_tokens, 1, kv_bits) + compute_buffer_bytes(model, 1)) / GB
    if per_slot <= 0:
        return 1
    return max(0, int((budget - fixed) / per_slot))
