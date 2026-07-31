"""Roofline throughput prediction for CPU-only llama.cpp inference.

Purely analytical. The simulator never runs a model and never loads the local
CPU -- it is sizing servers nobody here has access to, so benchmarking whatever
machine it happens to be running on would answer the wrong question. Everything
comes from published hardware specifications plus the efficiency coefficients in
`catalog/coefficients.json`.

Two independent bottlenecks decide everything, and they are not the same one:

  decode (token generation)
      Every generated token requires reading the whole active weight set from
      DRAM. Arithmetic is negligible. So tok/s is memory bandwidth divided by
      bytes-per-token, and what matters about a CPU is its memory channel count
      and DDR grade -- not its core count.

  prefill (prompt processing)
      A batch of tokens is a GEMM, so this is compute bound. What matters is
      cores x sustained all-core clock x vector width, which makes ISA support
      (AVX2 vs AVX-512 vs AMX) a multiplier, not a detail.

Each coefficient declares whether it was measured, derived, or estimated, and
the reported uncertainty is built from the ones a given prediction actually
used. So a prediction for an AMX Xeon -- where the coefficient is an estimate --
comes back with visibly wider error bars than one for a platform class where a
real measurement exists.
"""

from __future__ import annotations

from dataclasses import dataclass

from .memory import decode_weight_bytes, kv_bytes_per_token
from .types import (
    CoefficientSpec,
    CpuSpec,
    LatencyPrediction,
    MemoryOption,
    ModelSpec,
    QuantSpec,
    ThroughputPrediction,
    TokenProfile,
)

# --------------------------------------------------------------------------
# Vector throughput per core per cycle, fp32-equivalent FLOPs. These are
# architectural facts, not coefficients -- how much of them llama.cpp converts
# into prefill speed is what the eta_compute coefficients describe.
# --------------------------------------------------------------------------

#: 2 FMA ports x 8 fp32 lanes x 2 flops (multiply + add).
FLOP_AVX2 = 32.0
#: 2 FMA ports x 16 fp32 lanes x 2.
FLOP_AVX512 = 64.0
#: One TMUL issues a 16x16x32 bf16 block MAC; amortised, ~1024 flop/cycle/core.
FLOP_AMX_BF16 = 1024.0

FLOP_PER_CYCLE = {
    "amx-bf16": FLOP_AMX_BF16,
    "avx512": FLOP_AVX512,
    "avx2": FLOP_AVX2,
}

#: Error bar contributed by a coefficient, by how well it is known.
CONFIDENCE_UNCERTAINTY = {"measured": 0.15, "derived": 0.25, "estimate": 0.40}

#: Added when the CPU's own specification is not confirmed against a datasheet.
UNVERIFIED_SPEC_UNCERTAINTY = 0.05

MAX_UNCERTAINTY = 0.75


@dataclass(frozen=True)
class Efficiency:
    """The coefficient set a prediction runs on, resolved from the catalogue."""

    coefficients: tuple[CoefficientSpec, ...]

    @classmethod
    def from_catalog(cls, coefficients: list[CoefficientSpec]) -> Efficiency:
        return cls(tuple(coefficients))

    def get(self, kind: str, key: str = "*") -> CoefficientSpec:
        for c in self.coefficients:
            if c.kind == kind and c.key == key:
                return c
        for c in self.coefficients:
            if c.kind == kind and c.key == "*":
                return c
        raise KeyError(f"no {kind!r} coefficient for key {key!r}")

    def eta_bw(self, ddr_gen: str) -> CoefficientSpec:
        return self.get("eta_bw", ddr_gen)

    def eta_compute(self, isa: str) -> CoefficientSpec:
        return self.get("eta_compute", isa)

    @property
    def per_core_bw_gbs(self) -> float:
        return self.get("per_core_bw_gbs").value

    def dual_socket_efficiency(self, moe: bool = False) -> float:
        """Per-socket bandwidth multiplier on a 2P box.

        Keyed by model kind because the measured scaling differs sharply: dense
        models distribute across sockets well, MoE ones do not -- their expert
        matrices are small enough that synchronisation overhead dominates.
        """
        return self.get("dual_socket_efficiency", "moe" if moe else "dense").value


def widest_isa(cpu: CpuSpec) -> str:
    """The widest vector unit llama.cpp can use on this part."""
    for isa in ("amx-bf16", "avx512", "avx2"):
        if cpu.has(isa):
            return isa
    return "avx2"


def peak_flops(cpu: CpuSpec, sockets: int = 1) -> tuple[float, str]:
    """Theoretical sustained vector throughput in FLOP/s across all cores."""
    isa = widest_isa(cpu)
    return cpu.cores * sockets * cpu.all_core_turbo_ghz * 1e9 * FLOP_PER_CYCLE[isa], isa


def effective_bandwidth(
    cpu: CpuSpec,
    memory: MemoryOption,
    eff: Efficiency,
    sockets: int = 1,
    moe: bool = False,
    channels_populated: int | None = None,
) -> tuple[float, str]:
    """Achieved DRAM read bandwidth in bytes/s, and what limits it.

    `channels_populated` is per socket, and defaults to every channel filled --
    which is what a sizing report should assume, since nobody specifies a
    half-populated board on purpose. The parameter exists because people build
    them by accident: two 64 GB DIMMs in an eight-channel board is 128 GB and a
    quarter of the bandwidth, and decode is bandwidth bound, so token
    generation drops by the same quarter. Being unable to express that
    configuration meant the tool could not warn about it.

    Channels contribute their share linearly. Real controllers also lose some
    interleaving efficiency at awkward counts, which is not modelled here --
    so an under-populated figure is an optimistic bound, not a promise.
    """
    filled = cpu.mem_channels if channels_populated is None else channels_populated
    filled = max(0, min(filled, cpu.mem_channels))
    per_socket_gbs = filled * memory.effective_mts * 8 / 1000.0
    numa_factor = 1.0 if sockets == 1 else eff.dual_socket_efficiency(moe) * sockets
    # eta_bw derates the DRAM ceiling; the per-core figure is already achieved.
    dram_gbs = per_socket_gbs * numa_factor * eff.eta_bw(memory.ddr_gen).value
    core_gbs = cpu.cores * sockets * eff.per_core_bw_gbs

    if core_gbs < dram_gbs:
        return core_gbs * 1e9, "core-bandwidth"
    return dram_gbs * 1e9, "bandwidth"


def predict_throughput(
    model: ModelSpec,
    quant: QuantSpec,
    cpu: CpuSpec,
    memory: MemoryOption,
    tokens: TokenProfile,
    eff: Efficiency,
    slots: int = 1,
    sockets: int = 1,
    measured_bpw: float | None = None,
    channels_populated: int | None = None,
) -> ThroughputPrediction:
    warnings: list[str] = []

    is_moe = model.active_params_b is not None
    bw_bytes, bw_limit = effective_bandwidth(
        cpu, memory, eff, sockets, moe=is_moe, channels_populated=channels_populated
    )
    raw_flops, isa_used = peak_flops(cpu, sockets)
    eta_c = eff.eta_compute(isa_used)
    flops = raw_flops * eta_c.value

    w_bytes = decode_weight_bytes(model, quant, measured_bpw)
    # Attention reads the KV cache accumulated so far. Averaging over the
    # generated span is the honest figure for a whole request.
    avg_ctx = tokens.prefill_tokens + tokens.output_tokens / 2.0
    kv_bytes = kv_bytes_per_token(model) * avg_ctx

    # --- decode ---------------------------------------------------------
    bytes_per_token_single = w_bytes + kv_bytes
    decode_single = bw_bytes / bytes_per_token_single

    # Continuous batching amortises the weight read across the slots decoding
    # together: one sweep over the weights yields one token for each slot.
    bytes_per_token_batched = w_bytes / max(slots, 1) + kv_bytes
    decode_bw_bound = bw_bytes / bytes_per_token_batched
    # Batched decode is still 2*P flops per token, so compute can take over --
    # on a part with few or narrow vector units it sometimes does even at one
    # slot, because Q4 weights put the arithmetic intensity (~3.3 flop/byte)
    # above such a machine's balance point. Note this reuses the GEMM-derived
    # eta_compute, and a single-sequence decode is a GEMV that cannot reach GEMM
    # efficiency, so where this ceiling binds the prediction is optimistic.
    decode_compute_bound = flops / (2.0 * model.decode_params_b * 1e9)
    decode_aggregate = min(decode_bw_bound, decode_compute_bound)
    decode_bound = bw_limit if decode_bw_bound <= decode_compute_bound else "compute"

    # --- prefill --------------------------------------------------------
    prefill_compute = flops / (2.0 * model.decode_params_b * 1e9)
    # Prefill streams the weights once per micro-batch, so bandwidth rarely
    # binds, but check rather than assume.
    prefill_bw = bw_bytes / (w_bytes / 512.0)
    prefill = min(prefill_compute, prefill_bw)
    prefill_bound = "compute" if prefill_compute <= prefill_bw else bw_limit

    # --- honesty about the error bars -----------------------------------
    eta_b = eff.eta_bw(memory.ddr_gen)
    uncertainty = max(
        CONFIDENCE_UNCERTAINTY[eta_b.confidence],
        CONFIDENCE_UNCERTAINTY[eta_c.confidence],
    )
    for coeff, label in ((eta_b, "대역폭"), (eta_c, "연산")):
        if coeff.confidence == "estimate":
            warnings.append(
                f"{label} 효율 계수({coeff.id})가 추정값이다 — 실측 근거가 없어 "
                f"이 예측의 오차 범위가 넓다"
            )
    if sockets > 1:
        uncertainty += 0.10
        kind = "MoE" if is_moe else "dense"
        warnings.append(
            f"{sockets}소켓 구성: {kind} 모델의 실측 소켓당 확장 효율 "
            f"{eff.dual_socket_efficiency(is_moe):.0%}를 적용했다. NUMA 배치가 나쁘면 "
            f"원격 접근으로 이보다 낮아진다"
        )
    if bw_limit == "core-bandwidth":
        warnings.append(
            f"코어 수({cpu.cores * sockets})가 부족해 메모리 채널을 다 채우지 못한다: "
            f"채널 대역폭이 아니라 코어가 병목"
        )
    if channels_populated is not None and channels_populated < cpu.mem_channels:
        # Loud on purpose. Decode is bandwidth bound, so this is not a footnote:
        # it is a proportional cut to token generation speed.
        share = max(channels_populated, 0) / cpu.mem_channels
        if share <= 0:
            # A board with nothing in it. Answering "how many times faster
            # would it be if populated" is a division by zero, and the honest
            # answer is that this machine does not run at all.
            warnings.append(
                f"메모리가 한 장도 장착되지 않았다 — {cpu.model}은 "
                f"{cpu.mem_channels}채널을 채워야 동작한다"
            )
        else:
            warnings.append(
                f"메모리 채널을 {channels_populated}/{cpu.mem_channels}만 장착했다 — "
                f"대역폭이 {1 - share:.0%} 깎이고 토큰 생성 속도가 그만큼 떨어진다. "
                f"같은 용량이라도 전 채널에 나눠 꽂으면 {1 / share:.1f}배 빠르다"
            )
    if cpu.source == "unverified":
        # No warning text: with a whole catalogue unverified this would repeat
        # once per candidate and bury the warnings that differ between them. The
        # report has a dedicated unverified-rows section.
        uncertainty += UNVERIFIED_SPEC_UNCERTAINTY

    return ThroughputPrediction(
        prefill_tps=prefill,
        decode_tps_single=decode_single,
        decode_tps_aggregate=decode_aggregate,
        prefill_bound_by=prefill_bound,
        decode_bound_by=decode_bound,
        effective_bandwidth_gbs=bw_bytes / 1e9,
        peak_flops_tflops=raw_flops / 1e12,
        uncertainty=min(uncertainty, MAX_UNCERTAINTY),
        warnings=warnings,
    )


def predict_latency(
    throughput: ThroughputPrediction,
    tokens: TokenProfile,
    teams_rtt_s: float = 0.5,
) -> LatencyPrediction:
    """Single-request, no-queueing latency. The floor the simulator builds on."""
    ttft = tokens.billed_prefill_tokens / throughput.prefill_tps if throughput.prefill_tps else float("inf")
    generate = tokens.output_tokens / throughput.decode_tps_single if throughput.decode_tps_single else float("inf")
    return LatencyPrediction(
        ttft_s=ttft,
        generate_s=generate,
        teams_s=teams_rtt_s,
        total_s=ttft + generate + teams_rtt_s,
    )
