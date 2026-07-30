"""Frozen contract shared by every module and both catalog files.

Nothing here may be changed by catalog owners; the JSON files must conform to
these field names exactly. `svrspec catalog validate` enforces it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Provenance. Delivery documents live or die on this field: a sizing report
# built from guessed silicon specs is worthless, so every catalog row must
# declare where its numbers came from.
# --------------------------------------------------------------------------

#: Values transcribed from the vendor's own datasheet / ark.intel.com / AMD.
SOURCE_VENDOR = "vendor_datasheet"
#: Values from the model card / config.json of the published checkpoint.
SOURCE_MODEL_CARD = "model_card"
#: Values read off a citable third-party specification database -- Wikipedia's
#: per-SKU tables, PassMark, WikiChip. Not the vendor, but a real page that was
#: actually opened and can be re-checked. Used because Intel's and AMD's own
#: sites answer 403 to anything that is not a browser, and a transcribed
#: third-party table beats an estimate.
SOURCE_THIRD_PARTY = "third_party_db"
#: Values derived from a benchmark log somebody actually ran -- llama-bench
#: output, or a llama-server startup/inference log -- and imported into the
#: tool. Ranks above a third-party spec table because it is a measurement of
#: the thing being predicted rather than a specification of the hardware, but
#: it is not a citable public page, so `source_url` carries the log's identity
#: (file path, ticket, or run label) instead of a URL.
SOURCE_MEASUREMENT = "measurement"
#: Values recalled without a datasheet at hand. Usable for exploration,
#: flagged in reports, must be confirmed before a spec sheet ships.
SOURCE_UNVERIFIED = "unverified"

VALID_SOURCES = (
    SOURCE_VENDOR,
    SOURCE_MODEL_CARD,
    SOURCE_THIRD_PARTY,
    SOURCE_MEASUREMENT,
    SOURCE_UNVERIFIED,
)


@dataclass(frozen=True)
class ModelSpec:
    """Architecture of one LLM checkpoint, as needed for memory + roofline math.

    Everything here is obtainable from the HF `config.json` without downloading
    weights. `gguf.py` can also produce this from a local .gguf header.
    """

    id: str
    name: str
    family: str
    params_b: float  # total parameter count in billions
    n_layer: int
    n_embd: int
    n_head: int
    n_kv_head: int  # == n_head for MHA, < n_head for GQA, 1 for MQA
    n_vocab: int
    ctx_train: int
    #: Explicit per-head dim. Usually n_embd // n_head, but some families
    #: (Gemma 3, some Qwen3) decouple it, which changes KV cache size.
    head_dim: int | None = None
    #: Active params per token for MoE checkpoints; None for dense models.
    #: Decode bandwidth cost scales with ACTIVE params, not total.
    active_params_b: float | None = None
    tie_embeddings: bool = False
    korean: bool = False  # vendor claims Korean support
    license: str = ""
    source: str = SOURCE_UNVERIFIED
    source_url: str = ""
    notes: str = ""

    @property
    def kv_head_dim(self) -> int:
        return self.head_dim if self.head_dim else self.n_embd // self.n_head

    @property
    def decode_params_b(self) -> float:
        """Params actually read per generated token."""
        return self.active_params_b if self.active_params_b else self.params_b

    @property
    def size_class(self) -> str:
        p = self.params_b
        if p < 2:
            return "sub-2B"
        if p < 5:
            return "2-5B"
        if p < 10:
            return "7-9B"
        if p < 20:
            return "13-15B"
        if p < 40:
            return "30-35B"
        return "70B+"


@dataclass(frozen=True)
class QuantSpec:
    """A llama.cpp quantization type.

    `bits_per_weight` is the *effective* file-size BPW including the tensors
    llama.cpp keeps at higher precision (embeddings / output), not the nominal
    block width. Calibrated against real GGUF file sizes.
    """

    id: str
    bits_per_weight: float
    quality: str  # "lossless" | "near-lossless" | "good" | "degraded"
    notes: str = ""


@dataclass(frozen=True)
class CpuSpec:
    """One server CPU SKU. Fields chosen to feed both roofline bottlenecks."""

    id: str
    vendor: str  # "Intel" | "AMD"
    family: str  # e.g. "Xeon Scalable 4th Gen (Sapphire Rapids)"
    model: str  # e.g. "Silver 4410Y"
    cores: int
    threads: int
    base_ghz: float
    #: All-core sustained turbo. This, not max single-core turbo, drives
    #: prefill throughput. If the vendor does not publish it, estimate from
    #: base and say so in `notes`.
    all_core_turbo_ghz: float
    max_turbo_ghz: float
    #: ISA tokens understood by perf.py: "avx2", "avx512", "amx-bf16",
    #: "amx-int8", "avx512-vnni". Order irrelevant.
    isa: list[str]
    mem_channels: int
    ddr_gen: str  # "DDR4" | "DDR5" | "MRDIMM"
    max_ddr_mts: int
    max_mem_gb: int
    sockets_max: int
    l3_mb: float
    tdp_w: int
    launched: str = ""
    price_usd: int | None = None
    #: PassMark CPU Mark multithread score. A real measured figure, unlike
    #: everything else here, so it serves as an independent cross-check on the
    #: compute roofline: two parts with similar cores x clock x vector width
    #: should score similarly, and a big divergence means a spec is wrong.
    passmark_multithread: int | None = None
    socket: str = ""
    source: str = SOURCE_UNVERIFIED
    source_url: str = ""
    #: Second citation, when the memory subsystem figures came from a different
    #: page than the core/clock figures.
    memory_source_url: str = ""
    notes: str = ""

    def has(self, feature: str) -> bool:
        return feature in self.isa

    @property
    def peak_bandwidth_gbs(self) -> float:
        """Theoretical per-socket DRAM bandwidth at max rated speed."""
        return self.mem_channels * self.max_ddr_mts * 8 / 1000.0


@dataclass(frozen=True)
class CoefficientSpec:
    """One efficiency coefficient: how much of a theoretical ceiling llama.cpp
    actually reaches on a given class of hardware.

    These are what turn a roofline ceiling into a prediction, and they are
    catalogue data rather than something the tool measures on the machine it
    happens to be running on. That keeps the simulator purely analytical: it
    never loads the local CPU, and it can size a server it has no access to.

    `confidence` drives the reported error bars, so an estimate can never be
    passed off as a measurement:
      measured  -- read off a real llama.cpp run on hardware of this class
      derived   -- computed from a measurement plus a documented assumption
      estimate  -- literature or reasoning only; widens the uncertainty band
    """

    id: str
    #: "eta_bw" | "eta_compute" | "per_core_bw_gbs" | "dual_socket_efficiency"
    kind: str
    #: What it applies to: a DDR generation for eta_bw, an ISA token for
    #: eta_compute, or "*" for globals.
    key: str
    value: float
    confidence: str
    source: str = SOURCE_UNVERIFIED
    source_url: str = ""
    notes: str = ""


VALID_CONFIDENCE = ("measured", "derived", "estimate")


@dataclass(frozen=True)
class MemoryOption:
    """A populated DIMM configuration, and what it actually clocks at.

    DDR5 at 2 DIMMs per channel derates (e.g. 4800 -> 4400 MT/s), which
    directly cuts decode throughput. Modelling this is not a detail: it is
    often the difference between two candidate builds.
    """

    id: str
    ddr_gen: str
    rated_mts: int
    dimms_per_channel: int
    #: Actual operating speed once populated at this DPC.
    effective_mts: int
    dimm_gb: int
    ecc: bool = True
    kind: str = "RDIMM"
    source: str = SOURCE_UNVERIFIED
    source_url: str = ""
    notes: str = ""


# --------------------------------------------------------------------------
# Workload + results. Owned by the engine, listed here so report.py and the
# catalog validator agree on shapes.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TokenProfile:
    """Per-alarm token budget of the refine-and-analyse prompt."""

    system_tokens: int = 300
    fewshot_tokens: int = 400
    alarm_tokens: int = 250
    output_tokens: int = 250
    #: System + few-shot are identical for every alarm, so llama.cpp slot
    #: reuse can skip re-prefilling them. Large latency lever.
    prompt_cache: bool = True

    @property
    def prefill_tokens(self) -> int:
        return self.system_tokens + self.fewshot_tokens + self.alarm_tokens

    @property
    def cached_prefix_tokens(self) -> int:
        return self.system_tokens + self.fewshot_tokens if self.prompt_cache else 0

    @property
    def billed_prefill_tokens(self) -> int:
        """Tokens actually run through prefill for a steady-state request."""
        return self.prefill_tokens - self.cached_prefix_tokens

    @property
    def peak_ctx_tokens(self) -> int:
        return self.prefill_tokens + self.output_tokens


@dataclass(frozen=True)
class Workload:
    alarms_per_day: int = 150
    #: Fraction of daily volume inside business hours.
    business_hours: tuple[int, int] = (8, 20)
    business_share: float = 0.8
    #: Alarm storm: `storm_size` alarms arriving within `storm_window_s`.
    storm_size: int = 40
    storm_window_s: float = 30.0
    storms_per_day: int = 2
    #: Concurrent llama.cpp slots.
    slots: int = 2
    #: Alarm -> Teams end-to-end latency target.
    sla_seconds: float = 30.0
    #: Storm must be fully drained within this.
    storm_drain_sla_s: float = 300.0
    teams_rtt_ms: tuple[float, float] = (200.0, 800.0)
    tokens: TokenProfile = field(default_factory=TokenProfile)
    seed: int = 20260730


@dataclass(frozen=True)
class MemoryBreakdown:
    weights_gb: float
    kv_cache_gb: float
    compute_buffer_gb: float
    runtime_os_gb: float
    subtotal_gb: float
    headroom_factor: float
    recommended_gb: float
    #: Nearest sane DIMM-populated capacity >= recommended_gb.
    provision_gb: int


@dataclass(frozen=True)
class ThroughputPrediction:
    prefill_tps: float
    decode_tps_single: float
    decode_tps_aggregate: float
    #: Which roofline is binding: "bandwidth" | "compute" | "core-bandwidth".
    prefill_bound_by: str
    decode_bound_by: str
    effective_bandwidth_gbs: float
    peak_flops_tflops: float
    #: Multiplicative uncertainty, e.g. 0.25 -> +/-25%.
    uncertainty: float
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class LatencyPrediction:
    ttft_s: float
    generate_s: float
    teams_s: float
    total_s: float


@dataclass(frozen=True)
class SimResult:
    p50_s: float
    p95_s: float
    p99_s: float
    max_s: float
    max_queue_depth: int
    storm_drain_s: float
    slot_utilisation: float
    sla_met: bool
    storm_sla_met: bool
    completed: int
    #: Fraction of wall-clock time at least one request is in flight. This is
    #: the number a task manager would show as CPU load: llama.cpp pins every
    #: thread while a request runs, so the box is effectively saturated whenever
    #: it is not idle. Distinct from `slot_utilisation`, which divides by the
    #: slot count and so answers "how much of my concurrency am I using".
    busy_fraction: float = 0.0
    #: p95 over NON-storm alarms. This is what the per-alarm SLA is judged on.
    #: A storm queues by definition -- forty alarms in thirty seconds cannot
    #: each be answered within thirty seconds unless the box is sized for the
    #: burst rate rather than the daily rate -- so bursts are judged separately,
    #: by `storm_drain_s` against the drain target. Mixing the two would let a
    #: storm-heavy day condemn a server that handles normal traffic comfortably.
    p95_steady_s: float = 0.0
    steady_completed: int = 0


@dataclass(frozen=True)
class Candidate:
    """One (model, quant, cpu, memory) combination, fully evaluated."""

    model: ModelSpec
    quant: QuantSpec
    cpu: CpuSpec
    memory: MemoryOption
    sockets: int
    memory_gb: int
    throughput: ThroughputPrediction
    latency: LatencyPrediction
    ram: MemoryBreakdown
    sim: SimResult
    verdict: str  # "pass" | "marginal" | "fail"
    #: Copied from the workload so a Candidate is self-describing in reports.
    sla_seconds: float = 30.0
    #: The same day re-run with throughput derated by the prediction's own
    #: uncertainty. The verdict is taken from THIS, not from the nominal run:
    #: a spec sheet that only holds if every estimate lands optimistically is
    #: not a spec sheet.
    sim_pessimistic: SimResult | None = None
    reasons: list[str] = field(default_factory=list)

    @property
    def headroom(self) -> float:
        """SLA budget divided by steady-state p95. >1 fits, <1 misses.

        Taken from the pessimistic run when one exists, and from steady-state
        alarms rather than storm alarms -- see `SimResult.p95_steady_s`.
        """
        sim = self.sim_pessimistic or self.sim
        p95 = sim.p95_steady_s or sim.p95_s
        if p95 <= 0:
            return float("inf")
        return self.sla_seconds / p95
