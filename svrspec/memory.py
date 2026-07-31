"""RAM sizing.

Unlike throughput, memory is almost pure arithmetic — no calibration constant
worth arguing about, and llama.cpp prints the two biggest terms (`KV self size`,
`compute buffer size`) in its startup log, so `svrspec verify` can check this
module against ground truth rather than trusting it.
"""

from __future__ import annotations

from dataclasses import dataclass

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
#: Kept as the default so every existing caller behaves exactly as before; the
#: per-OS figures live in `OS_PROFILES` and are opt-in.
RUNTIME_OS_GB = 2.5

#: Applied to the computed subtotal. Covers allocator slack, page cache the
#: kernel wants for the mmap'd model, log/metric agents, and the fact that a
#: server sized to exactly its working set starts swapping the day someone
#: raises the context window.
DEFAULT_HEADROOM = 1.5

#: Capacities you can actually populate with sane DIMM counts.
PROVISIONABLE_GB = (16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048)


@dataclass(frozen=True)
class OsProfile:
    """What the operating system costs before the model is loaded.

    The same server does not offer the same memory to llama.cpp on every OS,
    and sizing that ignores it under-provisions the box that needs the most.
    Two effects, both real and both modelled here as one number each:

      `runtime_gb`   resident before any model data -- kernel, services, the
                     desktop shell where there is one.
      `headroom`     how much slack the allocator and page cache want on top
                     of the working set. A kernel that keeps an mmap'd model in
                     page cache needs room for it.

    These are estimates. Nothing in this project has measured a Windows Server
    idle footprint, and the numbers below are ordinary operator experience
    rather than a datasheet. `note` says so, and it travels with the sizing.
    """

    id: str
    label: str
    runtime_gb: float
    headroom: float
    note: str
    #: True when going over the limit kills the process instead of slowing it.
    #: A cgroup limit is not a soft ceiling: exceed it and the kernel OOM-kills
    #: the container, where the same overshoot on a normal host would page to
    #: disk and merely get slow. The headroom factor cannot express that
    #: difference -- it is one number, and the two failure modes are not one
    #: kind of bad -- so the flag rides alongside it and the sizing says so.
    hard_limit: bool = False

    @property
    def overrun_consequence(self) -> str:
        """What happens if the working set exceeds what was provisioned."""
        return (
            "메모리를 넘기면 OOM kill로 프로세스가 죽는다 — 느려지는 것이 아니다"
            if self.hard_limit
            else "메모리를 넘기면 스왑으로 넘어가 느려진다 — 죽지는 않는다"
        )


#: Ordered cheapest-first, which is also the order an operator should prefer:
#: an inference box has no reason to carry a desktop.
OS_PROFILES: dict[str, OsProfile] = {
    "linux-headless": OsProfile(
        id="linux-headless",
        label="Linux 서버 (GUI 없음)",
        runtime_gb=1.0,
        headroom=1.4,
        note="커널·systemd·sshd 정도만 상주한다고 본 값이다. 배포판과 설치 옵션에 따라 달라진다",
    ),
    "linux-container": OsProfile(
        id="linux-container",
        label="Linux 컨테이너",
        runtime_gb=0.6,
        # Deliberately the *largest* headroom, not the smallest. The container
        # frees the most memory and then punishes overrun the hardest: a host
        # that overshoots swaps, a cgroup that overshoots is killed. Sizing it
        # tightest because its baseline is lightest gets the trade backwards.
        headroom=1.6,
        note=(
            "컨테이너는 호스트 커널을 공유하므로 이미지 안의 상주분만 센다. "
            "상주량은 가장 적지만 실패 방식이 가장 나쁘다 — 아래 hard_limit 참조"
        ),
        hard_limit=True,
    ),
    "windows-server": OsProfile(
        id="windows-server",
        label="Windows Server",
        runtime_gb=2.5,
        headroom=1.5,
        note="이 프로젝트의 기존 기본값이다. Windows Server 2022 유휴 상주분을 기준으로 잡았다",
    ),
    "windows-desktop": OsProfile(
        id="windows-desktop",
        label="Windows 데스크톱",
        runtime_gb=4.0,
        headroom=1.6,
        note=(
            "셸·검색 색인·백신이 함께 도는 사무용 PC 기준이다. "
            "서버 산정에 쓸 값은 아니고, 개발자 노트북에서 미리 돌려보는 경우를 위한 것이다"
        ),
    ),
}

#: The profile assumed when a caller does not choose one. Windows Server,
#: because that is what the packaged app installs onto and because it is the
#: most expensive of the server profiles -- defaulting to the cheapest would
#: quietly under-size every build that did not think about it.
DEFAULT_OS_PROFILE = "windows-server"


def os_profile(name: str | None) -> OsProfile:
    """Look up a profile by id, falling back to the default."""
    return OS_PROFILES.get(name or DEFAULT_OS_PROFILE, OS_PROFILES[DEFAULT_OS_PROFILE])


def measured_os_profile(
    runtime_gb: float,
    source: str,
    *,
    base: str | None = None,
    label: str | None = None,
    headroom: float | None = None,
) -> OsProfile:
    """A profile whose resident figure came from a real machine.

    The shipped numbers are operator experience, not measurements -- nothing in
    this project has watched a Windows Server idle. That is fine as a default
    and bad as a final answer, so this is the way out: read the number off the
    box that will actually run the model and pass it in.

        free -g            # Linux, "used" with nothing else running
        Get-Counter '\\Memory\\Committed Bytes'   # Windows

    `source` is required and is not decoration. A sizing report that quotes a
    measured figure has to say which machine it was measured on, exactly as
    `measured.py` requires a log identity before promoting a coefficient.

    Only the resident cost is measurable this way; `headroom` stays a policy
    choice and is inherited from `base` unless given, and so does `hard_limit`,
    because whether overrun kills you is a property of the runtime and not
    something a memory reading can tell you.
    """
    if runtime_gb < 0:
        raise ValueError(f"runtime_gb는 음수일 수 없다: {runtime_gb}")
    if not source.strip():
        raise ValueError(
            "source가 비어 있다 — 어느 기계에서 잰 값인지 없으면 실측이라고 할 수 없다"
        )
    parent = os_profile(base)
    return OsProfile(
        id=f"{parent.id}-measured",
        label=label or f"{parent.label} (실측)",
        runtime_gb=float(runtime_gb),
        headroom=parent.headroom if headroom is None else float(headroom),
        note=f"상주량 {runtime_gb:.2f}GB는 실측값이다 — 출처: {source.strip()}",
        hard_limit=parent.hard_limit,
    )

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
    headroom: float | None = None,
    measured_bpw: float | None = None,
    flash_attention: bool = True,
    os_name: str | OsProfile | None = None,
) -> MemoryBreakdown:
    """Full RAM requirement for one (model, quant, concurrency) deployment.

    `os_name` picks an `OS_PROFILES` entry, which supplies both the resident
    cost of the operating system and the slack its allocator wants. An explicit
    `headroom` still wins, so callers that had tuned it keep their value.
    """
    # An OsProfile may be passed directly, which is how a measured figure gets
    # in: `measured_os_profile()` builds one from a reading off the real box.
    if isinstance(os_name, OsProfile):
        profile, chosen = os_name, True
    else:
        profile, chosen = os_profile(os_name), os_name is not None
    if headroom is None:
        headroom = profile.headroom if chosen else DEFAULT_HEADROOM
    runtime_gb = profile.runtime_gb if chosen else RUNTIME_OS_GB

    ctx = ctx_tokens if ctx_tokens else _rounded_ctx(tokens.peak_ctx_tokens)

    weights = weight_bytes(model, quant, measured_bpw)
    kv = kv_cache_bytes(model, ctx, slots, kv_bits)
    compute = compute_buffer_bytes(model, slots, flash_attention=flash_attention)

    subtotal_gb = (weights + kv + compute) / GB + runtime_gb
    recommended = subtotal_gb * headroom

    return MemoryBreakdown(
        weights_gb=weights / GB,
        kv_cache_gb=kv / GB,
        compute_buffer_gb=compute / GB,
        runtime_os_gb=runtime_gb,
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
