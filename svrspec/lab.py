"""Assembling a virtual server out of catalogue parts, and auditing the result.

Why this exists
---------------
Everything else in this tool *sizes*: given a model and a workload it computes
the RAM the deployment needs and picks the fastest memory the CPU can clock.
That is the right answer to "what should I buy", and it is the wrong shape for
the question an engineer with a quote in hand actually asks -- "this is what the
vendor offered me; what happens if I build it?"

The gap is not cosmetic. A sizing run derives capacity, so it can only ever
describe a sanely populated board: it has no way to *express* two 64 GB DIMMs in
an eight-channel server. That configuration has exactly the capacity the model
needs and a quarter of the memory bandwidth, and decode is bandwidth bound, so
token generation runs at a quarter speed. Capacity looks fine, every part is on
the compatibility list, and the box is four times slower than the one on the
spec sheet. It is the most common and most expensive way to get a CPU inference
server wrong, and until this module existed the tool could not even represent
it, let alone warn about it.

So an `Assembly` is a *chosen* build rather than a derived one: catalogue ids and
DIMM counts in, physics and a list of `Finding`s out. The physics is not
re-derived here -- `assemble` fills in `perf.predict_throughput`'s
`channels_populated` argument and reports what comes back. Any number in an
`Assembly` that looks like throughput came from `perf.py`, and a test pins that
equality so it stays true.

Why `assemble` never raises on a bad build
------------------------------------------
A GUI calls this on every keystroke. Raising would mean the caller has to guess
which half-typed configurations are legal before asking, which is the job it
wanted done in the first place. So a nonsensical build still returns an
`Assembly`, carrying `level="error"` findings and whatever numbers the physics
produced for it; `Assembly.ok` is the one thing a caller has to check. An
unknown catalogue id is different -- that is a caller bug, not a user choice, so
`CatalogError` propagates untouched.

Findings, and what each one is for
----------------------------------
    channels-underfilled  error's louder cousin: capacity fine, bandwidth cut.
                          The one this module was written for.
    dpc-derate            two DIMMs per channel, so the whole bus clocks down.
    dimms-not-divisible   DIMMs cannot be spread evenly; the model assumes they
                          are, so reality is at or below what is reported.
    odd-channels          populated channel count is not a power of two, which
                          interleaves less cleanly than the numbers assume.
    ram-too-small         the model does not fit. Nothing else matters.
    ram-exceeds-cpu       more RAM than the part can address.
    sockets-exceeded      more sockets than the part supports.
    no-dimms              empty board.
    too-many-dimms        more than two DIMMs per channel; no server does that.
    dimm-not-catalogued   that module capacity is not in the catalogue, so its
                          speed grade was assumed from a catalogued one.
    no-memory-option      the catalogue has nothing this CPU can clock.

User-visible strings are Korean because the operator reading them is; the
reasoning lives in English comments like the rest of the codebase.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from math import ceil
from pathlib import Path
from typing import Any

from .catalog import Catalog, CatalogError
from .memory import size_memory
from .perf import Efficiency, predict_throughput
from .pipeline import ServiceModel
from .types import (
    CpuSpec,
    MemoryOption,
    ModelSpec,
    QuantSpec,
    ThroughputPrediction,
    TokenProfile,
    Workload,
)

#: No server platform in this catalogue's era accepts a third DIMM per channel.
MAX_DPC = 2

#: Levels a `Finding` may carry, worst first. Also the display order.
LEVELS = ("error", "warn", "info")

#: File format written by `save`. Versioned because a saved build is something an
#: operator mails to a colleague, and a silently reinterpreted field would turn
#: into a wrong spec sheet rather than an error.
SAVE_SCHEMA = "svrspec-vm/v1"


# --------------------------------------------------------------------------
# What the caller chooses, and what falls out of it
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class VirtualMachine:
    """사용자가 고른 것. 카탈로그 id와 개수만 담는다(직렬화 가능).

    Deliberately holds no derived value at all -- no bandwidth, no RAM total, no
    memory option. Anything computed lives in `Assembly`, so a saved build cannot
    disagree with the catalogue it is re-opened against: a corrected DDR speed in
    `memory.json` changes the prediction the next time the file is loaded, which
    is the behaviour a bug fix is supposed to have.
    """

    name: str
    cpu_id: str
    sockets: int
    dimm_gb: int          # 한 장의 용량
    dimm_count: int       # 총 장수 (전 소켓 합계)
    model_id: str
    quant_id: str
    slots: int


@dataclass(frozen=True)
class Finding:
    """조립 결과에서 발견한 문제 한 건."""

    level: str      # "error" | "warn" | "info"
    code: str
    message: str    # 한국어 한 줄 — 무엇이 문제인지
    remedy: str = ""  # 한국어 한 줄 — 어떻게 고치는지

    def __post_init__(self) -> None:
        if self.level not in LEVELS:
            raise ValueError(f"level must be one of {LEVELS}, got {self.level!r}")


@dataclass(frozen=True)
class Assembly:
    """VM + 카탈로그에서 유도된 사실 전부."""

    vm: VirtualMachine
    cpu: CpuSpec
    memory: MemoryOption
    model: ModelSpec
    quant: QuantSpec
    channels_total: int          # cpu.mem_channels * sockets
    channels_populated: int      # 전체 (소켓당이 아니라)
    dimms_per_channel: int       # 1 또는 2
    ram_total_gb: int            # dimm_gb * dimm_count
    ram_used_gb: float           # size_memory 소계
    bandwidth_gbs: float         # 실제 구성의 실효 대역폭
    bandwidth_full_gbs: float    # 전 채널 장착이었다면
    prefill_tps: float
    decode_tps_single: float
    decode_tps_full: float       # 전 채널 장착이었다면
    uncertainty: float
    findings: list[Finding] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """`error` 수준 지적이 없으면 True. 호출자가 볼 유일한 관문."""
        return not any(f.level == "error" for f in self.findings)

    @property
    def sockets(self) -> int:
        """Socket count the physics was run at: at least one, whatever was asked."""
        return max(int(self.vm.sockets), 1)

    @property
    def channels_per_socket(self) -> int:
        """`perf` takes populated channels per socket, not per board."""
        return _per_socket(self.channels_populated, self.sockets)

    @property
    def bandwidth_loss(self) -> float:
        """Fraction of full-population bandwidth given up, 0.0 when none is."""
        if self.bandwidth_full_gbs <= 0:
            return 0.0
        return max(0.0, 1.0 - self.bandwidth_gbs / self.bandwidth_full_gbs)

    def summary(self) -> str:
        return (
            f"{self.vm.name or self.cpu.model}: {self.cpu.model} x{self.sockets}, "
            f"{self.vm.dimm_count}x{self.vm.dimm_gb}GB = {self.ram_total_gb}GB "
            f"({self.channels_populated}/{self.channels_total}ch, "
            f"{self.memory.effective_mts} MT/s), "
            f"decode {self.decode_tps_single:.1f} tok/s"
        )


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------


def assemble(
    cat: Catalog,
    vm: VirtualMachine,
    tokens: TokenProfile | None = None,
) -> Assembly:
    """Resolve a chosen build against the catalogue and audit it.

    Never raises for a bad *configuration* -- see the module docstring. Only a
    catalogue id that does not exist propagates (`CatalogError`).
    """
    tokens = tokens or TokenProfile()
    cpu = cat.cpu(vm.cpu_id)
    model = cat.model(vm.model_id)
    quant = cat.quant(vm.quant_id)
    eff = Efficiency.from_catalog(cat.coefficients)

    findings: list[Finding] = []

    sockets = max(int(vm.sockets), 1)
    count = max(int(vm.dimm_count), 0)
    dimm_gb = max(int(vm.dimm_gb), 0)
    slots = max(int(vm.slots), 1)
    channels_total = cpu.mem_channels * sockets

    # --- population geometry ------------------------------------------------
    # DPC first, because it decides which catalogue rows are even eligible: the
    # 2 DPC rows already carry the derated `effective_mts`, so picking the row is
    # how the clock-down gets into the prediction.
    dpc = max(1, ceil(count / channels_total)) if count else 1
    if dpc <= 1:
        populated = min(count, channels_total)
    else:
        # Past one DIMM per channel every channel is occupied by definition, so
        # bandwidth stops improving and only the DPC derate is left.
        populated = channels_total

    memory = _pick_memory(cat, cpu, min(dpc, MAX_DPC), dimm_gb, findings)

    ram_total_gb = dimm_gb * count
    ram = size_memory(model, quant, tokens, slots=slots)

    # --- physics, from perf.py and nowhere else -----------------------------
    def predict(channels: int | None) -> ThroughputPrediction:
        return predict_throughput(
            model, quant, cpu, memory, tokens, eff,
            slots=slots, sockets=sockets, channels_populated=channels,
        )

    # `None` rather than an explicit channel count when the board is full: that
    # is `perf`'s own default path, so a fully populated Assembly is bit-for-bit
    # the same prediction a caller gets from `predict_throughput` directly.
    channels_arg = (
        None if populated >= channels_total else _per_socket(populated, sockets)
    )
    full = predict(None)
    if populated < 1:
        # An empty board is not a slow machine, it is not a machine. `perf` is
        # asked nothing here -- it reasonably divides by the populated share, and
        # zero channels has no share -- so the throughput fields are zeroed and
        # `no-dimms` carries the explanation. `full` is still computed so the
        # operator can see what populating the board would buy, and its error
        # bars are kept because they describe the coefficients, not the board.
        actual = replace(
            full,
            prefill_tps=0.0,
            decode_tps_single=0.0,
            decode_tps_aggregate=0.0,
            effective_bandwidth_gbs=0.0,
        )
    elif channels_arg is None:
        actual = full
    else:
        actual = predict(channels_arg)

    # --- audit --------------------------------------------------------------
    if count < 1:
        findings.append(Finding(
            "error", "no-dimms",
            "DIMM이 한 장도 없다 — 메모리 없이는 모델을 올릴 수 없다",
            f"최소 1장, 대역폭을 위해서는 전 채널 {channels_total}장을 꽂아라",
        ))
    if int(vm.sockets) < 1:
        findings.append(Finding(
            "error", "sockets-invalid",
            f"소켓 수가 {vm.sockets}이다 — 1 이상이어야 한다",
            f"이 부품은 1~{cpu.sockets_max}소켓까지 지원한다",
        ))
    elif int(vm.sockets) > cpu.sockets_max:
        findings.append(Finding(
            "error", "sockets-exceeded",
            f"{vm.sockets}소켓을 요청했지만 {cpu.model}의 최대 소켓 수는 "
            f"{cpu.sockets_max}이다",
            f"{cpu.sockets_max}소켓 이하로 줄이거나 더 많은 소켓을 지원하는 "
            f"부품을 선택해 주세요",
        ))

    if dpc > MAX_DPC:
        findings.append(Finding(
            "error", "too-many-dimms",
            f"채널당 {dpc}장({count}장 / {channels_total}채널)은 서버 보드가 받지 "
            f"못한다 — 채널당 최대 {MAX_DPC}장이다",
            f"{channels_total * MAX_DPC}장 이하로 줄이고, 용량이 부족하면 장당 "
            f"용량을 올려라",
        ))

    if count >= 1 and ram_total_gb < ram.subtotal_gb:
        findings.append(Finding(
            "error", "ram-too-small",
            f"{model.id} / {quant.id}를 {slots}슬롯으로 돌리는 데 "
            f"{ram.subtotal_gb:.0f}GB가 필요한데 장착량은 {ram_total_gb}GB뿐이다",
            f"여유분까지 고려하면 {ram.provision_gb}GB 이상을 장착하셔야 합니다 "
            f"(권장 {ram.recommended_gb:.0f}GB)",
        ))

    installable = cpu.max_mem_gb * sockets
    if ram_total_gb > installable:
        findings.append(Finding(
            "error", "ram-exceeds-cpu",
            f"{ram_total_gb}GB는 {cpu.model} {sockets}소켓의 최대 장착량 "
            f"{installable}GB를 넘는다",
            f"{installable}GB 이하로 줄이거나 소켓을 늘려라",
        ))

    if count >= 1 and dpc == 1 and populated < channels_total:
        findings.append(_underfilled(cat, cpu, populated, channels_total,
                                     ram_total_gb, actual, full))

    if memory.effective_mts < memory.rated_mts:
        loss = 1.0 - memory.effective_mts / memory.rated_mts
        findings.append(Finding(
            "warn", "dpc-derate",
            f"채널당 {memory.dimms_per_channel}장을 꽂아 메모리가 "
            f"{memory.rated_mts} MT/s에서 {memory.effective_mts} MT/s로 "
            f"떨어진다 (대역폭 {loss:.0%} 손실)",
            f"채널당 1장으로 줄이면 {memory.rated_mts} MT/s로 돌아온다 — "
            f"같은 용량을 장당 2배 용량 모듈로 채워라",
        ))

    if count >= 1 and _uneven(count, channels_total, sockets):
        findings.append(Finding(
            "warn", "dimms-not-divisible",
            f"{count}장을 {sockets}소켓 {channels_total}채널에 균등하게 나눌 수 "
            f"없다 — 채널 간 용량이 달라 인터리빙이 깨진다",
            f"{sockets}소켓과 채널 수의 공배수({_even_counts(channels_total, sockets)}장 "
            f"등)로 맞춰라",
        ))

    if count >= 1 and populated < channels_total and not _power_of_two(populated):
        # Only when the board is *not* full: a fully populated 12-channel EPYC
        # is the vendor's intended configuration, and flagging it would train the
        # operator to ignore this level.
        findings.append(Finding(
            "info", "odd-channels",
            f"장착 채널이 {populated}개로 2의 거듭제곱이 아니다 — 인터리빙이 "
            f"최적이 아닐 수 있어 실제 대역폭은 예측보다 낮을 수 있다",
            f"{_nearest_power_of_two(populated)}채널 또는 전 채널 "
            f"{channels_total}개로 맞춰라",
        ))

    findings.sort(key=lambda f: LEVELS.index(f.level))

    return Assembly(
        vm=vm,
        cpu=cpu,
        memory=memory,
        model=model,
        quant=quant,
        channels_total=channels_total,
        channels_populated=populated,
        dimms_per_channel=min(dpc, MAX_DPC),
        ram_total_gb=ram_total_gb,
        ram_used_gb=ram.subtotal_gb,
        bandwidth_gbs=actual.effective_bandwidth_gbs,
        bandwidth_full_gbs=full.effective_bandwidth_gbs,
        prefill_tps=actual.prefill_tps,
        decode_tps_single=actual.decode_tps_single,
        decode_tps_full=full.decode_tps_single,
        uncertainty=actual.uncertainty,
        findings=findings,
    )


def _underfilled(
    cat: Catalog,
    cpu: CpuSpec,
    populated: int,
    channels_total: int,
    ram_total_gb: int,
    actual: ThroughputPrediction,
    full: ThroughputPrediction,
) -> Finding:
    """The finding this module exists for: right capacity, wrong bandwidth.

    The loss is read off the two predictions rather than computed from the
    channel ratio, so it stays honest when something else (a core-count
    bandwidth ceiling, for one) is binding instead and the ratio would overstate
    what is actually lost.
    """
    lost = 0.0
    if full.effective_bandwidth_gbs > 0:
        lost = max(0.0, 1.0 - actual.effective_bandwidth_gbs / full.effective_bandwidth_gbs)
    message = (
        f"메모리 채널을 {populated}/{channels_total}만 채웠다 — 용량은 "
        f"{ram_total_gb}GB로 멀쩡하지만 대역폭이 {lost:.0%} 깎이고, decode는 "
        f"대역폭 바운드이므로 토큰 생성 속도가 "
        f"{actual.decode_tps_single:.1f} tok/s로 떨어진다 "
        f"(전 채널이면 {full.decode_tps_single:.1f} tok/s)"
    )
    return Finding("warn", "channels-underfilled", message,
                   _fill_remedy(cat, cpu, channels_total, ram_total_gb, lost))


def _fill_remedy(
    cat: Catalog,
    cpu: CpuSpec,
    channels_total: int,
    ram_total_gb: int,
    lost: float,
) -> str:
    """How to keep the same capacity and get the bandwidth back.

    Spelling out the per-DIMM capacity that would be needed matters more than it
    looks: the honest answer is often "you cannot", because the module size that
    divides evenly is not sold in this catalogue. Saying so is what stops the
    remedy from being advice nobody can follow.
    """
    speedup = 1.0 / (1.0 - lost) if lost < 1.0 else float("inf")
    gain = f"대역폭이 {speedup:.1f}배로 돌아온다" if speedup != float("inf") else "대역폭이 회복된다"
    available = _catalogued_dimm_gb(cat, cpu)
    per_dimm = ram_total_gb / channels_total

    if per_dimm == int(per_dimm) and int(per_dimm) in available:
        return (
            f"같은 {ram_total_gb}GB를 {channels_total}×{int(per_dimm)}GB로 전 채널에 "
            f"나눠 꽂으면 {gain}"
        )

    caps = ", ".join(f"{c}GB" for c in sorted(available)) or "없음"
    smallest = min(available) if available else 0
    alt = smallest * channels_total
    return (
        f"같은 {ram_total_gb}GB를 전 채널({channels_total}장)에 나누려면 장당 "
        f"{per_dimm:g}GB가 필요한데 카탈로그에 있는 용량은 {caps}뿐이다 — "
        f"{channels_total}×{smallest}GB={alt}GB로 올려 전 채널을 채우면 {gain}"
    )


def _catalogued_dimm_gb(cat: Catalog, cpu: CpuSpec) -> set[int]:
    """DIMM capacities this CPU could actually clock, at one per channel."""
    return {
        m.dimm_gb
        for m in cat.memory
        if m.ddr_gen == cpu.ddr_gen
        and m.dimms_per_channel == 1
        and m.rated_mts <= cpu.max_ddr_mts
    }


def _pick_memory(
    cat: Catalog,
    cpu: CpuSpec,
    dpc: int,
    dimm_gb: int,
    findings: list[Finding],
) -> MemoryOption:
    """Fastest catalogued module of this capacity that the CPU can clock.

    `Catalog.memory_for` cannot be used here: it picks the fastest option at a
    DPC and ignores capacity, which is exactly the freedom this module has to
    take away from itself -- the operator chose the module size, and the whole
    point is to model what that choice costs.
    """
    eligible = [
        m for m in cat.memory
        if m.ddr_gen == cpu.ddr_gen
        and m.dimms_per_channel == dpc
        and m.rated_mts <= cpu.max_ddr_mts
    ]
    if not eligible:
        # Nothing at this DPC. Fall back across DPC so the caller still gets
        # numbers to look at next to the error, rather than an exception.
        eligible = [
            m for m in cat.memory
            if m.ddr_gen == cpu.ddr_gen and m.rated_mts <= cpu.max_ddr_mts
        ]
        findings.append(Finding(
            "error", "no-memory-option",
            f"카탈로그에 {cpu.model}이 채널당 {dpc}장으로 돌릴 수 있는 "
            f"{cpu.ddr_gen} 모듈이 없다",
            "채널당 장수를 바꾸거나 memory.json에 해당 구성을 추가해 주세요",
        ))
        if not eligible:
            raise CatalogError(
                f"no {cpu.ddr_gen} memory option within {cpu.id}'s "
                f"{cpu.max_ddr_mts} MT/s limit"
            )

    exact = [m for m in eligible if m.dimm_gb == dimm_gb]
    if exact:
        return max(exact, key=lambda m: m.effective_mts)

    chosen = max(eligible, key=lambda m: m.effective_mts)
    # Capacity still comes from what the operator asked for -- `ram_total_gb` is
    # `dimm_gb * dimm_count` regardless. Only the speed grade is borrowed, which
    # is the optimistic assumption, hence the warning.
    findings.append(Finding(
        "warn", "dimm-not-catalogued",
        f"{dimm_gb}GB {cpu.ddr_gen} 모듈이 카탈로그에 없다 — 속도는 "
        f"{chosen.dimm_gb}GB 모듈({chosen.effective_mts} MT/s)과 같다고 가정했다",
        f"카탈로그에 있는 용량({', '.join(f'{c}GB' for c in sorted(_dimm_gb_of(eligible)))}) "
        f"중에서 고르시거나 memory.json에 이 모듈을 추가해 주세요",
    ))
    return chosen


def _dimm_gb_of(options: list[MemoryOption]) -> set[int]:
    return {m.dimm_gb for m in options}


def _per_socket(populated: int, sockets: int) -> int:
    """Populated channels per socket, floored at one where any DIMM exists.

    An odd count cannot divide evenly, and a real machine with all its DIMMs on
    one socket of two is *worse* than this even split -- the far socket reaches
    memory over the interconnect. `dimms-not-divisible` says so; the arithmetic
    here stays on the optimistic side rather than pretending to model placement.
    """
    if populated <= 0 or sockets <= 0:
        return 0
    return max(1, populated // sockets)


def _uneven(count: int, channels_total: int, sockets: int) -> bool:
    """True when the DIMMs cannot be spread evenly over sockets and channels."""
    if sockets > 1 and count % sockets:
        return True
    if count > channels_total and count % channels_total:
        return True
    return False


def _even_counts(channels_total: int, sockets: int) -> str:
    """A couple of counts that do divide evenly, for the remedy line."""
    per_socket_channels = channels_total // max(sockets, 1)
    good = sorted({sockets, sockets * per_socket_channels, channels_total,
                   channels_total * MAX_DPC})
    return ", ".join(str(c) for c in good if c >= 1)


def _power_of_two(n: int) -> bool:
    return n > 0 and not (n & (n - 1))


def _nearest_power_of_two(n: int) -> int:
    p = 1
    while p * 2 <= n:
        p *= 2
    return p


# --------------------------------------------------------------------------
# Running the assembled build
# --------------------------------------------------------------------------


def to_service(cat: Catalog, asm: Assembly, workload: Workload) -> ServiceModel:
    """Service rates for this exact build, ready for `pipeline.run_pipeline`.

    `pipeline.build_service_model` is the normal route and is not used here for
    one reason: it has no `channels_populated`, so it would silently price the
    build as though every channel were filled -- the precise lie this module
    exists to prevent. The slot table is therefore built here, one
    `predict_throughput` call per slot count, same shape as `sizing.decode_table`
    produces (that function belongs to another module and also lacks the
    argument).

    Refuses a build with `error` findings. A pipeline run over a server that
    cannot load the model is a fiction with a plausible-looking timeline
    attached, and that is worse output than an exception.
    """
    if not asm.ok:
        codes = ", ".join(f.code for f in asm.findings if f.level == "error")
        raise ValueError(
            f"조립에 error 수준 문제가 있어 서비스 모델을 만들 수 없다: {codes}"
        )

    eff = Efficiency.from_catalog(cat.coefficients)
    slots = max(int(workload.slots), 1)
    channels_arg = (
        None if asm.channels_populated >= asm.channels_total
        else asm.channels_per_socket
    )

    def predict(k: int):
        return predict_throughput(
            asm.model, asm.quant, asm.cpu, asm.memory, workload.tokens, eff,
            slots=k, sockets=asm.sockets, channels_populated=channels_arg,
        )

    at_slots = predict(slots)
    decode_by_active = {k: predict(k).decode_tps_aggregate for k in range(1, slots + 1)}

    return ServiceModel(
        prefill_tps=at_slots.prefill_tps,
        decode_by_active=decode_by_active,
        slots=slots,
        uncertainty=at_slots.uncertainty,
        label=(
            f"{asm.vm.name or asm.cpu.model} / {asm.model.id} / {asm.quant.id} / "
            f"{asm.cpu.model} x{asm.sockets} / "
            f"{asm.channels_populated}/{asm.channels_total}ch"
        ),
    )


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------


def save(asm_or_vm: Assembly | VirtualMachine, path: str | Path) -> None:
    """Write the *choice* to disk, never the derived numbers.

    Storing a prediction would let a saved file contradict a corrected
    catalogue, and the file would win. Only the ids and counts go out; `load`
    re-derives everything.
    """
    vm = asm_or_vm.vm if isinstance(asm_or_vm, Assembly) else asm_or_vm
    if not isinstance(vm, VirtualMachine):
        raise TypeError("save() takes an Assembly or a VirtualMachine")
    doc = {"schema": SAVE_SCHEMA, "vm": asdict(vm)}
    Path(path).write_text(
        json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def load(cat: Catalog, path: str | Path) -> Assembly:
    """Read a saved build and re-assemble it against the current catalogue.

    The file is untrusted input -- it may have been hand-edited or written by an
    older version -- so every field is checked before it becomes a
    `VirtualMachine`. A wrong type here would otherwise surface much later as a
    nonsensical spec sheet.
    """
    p = Path(path)
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{p.name}: JSON이 깨졌다 ({exc.lineno}행: {exc.msg})") from exc

    if not isinstance(doc, dict):
        raise ValueError(f"{p.name}: 최상위가 객체여야 한다")
    if doc.get("schema") != SAVE_SCHEMA:
        raise ValueError(
            f"{p.name}: schema가 {SAVE_SCHEMA!r}여야 하는데 {doc.get('schema')!r}이다"
        )
    raw = doc.get("vm")
    if not isinstance(raw, dict):
        raise ValueError(f"{p.name}: 'vm' 객체가 없다")

    return assemble(cat, _vm_from_dict(raw, p.name))


#: field name -> (type, must be >= 1). The validation table for a loaded file.
_VM_SCHEMA: tuple[tuple[str, type, bool], ...] = (
    ("name", str, False),
    ("cpu_id", str, False),
    ("sockets", int, True),
    ("dimm_gb", int, True),
    ("dimm_count", int, True),
    ("model_id", str, False),
    ("quant_id", str, False),
    ("slots", int, True),
)


def _vm_from_dict(raw: dict[str, Any], where: str) -> VirtualMachine:
    unknown = set(raw) - {name for name, _, _ in _VM_SCHEMA}
    if unknown:
        raise ValueError(f"{where}: 모르는 필드 {sorted(unknown)}")

    kwargs: dict[str, Any] = {}
    for name, kind, positive in _VM_SCHEMA:
        if name not in raw:
            raise ValueError(f"{where}: 필수 필드 '{name}'가 없다")
        value = raw[name]
        # `bool` is an `int` in Python, and a JSON `true` where a count belongs
        # is a mistake worth naming rather than coercing to 1.
        if isinstance(value, bool) or not isinstance(value, kind):
            raise ValueError(
                f"{where}: '{name}'는 {kind.__name__}여야 하는데 "
                f"{type(value).__name__}이다"
            )
        if positive and value < 1:
            raise ValueError(f"{where}: '{name}'는 1 이상이어야 하는데 {value}이다")
        kwargs[name] = value
    return VirtualMachine(**kwargs)


# --------------------------------------------------------------------------
# What a picker can offer
# --------------------------------------------------------------------------


def dimm_options(cat: Catalog, cpu: CpuSpec, sockets: int) -> list[dict]:
    """Buildable (capacity, count) combinations for this CPU, for a dropdown.

    Only combinations that exist: a module capacity the catalogue carries for
    this memory generation and speed limit, a count that divides evenly across
    sockets and channels, at most `MAX_DPC` per channel, and a total the part can
    address. Offering anything else would push the audit work back onto the
    operator, which is the job this module took over.

    Under-populated counts are deliberately still offered -- the tool has to be
    able to express the bad build in order to warn about it -- so every row
    carries `channels_populated` and `full_channels` for the picker to show.
    """
    sockets = max(int(sockets), 1)
    channels_total = cpu.mem_channels * sockets
    installable = cpu.max_mem_gb * sockets

    rows: list[dict] = []
    for dpc in range(1, MAX_DPC + 1):
        eligible = [
            m for m in cat.memory
            if m.ddr_gen == cpu.ddr_gen
            and m.dimms_per_channel == dpc
            and m.rated_mts <= cpu.max_ddr_mts
        ]
        if not eligible:
            continue
        speed = max(m.effective_mts for m in eligible)
        for dimm_gb in sorted(_dimm_gb_of(eligible)):
            for count in _counts_for(channels_total, sockets, dpc):
                total = dimm_gb * count
                if total > installable:
                    continue
                populated = channels_total if dpc > 1 else min(count, channels_total)
                rows.append({
                    "dimm_gb": dimm_gb,
                    "dimm_count": count,
                    "ram_total_gb": total,
                    "channels_populated": populated,
                    "channels_total": channels_total,
                    "dimms_per_channel": dpc,
                    "effective_mts": speed,
                    "full_channels": populated >= channels_total,
                    "label": (
                        f"{count}×{dimm_gb}GB = {total}GB "
                        f"({populated}/{channels_total}ch, {speed} MT/s)"
                    ),
                })
    rows.sort(key=lambda r: (r["ram_total_gb"], r["dimms_per_channel"], r["dimm_count"]))
    return rows


def _counts_for(channels_total: int, sockets: int, dpc: int) -> list[int]:
    """DIMM counts worth offering at this DPC.

    Powers of two per socket plus the full channel count, because those are the
    populations a server vendor's own memory guide lists; the arbitrary counts in
    between interleave badly and exist in the picker only if someone types them
    in by hand, where `assemble` will flag them.
    """
    per_socket = channels_total // sockets
    if dpc > 1:
        # Every channel occupied twice, or nothing -- a partial second DIMM per
        # channel is the worst of both worlds and no vendor guide lists it.
        return [channels_total * dpc]
    counts = {sockets * p for p in _powers_of_two_upto(per_socket)}
    counts.add(channels_total)
    return sorted(c for c in counts if 1 <= c <= channels_total)


def _powers_of_two_upto(n: int) -> list[int]:
    out, p = [], 1
    while p <= n:
        out.append(p)
        p *= 2
    return out
