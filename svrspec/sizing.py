"""Putting it together: evaluate a (model, quant, CPU, memory) build and rank.

The verdict for every candidate is taken from the *pessimistic* run -- the same
simulated day with throughput derated by the prediction's own uncertainty. A
recommendation that only holds when every estimate lands on the optimistic side
is not a recommendation, and this is a delivery document.
"""

from __future__ import annotations

from dataclasses import replace

from .catalog import Catalog
from .memory import size_memory
from .perf import Efficiency, predict_latency, predict_throughput
from .simulate import simulate
from .types import (
    Candidate,
    CpuSpec,
    MemoryOption,
    ModelSpec,
    QuantSpec,
    ThroughputPrediction,
    Workload,
)

PASS = "pass"
MARGINAL = "marginal"
FAIL = "fail"

#: p95 must sit this far inside the SLA to count as comfortable rather than
#: merely passing. Alarm volume grows, prompts get longer, models get swapped.
COMFORTABLE_HEADROOM = 2.0


def decode_table(
    model: ModelSpec,
    quant: QuantSpec,
    cpu: CpuSpec,
    memory: MemoryOption,
    workload: Workload,
    sockets: int,
    eff: Efficiency,
    derate: float = 1.0,
) -> dict[int, float]:
    """Aggregate decode tok/s for each possible number of busy slots."""
    table: dict[int, float] = {}
    for k in range(1, max(workload.slots, 1) + 1):
        pred = predict_throughput(
            model, quant, cpu, memory, workload.tokens, eff, slots=k, sockets=sockets
        )
        table[k] = pred.decode_tps_aggregate * derate
    return table


def evaluate(
    model: ModelSpec,
    quant: QuantSpec,
    cpu: CpuSpec,
    memory: MemoryOption,
    eff: Efficiency,
    workload: Workload,
    sockets: int = 1,
) -> Candidate:
    throughput = predict_throughput(
        model, quant, cpu, memory, workload.tokens, eff, slots=workload.slots, sockets=sockets
    )
    mean_rtt = sum(workload.teams_rtt_ms) / 2000.0
    latency = predict_latency(throughput, workload.tokens, teams_rtt_s=mean_rtt)

    ram = size_memory(model, quant, workload.tokens, slots=workload.slots)

    nominal, _ = simulate(
        workload,
        prefill_tps=throughput.prefill_tps,
        decode_by_active=decode_table(model, quant, cpu, memory, workload, sockets, eff),
    )

    derate = max(1.0 - throughput.uncertainty, 0.05)
    pessimistic, _ = simulate(
        workload,
        prefill_tps=throughput.prefill_tps * derate,
        decode_by_active=decode_table(
            model, quant, cpu, memory, workload, sockets, eff, derate=derate
        ),
    )

    verdict, reasons = _judge(
        workload=workload,
        cpu=cpu,
        sockets=sockets,
        ram_needed_gb=ram.provision_gb,
        throughput=throughput,
        pessimistic=pessimistic,
    )

    return Candidate(
        model=model,
        quant=quant,
        cpu=cpu,
        memory=memory,
        sockets=sockets,
        memory_gb=ram.provision_gb,
        throughput=throughput,
        latency=latency,
        ram=ram,
        sim=nominal,
        verdict=verdict,
        sla_seconds=workload.sla_seconds,
        sim_pessimistic=pessimistic,
        reasons=reasons,
    )


def _judge(
    workload: Workload,
    cpu: CpuSpec,
    sockets: int,
    ram_needed_gb: int,
    throughput: ThroughputPrediction,
    pessimistic,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    verdict = PASS

    installable = cpu.max_mem_gb * sockets
    if ram_needed_gb > installable:
        return FAIL, [
            f"RAM {ram_needed_gb}GB 필요하지만 {cpu.model} {sockets}소켓 최대 장착량은 "
            f"{installable}GB"
        ]

    if not pessimistic.sla_met:
        verdict = FAIL
        reasons.append(
            f"평상시 p95 {pessimistic.p95_steady_s:.1f}초가 SLA "
            f"{workload.sla_seconds:.0f}초를 초과 (불리한 추정 기준)"
        )
    if not pessimistic.storm_sla_met:
        verdict = FAIL
        reasons.append(
            f"스톰 소진 {pessimistic.storm_drain_s / 60:.1f}분이 목표 "
            f"{workload.storm_drain_sla_s / 60:.0f}분을 초과"
        )

    if verdict == PASS:
        steady_p95 = pessimistic.p95_steady_s or pessimistic.p95_s
        headroom = workload.sla_seconds / steady_p95 if steady_p95 else float("inf")
        if headroom < COMFORTABLE_HEADROOM:
            verdict = MARGINAL
            reasons.append(
                f"SLA 여유 {headroom:.1f}배뿐 — 알람량 증가나 프롬프트 확장을 흡수할 "
                f"버퍼가 얇다"
            )
        if pessimistic.slot_utilisation > 0.5:
            verdict = MARGINAL
            reasons.append(f"슬롯 점유율 {pessimistic.slot_utilisation:.0%}: 상시 부하가 높다")

    if throughput.uncertainty >= 0.45:
        reasons.append(
            f"예측 불확실도 ±{throughput.uncertainty:.0%}: 이 하드웨어 등급의 효율 계수에 "
            f"실측 근거가 없다"
        )
    return verdict, reasons


# --------------------------------------------------------------------------
# Sweeps
# --------------------------------------------------------------------------


def cost_proxy(candidate: Candidate) -> tuple:
    """Ordering key standing in for money, cheapest first.

    Price is only populated for some SKUs, so rank on what is always known:
    socket count, then core count, then TDP. Those track cost and also track
    the things the customer's data centre cares about anyway.
    """
    price = candidate.cpu.price_usd if candidate.cpu.price_usd else 10**9
    return (
        candidate.sockets,
        candidate.cpu.cores * candidate.sockets,
        candidate.cpu.tdp_w * candidate.sockets,
        price,
        candidate.memory_gb,
    )


def sweep_cpus(
    catalog: Catalog,
    model: ModelSpec,
    quant: QuantSpec,
    workload: Workload,
    sockets: int = 1,
    dimms_per_channel: int = 1,
    cpu_ids: list[str] | None = None,
) -> list[Candidate]:
    """Evaluate every catalogued CPU against one model, cheapest-first."""
    eff = Efficiency.from_catalog(catalog.coefficients)
    cpus = [catalog.cpu(c) for c in cpu_ids] if cpu_ids else catalog.cpus
    out: list[Candidate] = []
    for cpu in cpus:
        if sockets > cpu.sockets_max:
            continue
        try:
            memory = catalog.memory_for(cpu, dimms_per_channel)
        except Exception:
            continue
        out.append(evaluate(model, quant, cpu, memory, eff, workload, sockets))
    out.sort(key=cost_proxy)
    return out


def sweep_models(
    catalog: Catalog,
    cpu: CpuSpec,
    quant: QuantSpec,
    workload: Workload,
    sockets: int = 1,
    dimms_per_channel: int = 1,
) -> list[Candidate]:
    """Evaluate every catalogued model against one CPU, largest model first."""
    eff = Efficiency.from_catalog(catalog.coefficients)
    memory = catalog.memory_for(cpu, dimms_per_channel)
    out = [
        evaluate(m, quant, cpu, memory, eff, workload, sockets)
        for m in catalog.models
    ]
    out.sort(key=lambda c: c.model.params_b, reverse=True)
    return out


def tiers(candidates: list[Candidate]) -> dict[str, Candidate | None]:
    """The three rows a delivery document actually needs.

    minimum      cheapest build that meets the SLA under pessimistic estimates
    recommended  cheapest with 2x SLA headroom -- what to actually quote
    comfortable  cheapest that also keeps a slot spare and 3x headroom
    """
    ok = [c for c in candidates if c.verdict in (PASS, MARGINAL)]
    ok.sort(key=cost_proxy)

    def first(predicate) -> Candidate | None:
        return next((c for c in ok if predicate(c)), None)

    minimum = first(lambda c: True)
    recommended = first(lambda c: c.verdict == PASS)
    comfortable = first(
        lambda c: c.verdict == PASS
        and _headroom(c) >= 3.0
        and c.sim_pessimistic is not None
        and c.sim_pessimistic.slot_utilisation < 0.25
    )
    return {"minimum": minimum, "recommended": recommended, "comfortable": comfortable}


def _headroom(candidate: Candidate) -> float:
    return candidate.headroom


def with_slots(workload: Workload, slots: int) -> Workload:
    return replace(workload, slots=slots)
