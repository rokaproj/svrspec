"""What this server does with this model, before anyone mentions alarms.

Why this module is separate from the alarm pipeline
---------------------------------------------------
Everything downstream of `pipeline.py` answers one applied question: can this box
turn a monitoring alarm into a Teams message inside the SLA. That is a real
question -- it is the one that was asked first -- but it is *an application*, not
the only question, and building only that answer made the tool unable to say the
more basic thing an engineer wants when a quote lands on the desk:

    "If I put this model on this server, how many tokens per second do I get,
     how many people can use it at once, and can I fine-tune anything on it?"

An alarm pipeline answers that only through a wrapper of prompt caching, storm
arrival patterns, queueing and an SLA clause. All four of those are assumptions
about one deployment. Change any of them and the pipeline verdict changes while
the machine has not: the tok/s did not move. So the throughput question deserves
its own module with no workload model in it at all, and the alarm pipeline
becomes what it always was -- one application sitting on top of these numbers.

Concretely that means: no arrival process, no SLA, no queue, no prompt cache.
Just a grid -- batch x context -- plus a concurrency sweep, a bottleneck
decomposition, and a training verdict.

Where the physics comes from
----------------------------
Nowhere in here. Every throughput number is a `perf.predict_throughput` return
value, called once per grid point; every byte count is `memory.py`'s. This module
chooses the *operating points* and reports what the physics says about them. A
test pins that by making `predict_throughput` raise, and this module has no
fallback path to answer with -- which is the point. The one thing computed here
that `perf` does not model is the training axis, and see below for how honest
that is.

How a context length becomes a prediction
-----------------------------------------
`perf` charges the KV read at the average context over a generated span --
`prefill_tokens + output_tokens/2`. So a bench row labelled "ctx 4096" is a
`TokenProfile` built backwards from that identity, which makes the label mean
what a reader assumes it means: the request's mean context while decoding was
4096 tokens. Prompt caching is off, because a bench of a model has no shared
prefix to reuse; the alarm pipeline turns it back on because *that* workload
does.

Why the per-sequence rate is the aggregate divided by the batch
--------------------------------------------------------------
`perf` reports `decode_tps_single` as a true single-stream figure -- it ignores
`slots` -- and `decode_tps_aggregate` as the batched total. At batch B, what one
user feels is the total split B ways, and that is what `ThroughputPoint`
publishes as `decode_tps_single`. The two agree at B=1 by construction. Reporting
`perf`'s single-stream number at every batch would draw a flat line through the
most important trade in CPU inference: batching buys throughput and spends
per-user latency.

The training axis is an estimate, and says so
---------------------------------------------
There is no coefficient in `catalog/coefficients.json` for training anything.
Nothing in this project has ever measured a backward pass. So every constant
under "training" below is a literature rule of thumb, marked as such, repeated in
each `TrainingVerdict.reasons` and in `ModelBench.notes` -- because the useful
output of this axis is usually the word "no", and a "no" whose basis is hidden is
indistinguishable from an opinion. The GPU comparison line is the same kind of
estimate: a published peak FLOPS figure with an assumed utilisation, converted
arithmetically. It is there to size the gap (tens of times, not tens of percent),
not to price a purchase order.

User-visible strings are Korean because the operator reading them is; the
reasoning lives in English comments, like the rest of the codebase.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, fields
from typing import Any

from .memory import (
    DEFAULT_UBATCH,
    GB,
    decode_weight_bytes,
    kv_bytes_per_token,
    size_memory,
)
from .perf import Efficiency, peak_flops, widest_isa
# Module-qualified on purpose: the call site stays visible as "this came from
# perf", and a test can substitute the function on either module to prove there
# is no second copy of the roofline hiding in here.
from . import perf
from .types import TokenProfile

# --------------------------------------------------------------------------
# Inference axes
# --------------------------------------------------------------------------

#: Grid defaults. Powers of two because that is how llama.cpp's `-np` and `-c`
#: get set in practice, and because a doubling grid shows a saturation knee in
#: six rows instead of thirty.
DEFAULT_BATCHES = (1, 2, 4, 8, 16, 32)
DEFAULT_CONTEXTS = (512, 2048, 4096, 8192, 16384)
DEFAULT_USERS = (1, 2, 4, 8, 16, 32, 64)
DEFAULT_OUTPUT_TOKENS = 256

# --------------------------------------------------------------------------
# Training coefficients. Every one of these is an estimate -- see the module
# docstring. They are named and grouped here rather than inlined so that a
# reader can audit the whole basis of the training verdict in one screen.
# --------------------------------------------------------------------------

#: Bytes of state per trainable parameter for mixed-precision Adam:
#: bf16 weight (2) + bf16 gradient (2) + fp32 master copy (4) + Adam m (4) +
#: Adam v (4). The widely quoted "16 bytes per parameter" figure.
TRAIN_BYTES_PER_TRAINABLE_PARAM = 16.0

#: Bytes per parameter for a *frozen* base, by scheme. LoRA keeps the base in
#: bf16; QLoRA keeps it in a 4-bit format (NF4 with its scales, ~0.5 B/param).
FROZEN_BYTES_PER_PARAM = {"lora": 2.0, "qlora": 0.5}

#: Trainable fraction of the parameter count for a rank-16 LoRA on the attention
#: and MLP projections. Reported LoRA configurations land between 0.1% and 1%;
#: the middle of that range is used and the range is disclosed.
LORA_TRAINABLE_FRACTION = 0.005

#: Activation bytes kept live per token per layer, as a multiple of n_embd, at
#: bf16 and *without* gradient checkpointing. Counts the residual stream, the
#: attention projections and the SwiGLU intermediate. Same shape of estimate as
#: `memory.ACTIVATION_TENSORS`, and about as soft.
TRAIN_ACTIVATION_TENSORS_PER_LAYER = 10.0
TRAIN_ACTIVATION_BYTES = 2.0

#: One optimiser step processes this many tokens: one sequence of this length.
#: Bigger batches trade memory for a shorter epoch, and the tool reports the
#: point it picked rather than the best one.
TRAIN_SEQ_TOKENS = 2048
TRAIN_SEQUENCES_PER_STEP = 1

#: FLOPs per parameter per token. Forward is 2 (one multiply, one add per
#: weight); a full backward pass costs about twice the forward, hence 6. With a
#: frozen base only activation gradients propagate -- no weight gradients to
#: accumulate for the frozen part -- which is about 4.
TRAIN_FLOPS_PER_PARAM = {"full": 6.0, "lora": 4.0, "qlora": 4.0}

#: QLoRA dequantises the base weights on every use. Penalty on step time.
QLORA_COMPUTE_PENALTY = 1.3

#: An epoch that cannot finish overnight cannot be iterated on, and a
#: fine-tuning run nobody can iterate on is not a capability. This is a
#: judgement call, not physics, so the hour count is always printed next to the
#: verdict for a reader who disagrees with the threshold.
TRAIN_PRACTICAL_EPOCH_HOURS = 24.0

#: Published dense bf16 peak, and an assumed achieved fraction of it. 40% MFU is
#: a good real-world figure for a well-tuned single-GPU fine-tune.
GPU_REFERENCE = ("A100 80GB", 312e12, 80.0)
GPU_MFU = 0.4


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ThroughputPoint:
    """One (batch, context) cell of the inference grid."""

    batch: int
    ctx_tokens: int
    prefill_tps: float
    decode_tps_single: float
    decode_tps_total: float
    prefill_bound: str
    decode_bound: str
    #: Time to first token: how long the prompt takes before anything appears.
    #: `ctx_tokens / prefill_tps`, with the prefix cache off -- see `notes`.
    ttft_s: float = 0.0
    #: The same number as `decode_tps_single`, under the name the industry uses
    #: for it. Not redundancy: a screen that says "Generation tok/s" while the
    #: engine says `decode_tps_single` makes every reader translate, and the
    #: two figures people actually feel are this one and TTFT.
    gen_tps: float = 0.0


@dataclass(frozen=True)
class ConcurrencyPoint:
    """What `users` simultaneous requests feel like, each and together."""

    users: int
    ttft_s: float
    decode_tps_each: float
    response_s: float
    total_tps: float


@dataclass(frozen=True)
class ResourceSplit:
    """How full each ceiling is in one phase, at the reference operating point.

    The percentages are against the *achievable* ceilings -- DRAM bandwidth after
    `eta_bw`, vector throughput after `eta_compute` -- not the datasheet peaks,
    because those are the two numbers `perf` actually binds against. So the
    binding phase reads 100% by construction and the other figure says how much
    of the machine's other half is left idle while it does.
    """

    phase: str
    bandwidth_pct: float
    compute_pct: float
    bound_by: str
    bytes_per_token: float
    flops_per_token: float


@dataclass(frozen=True)
class TrainingVerdict:
    """Whether one training scheme fits, in memory and in wall clock."""

    kind: str
    feasible: bool
    memory_needed_gb: float
    memory_available_gb: float
    step_seconds: float | None
    epoch_hours: float | None
    reasons: list[str]
    gpu_comparison: str


@dataclass(frozen=True)
class ModelBench:
    """Everything this module has to say about one (model, quant, machine)."""

    model_name: str
    quant_id: str
    hardware: str
    throughput: list[ThroughputPoint]
    concurrency: list[ConcurrencyPoint]
    resources: list[ResourceSplit]
    training: list[TrainingVerdict]
    memory_gb: float
    uncertainty: float
    notes: list[str]
    warnings: list[str]


# --------------------------------------------------------------------------
# The bench
# --------------------------------------------------------------------------


def bench_model(
    asm: Any,
    *,
    batches: tuple[int, ...] = DEFAULT_BATCHES,
    contexts: tuple[int, ...] = DEFAULT_CONTEXTS,
    users: tuple[int, ...] = DEFAULT_USERS,
    output_tokens: int = DEFAULT_OUTPUT_TOKENS,
    train_samples: int = 10_000,
    eff: Efficiency | None = None,
) -> ModelBench:
    """Benchmark an assembled build against its model, analytically.

    `asm` is duck-typed rather than annotated as `lab.Assembly`: only `cpu`,
    `memory`, `model`, `quant`, `vm.sockets`, `channels_populated` and
    `uncertainty` are read. A machine the operator assembled in the lab is the
    intended input, but a caller with a CPU and a DIMM in hand should not have to
    build one to ask what the tok/s is.

    `eff` is the one argument not in the caller-facing contract. It defaults to
    the shipped catalogue's coefficients, which is what every other entry point
    resolves anyway; it is injectable so a test can bench against a synthetic
    coefficient set without a catalogue on disk.
    """
    batches = _axis(batches, DEFAULT_BATCHES)
    contexts = _axis(contexts, DEFAULT_CONTEXTS)
    users = _axis(users, DEFAULT_USERS)
    output_tokens = max(int(output_tokens), 1)

    model = asm.model
    quant = asm.quant
    cpu = asm.cpu
    memory = asm.memory
    sockets = max(int(getattr(getattr(asm, "vm", None), "sockets", 1) or 1), 1)
    eff = eff or _default_efficiency()

    channels_total = cpu.mem_channels * sockets
    populated = getattr(asm, "channels_populated", None)
    channels_arg = _channels_per_socket(populated, sockets, channels_total)

    warnings: list[str] = []

    def predict(ctx_tokens: int, slots: int):
        pred = perf.predict_throughput(
            model, quant, cpu, memory, _profile(ctx_tokens, output_tokens), eff,
            slots=slots, sockets=sockets, channels_populated=channels_arg,
        )
        for w in pred.warnings:
            if w not in warnings:
                warnings.append(w)
        return pred

    # --- inference grid --------------------------------------------------
    throughput: list[ThroughputPoint] = []
    for ctx in contexts:
        for batch in batches:
            pred = predict(ctx, batch)
            total = pred.decode_tps_aggregate
            per_sequence = total / batch if batch else 0.0
            throughput.append(ThroughputPoint(
                batch=batch,
                ctx_tokens=ctx,
                prefill_tps=pred.prefill_tps,
                # Not `pred.decode_tps_single` -- see the module docstring.
                decode_tps_single=per_sequence,
                decode_tps_total=total,
                prefill_bound=pred.prefill_bound_by,
                decode_bound=pred.decode_bound_by,
                ttft_s=(ctx / pred.prefill_tps) if pred.prefill_tps else 0.0,
                gen_tps=per_sequence,
            ))

    # --- concurrency ----------------------------------------------------
    ref_ctx = contexts[len(contexts) // 2]
    ref_profile = _profile(ref_ctx, output_tokens)
    concurrency: list[ConcurrencyPoint] = []
    for n in users:
        pred = predict(ref_ctx, n)
        # Prefill is one shared compute ceiling, so n prompts arriving together
        # each get 1/n of it. This is the queue-free view: nobody waits for a
        # slot, they just share the machine. `pipeline.py` is where waiting for
        # a slot gets modelled, and that is a workload question, not this one.
        share = _div(pred.prefill_tps, n)
        ttft = _div(ref_profile.billed_prefill_tokens, share)
        each = _div(pred.decode_tps_aggregate, n)
        concurrency.append(ConcurrencyPoint(
            users=n,
            ttft_s=ttft,
            decode_tps_each=each,
            response_s=ttft + _div(output_tokens, each),
            total_tps=pred.decode_tps_aggregate,
        ))

    # --- bottleneck decomposition ---------------------------------------
    ref_batch = batches[0]
    ref_pred = predict(ref_ctx, ref_batch)
    resources = _resource_splits(model, quant, cpu, eff, ref_pred, ref_batch, ref_ctx)

    # --- training -------------------------------------------------------
    available_gb = _available_ram_gb(asm, cpu, sockets)
    flops_achievable = _achievable_flops(cpu, eff, sockets)
    training = [
        _training_verdict(kind, model, quant, available_gb, flops_achievable,
                          train_samples)
        for kind in ("full", "lora", "qlora")
    ]

    # --- headline memory and honesty ------------------------------------
    ram = size_memory(
        model, quant, _profile(max(contexts), output_tokens),
        slots=1, ctx_tokens=max(contexts),
    )
    uncertainty = max(ref_pred.uncertainty, float(getattr(asm, "uncertainty", 0.0) or 0.0))

    return ModelBench(
        model_name=model.name,
        quant_id=quant.id,
        hardware=_hardware_label(cpu, memory, sockets, populated, channels_total),
        throughput=throughput,
        concurrency=concurrency,
        resources=resources,
        training=training,
        memory_gb=ram.subtotal_gb,
        uncertainty=uncertainty,
        notes=_notes(model, ref_ctx, ref_batch, max(contexts), output_tokens,
                     train_samples, populated, channels_total, uncertainty,
                     available_gb, getattr(asm, "ram_total_gb", None) is not None),
        warnings=warnings,
    )


def _axis(values: tuple[int, ...] | None, default: tuple[int, ...]) -> tuple[int, ...]:
    """Sorted, deduplicated, positive. An empty axis falls back to the default.

    Returning the default rather than raising: the axes are display choices, and
    a GUI that clears a checkbox list should get a bench back, not an exception.
    """
    cleaned = sorted({int(v) for v in (values or ()) if int(v) >= 1})
    return tuple(cleaned) if cleaned else default


def _profile(ctx_tokens: int, output_tokens: int) -> TokenProfile:
    """A `TokenProfile` whose mean decode context is exactly `ctx_tokens`.

    `perf` charges KV reads at `prefill_tokens + output_tokens/2`, so the prompt
    length is solved for from the label rather than set to it. Without this a row
    marked "ctx 512" would silently be a 640-token measurement.
    """
    prefill = max(1, int(round(ctx_tokens - output_tokens / 2.0)))
    return TokenProfile(
        system_tokens=prefill,
        fewshot_tokens=0,
        alarm_tokens=0,
        output_tokens=output_tokens,
        # No shared prefix in a model bench: every prompt is billed in full.
        prompt_cache=False,
    )


def _channels_per_socket(
    populated: int | None, sockets: int, channels_total: int
) -> int | None:
    """`perf`'s `channels_populated` argument, which is per socket, not per board.

    `None` when the board is full, so a fully populated build is bit-for-bit the
    prediction `predict_throughput` gives on its own default path -- the same
    convention `lab.assemble` uses, for the same reason.
    """
    if populated is None or populated >= channels_total:
        return None
    if populated <= 0:
        return 0
    return max(1, int(populated) // sockets)


def _div(numerator: float, denominator: float) -> float:
    """Division that answers "forever" instead of raising.

    An unpopulated board really does have zero bandwidth and zero prefill rate,
    and the honest time to first token there is infinite. Reporting that beats
    both an exception and a zero.
    """
    if denominator <= 0:
        return float("inf")
    return numerator / denominator


# --------------------------------------------------------------------------
# Bottleneck decomposition
# --------------------------------------------------------------------------


def _resource_splits(
    model, quant, cpu, eff: Efficiency, pred, batch: int, ctx: int,
) -> list[ResourceSplit]:
    """Bandwidth and compute occupancy for both phases at one operating point.

    Both phases are reported at the same point so the two rows are comparable,
    and the point is the *single-sequence* one because that is the machine an
    operator meets first. The per-token costs are recomputed from `memory.py`'s
    functions here, which is not a second copy of the physics: they are the same
    calls `perf` makes, and the achieved rate they are multiplied by comes back
    from `perf`. A drift between the two would show up as a percentage above
    100, which a test checks for.
    """
    w_bytes = decode_weight_bytes(model, quant)
    kv_per_token = kv_bytes_per_token(model)
    avg_ctx = ctx
    flops_per_token = 2.0 * model.decode_params_b * 1e9

    bw_ceiling = pred.effective_bandwidth_gbs * 1e9
    compute_ceiling = pred.peak_flops_tflops * 1e12 * eff.eta_compute(widest_isa(cpu)).value

    # Prefill streams the weights once per micro-batch and amortises them over
    # the whole batch of tokens -- `perf`'s own assumption, and the reason
    # prefill almost never touches the bandwidth ceiling.
    prefill_bytes = w_bytes / float(DEFAULT_UBATCH)
    # Decode reads the whole active weight set per sweep, shared across the
    # sequences decoding together, plus every sequence's own KV history.
    decode_bytes = w_bytes / max(batch, 1) + kv_per_token * avg_ctx

    return [
        ResourceSplit(
            phase="prefill",
            bandwidth_pct=_pct(pred.prefill_tps * prefill_bytes, bw_ceiling),
            compute_pct=_pct(pred.prefill_tps * flops_per_token, compute_ceiling),
            bound_by=pred.prefill_bound_by,
            bytes_per_token=prefill_bytes,
            flops_per_token=flops_per_token,
        ),
        ResourceSplit(
            phase="decode",
            bandwidth_pct=_pct(pred.decode_tps_aggregate * decode_bytes, bw_ceiling),
            compute_pct=_pct(pred.decode_tps_aggregate * flops_per_token, compute_ceiling),
            bound_by=pred.decode_bound_by,
            bytes_per_token=decode_bytes,
            flops_per_token=flops_per_token,
        ),
    ]


def _pct(used: float, ceiling: float) -> float:
    return 100.0 * used / ceiling if ceiling > 0 else 0.0


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------


def _training_verdict(
    kind: str, model, quant, available_gb: float, flops_achievable: float,
    train_samples: int,
) -> TrainingVerdict:
    """Can this scheme run here, and what would it cost in hours.

    Memory and time are judged separately and both can veto. A run that fits in
    RAM and takes three weeks per epoch is not a capability this server has, and
    saying "feasible" about it would be the tool lying by omission -- so the hour
    count is in `reasons` either way, for a reader whose patience differs from
    `TRAIN_PRACTICAL_EPOCH_HOURS`.
    """
    params = model.params_b * 1e9
    trainable = params if kind == "full" else params * LORA_TRAINABLE_FRACTION
    frozen_bytes = 0.0 if kind == "full" else params * FROZEN_BYTES_PER_PARAM[kind]

    state_bytes = trainable * TRAIN_BYTES_PER_TRAINABLE_PARAM
    tokens_per_step = TRAIN_SEQ_TOKENS * TRAIN_SEQUENCES_PER_STEP
    activation_bytes = (
        tokens_per_step * model.n_layer * model.n_embd
        * TRAIN_ACTIVATION_TENSORS_PER_LAYER * TRAIN_ACTIVATION_BYTES
    )
    needed_gb = (frozen_bytes + state_bytes + activation_bytes) / GB

    # Memory is charged on the whole parameter count and compute on the active
    # one: an MoE checkpoint has to hold every expert in RAM, but a token only
    # runs through the experts it routed to, in the backward pass as in the
    # forward one.
    flops_per_step = (
        tokens_per_step * TRAIN_FLOPS_PER_PARAM[kind] * model.decode_params_b * 1e9
    )
    if kind == "qlora":
        flops_per_step *= QLORA_COMPUTE_PENALTY

    steps = max(int(train_samples), 1) / max(TRAIN_SEQUENCES_PER_STEP, 1)
    step_s: float | None = None
    epoch_h: float | None = None
    if flops_achievable > 0:
        step_s = flops_per_step / flops_achievable
        epoch_h = step_s * steps / 3600.0

    fits = needed_gb <= available_gb
    in_time = epoch_h is not None and epoch_h <= TRAIN_PRACTICAL_EPOCH_HOURS
    reasons = _training_reasons(
        kind, model, quant, needed_gb, available_gb, fits, step_s, epoch_h,
        train_samples, frozen_bytes, state_bytes, activation_bytes,
    )

    return TrainingVerdict(
        kind=kind,
        feasible=bool(fits and in_time),
        memory_needed_gb=needed_gb,
        memory_available_gb=available_gb,
        step_seconds=step_s,
        epoch_hours=epoch_h,
        reasons=reasons,
        gpu_comparison=_gpu_comparison(flops_per_step, step_s, steps, needed_gb),
    )


TRAIN_LABEL = {
    "full": "전체 파인튜닝",
    "lora": "LoRA (기저 가중치 bf16 동결)",
    "qlora": "QLoRA (기저 가중치 4bit 동결)",
}


def _training_reasons(
    kind: str, model, quant, needed_gb: float, available_gb: float, fits: bool,
    step_s: float | None, epoch_h: float | None, train_samples: int,
    frozen_bytes: float, state_bytes: float, activation_bytes: float,
) -> list[str]:
    """Korean, and specific about which of the two vetoes fired."""
    out = [f"{TRAIN_LABEL[kind]}: {model.name} {model.params_b:g}B 기준"]

    breakdown = [f"학습 상태 {state_bytes / GB:.1f}GB"]
    if frozen_bytes > 0:
        breakdown.insert(0, f"동결 기저 {frozen_bytes / GB:.1f}GB")
    breakdown.append(
        f"활성값 {activation_bytes / GB:.1f}GB"
        f"(시퀀스 {TRAIN_SEQ_TOKENS}토큰 x {TRAIN_SEQUENCES_PER_STEP}개, "
        f"gradient checkpointing 없음)"
    )
    out.append(
        f"메모리 {needed_gb:.1f}GB 필요 = " + " + ".join(breakdown)
        + f" / 장착 {available_gb:.0f}GB"
    )

    if fits:
        out.append(f"메모리는 들어간다 (여유 {available_gb - needed_gb:.0f}GB)")
    else:
        out.append(
            f"메모리가 {needed_gb - available_gb:.1f}GB 모자라다 — "
            f"이 서버에서는 {TRAIN_LABEL[kind]}을 시작할 수 없다"
        )

    if step_s is None or epoch_h is None:
        out.append("연산 천장이 0이라 step 시간을 낼 수 없다 — 이 구성은 학습을 돌리지 못한다")
    else:
        out.append(
            f"1 step({TRAIN_SEQ_TOKENS * TRAIN_SEQUENCES_PER_STEP:,}토큰) {step_s:,.1f}초, "
            f"{train_samples:,}샘플 1 epoch {epoch_h:,.1f}시간"
        )
        if epoch_h > TRAIN_PRACTICAL_EPOCH_HOURS:
            out.append(
                f"1 epoch이 {TRAIN_PRACTICAL_EPOCH_HOURS:.0f}시간을 넘는다 — "
                f"메모리와 무관하게 반복 실험이 불가능하므로 '된다'고 하지 않는다"
            )

    if kind == "full":
        out.append(
            f"가정: 파라미터당 {TRAIN_BYTES_PER_TRAINABLE_PARAM:.0f}바이트"
            "(bf16 가중치 2 + bf16 그래디언트 2 + fp32 마스터 4 + Adam m 4 + v 4), "
            "역전파는 순전파의 2배 연산 — 문헌의 통용 수치를 그대로 쓴 추정이고 "
            "이 프로젝트가 역전파를 실측한 값이 아니다"
        )
    else:
        out.append(
            f"가정: 학습 대상이 전체 파라미터의 {LORA_TRAINABLE_FRACTION:.1%}"
            "(rank 16 어댑터, 보고된 범위는 0.1~1%), 동결 기저는 "
            f"{FROZEN_BYTES_PER_PARAM[kind]:g}바이트/파라미터 — 추정이다"
        )
        out.append(
            "동결 기저라도 활성값 역전파는 전 계층을 통과하므로 연산은 순전파의 약 2배다"
            + ("; 4bit 기저는 사용할 때마다 역양자화하므로 "
               f"step 시간에 {QLORA_COMPUTE_PENALTY:g}배를 곱했다" if kind == "qlora" else "")
        )
    out.append(
        f"추론용 {quant.id} 파일은 학습에 쓰지 않는다 — 위 숫자는 bf16/4bit 학습 기준이다"
    )
    return out


def _gpu_comparison(
    flops_per_step: float, cpu_step_s: float | None, steps: float, needed_gb: float
) -> str:
    """One Korean line sizing the gap against a single reference GPU.

    Arithmetic on a published peak and an assumed utilisation, nothing more. It
    is here because "CPU training is impractical" only lands when the reader can
    see what practical looks like -- and because the memory clause is often the
    more actionable half: a full fine-tune that needs 120 GB does not fit one
    80 GB card either. The step count is the same on both sides, so the epoch
    ratio is the step ratio and only one division is needed.
    """
    name, peak_flops_gpu, vram_gb = GPU_REFERENCE
    achieved = peak_flops_gpu * GPU_MFU
    gpu_step = _div(flops_per_step, achieved)
    gpu_epoch_h = gpu_step * steps / 3600.0

    parts = [
        f"{name} 1장(bf16 {peak_flops_gpu / 1e12:.0f} TFLOPS, MFU {GPU_MFU:.0%} 가정)이면 "
        f"1 step {gpu_step:,.2f}초, 1 epoch {gpu_epoch_h:,.1f}시간"
    ]
    if cpu_step_s is not None and gpu_step > 0:
        parts.append(f"— 이 서버보다 약 {cpu_step_s / gpu_step:,.0f}배 빠르다")
    if needed_gb > vram_gb:
        parts.append(
            f"단 필요 메모리 {needed_gb:.0f}GB는 {vram_gb:.0f}GB 1장에 들어가지 않아 "
            f"여러 장 또는 ZeRO/오프로딩이 필요하다"
        )
    parts.append("(공개 피크 FLOPS로 단순 환산한 추정치다)")
    return " ".join(parts)


def _achievable_flops(cpu, eff: Efficiency, sockets: int) -> float:
    """Vector throughput this part actually reaches, after the coefficient.

    The same product `perf` uses for prefill, reused here rather than re-derived.
    Training in bf16 through AMX is exactly the shape of GEMM the `eta_compute`
    coefficient was fitted to for inference, so this is the least wrong number
    available -- which is not the same as it being measured.
    """
    raw, isa = peak_flops(cpu, sockets)
    return raw * eff.eta_compute(isa).value


def _available_ram_gb(asm: Any, cpu, sockets: int) -> float:
    """RAM the training run may use.

    An assembled build knows what was actually installed, so that wins -- zero
    included, which is why the check is against `None` and not truthiness: an
    empty board cannot train anything, and silently substituting the CPU's
    addressable maximum there would turn "no DIMMs" into "1.5 TB available".
    Without an installed figure at all, the part's maximum is the only defensible
    answer, and it is the optimistic one, so `notes` says which was used.
    """
    installed = getattr(asm, "ram_total_gb", None)
    if installed is not None:
        return float(installed)
    return float(cpu.max_mem_gb * sockets)


# --------------------------------------------------------------------------
# Labels, notes, export
# --------------------------------------------------------------------------


def _hardware_label(cpu, memory, sockets: int, populated, channels_total: int) -> str:
    filled = channels_total if populated is None else int(populated)
    return (
        f"{cpu.model} x{sockets}, {filled}/{channels_total}ch "
        f"{memory.ddr_gen}-{memory.effective_mts}"
    )


def _notes(
    model, ref_ctx: int, ref_batch: int, max_ctx: int, output_tokens: int,
    train_samples: int, populated, channels_total: int, uncertainty: float,
    available_gb: float, installed_known: bool,
) -> list[str]:
    """The assumptions, in Korean, in the order a reader needs them."""
    notes = [
        f"추론 격자는 배치 x 컨텍스트다. 컨텍스트는 생성 중 평균 KV 길이이고 "
        f"(prefill + 출력/2), 출력 길이는 {output_tokens}토큰으로 고정했다",
        "프롬프트 캐시를 끈 상태다 — 모델 자체의 성능을 재는 것이므로 공유 프리픽스를 "
        "가정하지 않는다. 알람 파이프라인은 캐시를 켜므로 TTFT가 더 짧게 나온다",
        f"배치 B의 개별 체감 속도는 총량/B다. 배치는 총 처리량을 사고 개별 지연을 판다",
        f"동시 사용자 축과 자원 분해는 컨텍스트 {ref_ctx:,}토큰"
        f"(자원 분해는 배치 {ref_batch})에서 쟀다",
        "동시 사용자 축에는 대기열이 없다 — 모두 동시에 기계를 나눠 쓰는 경우다. "
        "슬롯을 기다리는 시간은 워크로드 문제이고 pipeline이 다룬다",
        "자원 분해의 %는 실효 천장(계수 적용 후) 기준이다. 데이터시트 피크가 아니다",
        f"추론 실사용 메모리는 슬롯 1개를 격자의 최대 컨텍스트 {max_ctx:,}토큰으로 "
        f"띄웠을 때 기준이다 — 슬롯을 늘리면 KV 캐시가 슬롯 수만큼 늘어난다",
        f"학습 축의 계수는 카탈로그에 없다 — 전부 문헌 기반 추정이고 이 프로젝트가 "
        f"역전파를 실측한 적은 없다. 1 epoch은 {train_samples:,}샘플 x "
        f"{TRAIN_SEQ_TOKENS}토큰 기준이다",
        f"GPU 비교는 공개 피크 FLOPS에 MFU {GPU_MFU:.0%}를 가정한 산술 환산이다 — "
        f"자릿수를 보여주려는 값이고 견적 근거가 아니다",
        (f"학습 가용 메모리 {available_gb:.0f}GB는 조립에 장착된 양이다"
         if installed_known else
         f"학습 가용 메모리 {available_gb:.0f}GB는 이 CPU가 주소지정할 수 있는 최대치다 "
         f"— 장착량을 모르는 입력이므로 낙관적인 쪽으로 잡았다"),
        f"예측 오차 범위는 ±{uncertainty:.0%}다. 이 도구는 모델을 실행하지 않고 "
        f"공개 스펙과 효율 계수만으로 계산한다",
    ]
    if populated is not None and populated < channels_total:
        notes.append(
            f"메모리 채널이 {populated}/{channels_total}만 장착된 조립이다 — decode는 "
            f"대역폭 바운드이므로 아래 모든 생성 속도가 그만큼 깎여 있다"
        )
    if model.active_params_b:
        notes.append(
            f"MoE 모델이다: 토큰당 읽는 가중치와 학습 step 연산은 활성 "
            f"{model.active_params_b:g}B분, 학습 메모리는 전체 {model.params_b:g}B분으로 "
            f"계산했다 — 전문가 전부를 RAM에 얹어야 하지만 토큰은 라우팅된 것만 지난다"
        )
    return notes


SECTIONS = ("throughput", "concurrency", "training")

#: Which `ModelBench` field each CSV section exports, and its row type.
_SECTION_FIELDS = {
    "throughput": (ThroughputPoint, "throughput"),
    "concurrency": (ConcurrencyPoint, "concurrency"),
    "training": (TrainingVerdict, "training"),
}


def to_csv(bench: ModelBench, section: str) -> str:
    """One section as CSV: a header row plus one row per point.

    Columns are read off the dataclass rather than listed here, so a field added
    to a point type cannot silently go missing from the export. `reasons` is a
    list, joined into one cell with a separator no Korean sentence uses, because
    a variable column count would stop the file being a table.
    """
    if section not in _SECTION_FIELDS:
        raise ValueError(
            f"unknown section {section!r}; expected one of {SECTIONS}"
        )
    row_type, attr = _SECTION_FIELDS[section]
    columns = [f.name for f in fields(row_type)]

    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(columns)
    for point in getattr(bench, attr):
        writer.writerow([_cell(getattr(point, name)) for name in columns])
    return out.getvalue()


def _cell(value: Any) -> str:
    if isinstance(value, list):
        return " | ".join(str(v) for v in value)
    if isinstance(value, bool) or isinstance(value, int) or value is None:
        return "" if value is None else str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


_EFFICIENCY: Efficiency | None = None


def _default_efficiency() -> Efficiency:
    """The shipped coefficient set, loaded once.

    Imported inside the function so this module can be used with an injected
    `eff` on a tree where the catalogue is not readable, and so the import error,
    if it happens, names the thing that is actually missing.
    """
    global _EFFICIENCY
    if _EFFICIENCY is None:
        from .catalog import Catalog

        _EFFICIENCY = Efficiency.from_catalog(Catalog().coefficients)
    return _EFFICIENCY
