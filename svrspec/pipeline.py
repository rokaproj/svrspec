"""Runnable alarm -> LLM -> Teams pipeline, driven by simulated time.

Why no model is ever loaded, and why that is the correct decision
-----------------------------------------------------------------
The server being sized is not this machine. It is a CPU-only box in the
customer's rack that nobody here can log into, so the only honest way to answer
"how long does one alarm take" is the analytic prediction in `perf.py`: a
roofline throughput figure derived from published hardware specifications and the
catalogued efficiency coefficients. Starting a real inference runtime here would
measure a laptop nobody is buying -- a different question, answered at the cost
of pinning every core on the operator's PC.

So every service time in this module is *computed*, never observed. That makes
this a **simulated run**, not a run: the control flow, the queue, the slot
contention, the backpressure and the Teams payload are all real code paths that
the operator can watch, while the durations come from a model with error bars
(`ServiceModel.uncertainty`). Nothing here loads weights, starts a child program,
or opens a network connection -- and a test enforces that by inspecting the
import graph.

Why this exists next to simulate.py
-----------------------------------
`simulate.py` summarises a day into percentiles. That answers "does this build
hold the SLA", which is what the sizing report needs, but it cannot show what
happened to one alarm. This module records the journey of every single alarm --
arrival, queue wait, first token, generation end, delivery -- and produces the
Teams card that would have been posted. That is what makes a dry run inspectable
by a human, and it is what a UI can animate.

The two must agree. They share one physical model of llama.cpp's continuous
batching, taken from `simulate.py` deliberately rather than re-derived:

  * prefill is compute bound, so the aggregate prompt-processing rate is fixed
    and split evenly across the alarms in flight;
  * decode is bandwidth bound, so the aggregate generation rate *rises* with the
    number of occupied slots -- one sweep over the weights emits a token for
    every slot -- and is likewise split evenly.

Rates are piecewise constant between events, so the loop jumps to the next
arrival or the next phase completion, whichever comes first. No time stepping, no
accumulated drift, and identical input gives identical output.

Replay pacing
-------------
`speed=0.0` is virtual time: the run completes as fast as the CPU can walk the
event list, and it is the default because it is what tests and reports want.
`speed=60.0` replays the same event list at sixty times real time, pausing
between events so a human can watch the queue fill. The pause is the only place
this module ever waits, and it is bounded by the compression factor.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Protocol

from .perf import Efficiency, predict_throughput
from .sizing import decode_table
from .types import CpuSpec, MemoryOption, ModelSpec, QuantSpec, TokenProfile, Workload
from .workload import teams_rtt

#: Lifecycle of one alarm. "queued" -> "prefill" -> "decode" -> "deliver" are
#: emitted for every served alarm; "done" is the terminal record for an alarm
#: that was shed by backpressure and therefore never reached a slot.
PHASES = ("queued", "prefill", "decode", "deliver", "done")

#: Token-remainder tolerance. Rates are floats, so a phase is finished when it is
#: within this of zero rather than exactly at it.
EPS = 1e-9

#: Adaptive-card accent per severity. Kept as data so the mapping is visible in
#: one place instead of scattered through the card builder.
SEVERITY_COLOUR = {
    "critical": "attention",
    "major": "warning",
    "minor": "accent",
    "warning": "good",
}


# --------------------------------------------------------------------------
# What the pipeline consumes and produces
# --------------------------------------------------------------------------


class AlarmLike(Protocol):
    """The alarm record this module needs.

    Declared structurally on purpose: `mockdata.Alarm` satisfies it, and so does
    anything an operator feeds in from a file, without this module importing the
    generator. `device`, `code`, `message` and `site` are read with `getattr`
    defaults because they only affect the card text, not the timing.
    """

    id: str
    at_s: float
    severity: str
    storm_id: int | None
    prompt_tokens: int


@dataclass(frozen=True)
class Delivery:
    """알람 한 건이 Teams에 도달하기까지의 전 여정."""

    alarm_id: str
    arrived_s: float
    started_s: float          # 슬롯에 올라간 시각
    first_token_s: float      # prefill 끝 = 첫 토큰
    generated_s: float        # decode 끝
    delivered_s: float        # Teams 응답 완료
    slot: int
    prompt_tokens: int
    output_tokens: int
    queue_wait_s: float
    ttft_s: float
    generate_s: float
    deliver_s: float
    total_s: float
    storm_id: int | None
    severity: str
    dropped: bool = False     # 큐 상한 초과로 버려졌으면 True


@dataclass(frozen=True)
class RunStats:
    received: int
    delivered: int
    dropped: int
    p50_s: float
    p95_s: float
    p99_s: float
    max_s: float
    p95_steady_s: float        # 스톰 아닌 알람만
    storm_drain_s: float
    max_queue: int
    mean_queue: float
    busy_fraction: float
    slot_utilisation: float
    sla_met: bool
    storm_sla_met: bool
    tokens_prefill: int
    tokens_generated: int
    wall_clock_s: float        # 실제로 돌린 시간 (가상시간이면 ~0)


class Sink(Protocol):
    """Where a finished alarm goes. One method, so a webhook can be dropped in.

    The pipeline never learns whether the far side is a channel, a log file or a
    list in memory -- which is exactly why a dry run is safe to hand to an
    operator.
    """

    def send(self, delivery: "Delivery", card: dict) -> None: ...


@dataclass
class TeamsSink:
    """전송 대상. 기본 구현은 메모리에 모은다 — 여기서 네트워크를 치지 않는다.

    This is the seam where a real webhook goes: subclass and override `send` to
    post `card` to the channel's incoming-webhook URL. Deliberately not written
    here -- the default must stay usable on an air-gapped box, and a dry run must
    never notify a live operations channel by accident.
    """

    sent: list = field(default_factory=list)

    def send(self, delivery: "Delivery", card: dict) -> None:
        self.sent.append((delivery, card))


@dataclass(frozen=True)
class ServiceModel:
    """Service rates for one build, as the queue engine needs them.

    `decode_by_active` maps occupied-slot count -> aggregate decode tok/s, which
    is the shape `sizing.decode_table()` already produces. `uncertainty` is
    carried through from the throughput prediction so a caller can show the error
    bar next to the run and never mistake a computed figure for a measured one.
    """

    prefill_tps: float
    decode_by_active: dict[int, float]
    slots: int = 1
    uncertainty: float = 0.0
    label: str = ""

    def __post_init__(self) -> None:
        if self.prefill_tps <= 0:
            raise ValueError("prefill_tps must be positive")
        if not self.decode_by_active:
            raise ValueError("decode_by_active must not be empty")
        if any(v <= 0 for v in self.decode_by_active.values()):
            raise ValueError("decode rates must be positive")

    def decode_rate(self, active: int) -> float:
        """Aggregate decode rate at `active` in flight.

        Clamps to the tabulated range rather than extrapolating past it: the
        table was built from predictions at specific slot counts, and inventing
        capacity beyond the last one would flatter a build that was never
        evaluated. Same rule as `simulate._decode_rate`, on purpose.
        """
        table = self.decode_by_active
        if active in table:
            return table[active]
        keys = sorted(table)
        if active < keys[0]:
            return table[keys[0]]
        return table[keys[-1]]


def build_service_model(
    model: ModelSpec,
    quant: QuantSpec,
    cpu: CpuSpec,
    memory: MemoryOption,
    eff: Efficiency,
    workload: Workload,
    sockets: int = 1,
) -> ServiceModel:
    """Turn a hardware build into the service rates the pipeline runs on.

    The single route from silicon to seconds, so there is one place where the
    analytic prediction enters this module and no chance of a second, divergent
    derivation appearing later.
    """
    prediction = predict_throughput(
        model, quant, cpu, memory, workload.tokens, eff,
        slots=max(workload.slots, 1), sockets=sockets,
    )
    return ServiceModel(
        prefill_tps=prediction.prefill_tps,
        decode_by_active=decode_table(
            model, quant, cpu, memory, workload, sockets, eff
        ),
        slots=max(workload.slots, 1),
        uncertainty=prediction.uncertainty,
        label=f"{model.id} / {quant.id} / {cpu.model} x{sockets}",
    )


# --------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------


@dataclass
class _Job:
    """One alarm in flight. Mutable; `Delivery` is the frozen record it becomes."""

    alarm: Any
    alarm_id: str
    arrived_s: float
    storm_id: int | None
    severity: str
    prompt_tokens: int          # billed prefill tokens for THIS alarm
    output_tokens: int
    rtt_s: float
    remaining_prefill: float
    remaining_decode: float
    phase: str = "prefill"
    slot: int = -1
    started_s: float = 0.0
    first_token_s: float = 0.0
    generated_s: float = 0.0
    delivered_s: float = 0.0
    dropped: bool = False

    def to_delivery(self) -> Delivery:
        return Delivery(
            alarm_id=self.alarm_id,
            arrived_s=self.arrived_s,
            started_s=self.started_s,
            first_token_s=self.first_token_s,
            generated_s=self.generated_s,
            delivered_s=self.delivered_s,
            slot=self.slot,
            prompt_tokens=self.prompt_tokens,
            output_tokens=self.output_tokens,
            queue_wait_s=self.started_s - self.arrived_s,
            ttft_s=self.first_token_s - self.arrived_s,
            generate_s=self.generated_s - self.first_token_s,
            deliver_s=self.delivered_s - self.generated_s,
            total_s=self.delivered_s - self.arrived_s,
            storm_id=self.storm_id,
            severity=self.severity,
            dropped=self.dropped,
        )


class _Pacer:
    """Maps simulated seconds onto real ones for the replay mode.

    Factored out so the pacing decision is testable and so the wait lives in
    exactly one place. At `speed <= 0` every call is a no-op, which is what makes
    virtual time free.
    """

    def __init__(
        self,
        speed: float,
        monotonic: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if speed < 0:
            raise ValueError("speed must be >= 0 (0 = virtual time)")
        self.speed = speed
        self._monotonic = monotonic
        self._sleep = sleeper
        self._origin_real = monotonic()
        self._origin_sim = 0.0

    def start(self, sim_t: float) -> None:
        self._origin_real = self._monotonic()
        self._origin_sim = sim_t

    def wait_until(self, sim_t: float) -> None:
        if self.speed <= 0:
            return
        target = self._origin_real + (sim_t - self._origin_sim) / self.speed
        gap = target - self._monotonic()
        if gap > 0:
            self._sleep(gap)

    def elapsed(self) -> float:
        return self._monotonic() - self._origin_real


def teams_card(alarm: AlarmLike, delivery: Delivery) -> dict:
    """The payload that would be posted to the Teams channel.

    Built as an Adaptive Card so the shape is the real one, with the alarm id,
    severity and a Korean one-line summary lifted to the top level as well --
    a sink that only logs should not have to walk the card body to say what it
    just handled.
    """
    device = str(getattr(alarm, "device", "") or "")
    code = str(getattr(alarm, "code", "") or "")
    message = str(getattr(alarm, "message", "") or "")
    site = str(getattr(alarm, "site", "") or "")
    device_type = str(getattr(alarm, "device_type", "") or "")

    head = " ".join(part for part in (device, code) if part)
    summary = f"[{delivery.severity}] {head} — {message}".strip(" —")

    return {
        "type": "message",
        "alarm_id": delivery.alarm_id,
        "severity": delivery.severity,
        "summary": summary,
        "attachments": [
            {
                "contentType": "application/vnd.microsoft.card.adaptive",
                "content": {
                    "type": "AdaptiveCard",
                    "version": "1.4",
                    "body": [
                        {
                            "type": "TextBlock",
                            "text": summary,
                            "weight": "bolder",
                            "wrap": True,
                            "color": SEVERITY_COLOUR.get(delivery.severity, "default"),
                        },
                        {
                            "type": "FactSet",
                            "facts": [
                                {"title": "장비", "value": device or "-"},
                                {"title": "유형", "value": device_type or "-"},
                                {"title": "위치", "value": site or "-"},
                                {"title": "코드", "value": code or "-"},
                                {"title": "심각도", "value": delivery.severity},
                                {"title": "대기", "value": f"{delivery.queue_wait_s:.1f}초"},
                                {"title": "첫 토큰", "value": f"{delivery.ttft_s:.1f}초"},
                                {"title": "총 소요", "value": f"{delivery.total_s:.1f}초"},
                            ],
                        },
                    ],
                },
            }
        ],
    }


def run_pipeline(
    alarms: Iterable[AlarmLike],
    service: ServiceModel,
    workload: Workload,
    *,
    sink: Sink | None = None,
    speed: float = 0.0,
    queue_limit: int | None = None,
    on_event: Callable[[dict], None] | None = None,
) -> tuple[RunStats, list[Delivery]]:
    """Replay a day of alarms through the sized server and out to Teams.

    `speed=0.0` (default) runs in virtual time and is deterministic: identical
    input yields an identical `Delivery` list. `speed=60.0` paces the same event
    list at sixty times real time so a human can watch it.

    `queue_limit` applies backpressure. When the queue is full the *newest*
    arrival is shed, not the oldest -- an alarm already waiting has burned queue
    time that would be wasted by discarding it, and the operator would rather
    lose the tail of a storm than its head.
    """
    if queue_limit is not None and queue_limit < 1:
        raise ValueError("queue_limit must be >= 1, or None for unbounded")
    slots = max(int(workload.slots), 1)

    pacer = _Pacer(speed)
    tokens: TokenProfile = workload.tokens
    sink = TeamsSink() if sink is None else sink
    emit = _emitter(on_event)

    # Arrival order decides everything downstream, including the Teams round-trip
    # draw, so sort defensively and break ties by input position: a generator
    # that emits two alarms at the same instant must still replay identically.
    incoming = sorted(enumerate(alarms), key=lambda pair: (float(pair[1].at_s), pair[0]))

    # Same stream as `simulate` uses, seeded from the workload, so the two
    # engines see identical round trips and their percentiles are comparable.
    rng = random.Random(workload.seed + 1)
    jobs = [_new_job(alarm, tokens, teams_rtt(workload, rng)) for _, alarm in incoming]

    queue: list[_Job] = []
    active: list[_Job] = []
    free_slots = list(range(slots))
    served: list[_Job] = []
    finished: list[_Job] = []      # served + dropped, in the order they left

    now = jobs[0].arrived_s if jobs else 0.0
    first_event_s = now
    next_arrival = 0
    busy_slot_seconds = 0.0
    any_busy_seconds = 0.0
    queue_seconds = 0.0
    max_queue = 0

    pacer.start(now)

    while next_arrival < len(jobs) or queue or active:
        pacer.wait_until(now)

        # --- admit everything that has arrived, shedding if the queue is full
        while next_arrival < len(jobs) and jobs[next_arrival].arrived_s <= now + EPS:
            job = jobs[next_arrival]
            next_arrival += 1
            if queue_limit is not None and len(queue) >= queue_limit:
                job.dropped = True
                job.started_s = job.first_token_s = job.arrived_s
                job.generated_s = job.delivered_s = job.arrived_s
                finished.append(job)
                emit("done", job, now, len(queue), len(active), dropped=True)
                continue
            queue.append(job)
            emit("queued", job, now, len(queue), len(active))

        max_queue = max(max_queue, len(queue))

        # --- fill free slots, oldest first
        while len(active) < slots and queue:
            job = queue.pop(0)
            job.started_s = now
            job.slot = free_slots.pop(0)
            active.append(job)
            emit("prefill", job, now, len(queue), len(active), slot=job.slot)

        if not active:
            # Idle. Jump to the next arrival instead of stepping through the gap;
            # in replay mode the pause at the top of the loop covers the wait.
            if next_arrival < len(jobs):
                now = jobs[next_arrival].arrived_s
                continue
            break

        share = len(active)
        prefill_rate = service.prefill_tps / share
        decode_rate = service.decode_rate(share) / share

        horizons = [
            (job.remaining_prefill / prefill_rate) if job.phase == "prefill"
            else (job.remaining_decode / decode_rate)
            for job in active
        ]
        step = min(horizons)
        if next_arrival < len(jobs):
            step = min(step, max(jobs[next_arrival].arrived_s - now, 0.0))
        if step <= 0 or step == float("inf"):
            # Nothing can progress and nothing can arrive. Only reachable from a
            # zero-rate configuration, which `ServiceModel` already rejects, so
            # this is a guard against spinning rather than an expected path.
            break

        end = now + step
        first_tokens: list[_Job] = []
        # Pace to the end of the step, not just its start: the transitions below
        # happen at `end`, and in replay mode a watcher must see them then rather
        # than one step early. Absolute targets, so a late wake-up does not
        # accumulate into drift.
        pacer.wait_until(end)
        for job in active:
            if job.phase == "prefill":
                job.remaining_prefill -= prefill_rate * step
                if job.remaining_prefill <= EPS:
                    job.remaining_prefill = 0.0
                    job.phase = "decode"
                    job.first_token_s = end
                    first_tokens.append(job)
            else:
                job.remaining_decode -= decode_rate * step

        busy_slot_seconds += share * step
        any_busy_seconds += step
        queue_seconds += len(queue) * step
        now = end

        # Emit first-token transitions before completions so the event stream
        # reads in the order a watcher would see it.
        for job in first_tokens:
            emit("decode", job, now, len(queue), len(active), slot=job.slot)

        completed_now: list[_Job] = []
        still_active: list[_Job] = []
        for job in active:
            if job.phase == "decode" and job.remaining_decode <= EPS:
                # The slot frees the moment generation ends: posting to Teams is
                # network wait, not server work, so it does not hold capacity.
                job.remaining_decode = 0.0
                job.generated_s = now
                job.delivered_s = now + job.rtt_s
                free_slots.append(job.slot)
                free_slots.sort()
                completed_now.append(job)
            else:
                still_active.append(job)
        active = still_active

        # Hand off in completion order: one event, one card, one sink call each.
        # The event is stamped at `generated_s` -- the moment the answer existed
        # and the post went out -- so the stream stays in time order. The round
        # trip itself lands in `Delivery.delivered_s`, and the loop does not
        # advance for it because it costs the server nothing.
        for job in completed_now:
            served.append(job)
            finished.append(job)
            delivery = job.to_delivery()
            emit("deliver", job, job.generated_s, len(queue), len(active),
                 slot=job.slot, delivered_s=job.delivered_s)
            sink.send(delivery, teams_card(job.alarm, delivery))

    deliveries = [job.to_delivery() for job in finished]
    stats = _summarise(
        workload=workload,
        slots=slots,
        received=len(jobs),
        deliveries=deliveries,
        span=max(now - first_event_s, 1e-9),
        busy_slot_seconds=busy_slot_seconds,
        any_busy_seconds=any_busy_seconds,
        queue_seconds=queue_seconds,
        max_queue=max_queue,
        wall_clock_s=pacer.elapsed(),
    )
    return stats, deliveries


def _new_job(alarm: AlarmLike, tokens: TokenProfile, rtt_s: float) -> _Job:
    """Freeze one alarm into the work item the queue engine moves around."""
    billed = _billed_prefill(alarm, tokens)
    return _Job(
        alarm=alarm,
        alarm_id=str(alarm.id),
        arrived_s=float(alarm.at_s),
        storm_id=getattr(alarm, "storm_id", None),
        severity=str(getattr(alarm, "severity", "minor")),
        prompt_tokens=billed,
        output_tokens=int(tokens.output_tokens),
        rtt_s=rtt_s,
        remaining_prefill=float(billed),
        remaining_decode=float(tokens.output_tokens),
    )


def _emitter(on_event: Callable[[dict], None] | None):
    """Wrap the callback so the event shape is built in exactly one place."""

    def emit(phase: str, job: _Job, t: float, queued: int, active: int, **extra) -> None:
        if on_event is None:
            return
        event = {
            "t": t,
            "phase": phase,
            "alarm_id": job.alarm_id,
            "queue": queued,
            "active": active,
        }
        event.update(extra)
        on_event(event)

    return emit


def _billed_prefill(alarm: AlarmLike, tokens: TokenProfile) -> int:
    """Prefill tokens actually charged for this alarm.

    The shared system + few-shot prefix is identical for every alarm, so with
    llama.cpp slot reuse it is prefilled once and skipped thereafter; only the
    alarm body is billed. Falls back to the workload's nominal alarm size when
    the record does not carry its own token count, which is what makes a run over
    plain `Arrival`-shaped input agree with `simulate`.
    """
    body = getattr(alarm, "prompt_tokens", None)
    body = int(body) if body else int(tokens.alarm_tokens)
    if tokens.prompt_cache:
        return max(body, 1)
    # Cache off: the shared prefix is re-processed for every alarm.
    return max(body + tokens.system_tokens + tokens.fewshot_tokens, 1)


def _summarise(
    workload: Workload,
    slots: int,
    received: int,
    deliveries: list[Delivery],
    span: float,
    busy_slot_seconds: float,
    any_busy_seconds: float,
    queue_seconds: float,
    max_queue: int,
    wall_clock_s: float,
) -> RunStats:
    served = [d for d in deliveries if not d.dropped]
    dropped = len(deliveries) - len(served)
    totals = sorted(d.total_s for d in served)
    steady = sorted(d.total_s for d in served if d.storm_id is None)
    # With no quiet-hours traffic at all, judge the per-alarm SLA on the whole
    # population rather than on nothing. Same fallback as `simulate`.
    p95_steady = _percentile(steady or totals, 0.95)
    drain = _storm_drain(served)

    return RunStats(
        received=received,
        delivered=len(served),
        dropped=dropped,
        p50_s=_percentile(totals, 0.50),
        p95_s=_percentile(totals, 0.95),
        p99_s=_percentile(totals, 0.99),
        max_s=totals[-1] if totals else 0.0,
        p95_steady_s=p95_steady,
        storm_drain_s=drain,
        max_queue=max_queue,
        mean_queue=queue_seconds / span if span > 0 else 0.0,
        busy_fraction=min(any_busy_seconds / span, 1.0) if span > 0 else 0.0,
        slot_utilisation=min(busy_slot_seconds / (span * slots), 1.0) if span > 0 else 0.0,
        sla_met=p95_steady <= workload.sla_seconds,
        storm_sla_met=drain <= workload.storm_drain_sla_s,
        tokens_prefill=sum(d.prompt_tokens for d in served),
        tokens_generated=sum(d.output_tokens for d in served),
        wall_clock_s=wall_clock_s,
    )


def _storm_drain(served: list[Delivery]) -> float:
    """Longest span from a storm's first alarm arriving to its last one sent."""
    by_storm: dict[int, list[Delivery]] = {}
    for d in served:
        if d.storm_id is not None:
            by_storm.setdefault(d.storm_id, []).append(d)

    worst = 0.0
    for group in by_storm.values():
        begin = min(d.arrived_s for d in group)
        end = max(d.delivered_s for d in group)
        worst = max(worst, end - begin)
    return worst


def _percentile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolated percentile of an already-sorted list.

    Deliberately a local copy of `simulate`'s definition rather than an import of
    its private helper: the agreement test compares the two engines' p95, and
    that comparison is only meaningful if both compute the percentile the same
    way. Keeping the definition identical is the point; sharing the function
    would couple this module to another module's internals.
    """
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    low = int(pos)
    high = min(low + 1, len(sorted_values) - 1)
    frac = pos - low
    return sorted_values[low] * (1 - frac) + sorted_values[high] * frac
