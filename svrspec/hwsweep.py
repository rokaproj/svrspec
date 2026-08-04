"""How far each CPU in the catalogue gets on one model, at one setup.

The question this answers
------------------------
`modelbench` answers "what does this model do on *this* machine". That is the
right question once a machine exists. Before one does, the operator has a
different one: given this model and this way of using it, **which box do I buy,
and what does each one actually get me** -- not as a pass/fail against somebody
else's alarm count, but as a ceiling: how many concurrent users hold the
response time, how many tokens per second come out, and what runs out first.

So this module runs the same engine across every CPU in the catalogue and
reports one row per CPU. The setup -- concurrent users, context length, output
length, sockets, DIMMs -- is the caller's, not a fixed profile: two operators
sizing the same model for a chat assistant and for a batch summariser are asking
about different machines, and a single hardcoded workload would answer neither.

Why the ceiling is a search, not a grid cell
--------------------------------------------
"Where does it stop" cannot be read off a fixed user list. A ladder of
1, 2, 4, ... 64 users tells you the box was fine at 32 and broken at 64, which
is a factor-of-two answer to a question the operator will spend real money on.
`max_users_within` bisects instead, so the reported ceiling is the largest user
count that actually held the target -- and it is bounded, so a machine that
holds the target at any load says so rather than searching forever.

What is deliberately not here
-----------------------------
No alarm counts, no storms, no SLA-per-day. Those belong to the delivery report
and still live in `pipeline`/`simulate` unchanged; this module exists precisely
because the hardware question does not need them. It also does not rank by a
single score: the rows carry tok/s, response time and RAM, and which of those
matters is the operator's call, not this module's.

Nothing here runs a model or touches hardware. Same analytical engine as every
other screen, so a laptop can answer it for a machine nobody has bought yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .catalog import Catalog, CatalogError
from .modelbench import bench_model
from .types import TokenProfile

#: Response-time targets the ceiling is measured against, in seconds. Not an
#: SLA -- an SLA is a contract about a specific deployment, and this screen is
#: for choosing hardware before there is one. These are the thresholds at which
#: an interactive session stops feeling interactive, which is the thing a
#: reader can judge without knowing the customer.
DEFAULT_TARGET_S = 30.0
#: Never search past this. A machine that holds the target at 4096 concurrent
#: users is not going to be limited by concurrency, and saying "at least 4096"
#: is both true and more useful than a number with six digits of false
#: precision. Also the loop bound: bisection over this range is 12 steps.
MAX_USERS = 4096
#: Below this, a build is reported as not usable at any concurrency rather than
#: given a ceiling of zero -- the distinction the operator needs is "too slow to
#: use" versus "fine but only for one person".
MIN_USERS = 1


@dataclass(frozen=True)
class SweepRow:
    """One CPU's ceiling on one model, at the caller's setup."""

    cpu_id: str
    label: str
    vendor: str
    cores: int
    threads: int
    sockets: int
    #: Total memory channels actually populated across all sockets, and the
    #: effective bandwidth that geometry gives. Two CPUs with the same core
    #: count and different channel counts are the whole reason this column
    #: exists -- see the Gold 6234 row in the catalogue.
    channels: int
    bandwidth_gbs: float
    ram_total_gb: int
    ram_needed_gb: float
    fits: bool
    #: Single-stream figures: what one user feels with nobody else on the box.
    gen_tps: float
    ttft_s: float
    #: The ceiling. `max_users` is the largest concurrency that held
    #: `target_s`; `capped` says the search hit MAX_USERS rather than a real
    #: limit, so the number is a floor, not a measurement.
    max_users: int
    capped: bool
    #: Aggregate throughput at that ceiling, and what each user saw there.
    total_tps_at_max: float
    response_s_at_max: float
    #: What ran out first at the reference point: bandwidth, compute, or cores.
    bound_by: str
    #: What a single user waits, with nobody else on the box. Carried even when
    #: `max_users` is 0, because that is exactly the case where the operator
    #: needs a number: "미달" on all 43 rows with nothing else on screen does
    #: not say whether the target was missed by a second or by a minute.
    response_s_single: float
    #: How the response time at the ceiling splits, and which half dominates.
    #: Reporting only `bound_by` was misleading here: that is the *decode*
    #: bound, and decode is bandwidth-bound on nearly every part in this
    #: catalogue, so the column read "bandwidth" for all 43 rows while the
    #: thing actually setting the ceiling was prefill. At a long context the
    #: prompt is compute-bound and shared n ways, so a part with more cores
    #: takes more concurrent users even when a higher-bandwidth part beats it
    #: on tokens per second -- which is the single most decision-changing fact
    #: this screen has, and it was invisible.
    ttft_s_at_max: float
    decode_s_at_max: float
    limited_by: str
    tdp_w: int
    price_usd: int | None
    #: Non-fatal notes about this row specifically (derated memory, socket
    #: count clamped, and so on). Fatal problems put the row in `blocked`.
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Sweep:
    """Every CPU's answer to one question, plus what was asked."""

    model_name: str
    quant_id: str
    target_s: float
    ctx_tokens: int
    output_tokens: int
    sockets: int
    dimm_gb: int
    dimm_count: int
    rows: list[SweepRow]
    #: Set only when no CPU met the target. The fastest single-user response
    #: any part managed, so the page can say how far off the setup is rather
    #: than showing 43 identical failures and leaving the reader to guess
    #: whether to buy a bigger box or ask a smaller question.
    best_single_s: float | None
    #: CPUs that could not be evaluated at all, as (cpu_id, why). A build that
    #: cannot be assembled is not a slow build and must not be ranked as one.
    blocked: list[tuple[str, str]]
    notes: list[str] = field(default_factory=list)


def max_users_within(
    probe: Any,
    target_s: float,
    *,
    lo: int = MIN_USERS,
    hi: int = MAX_USERS,
) -> tuple[int, bool]:
    """Largest user count whose response time still holds `target_s`.

    `probe(users) -> float` returns the response time at that concurrency, and
    is assumed monotonically non-decreasing in `users` -- more load never makes
    a machine faster. Returns `(users, capped)`, where `capped` marks a search
    that ran into `hi` rather than finding a real limit.

    Bisection rather than walking a ladder, because the ladder answer is only
    ever accurate to a factor of two and this number is what somebody buys
    hardware against. Returns 0 when even one user misses the target: a box too
    slow for a single session has no concurrency ceiling to report, and calling
    that "1" would read as usable.
    """
    target_s = float(target_s)
    if target_s <= 0:
        return 0, False
    lo, hi = max(1, int(lo)), max(1, int(hi))
    if lo > hi:
        lo, hi = hi, lo

    first = _finite(probe(lo))
    if first is None or first > target_s:
        return 0, False
    if hi == lo:
        return lo, False

    last = _finite(probe(hi))
    if last is not None and last <= target_s:
        # Held all the way to the ceiling: report it as a floor, not a limit.
        return hi, True

    good, bad = lo, hi
    while bad - good > 1:
        mid = (good + bad) // 2
        got = _finite(probe(mid))
        if got is not None and got <= target_s:
            good = mid
        else:
            bad = mid
    return good, False


def sweep(
    cat: Catalog,
    *,
    model_id: str,
    quant_id: str,
    ctx_tokens: int = 4096,
    output_tokens: int = 256,
    target_s: float = DEFAULT_TARGET_S,
    sockets: int = 1,
    dimm_gb: int = 64,
    dimm_count: int = 0,
    slots: int = 1,
    cpu_ids: tuple[str, ...] | None = None,
) -> Sweep:
    """Run every catalogue CPU against one model at one setup.

    `dimm_count=0` means "populate one DIMM per channel", resolved per CPU --
    which is the only fair default across a catalogue whose parts have 6, 8 and
    12 channels each. Pinning a single DIMM count instead would hand the
    8-channel parts a half-populated memory subsystem and call the result a
    hardware comparison.
    """
    from .lab import VirtualMachine, assemble

    model = cat.model(model_id)
    quant = cat.quant(quant_id)
    ctx_tokens = max(int(ctx_tokens), 1)
    output_tokens = max(int(output_tokens), 1)
    sockets = max(int(sockets), 1)
    slots = max(int(slots), 1)

    wanted = tuple(cpu_ids) if cpu_ids else tuple(c.id for c in cat.cpus)
    rows: list[SweepRow] = []
    blocked: list[tuple[str, str]] = []

    for cpu_id in wanted:
        try:
            cpu = cat.cpu(cpu_id)
        except CatalogError as exc:
            blocked.append((cpu_id, str(exc)))
            continue

        # Clamp rather than skip: a 1S part asked to be a 2S build is a real
        # answer about that part, not a reason to drop it from the comparison.
        use_sockets = min(sockets, max(int(cpu.sockets_max), 1))
        count = int(dimm_count) or cpu.mem_channels * use_sockets

        vm = VirtualMachine(
            name=cpu_id[:40] or "sweep",
            cpu_id=cpu_id,
            sockets=use_sockets,
            dimm_gb=max(int(dimm_gb), 1),
            dimm_count=count,
            model_id=model.id,
            quant_id=quant.id,
            slots=slots,
        )
        tokens = TokenProfile(
            system_tokens=0,
            fewshot_tokens=0,
            alarm_tokens=ctx_tokens,
            output_tokens=output_tokens,
            prompt_cache=False,
        )
        try:
            asm = assemble(cat, vm, tokens)
        except CatalogError as exc:
            blocked.append((cpu_id, str(exc)))
            continue

        notes = [f.message for f in asm.findings if f.level != "error"]
        if not asm.ok:
            why = "; ".join(f.message for f in asm.findings if f.level == "error")
            blocked.append((cpu_id, why or "구성 오류"))
            continue

        row = _row(
            asm, cpu, use_sockets,
            ctx_tokens=ctx_tokens, output_tokens=output_tokens,
            target_s=target_s, notes=notes,
        )
        if row is not None:
            rows.append(row)

    rows.sort(key=lambda r: (-r.max_users, -r.total_tps_at_max, r.cpu_id))
    return Sweep(
        model_name=model.name,
        quant_id=quant.id,
        target_s=float(target_s),
        ctx_tokens=ctx_tokens,
        output_tokens=output_tokens,
        sockets=sockets,
        dimm_gb=max(int(dimm_gb), 1),
        dimm_count=int(dimm_count),
        rows=rows,
        best_single_s=_best_single(rows, target_s),
        blocked=blocked,
        notes=_notes(dimm_count, sockets),
    )


def _row(
    asm: Any,
    cpu: Any,
    sockets: int,
    *,
    ctx_tokens: int,
    output_tokens: int,
    target_s: float,
    notes: list[str],
) -> SweepRow | None:
    """One CPU's row, or None if the engine could not bench it.

    The concurrency ladder is benched once per probed user count rather than
    once for a fixed list, because the ceiling is a search. `bench_model` is
    cheap -- arithmetic over the catalogue -- so a dozen bisection steps per
    CPU stays well inside the button-press budget the other screens use.
    """
    cache: dict[int, Any] = {}

    def at(users: int) -> Any:
        users = max(1, int(users))
        if users not in cache:
            mb = bench_model(
                asm,
                batches=(1,),
                contexts=(ctx_tokens,),
                users=(users,),
                output_tokens=output_tokens,
            )
            cache[users] = mb
        return cache[users]

    def response(users: int) -> float:
        mb = at(users)
        pts = list(mb.concurrency)
        return float(pts[0].response_s) if pts else float("inf")

    base = at(1)
    base_pts = list(base.concurrency)
    base_pt = base_pts[0] if base_pts else None
    tp = list(base.throughput)
    if not tp:
        return None
    ref = tp[0]

    ceiling, capped = max_users_within(response, target_s)
    top = at(ceiling) if ceiling >= 1 else base
    top_pts = list(top.concurrency)
    top_pt = top_pts[0] if top_pts else None

    return SweepRow(
        cpu_id=str(cpu.id),
        label=str(base.hardware),
        vendor=str(cpu.vendor),
        cores=int(cpu.cores) * sockets,
        threads=int(cpu.threads) * sockets,
        sockets=sockets,
        channels=int(getattr(asm, "channels_populated", 0) or 0),
        bandwidth_gbs=_f(getattr(asm, "bandwidth_gbs", 0.0)),
        ram_total_gb=int(getattr(asm, "ram_total_gb", 0) or 0),
        ram_needed_gb=_f(ref.ram_gb),
        fits=bool(ref.fits),
        gen_tps=_f(ref.gen_tps or ref.decode_tps_single),
        ttft_s=_f(ref.ttft_s),
        max_users=int(ceiling),
        capped=bool(capped),
        total_tps_at_max=_f(top_pt.total_tps) if top_pt else 0.0,
        response_s_at_max=_f(top_pt.response_s) if top_pt else 0.0,
        bound_by=str(ref.decode_bound or ""),
        response_s_single=_f(base_pt.response_s) if base_pt else 0.0,
        ttft_s_at_max=_f(top_pt.ttft_s) if top_pt else 0.0,
        decode_s_at_max=_decode_s(top_pt, output_tokens),
        limited_by=_limited_by(top_pt, output_tokens),
        tdp_w=int(getattr(cpu, "tdp_w", 0) or 0),
        price_usd=_price(cpu),
        notes=list(notes),
    )


def _best_single(rows: list[SweepRow], target_s: float) -> float | None:
    """The closest any part got, when none of them got there.

    None when something met the target -- the table answers the question then
    and this would only be noise.
    """
    if any(r.max_users >= 1 for r in rows):
        return None
    times = [r.response_s_single for r in rows if r.response_s_single > 0]
    return min(times) if times else None


def _decode_s(point: Any, output_tokens: int) -> float:
    """Seconds spent generating, as opposed to waiting for the first token."""
    if point is None:
        return 0.0
    each = _finite(getattr(point, "decode_tps_each", None))
    if not each or each <= 0:
        return 0.0
    return _f(output_tokens / each)


def _limited_by(point: Any, output_tokens: int) -> str:
    """Which half of the response time the ceiling is actually made of.

    Deliberately a plain "prefill"/"decode" rather than a percentage: the point
    is to tell a reader which way to spend money -- more cores or more memory
    bandwidth -- and a 55/45 split does not change that answer while pretending
    to more precision than the coefficients behind it support.
    """
    if point is None:
        return ""
    ttft = _finite(getattr(point, "ttft_s", None)) or 0.0
    decode = _decode_s(point, output_tokens)
    if ttft <= 0 and decode <= 0:
        return ""
    return "prefill" if ttft >= decode else "decode"


def _notes(dimm_count: int, sockets: int) -> list[str]:
    out = []
    if not dimm_count:
        out.append(
            "DIMM 수를 지정하지 않아 CPU 마다 채널당 1개로 채웠다 — "
            "채널 수가 6·8·12로 다르므로, 같은 개수를 강제하면 채널이 많은 부품만 "
            "절반만 채운 채 비교하게 된다."
        )
    if sockets > 1:
        out.append(
            f"{sockets}소켓으로 요청했다. 그보다 적게 지원하는 부품은 지원 한도로 "
            "낮춰 계산했고, 각 행의 소켓 수에 실제 값이 적혀 있다."
        )
    return out


def _price(cpu: Any) -> int | None:
    raw = getattr(cpu, "price_usd", None)
    try:
        return int(raw) if raw else None
    except (TypeError, ValueError):
        return None


def _finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out and out not in (float("inf"), float("-inf")) else None


def _f(value: Any) -> float:
    got = _finite(value)
    return round(got, 3) if got is not None else 0.0
