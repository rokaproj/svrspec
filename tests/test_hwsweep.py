"""Tests for `svrspec.hwsweep`: does the catalogue sweep rank hardware honestly?

Run against the **shipped** catalogue, like `test_modelbench` and `test_lab`,
because every row here is a claim about a part somebody can buy. Where a test
leans on a catalogue fact (the Gold 6234 is the six-channel part), it asserts
that fact first, so a catalogue edit is reported as a catalogue edit rather than
as a mystery.

The arithmetic belongs to `perf` and `modelbench` and is tested there. What is
tested here is the shape of the *comparison*: the ceiling search is a search and
not a ladder, every CPU is populated fairly rather than identically, a part that
cannot be built is excluded instead of ranked last, and the screen reports which
half of the response time actually sets the ceiling. Those are the claims the
buy-this-box screen rests on.
"""

from __future__ import annotations

import pytest

from svrspec.catalog import Catalog
from svrspec.hwsweep import MAX_USERS, max_users_within, sweep

MODEL = "exaone-3.5-2.4b-instruct"
QUANT = "Q4_K_M"


@pytest.fixture(scope="module")
def cat() -> Catalog:
    return Catalog()


# -- the ceiling search ----------------------------------------------------


def test_the_ceiling_is_the_largest_count_that_actually_held():
    """Bisection, not a ladder: the answer must be exact, not a power of two.

    A ladder of 1, 2, 4 ... 64 can only ever say "fine at 32, broken at 64",
    and somebody buys hardware against this number. So the search has to land
    on the true boundary.
    """
    # A machine whose response time is linear in load, breaking above 37.
    got, capped = max_users_within(lambda n: n * 1.0, target_s=37.0)
    assert got == 37
    assert not capped


def test_a_box_too_slow_for_one_user_gets_no_ceiling_rather_than_one():
    """0 and 1 must not be the same answer.

    "1" reads as "usable by one person". A build that misses the target with
    nobody else on it is not that, and rounding it up to 1 would put unusable
    hardware on the buy list.
    """
    got, capped = max_users_within(lambda n: 99.0, target_s=30.0)
    assert got == 0
    assert not capped


def test_holding_the_target_everywhere_is_reported_as_a_floor():
    """A search that runs into the bound must say so.

    Otherwise MAX_USERS reads as a measured limit, and "this box tops out at
    4096 users" is a very different claim from "we stopped looking at 4096".
    """
    got, capped = max_users_within(lambda n: 0.001, target_s=30.0)
    assert got == MAX_USERS
    assert capped


def test_the_search_never_reports_a_count_that_missed_the_target():
    """The invariant the whole screen rests on, checked across many shapes."""
    for breaking_point in (1, 2, 3, 17, 100, 1000):
        def probe(n: int, k: int = breaking_point) -> float:
            return 1.0 if n <= k else 999.0

        got, _ = max_users_within(probe, target_s=30.0)
        assert got == breaking_point
        assert probe(got) <= 30.0
        if got < MAX_USERS:
            assert probe(got + 1) > 30.0


# -- the sweep ------------------------------------------------------------


def test_every_catalogue_cpu_is_either_ranked_or_explained(cat: Catalog):
    """No part may silently vanish from a hardware comparison.

    A CPU missing from the table looks like a CPU that was never considered.
    Each one has to come back either as a row or as a blocked entry carrying
    the reason.
    """
    s = sweep(cat, model_id=MODEL, quant_id=QUANT)
    seen = {r.cpu_id for r in s.rows} | {c for c, _ in s.blocked}
    assert seen == {c.id for c in cat.cpus}
    assert all(why.strip() for _, why in s.blocked)


def test_each_cpu_is_populated_to_its_own_channel_count(cat: Catalog):
    """Fair population, not identical population.

    The catalogue mixes 6-, 8- and 12-channel parts. Handing all of them the
    same DIMM count would leave the wide parts half-populated and call the
    result a hardware comparison -- the exact mistake the lab screen exists to
    catch. So the default fills one DIMM per channel, per CPU.
    """
    six = cat.cpu("xeon-gold-6234")
    assert six.mem_channels == 6, "catalogue changed: 6234 is the 6-channel row"
    wide = cat.cpu("xeon-gold-6542y")
    assert wide.mem_channels == 8

    s = sweep(cat, model_id=MODEL, quant_id=QUANT, sockets=1)
    rows = {r.cpu_id: r for r in s.rows}
    assert rows["xeon-gold-6234"].channels == 6
    assert rows["xeon-gold-6542y"].channels == 8


def test_a_one_socket_part_is_clamped_and_says_so(cat: Catalog):
    """Asking for two sockets must not drop the 1S parts from the comparison.

    Excluding them would answer "which of these should I buy" by quietly
    removing candidates. Clamping and reporting the real socket count is the
    honest answer about that part.
    """
    bronze = cat.cpu("xeon-bronze-3508u")
    assert bronze.sockets_max == 1, "catalogue changed: 3508U is the 1S row"

    s = sweep(cat, model_id=MODEL, quant_id=QUANT, sockets=2)
    rows = {r.cpu_id: r for r in s.rows}
    assert "xeon-bronze-3508u" in rows
    assert rows["xeon-bronze-3508u"].sockets == 1
    assert any("소켓" in n for n in s.notes)


def test_more_concurrency_never_reads_as_faster_per_user(cat: Catalog):
    """Monotonicity, the assumption the bisection is built on.

    If response time ever fell as load rose, the search would be bisecting a
    non-monotonic function and the ceiling would be meaningless.
    """
    from svrspec.hwsweep import sweep as _sweep  # noqa: F401  (documents origin)
    from svrspec.lab import VirtualMachine, assemble
    from svrspec.modelbench import bench_model

    cpu = cat.cpu("xeon-gold-6542y")
    vm = VirtualMachine(
        name="probe", cpu_id=cpu.id, sockets=1,
        dimm_gb=64, dimm_count=cpu.mem_channels,
        model_id=MODEL, quant_id=QUANT, slots=1,
    )
    asm = assemble(cat, vm)
    last = 0.0
    for users in (1, 2, 4, 8, 16, 32):
        mb = bench_model(asm, batches=(1,), contexts=(4096,), users=(users,))
        got = mb.concurrency[0].response_s
        assert got >= last, f"{users} users answered faster than {last}"
        last = got


def test_a_longer_context_lowers_every_ceiling(cat: Catalog):
    """The setup has to actually move the answer.

    A screen whose knobs do not change the ranking is a screen that is not
    reading them. Longer prompts cost prefill, which is shared across users, so
    every ceiling must come down.
    """
    short = sweep(cat, model_id=MODEL, quant_id=QUANT, ctx_tokens=512)
    long = sweep(cat, model_id=MODEL, quant_id=QUANT, ctx_tokens=8192)
    s_rows = {r.cpu_id: r for r in short.rows}
    l_rows = {r.cpu_id: r for r in long.rows}
    shared = set(s_rows) & set(l_rows)
    assert shared
    for cpu_id in shared:
        assert l_rows[cpu_id].max_users <= s_rows[cpu_id].max_users


def test_the_ceiling_names_the_half_of_the_response_it_came_from(cat: Catalog):
    """`bound_by` alone was misleading, and this is why the split exists.

    `bound_by` is the *decode* bound, and decode is bandwidth-bound on nearly
    every part here -- so that column reads "bandwidth" for essentially the
    whole table while the thing setting the ceiling is prefill. An operator
    reading it would buy memory bandwidth to fix a core-count problem.
    """
    s = sweep(cat, model_id=MODEL, quant_id=QUANT, ctx_tokens=8192)
    usable = [r for r in s.rows if r.max_users >= 1]
    assert usable

    for r in usable:
        assert r.limited_by in {"prefill", "decode"}
        # The split has to add up to the reported response time.
        assert r.ttft_s_at_max + r.decode_s_at_max == pytest.approx(
            r.response_s_at_max, rel=0.02
        )
        bigger = "prefill" if r.ttft_s_at_max >= r.decode_s_at_max else "decode"
        assert r.limited_by == bigger


def test_cores_not_bandwidth_decide_the_ceiling_at_a_long_context(cat: Catalog):
    """The single most decision-changing fact this screen has.

    At a long prompt the prefill is compute-bound and shared n ways, so a part
    with more cores carries more concurrent users even when a rival beats it on
    memory bandwidth and on tokens per second. If this inverts, the screen is
    telling operators to buy the wrong machine.
    """
    many_cores = cat.cpu("xeon-6788p")
    much_bandwidth = cat.cpu("epyc-9575f")
    assert many_cores.cores > much_bandwidth.cores
    assert much_bandwidth.mem_channels > many_cores.mem_channels

    s = sweep(cat, model_id=MODEL, quant_id=QUANT, ctx_tokens=8192, sockets=1)
    rows = {r.cpu_id: r for r in s.rows}
    wide, fast = rows["epyc-9575f"], rows["xeon-6788p"]

    assert wide.bandwidth_gbs > fast.bandwidth_gbs
    assert wide.gen_tps > fast.gen_tps          # wins the single-stream figure
    assert fast.max_users > wide.max_users      # and still loses the ceiling
    assert fast.limited_by == "prefill"


def test_rows_are_ordered_by_the_ceiling_they_reached(cat: Catalog):
    s = sweep(cat, model_id=MODEL, quant_id=QUANT)
    ceilings = [r.max_users for r in s.rows]
    assert ceilings == sorted(ceilings, reverse=True)


def test_the_sweep_answers_fast_enough_to_be_button_driven(cat: Catalog):
    """43 CPUs x a bisection each still has to feel like a click.

    The engine is arithmetic over the catalogue, so this is a regression guard
    on somebody later putting a simulation inside the loop.
    """
    import time

    start = time.monotonic()
    s = sweep(cat, model_id=MODEL, quant_id=QUANT)
    elapsed = time.monotonic() - start
    assert s.rows
    assert elapsed < 5.0, f"sweep took {elapsed:.1f}s"


def test_what_was_asked_comes_back_with_the_answer(cat: Catalog):
    """The rows are meaningless without the setup that produced them."""
    s = sweep(
        cat, model_id=MODEL, quant_id=QUANT,
        ctx_tokens=2048, output_tokens=128, target_s=10.0, sockets=1,
    )
    assert s.ctx_tokens == 2048
    assert s.output_tokens == 128
    assert s.target_s == 10.0
    assert s.model_name and s.quant_id == QUANT
