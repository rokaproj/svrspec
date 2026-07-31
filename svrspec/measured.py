"""Import real llama.cpp benchmark logs and turn them into coefficients.

Why a log importer and not a benchmark runner
---------------------------------------------
The servers this tool sizes are not on this machine. They are air-gapped boxes
at a customer site, frequently ones nobody has bought yet, which is the whole
reason `perf.py` is purely analytical. So "measure it" cannot mean "run
llama.cpp here": whatever this machine scores answers a question nobody asked.

That leaves exactly one honest path to a measurement. Somebody who *does* have
the hardware -- the customer's own engineer -- runs `llama-bench` or reads a
`llama-server` startup log once, and sends the text. This module turns that text
into a `CoefficientSpec` with `confidence="measured"`, so `eta_compute[amx-bf16]`
stops being an estimate the moment a single AMX Xeon log arrives. Nothing here
executes anything: it reads text. A test enforces that -- it asserts this file
contains no way to spawn a process and no way to reach the network -- because a
tool that quietly benchmarks the machine it runs on would be answering a
question nobody asked.

What it will not do
-------------------
Every derivation refuses rather than assumes. If the log does not say what
context length the run used, `derive_eta_bw` raises instead of guessing one --
a coefficient built on an invented context is not a measurement, and labelling
it `measured` would poison the error bars for every prediction that later uses
it. Likewise a derived value above 1.0 is rejected: exceeding the hardware
ceiling means the log and the declared hardware disagree, which is a data-entry
bug, not a discovery.

Parsers are the opposite: deliberately forgiving. llama.cpp's output changes
shape between releases, so a field that cannot be found becomes `None` and a
line that cannot be understood is skipped. Broken input yields an empty result,
never an exception -- the caller of `parse_*` is a person pasting a log, and the
derivation step downstream is where the strictness belongs.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .memory import decode_weight_bytes, kv_bytes_per_token
from .perf import peak_flops, widest_isa
from .types import (
    SOURCE_MEASUREMENT,
    VALID_CONFIDENCE,
    CoefficientSpec,
    CpuSpec,
    MemoryOption,
    ModelSpec,
    QuantSpec,
    ThroughputPrediction,
)

# --------------------------------------------------------------------------
# What a parsed log looks like
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MeasuredPoint:
    """One throughput figure read out of a log, with its run conditions.

    `raw` keeps the row exactly as parsed. A coefficient that ships in a
    delivery document has to be re-checkable against the text it came from, and
    the normalised fields above are lossy by design.
    """

    #: Identity of the log: file path, ticket number, or run label. Becomes the
    #: coefficient's `source_url`, so a derivation without one is refused.
    source: str
    #: "pp" (prompt processing / prefill) or "tg" (token generation / decode).
    kind: str
    tokens_per_s: float
    stddev: float | None = None
    #: The model name exactly as the log wrote it, not normalised to a catalog id.
    model_label: str = ""
    params_b: float | None = None
    quant: str | None = None
    n_threads: int | None = None
    n_batch: int | None = None
    #: Tokens resident in the KV cache for this run. See `_run_ctx`.
    n_ctx: int | None = None
    backend: str | None = None
    raw: dict = field(default_factory=dict)


@dataclass(frozen=True)
class MeasuredMemory:
    """The allocation lines llama.cpp prints at startup.

    These are the ground truth `svrspec verify` checks `memory.py` against --
    the KV cache and compute buffer are the two terms the RAM model computes
    rather than looks up, so they are the two worth confirming.
    """

    source: str
    model_size_mib: float | None = None
    kv_self_mib: float | None = None
    compute_buffer_mib: float | None = None
    n_ctx: int | None = None
    n_slots: int | None = None


@dataclass(frozen=True)
class Calibration:
    """A coefficient derived from a measurement, plus the arithmetic behind it.

    The coefficient alone is not reviewable. `basis`, `measured_tps` and
    `implied_ceiling` are what let somebody check the derivation without
    re-reading this module, and `change_pct` is what tells them whether
    accepting it moves any prediction.
    """

    coefficient: CoefficientSpec
    basis: str
    measured_tps: float
    #: Back-computed achieved figure: bytes/s for eta_bw, FLOP/s for eta_compute.
    implied_ceiling: float
    previous_value: float | None = None
    change_pct: float | None = None


# --------------------------------------------------------------------------
# Small, forgiving field readers
# --------------------------------------------------------------------------

#: "46.31 ± 0.42", "46.31 +/- 0.42", or a bare "46.31".
_TPS_RE = re.compile(r"^([0-9]+(?:\.[0-9]+)?)(?:\s*(?:±|\+/-|\+-)\s*([0-9]+(?:\.[0-9]+)?))?$")

#: llama-bench's `test` column: pp512, tg128, and the depth form tg128@d4096.
_TEST_RE = re.compile(r"^(pp|tg)(\d+)(?:@d(\d+))?$", re.IGNORECASE)

#: "8.03 B", "8030 M".
_PARAMS_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*([BMK])\b", re.IGNORECASE)

#: llama-bench spells quantisation out: "llama 8B Q4_K - Medium".
_QUANT_LONG_RE = re.compile(
    r"\b(I?Q\d+_[K01])\s*-\s*(extra small|small|medium|large)\b", re.IGNORECASE
)
#: ...while filenames use the compact form: "...-Q4_K_M.gguf".
_QUANT_TOKEN_RE = re.compile(
    r"\b(IQ\d+_[A-Z]{1,3}|Q\d+_K_[SML]|Q\d+_K|Q\d+_[01]|BF16|F16|F32)\b", re.IGNORECASE
)
_QUANT_SUFFIX = {"extra small": "XS", "small": "S", "medium": "M", "large": "L"}

_MIB_PER = {"MIB": 1.0, "GIB": 1024.0, "MB": 1e6 / 1024**2, "GB": 1e9 / 1024**2}


def _float(text: str) -> float | None:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int | None:
    try:
        out = int(value)
    except (TypeError, ValueError):
        return None
    return out if out > 0 else None


def _parse_tps(cell: str) -> tuple[float, float | None] | None:
    m = _TPS_RE.match(cell.strip())
    if not m:
        return None
    value = _float(m.group(1))
    if value is None:
        return None
    return value, _float(m.group(2)) if m.group(2) else None


def _parse_params_b(cell: str) -> float | None:
    m = _PARAMS_RE.search(cell or "")
    if not m:
        return None
    value = _float(m.group(1))
    if value is None:
        return None
    scale = {"B": 1.0, "M": 1e-3, "K": 1e-6}[m.group(2).upper()]
    return value * scale


def quant_from_label(label: str) -> str | None:
    """Best-effort quantisation id from a model label or filename.

    Returns None rather than guessing. A wrong quant silently changes
    bytes-per-weight by 30% and therefore every derived bandwidth figure, so
    "unknown" has to stay distinguishable from "assumed Q4_K_M".
    """
    if not label:
        return None
    m = _QUANT_LONG_RE.search(label)
    if m:
        return f"{m.group(1).upper()}_{_QUANT_SUFFIX[m.group(2).lower()]}"
    m = _QUANT_TOKEN_RE.search(label)
    if m:
        return m.group(1).upper()
    return None


def _run_ctx(kind: str, count: int, depth: int = 0, prompt: int = 0) -> int | None:
    """Tokens in the KV cache by the end of the measured run.

    llama-bench's `tg128` generates 128 tokens from an empty cache, and
    `tg128@d4096` does it after a 4096-token prefill; a server log states its
    prompt and generated counts outright. Taking the end-of-run length (rather
    than the average over the span, or the *allocated* `n_ctx`) is the reading
    that stays closest to the log. For a dense 8B the KV term is under 2% of
    the weight bytes at these lengths, so the choice barely moves eta_bw -- but
    it does have to be the log's number and not one this module picked.

    `kind` is unused in the arithmetic -- prefill and decode both end with
    depth + prompt + count tokens resident -- but is kept in the signature so a
    future format whose phases differ has somewhere to say so.
    """
    total = depth + prompt + count
    return total or None


# --------------------------------------------------------------------------
# llama-bench: markdown table and -o json
# --------------------------------------------------------------------------

#: Header cell -> canonical name. Matching by name is what makes a reordered
#: table parse identically; llama-bench only prints the columns that vary
#: between runs, so column position is not stable across two invocations.
_HEADERS = {
    "model": "model",
    "size": "size",
    "params": "params",
    "backend": "backend",
    "backends": "backend",
    "threads": "n_threads",
    "n_threads": "n_threads",
    "test": "test",
    "t/s": "tps",
    "tokens/s": "tps",
    "n_batch": "n_batch",
    "n_ubatch": "n_ubatch",
    "n_ctx": "n_ctx",
    "n_depth": "n_depth",
    "type_k": "type_k",
    "type_v": "type_v",
    "fa": "fa",
}


def _split_row(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def _is_separator(cells: list[str]) -> bool:
    return bool(cells) and all(c and set(c) <= set("-: ") for c in cells)


def _table_blocks(text: str) -> list[list[list[str]]]:
    """Contiguous runs of `|`-delimited lines, each its own table."""
    blocks: list[list[list[str]]] = []
    current: list[list[str]] = []
    for line in text.splitlines():
        if line.strip().startswith("|"):
            current.append(_split_row(line))
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    return blocks


def _parse_markdown(text: str, source: str) -> list[MeasuredPoint]:
    points: list[MeasuredPoint] = []
    for block in _table_blocks(text):
        rows = [r for r in block if not _is_separator(r)]
        if len(rows) < 2:
            continue
        header = [_HEADERS.get(c.strip().lower(), c.strip().lower()) for c in rows[0]]
        if "tps" not in header or "test" not in header:
            continue  # not a llama-bench result table
        for cells in rows[1:]:
            if len(cells) != len(header):
                continue
            row = dict(zip(header, cells))
            point = _point_from_md_row(row, source)
            if point:
                points.append(point)
    return points


def _point_from_md_row(row: dict[str, str], source: str) -> MeasuredPoint | None:
    test = _TEST_RE.match(row.get("test", "").strip())
    tps = _parse_tps(row.get("tps", ""))
    # A combined "pp512+tg128" row cannot attribute its t/s to either phase,
    # so it is dropped rather than filed under a phase it does not describe.
    if not test or not tps:
        return None

    kind = test.group(1).lower()
    count = int(test.group(2))
    depth = int(test.group(3)) if test.group(3) else 0
    label = row.get("model", "")

    n_batch = _int(row.get("n_batch")) or (count if kind == "pp" else None)
    n_ctx = _int(row.get("n_ctx")) or _run_ctx(kind, count, depth)

    return MeasuredPoint(
        source=source,
        kind=kind,
        tokens_per_s=tps[0],
        stddev=tps[1],
        model_label=label,
        params_b=_parse_params_b(row.get("params", "")),
        quant=quant_from_label(label),
        n_threads=_int(row.get("n_threads")),
        n_batch=n_batch,
        n_ctx=n_ctx,
        backend=row.get("backend") or None,
        raw=dict(row),
    )


def _parse_bench_json(text: str, source: str) -> list[MeasuredPoint]:
    try:
        doc = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return []
    if isinstance(doc, dict):
        for key in ("results", "entries", "data"):
            if isinstance(doc.get(key), list):
                doc = doc[key]
                break
        else:
            doc = [doc]
    if not isinstance(doc, list):
        return []

    points: list[MeasuredPoint] = []
    for row in doc:
        if not isinstance(row, dict):
            continue
        point = _point_from_json_row(row, source)
        if point:
            points.append(point)
    return points


def _point_from_json_row(row: dict[str, Any], source: str) -> MeasuredPoint | None:
    n_prompt = _int(row.get("n_prompt")) or 0
    n_gen = _int(row.get("n_gen")) or 0
    if bool(n_prompt) == bool(n_gen):
        return None  # neither phase, or both at once -- not attributable
    kind = "pp" if n_prompt else "tg"
    count = n_prompt or n_gen

    tps = _float(str(row.get("avg_ts")))
    if tps is None:
        return None

    filename = str(row.get("model_filename") or "")
    model_type = str(row.get("model_type") or "")
    label = model_type or filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]

    backends = row.get("backends", row.get("backend"))
    if isinstance(backends, list):
        backends = ",".join(str(b) for b in backends)

    params = row.get("model_n_params")
    params_b = float(params) / 1e9 if isinstance(params, (int, float)) and params > 0 else None

    depth = _int(row.get("n_depth")) or 0
    return MeasuredPoint(
        source=source,
        kind=kind,
        tokens_per_s=tps,
        stddev=_float(str(row.get("stddev_ts"))) if row.get("stddev_ts") is not None else None,
        model_label=label,
        params_b=params_b,
        # The filename carries the quant more reliably than the type string.
        quant=quant_from_label(filename) or quant_from_label(model_type),
        n_threads=_int(row.get("n_threads")),
        n_batch=_int(row.get("n_batch")) or (count if kind == "pp" else None),
        n_ctx=_int(row.get("n_ctx")) or _run_ctx(kind, count, depth),
        backend=str(backends) if backends else None,
        raw=dict(row),
    )


def parse_llama_bench(text: str, source: str = "") -> list[MeasuredPoint]:
    """Parse llama-bench output, either the markdown table or `-o json`.

    Returns an empty list for anything it cannot understand. Column order in
    the markdown form is irrelevant: cells are matched by header name.
    """
    if not text or not text.strip():
        return []
    stripped = text.lstrip()
    if stripped[:1] in "[{":
        points = _parse_bench_json(stripped, source)
        if points:
            return points
        # A JSON-looking file that yielded nothing may still be a log with a
        # table in it, so fall through rather than declaring failure.
    return _parse_markdown(text, source)


# --------------------------------------------------------------------------
# llama-server / llama-cli logs
# --------------------------------------------------------------------------

#: "prompt eval time = 12345.67 ms / 512 tokens ( 24.11 ms per token, 41.47 tokens per second)"
_PERF_RE = re.compile(
    r"(?P<label>prompt eval time|eval time)\s*=\s*[0-9.]+\s*ms\s*/\s*(?P<count>\d+)\s*"
    r"(?:tokens|runs)\b.*?(?P<tps>[0-9]+(?:\.[0-9]+)?)\s*tokens per second",
    re.IGNORECASE,
)
_NCTX_RE = re.compile(r"\bn_ctx\s*[:=]\s*(\d+)")
_NSLOTS_RE = re.compile(r"\b(?:n_slots|n_parallel)\s*[:=]\s*(\d+)")
_MODEL_SIZE_RE = re.compile(r"\bmodel size\s*[:=]\s*([0-9.]+)\s*(MiB|GiB|MB|GB)\b", re.IGNORECASE)
_KV_RE = re.compile(
    r"\bKV\s+(?:self\s+|cache\s+)?(?:buffer\s+)?size\s*[:=]\s*([0-9.]+)\s*(MiB|GiB|MB|GB)\b",
    re.IGNORECASE,
)
_COMPUTE_RE = re.compile(
    r"\bcompute buffer size\s*[:=]\s*([0-9.]+)\s*(MiB|GiB|MB|GB)\b", re.IGNORECASE
)
_MODEL_FILE_RE = re.compile(r"([^\s/\\]+\.gguf)", re.IGNORECASE)


def parse_server_log(text: str, source: str = "") -> list[MeasuredPoint]:
    """Pull throughput out of a llama-server / llama-cli timing block.

    Both `llama_perf_context_print:` (current) and `llama_print_timings:`
    (older) prefixes work, because the numbers are matched on the phase label
    rather than the emitting function.
    """
    if not text or not text.strip():
        return []

    declared_ctx = _last_int(_NCTX_RE, text)
    n_threads = _last_int(re.compile(r"\bn_threads\s*[:=]\s*(\d+)"), text)
    label_match = _MODEL_FILE_RE.search(text)
    label = label_match.group(1) if label_match else ""
    quant = quant_from_label(label)

    prompt_tokens = 0
    points: list[MeasuredPoint] = []
    for m in _PERF_RE.finditer(text):
        kind = "pp" if m.group("label").lower().startswith("prompt") else "tg"
        tps = _float(m.group("tps"))
        count = _int(m.group("count"))
        if tps is None or tps <= 0:
            continue
        if kind == "pp":
            prompt_tokens = count or 0
            ctx = _run_ctx("pp", count or 0) or declared_ctx
        else:
            # Generation ran on top of whatever the prompt left in the cache.
            ctx = _run_ctx("tg", count or 0, prompt=prompt_tokens) or declared_ctx
        points.append(
            MeasuredPoint(
                source=source,
                kind=kind,
                tokens_per_s=tps,
                stddev=None,  # a single run reports no spread
                model_label=label,
                params_b=None,  # the timing block does not state it
                quant=quant,
                n_threads=n_threads,
                n_batch=count if kind == "pp" else None,
                n_ctx=ctx,
                backend="CPU" if "CPU" in text else None,
                raw={"line": m.group(0)},
            )
        )
    return points


def parse_memory(text: str, source: str = "") -> MeasuredMemory:
    """Read the allocation lines a llama.cpp startup prints.

    Every field is optional: these lines have been renamed several times
    (`llama_kv_cache_init` -> `llama_kv_cache_unified`, `llm_load_print_meta`
    -> `print_info`), so a miss means "this build did not print it", not "the
    log is broken".
    """
    return MeasuredMemory(
        source=source,
        model_size_mib=_first_size(_MODEL_SIZE_RE, text),
        kv_self_mib=_first_size(_KV_RE, text),
        compute_buffer_mib=_first_size(_COMPUTE_RE, text),
        n_ctx=_last_int(_NCTX_RE, text),
        n_slots=_last_int(_NSLOTS_RE, text),
    )


def _first_size(pattern: re.Pattern[str], text: str) -> float | None:
    if not text:
        return None
    m = pattern.search(text)
    if not m:
        return None
    value = _float(m.group(1))
    if value is None:
        return None
    return value * _MIB_PER[m.group(2).upper()]


def _last_int(pattern: re.Pattern[str], text: str) -> int | None:
    if not text:
        return None
    found = pattern.findall(text)
    return _int(found[-1]) if found else None


# --------------------------------------------------------------------------
# Derivation: measurement -> coefficient
# --------------------------------------------------------------------------


#: Which phase each coefficient may be derived from, and what to call it when
#: refusing. Deriving a bandwidth coefficient from a prefill number would be
#: measuring the wrong roofline entirely.
_DERIVABLE_FROM = {"tg": ("eta_bw", "토큰 생성(tg)"), "pp": ("eta_compute", "프롬프트 처리(pp)")}


def _check_derivable(point: MeasuredPoint, kind: str, confidence: str) -> None:
    if point.kind != kind:
        name, phase = _DERIVABLE_FROM[kind]
        raise ValueError(
            f"{name} 계수는 {phase} 측정에서만 유도할 수 있다 — 이 측정은 {point.kind!r}이다"
        )
    if point.tokens_per_s <= 0:
        raise ValueError(f"측정 처리량이 {point.tokens_per_s}이다 — 유도할 수 없다")
    if not point.source.strip():
        raise ValueError(
            "로그의 신원(source)이 비어 있다 — 출처 없는 값은 실측 계수로 승격할 수 없다"
        )
    if confidence not in VALID_CONFIDENCE:
        raise ValueError(f"confidence는 {VALID_CONFIDENCE} 중 하나여야 한다: {confidence!r}")


#: A log whose thread count differs from the target part's core count by more
#: than this is treated as evidence that the log is not from that part. Chosen
#: loose on purpose: operators legitimately pin below core count (leaving a
#: core for the OS, avoiding SMT siblings), and that is not a mismatch. What it
#: catches is the real accident -- a 10-thread desktop log pointed at a
#: 32-core EPYC, which derives an eta_bw of 0.07 and reports it as measured.
THREAD_MISMATCH_RATIO = 0.5


def _grade(
    confidence: str,
    sockets: int,
    point: MeasuredPoint | None = None,
    cpu: CpuSpec | None = None,
) -> tuple[str, str]:
    """Final confidence for a derivation, and a note when it was lowered.

    Two downgrades, both automatic, both because the caller has no more
    information than we do here and the mistake is silent otherwise.

    **Multi-socket.** eta_bw and eta_compute describe what one socket reaches
    against its own ceiling; cross-socket scaling loss is carried separately by
    `dual_socket_efficiency`. A 2P measurement divided by two sockets' worth of
    ceiling absorbs that loss, and every 2P prediction then subtracts it again.

    **Thread count far from core count.** The one piece of evidence a log
    carries about which machine produced it. A log run with a third of the
    target part's cores either came from a different machine or exercised a
    fraction of this one; either way the ceiling it is being divided by is not
    the ceiling it achieved against. This does not refuse the derivation --
    pinning below core count is legitimate and we cannot tell the two apart --
    it refuses to call the result measured.
    """
    reasons: list[str] = []
    if sockets > 1:
        reasons.append(
            f"{sockets}소켓 실측이라 소켓 간 확장 손실이 값에 섞인다"
            f"(그 손실은 dual_socket_efficiency가 따로 빼야 한다)"
        )
    if point is not None and cpu is not None and point.n_threads:
        cores = cpu.cores * max(sockets, 1)
        if cores > 0 and point.n_threads < cores * THREAD_MISMATCH_RATIO:
            reasons.append(
                f"로그는 {point.n_threads}스레드인데 {cpu.model}은 {cores}코어다 — "
                f"이 로그가 이 부품의 것인지 확인해 주세요"
            )
    if reasons and confidence == "measured":
        return "derived", " (실측유도로 낮췄다: " + "; ".join(reasons) + ")"
    return confidence, ""


def _check_fraction(value: float, what: str, detail: str) -> None:
    """A coefficient outside (0, 1] means the inputs disagree, not a discovery."""
    if not 0 < value <= 1.0:
        raise ValueError(
            f"{what} 유도값이 {value:.3f}로 (0, 1] 범위를 벗어났다 — "
            f"로그와 하드웨어 스펙이 서로 맞지 않는다는 뜻이다. {detail}"
        )


def _change(previous: CoefficientSpec | None, value: float) -> tuple[float | None, float | None]:
    if previous is None:
        return None, None
    if not previous.value:
        return previous.value, None
    return previous.value, (value - previous.value) / previous.value * 100.0


def derive_eta_bw(
    point: MeasuredPoint,
    model: ModelSpec,
    quant: QuantSpec,
    cpu: CpuSpec,
    memory: MemoryOption,
    *,
    sockets: int = 1,
    previous: CoefficientSpec | None = None,
    confidence: str = "measured",
) -> Calibration:
    """Invert a token-generation measurement into a bandwidth efficiency.

    Decode reads the whole active weight set plus the KV cache once per token,
    so tok/s converts directly into achieved DRAM bandwidth, and dividing by
    the platform's channel ceiling gives eta_bw.

    Raises `ValueError` when the run's context length is unknown: the KV term
    needs it, and a coefficient built on an assumed context is not measured.

    Lower `confidence` to "derived" when the log is not from this exact SKU
    (same generation, different part), or when the model or quant passed here
    had to be matched to the log by hand. `sockets > 1` is lowered
    automatically -- see `_grade`.
    """
    _check_derivable(point, "tg", confidence)
    if sockets < 1:
        raise ValueError(f"sockets는 1 이상이어야 한다: {sockets}")
    confidence, downgrade_note = _grade(confidence, sockets, point, cpu)

    ctx = point.n_ctx
    if not ctx or ctx <= 0:
        raise ValueError(
            "로그에 컨텍스트 길이가 없다 — KV 캐시 바이트를 계산할 수 없으므로 "
            "대역폭 계수를 유도하지 않는다 (가정한 ctx로 만든 계수는 실측이 아니다)"
        )

    weight_bytes = decode_weight_bytes(model, quant)
    kv_bytes = kv_bytes_per_token(model) * ctx
    bytes_per_token = weight_bytes + kv_bytes
    achieved = point.tokens_per_s * bytes_per_token

    per_socket_ceiling = cpu.mem_channels * memory.effective_mts * 8 / 1000.0 * 1e9
    ceiling = per_socket_ceiling * sockets
    if ceiling <= 0:
        raise ValueError(
            f"채널 천장이 0이다 — {cpu.id}의 채널 수({cpu.mem_channels})나 "
            f"{memory.id}의 실효 속도({memory.effective_mts})를 확인해 주세요"
        )

    value = achieved / ceiling
    _check_fraction(
        value,
        "eta_bw",
        f"{point.tokens_per_s:.2f} tok/s × {bytes_per_token / 1e9:.2f} GB/tok = "
        f"{achieved / 1e9:.1f} GB/s인데 {cpu.mem_channels}채널 × "
        f"{memory.effective_mts} MT/s × {sockets}소켓의 천장은 {ceiling / 1e9:.1f} GB/s다",
    )

    basis = (
        f"{model.name} {quant.id}, ctx {ctx}에서 {point.tokens_per_s:.2f} tok/s → "
        f"토큰당 {bytes_per_token / 1e9:.3f} GB × {point.tokens_per_s:.2f} = "
        f"{achieved / 1e9:.1f} GB/s 달성, {cpu.mem_channels}채널 "
        f"{memory.ddr_gen}-{memory.effective_mts} 천장 {ceiling / 1e9:.1f} GB/s 대비 {value:.3f}"
        + downgrade_note
    )
    previous_value, change_pct = _change(previous, value)

    return Calibration(
        coefficient=CoefficientSpec(
            id=f"eta-bw-{memory.ddr_gen.lower()}",
            kind="eta_bw",
            key=memory.ddr_gen,
            value=value,
            confidence=confidence,
            source=SOURCE_MEASUREMENT,
            source_url=point.source,
            notes=basis,
        ),
        basis=basis,
        measured_tps=point.tokens_per_s,
        implied_ceiling=achieved,
        previous_value=previous_value,
        change_pct=change_pct,
    )


def derive_eta_compute(
    point: MeasuredPoint,
    model: ModelSpec,
    cpu: CpuSpec,
    *,
    sockets: int = 1,
    previous: CoefficientSpec | None = None,
    confidence: str = "measured",
) -> Calibration:
    """Invert a prompt-processing measurement into a compute efficiency.

    Prefill is a GEMM at 2 FLOP per parameter per token, so tok/s converts into
    achieved FLOP/s, and dividing by the part's vector peak gives eta_compute
    for whichever ISA `widest_isa` says llama.cpp would have used.

    The result is only as good as `all_core_turbo_ghz`: the peak is
    cores x clock x width, so a guessed sustained clock propagates straight
    into the coefficient. Lower `confidence` to "derived" when that clock is an
    estimate, when the log came from a different SKU of the same generation, or
    when the thread count in the log does not correspond to the core count used
    here (llama.cpp pinned to 10 threads on a 12-core hybrid part is not
    running 12 equal vector units). `sockets > 1` is lowered automatically --
    see `_grade`.
    """
    _check_derivable(point, "pp", confidence)
    if sockets < 1:
        raise ValueError(f"sockets는 1 이상이어야 한다: {sockets}")
    confidence, downgrade_note = _grade(confidence, sockets, point, cpu)

    achieved = point.tokens_per_s * 2.0 * model.decode_params_b * 1e9
    peak, isa = peak_flops(cpu, sockets)
    if peak <= 0:
        raise ValueError(
            f"{cpu.id}의 이론 연산 천장이 0이다 — 코어 수나 전코어 터보 클럭을 확인해 주세요"
        )

    value = achieved / peak
    _check_fraction(
        value,
        "eta_compute",
        f"{point.tokens_per_s:.2f} tok/s × 2 × {model.decode_params_b:.2f}B = "
        f"{achieved / 1e12:.2f} TFLOP/s인데 {cpu.cores}코어 × {sockets}소켓 × "
        f"{cpu.all_core_turbo_ghz} GHz × {isa}의 천장은 {peak / 1e12:.2f} TFLOP/s다",
    )

    basis = (
        f"{model.name} 프롬프트 처리 {point.tokens_per_s:.2f} tok/s → "
        f"2 × {model.decode_params_b:.2f}B × {point.tokens_per_s:.2f} = "
        f"{achieved / 1e12:.2f} TFLOP/s 달성, {cpu.model} {cpu.cores}코어 × {sockets}소켓 "
        f"{cpu.all_core_turbo_ghz} GHz {isa} 천장 {peak / 1e12:.2f} TFLOP/s 대비 {value:.3f}"
        + downgrade_note
    )
    previous_value, change_pct = _change(previous, value)

    return Calibration(
        coefficient=CoefficientSpec(
            id=f"eta-compute-{isa}",
            kind="eta_compute",
            key=isa,
            value=value,
            confidence=confidence,
            source=SOURCE_MEASUREMENT,
            source_url=point.source,
            notes=basis,
        ),
        basis=basis,
        measured_tps=point.tokens_per_s,
        implied_ceiling=achieved,
        previous_value=previous_value,
        change_pct=change_pct,
    )


# --------------------------------------------------------------------------
# Checking a prediction against what actually happened
# --------------------------------------------------------------------------

#: Phase -> the prediction field it should be compared against. Both llama-bench
#: and a single llama-cli run generate one sequence at a time, so decode is
#: judged against the single-stream figure, never the batched aggregate.
_PREDICTED_FIELD = {"pp": "prefill_tps", "tg": "decode_tps_single"}


def compare_to_prediction(point: MeasuredPoint, prediction: ThroughputPrediction) -> dict:
    """Score a prediction against a measurement of the same configuration.

    `error_pct` is signed from the prediction's point of view: positive means
    the tool promised more than the hardware delivered. The verdict is only
    called wrong when the error leaves the prediction's own stated uncertainty
    band -- inside it the tool told the truth about how well it knew.
    """
    field_name = _PREDICTED_FIELD.get(point.kind)
    if field_name is None:
        raise ValueError(f"알 수 없는 측정 종류 {point.kind!r} — 'pp' 또는 'tg'여야 한다")
    if point.tokens_per_s <= 0:
        raise ValueError(f"측정 처리량이 {point.tokens_per_s}이다 — 비교할 수 없다")

    predicted = float(getattr(prediction, field_name))
    error_pct = (predicted - point.tokens_per_s) / point.tokens_per_s * 100.0
    within = abs(error_pct) <= prediction.uncertainty * 100.0

    if within:
        verdict = "확인"
    else:
        verdict = "과대예측" if error_pct > 0 else "과소예측"

    return {
        "measured_tps": point.tokens_per_s,
        "predicted_tps": predicted,
        "error_pct": error_pct,
        "within_uncertainty": within,
        "kind": point.kind,
        "verdict": verdict,
    }
