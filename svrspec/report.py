"""Renderers: console tables, a single-file HTML report, CSV and JSON.

The HTML is what gets attached to a delivery document, so it carries its own
caveats: which catalogue rows were unverified, what the error bars are, and
which numbers are extrapolations rather than measurements. A sizing report that
presents an estimate as a fact is the failure mode to avoid.
"""

from __future__ import annotations

import csv
import html
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from .perf import Efficiency
from .theme import stylesheet
from .types import Candidate, Workload
from .workload import describe as describe_workload

VERDICT_LABEL = {"pass": "통과", "marginal": "여유부족", "fail": "미달"}


# --------------------------------------------------------------------------
# Console
# --------------------------------------------------------------------------


def _table(headers: list[str], rows: list[list[str]], aligns: str = "") -> str:
    if not rows:
        return "(없음)"
    widths = [_width(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], _width(cell))

    aligns = (aligns + "l" * len(headers))[: len(headers)]

    def line(cells: list[str]) -> str:
        out = []
        for cell, w, a in zip(cells, widths, aligns):
            pad = w - _width(cell)
            out.append(" " * pad + cell if a == "r" else cell + " " * pad)
        return "  ".join(out).rstrip()

    sep = "  ".join("-" * w for w in widths)
    return "\n".join([line(headers), sep, *(line(r) for r in rows)])


def _width(text: str) -> int:
    """Display width, counting CJK glyphs as two columns."""
    import unicodedata

    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def clip(text: str, columns: int = 68) -> str:
    """Truncate to a display width, counting CJK glyphs as two columns."""
    if _width(text) <= columns:
        return text
    out = ""
    for ch in text:
        if _width(out + ch) > columns - 1:
            break
        out += ch
    return out.rstrip() + "…"


def candidate_rows(candidates: Iterable[Candidate]) -> list[list[str]]:
    rows = []
    for c in candidates:
        sim = c.sim_pessimistic or c.sim
        rows.append(
            [
                c.cpu.model,
                f"{c.cpu.cores * c.sockets}C",
                f"{c.throughput.effective_bandwidth_gbs:.0f}",
                f"{c.throughput.prefill_tps:.0f}",
                f"{c.throughput.decode_tps_single:.1f}",
                f"{c.latency.total_s:.1f}",
                f"{sim.p95_steady_s or sim.p95_s:.1f}",
                f"{sim.storm_drain_s / 60:.1f}",
                f"{c.memory_gb}",
                VERDICT_LABEL.get(c.verdict, c.verdict),
            ]
        )
    return rows


CANDIDATE_HEADERS = [
    "CPU",
    "코어",
    "대역폭GB/s",
    "prefill t/s",
    "decode t/s",
    "1건 지연s",
    "평상시 p95",
    "스톰 분",
    "RAM GB",
    "판정",
]


def render_console(
    candidates: list[Candidate],
    tier_map: dict[str, Candidate | None],
    workload: Workload,
    eff: Efficiency,
    unverified: list[tuple[str, str]],
) -> str:
    parts: list[str] = []
    head = candidates[0] if candidates else None
    if head:
        m = head.model
        parts.append(
            f"모델  {m.name}  ({m.params_b:.1f}B"
            + (f", 활성 {m.active_params_b:.1f}B MoE" if m.active_params_b else "")
            + f", {head.quant.id}, KV/토큰 {_kv_kib(head):.0f} KiB)"
        )
    parts.append(f"부하  {describe_workload(workload)}")
    parts.append(f"계수  {_coefficient_line(eff, candidates)}")
    parts.append("")
    parts.append("판정은 예측 불확실도를 반영한 불리한 추정 기준이다.")
    parts.append("")
    parts.append(_table(CANDIDATE_HEADERS, candidate_rows(candidates), aligns="lrrrrrrrrl"))
    parts.append("")
    parts.append(_render_tiers(tier_map))

    warnings = _collect_warnings(candidates)
    if warnings:
        parts.append("")
        parts.append("주의사항")
        parts.extend(f"  - {w}" for w in warnings)

    if unverified:
        parts.append("")
        parts.append(
            f"미확인 스펙 {len(unverified)}건 — 납품 전 벤더 데이터시트 확인 필요:"
        )
        shown = ", ".join(f"{k}:{i}" for k, i in unverified[:12])
        more = f" (외 {len(unverified) - 12}건)" if len(unverified) > 12 else ""
        parts.append(f"  {shown}{more}")

    return "\n".join(parts)


def _kv_kib(candidate: Candidate) -> float:
    from .memory import kv_bytes_per_token

    return kv_bytes_per_token(candidate.model) / 1024


def _render_tiers(tier_map: dict[str, Candidate | None]) -> str:
    labels = {
        "minimum": "최소 스펙",
        "recommended": "권장 스펙",
        "comfortable": "여유 스펙",
    }
    rows = []
    for key in ("minimum", "recommended", "comfortable"):
        c = tier_map.get(key)
        if not c:
            rows.append([labels[key], "해당 없음", "", "", "", ""])
            continue
        sim = c.sim_pessimistic or c.sim
        rows.append(
            [
                labels[key],
                f"{c.cpu.vendor} {c.cpu.model}",
                f"{c.cpu.cores * c.sockets}C / {c.sockets}소켓",
                f"{c.memory_gb} GB {c.memory.ddr_gen}-{c.memory.effective_mts}",
                f"p95 {sim.p95_steady_s or sim.p95_s:.1f}s (여유 {c.headroom:.1f}배)",
                f"스톰 {sim.storm_drain_s / 60:.1f}분",
            ]
        )
    return _table(["등급", "CPU", "구성", "메모리", "지연", "버스트"], rows)


def _collect_warnings(candidates: Iterable[Candidate]) -> list[str]:
    seen: dict[str, None] = {}
    for c in candidates:
        for w in c.throughput.warnings:
            seen.setdefault(w, None)
    return list(seen)


CONFIDENCE_LABEL = {"measured": "실측", "derived": "실측유도", "estimate": "추정"}


def _used_coefficients(eff: Efficiency, candidates: Iterable[Candidate]):
    """Only the coefficients these candidates actually depend on.

    Listing the whole table would bury the two or three rows that carry the
    result; a reader needs to know which numbers their answer rests on.
    """
    from .perf import widest_isa

    wanted: set[tuple[str, str]] = set()
    for c in candidates:
        wanted.add(("eta_bw", c.memory.ddr_gen))
        wanted.add(("eta_compute", widest_isa(c.cpu)))
        wanted.add(("per_core_bw_gbs", "*"))
        if c.sockets > 1:
            wanted.add(("dual_socket_efficiency", "*"))
    return [eff.get(kind, key) for kind, key in sorted(wanted)]


def _coefficient_line(eff: Efficiency, candidates: Iterable[Candidate]) -> str:
    used = _used_coefficients(eff, candidates)
    parts = [
        f"{c.kind}[{c.key}]={c.value:g} ({CONFIDENCE_LABEL.get(c.confidence, c.confidence)})"
        for c in used
        if c.kind in ("eta_bw", "eta_compute")
    ]
    return ", ".join(parts) or "카탈로그 기본값"


def _coefficient_html(eff: Efficiency, candidates: Iterable[Candidate]) -> str:
    e = html.escape
    used = _used_coefficients(eff, candidates)
    rows = "".join(
        "<tr>"
        f"<td>{e(c.kind)}</td><td>{e(c.key)}</td>"
        f'<td class="num">{c.value:g}</td>'
        f"<td>{e(CONFIDENCE_LABEL.get(c.confidence, c.confidence))}</td>"
        f'<td class="wrap">{e(c.notes)}</td>'
        "</tr>"
        for c in used
    )
    estimated = [c for c in used if c.confidence == "estimate"]
    caveat = ""
    if estimated:
        ids = ", ".join(e(c.id) for c in estimated)
        caveat = (
            f"<div class='caveat'><strong>추정 계수 {len(estimated)}건이 이 결과에 "
            f"들어갔다.</strong> 해당 하드웨어 등급에서 llama.cpp를 실측한 근거가 없다는 "
            f"뜻이고, 그만큼 오차 범위가 넓어져 있다: {ids}</div>"
        )
    return (
        "<p class='sub'>이 산정에 실제로 쓰인 계수만 나열한다. 예측은 순수 해석적이며 "
        "실측 벤치마크를 돌리지 않는다.</p>"
        "<div class='scroll'><table><thead><tr><th>종류</th><th>적용 대상</th>"
        "<th class='num'>값</th><th>근거 수준</th><th>비고</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></div>{caveat}"
    )


# --------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------

#: Report-specific layout on top of the shared design tokens.
_REPORT_CSS = """
.wrap{max-width:1180px; margin:0 auto; padding:var(--s7) var(--s5) 80px}
h1{font-size:var(--fs-xl); margin:0 0 var(--s1)}
h2{font-size:var(--fs-lg); margin:var(--s7) 0 var(--s3)}
h3{margin:var(--s5) 0 var(--s2)}
.sub{color:var(--text-secondary); margin:0 0 var(--s5); font-size:var(--fs-sm)}
.grid{display:grid; gap:var(--s3);
  grid-template-columns:repeat(auto-fit,minmax(250px,1fr))}
.card .big{font-size:var(--fs-lg); font-weight:600; margin:var(--s1) 0 2px}
.card .note{font-size:var(--fs-sm); color:var(--text-secondary);
  border:0; padding:0; background:none}
.caveat{background:var(--bg-secondary); border:1px solid var(--border);
  border-left:3px solid var(--warning); border-radius:var(--radius-sm);
  padding:var(--s3) var(--s4); margin:var(--s3) 0; font-size:var(--fs-sm)}
footer{margin-top:var(--s7); padding-top:var(--s4);
  border-top:1px solid var(--border);
  color:var(--text-tertiary); font-size:var(--fs-xs)}
"""

_CSS = stylesheet(_REPORT_CSS)


def render_html(
    candidates: list[Candidate],
    tier_map: dict[str, Candidate | None],
    workload: Workload,
    eff: Efficiency,
    unverified: list[tuple[str, str]],
    title: str = "LLM 서버 스펙 산정 리포트",
) -> str:
    e = html.escape
    head = candidates[0] if candidates else None
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")

    cards = []
    labels = {"minimum": "최소 스펙", "recommended": "권장 스펙", "comfortable": "여유 스펙"}
    for key in ("minimum", "recommended", "comfortable"):
        c = tier_map.get(key)
        if not c:
            cards.append(
                f'<div class="card"><div class="label">{e(labels[key])}</div>'
                f'<div class="big">해당 없음</div>'
                f'<div class="note">카탈로그의 어떤 CPU도 이 조건을 만족하지 못했다</div></div>'
            )
            continue
        sim = c.sim_pessimistic or c.sim
        cards.append(
            f'<div class="card"><div class="label">{e(labels[key])}</div>'
            f'<div class="big">{e(c.cpu.vendor)} {e(c.cpu.model)}</div>'
            f'<div class="note">{c.cpu.cores * c.sockets}코어 / {c.sockets}소켓 · '
            f"{c.memory_gb} GB {e(c.memory.ddr_gen)}-{c.memory.effective_mts}<br>"
            f"평상시 p95 {sim.p95_steady_s or sim.p95_s:.1f}초 (SLA 여유 {c.headroom:.1f}배) · "
            f"스톰 소진 {sim.storm_drain_s / 60:.1f}분</div></div>"
        )

    rows = []
    for c in candidates:
        sim = c.sim_pessimistic or c.sim
        rows.append(
            "<tr>"
            f"<td>{e(c.cpu.vendor)} {e(c.cpu.model)}</td>"
            f"<td>{e(c.cpu.family)}</td>"
            f'<td class="num">{c.cpu.cores * c.sockets}</td>'
            f'<td class="num">{c.cpu.mem_channels}ch {e(c.memory.ddr_gen)}-{c.memory.effective_mts}</td>'
            f'<td class="num">{c.throughput.effective_bandwidth_gbs:.0f}</td>'
            f'<td class="num">{c.throughput.prefill_tps:.0f}</td>'
            f'<td class="num">{c.throughput.decode_tps_single:.1f}</td>'
            f'<td class="num">{c.latency.total_s:.1f}</td>'
            f'<td class="num">{sim.p95_steady_s or sim.p95_s:.1f}</td>'
            f'<td class="num">{sim.storm_drain_s / 60:.1f}</td>'
            f'<td class="num">{c.memory_gb}</td>'
            f'<td class="v-{c.verdict}">{e(VERDICT_LABEL.get(c.verdict, c.verdict))}</td>'
            "</tr>"
        )

    model_line = ""
    if head:
        m = head.model
        moe = f", 활성 {m.active_params_b:.1f}B (MoE)" if m.active_params_b else ""
        model_line = (
            f"{e(m.name)} — {m.params_b:.1f}B{moe}, {e(head.quant.id)} "
            f"({head.quant.bits_per_weight:.2f} bpw), 가중치 {head.ram.weights_gb:.1f} GiB, "
            f"KV {_kv_kib(head):.0f} KiB/토큰"
        )

    warnings = _collect_warnings(candidates)
    warning_html = (
        "<ul class='notes'>" + "".join(f"<li>{e(w)}</li>" for w in warnings) + "</ul>"
        if warnings
        else "<p class='sub'>없음</p>"
    )

    unverified_html = ""
    if unverified:
        items = ", ".join(f"{e(k)}:{e(i)}" for k, i in unverified)
        unverified_html = (
            f"<div class='caveat'><strong>미확인 스펙 {len(unverified)}건.</strong> "
            f"아래 항목은 벤더 데이터시트로 확인되지 않은 값을 포함한다. 납품 문서로 "
            f"확정하기 전에 반드시 대조하셔야 합니다.<br><span class='note'>{items}</span></div>"
        )

    calib_html = _coefficient_html(eff, candidates)

    body = f"""<div class="wrap">
<h1>{e(title)}</h1>
<p class="sub">생성 {e(generated)} · svrspec · 관제 알람 → LLM 정제·분석 → Teams 전송 파이프라인</p>

<h2>권장 스펙</h2>
<div class="grid">{"".join(cards)}</div>

<h2>대상 모델과 부하</h2>
<div class="card"><div class="note">{model_line}<br>{e(describe_workload(workload))}</div></div>

<h2>CPU 후보별 예측</h2>
<p class="sub">판정은 예측 불확실도를 반영한 <strong>불리한 추정</strong> 기준이다.
p95와 스톰 소진 시간도 같은 기준의 값이다.</p>
<div class="scroll"><table>
<thead><tr>
<th>CPU</th><th>세대</th><th class="num">코어</th><th class="num">메모리</th>
<th class="num">대역폭 GB/s</th><th class="num">prefill t/s</th><th class="num">decode t/s</th>
<th class="num">1건 지연 s</th><th class="num">평상시 p95 s</th><th class="num">스톰 분</th>
<th class="num">RAM GB</th><th>판정</th>
</tr></thead>
<tbody>{"".join(rows)}</tbody>
</table></div>

<h2>산정 근거</h2>
<h3>토큰 생성은 메모리 대역폭 바운드</h3>
<div class="formula">tok/s = eta_bw × min(채널수 × DDR_MT/s × 8/1000, 코어수 × 코어당대역폭) / (활성가중치 + KV읽기)</div>
<h3>프롬프트 처리는 연산 바운드</h3>
<div class="formula">tok/s = eta_c × 코어수 × 전코어클럭 × FLOP_per_cycle / (2 × 활성파라미터)
FLOP_per_cycle: AVX2 32 · AVX-512 64 · AMX-BF16 1024</div>
<h3>RAM</h3>
<div class="formula">RAM = (가중치 + KV캐시×슬롯 + 컴퓨트버퍼 + OS·런타임) × 여유계수</div>

<h2>효율 계수와 그 근거</h2>
{calib_html}

<h2>주의사항</h2>
{warning_html}
{unverified_html}

<footer>이 리포트의 성능 수치는 하드웨어 스펙에 기반한 예측이며 실측이 아니다.
납품 확정 전 후보 서버 실기에서 <code>svrspec verify</code>로 예측 오차를 확인할 것을 권장한다.</footer>
</div>
"""
    return _document(title, body)


def _document(title: str, body: str) -> str:
    """Wrap the body into a standalone file: no external fonts, css or scripts,
    so it survives being emailed around inside a delivery package."""
    return (
        "<!doctype html>\n<html lang='ko'>\n<head>\n<meta charset='utf-8'>\n"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>\n"
        f"<title>{html.escape(title)}</title>\n<style>{_CSS}</style>\n</head>\n"
        f"<body>\n{body}\n</body>\n</html>\n"
    )


# --------------------------------------------------------------------------
# Machine-readable
# --------------------------------------------------------------------------


def write_csv(candidates: list[Candidate], path: Path) -> Path:
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "cpu_id",
                "vendor",
                "model",
                "family",
                "cores",
                "sockets",
                "mem_channels",
                "ddr",
                "bandwidth_gbs",
                "prefill_tps",
                "decode_tps",
                "single_request_s",
                "p95_steady_s",
                "p95_all_s",
                "p99_s",
                "storm_drain_min",
                "slot_utilisation",
                "ram_gb",
                "uncertainty",
                "verdict",
                "reasons",
                "cpu_source",
            ]
        )
        for c in candidates:
            sim = c.sim_pessimistic or c.sim
            w.writerow(
                [
                    c.cpu.id,
                    c.cpu.vendor,
                    c.cpu.model,
                    c.cpu.family,
                    c.cpu.cores * c.sockets,
                    c.sockets,
                    c.cpu.mem_channels,
                    f"{c.memory.ddr_gen}-{c.memory.effective_mts}",
                    round(c.throughput.effective_bandwidth_gbs, 1),
                    round(c.throughput.prefill_tps, 1),
                    round(c.throughput.decode_tps_single, 2),
                    round(c.latency.total_s, 2),
                    round(sim.p95_steady_s or sim.p95_s, 2),
                    round(sim.p95_s, 2),
                    round(sim.p99_s, 2),
                    round(sim.storm_drain_s / 60, 2),
                    round(sim.slot_utilisation, 3),
                    c.memory_gb,
                    round(c.throughput.uncertainty, 3),
                    c.verdict,
                    " | ".join(c.reasons),
                    c.cpu.source,
                ]
            )
    return path


def _plain(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _plain(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    return obj


def write_json(
    candidates: list[Candidate],
    tier_map: dict[str, Candidate | None],
    workload: Workload,
    eff: Efficiency,
    path: Path,
) -> Path:
    payload = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "workload": _plain(workload),
        "coefficients": _plain(list(eff.coefficients)),
        "tiers": {k: (v.cpu.id if v else None) for k, v in tier_map.items()},
        "candidates": [_plain(c) for c in candidates],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
