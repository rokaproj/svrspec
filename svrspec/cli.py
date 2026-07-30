"""svrspec command line.

    svrspec app                       데스크톱 창 실행 (서버 없음)
    svrspec gui                       브라우저 GUI 실행
    svrspec list cpus                 카탈로그 열람
    svrspec recommend --model X       모델 하나에 대한 CPU 후보 전체 산정
    svrspec fit --cpu Y               서버 하나에 어떤 모델까지 올라가는지
    svrspec size --model X --cpu Y    한 조합의 상세 내역
    svrspec verify                    메모리 모델과 BPW 표를 실제 GGUF로 검증
    svrspec catalog validate          카탈로그 정합성 검사
    svrspec bundle                    에어갭 서버 반입용 zip
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

from . import report
from .catalog import Catalog, CatalogError
from .gguf import GgufError, read_gguf, to_model_spec
from .memory import compute_buffer_bytes, kv_cache_bytes, weight_bytes
from .perf import Efficiency
from .sizing import evaluate, sweep_cpus, sweep_models, tiers
from .types import TokenProfile, Workload


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "handler", None):
        parser.print_help()
        return 2
    try:
        return args.handler(args)
    except (CatalogError, GgufError) as exc:
        print(f"svrspec: {exc}", file=sys.stderr)
        return 1
    except BrokenPipeError:  # `| head`
        return 0


# --------------------------------------------------------------------------
# Argument plumbing
# --------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="svrspec",
        description="CPU 전용 LLM 서버 스펙 산정 시뮬레이터 (관제 알람 → LLM → Teams)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--catalog-dir", type=Path, default=None, help="카탈로그 디렉터리 재지정")
    sub = p.add_subparsers(dest="command")

    # app ------------------------------------------------------------------
    ap = sub.add_parser("app", help="데스크톱 창 실행 (네이티브, 서버 없음)")
    ap.add_argument("--debug", action="store_true", help="WebView 개발자 도구를 연다")
    ap.set_defaults(handler=_cmd_app)

    # gui ------------------------------------------------------------------
    gu = sub.add_parser("gui", help="브라우저 GUI 실행 (로컬)")
    gu.add_argument("--port", type=int, default=8765)
    gu.add_argument("--host", default="127.0.0.1",
                    help="기본값은 localhost. 다른 PC에서 접속하려면 0.0.0.0")
    gu.add_argument("--no-browser", action="store_true", help="브라우저를 자동으로 열지 않는다")
    gu.set_defaults(handler=_cmd_gui)

    # list -----------------------------------------------------------------
    ls = sub.add_parser("list", help="카탈로그 열람")
    ls.add_argument("what", choices=["models", "cpus", "memory", "quants", "coefficients"])
    ls.add_argument("--filter", default="", help="id/이름 부분 일치")
    ls.add_argument("--korean", action="store_true", help="한국어 지원 모델만")
    ls.set_defaults(handler=_cmd_list)

    # recommend ------------------------------------------------------------
    rc = sub.add_parser("recommend", help="모델 하나에 대해 CPU 후보 전체를 산정")
    rc.add_argument("--model", required=True, help="모델 id (list models 로 확인)")
    _add_workload_args(rc)
    rc.add_argument("--sockets", type=int, default=1)
    rc.add_argument("--dpc", type=int, default=1, choices=[1, 2], help="채널당 DIMM 수")
    rc.add_argument("--cpu", action="append", default=None, help="특정 CPU만 (반복 가능)")
    rc.add_argument("--html", type=Path, default=None, help="HTML 리포트 출력 경로")
    rc.add_argument("--csv", type=Path, default=None)
    rc.add_argument("--json", type=Path, default=None)
    rc.add_argument("--only-pass", action="store_true", help="통과 후보만 표시")
    rc.set_defaults(handler=_cmd_recommend)

    # fit ------------------------------------------------------------------
    ft = sub.add_parser("fit", help="서버 하나에 어떤 모델까지 올릴 수 있는지")
    ft.add_argument("--cpu", required=True)
    _add_workload_args(ft)
    ft.add_argument("--sockets", type=int, default=1)
    ft.add_argument("--dpc", type=int, default=1, choices=[1, 2])
    ft.add_argument("--html", type=Path, default=None)
    ft.set_defaults(handler=_cmd_fit)

    # size -----------------------------------------------------------------
    sz = sub.add_parser("size", help="한 조합의 상세 산정 내역")
    sz.add_argument("--model", required=True)
    sz.add_argument("--cpu", required=True)
    _add_workload_args(sz)
    sz.add_argument("--sockets", type=int, default=1)
    sz.add_argument("--dpc", type=int, default=1, choices=[1, 2])
    sz.set_defaults(handler=_cmd_size)

    # gguf -----------------------------------------------------------------
    gg = sub.add_parser("gguf", help="GGUF 파일 헤더에서 모델 스펙 추출")
    gg.add_argument("path", type=Path)
    gg.set_defaults(handler=_cmd_gguf)

    # verify ---------------------------------------------------------------
    vf = sub.add_parser("verify", help="메모리 모델·BPW 표를 실제 GGUF 파일로 검증")
    vf.add_argument("--gguf", type=Path, action="append", default=None,
                    help="검증할 GGUF (미지정 시 ~/models/*.gguf)")
    vf.set_defaults(handler=_cmd_verify)

    # catalog --------------------------------------------------------------
    ct = sub.add_parser("catalog", help="카탈로그 관리")
    ct_sub = ct.add_subparsers(dest="catalog_command")
    ctv = ct_sub.add_parser("validate", help="정합성 검사")
    ctv.set_defaults(handler=_cmd_catalog_validate)
    ct.set_defaults(handler=_cmd_catalog_validate)

    # bundle ---------------------------------------------------------------
    bd = sub.add_parser("bundle", help="에어갭 서버 반입용 zip 생성")
    bd.add_argument("--out", type=Path, default=Path("svrspec-bundle.zip"))
    bd.set_defaults(handler=_cmd_bundle)

    return p


def _add_workload_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--quant", default="Q4_K_M")
    p.add_argument("--alarms-per-day", type=int, default=150)
    p.add_argument("--slots", type=int, default=2, help="동시 처리 슬롯 수")
    p.add_argument("--sla", type=float, default=30.0, help="알람→Teams 목표 지연(초)")
    p.add_argument("--storm", default="40/30", help="스톰 크기/창(초). 예 40/30")
    p.add_argument("--storms-per-day", type=int, default=2)
    p.add_argument("--storm-drain-min", type=float, default=5.0, help="스톰 소진 목표(분)")
    p.add_argument("--prompt-tokens", type=int, default=300, help="시스템 프롬프트 토큰")
    p.add_argument("--fewshot-tokens", type=int, default=400)
    p.add_argument("--alarm-tokens", type=int, default=250)
    p.add_argument("--output-tokens", type=int, default=250)
    p.add_argument("--no-prompt-cache", action="store_true",
                   help="시스템+few-shot 프리픽스 캐시를 쓰지 않는 경우")
    p.add_argument("--seed", type=int, default=20260730)


def _workload_from(args) -> Workload:
    try:
        size_s, window_s = args.storm.split("/")
        storm_size, storm_window = int(size_s), float(window_s)
    except ValueError:
        raise SystemExit(f"--storm 형식은 '개수/초' 이다 (받은 값: {args.storm!r})")

    return Workload(
        alarms_per_day=args.alarms_per_day,
        storm_size=storm_size,
        storm_window_s=storm_window,
        storms_per_day=args.storms_per_day,
        slots=args.slots,
        sla_seconds=args.sla,
        storm_drain_sla_s=args.storm_drain_min * 60.0,
        tokens=TokenProfile(
            system_tokens=args.prompt_tokens,
            fewshot_tokens=args.fewshot_tokens,
            alarm_tokens=args.alarm_tokens,
            output_tokens=args.output_tokens,
            prompt_cache=not args.no_prompt_cache,
        ),
        seed=args.seed,
    )


def _efficiency(cat: Catalog) -> Efficiency:
    """The coefficient set, straight from the catalogue.

    There is no runtime measurement step: the simulator is analytical and never
    loads the local CPU. Coefficients live in catalog/coefficients.json with
    their provenance attached.
    """
    return Efficiency.from_catalog(cat.coefficients)


def _catalog(args) -> Catalog:
    return Catalog(args.catalog_dir)


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


def _cmd_app(args) -> int:
    from .desktop import run

    return run(catalog=_catalog(args), debug=args.debug)


def _cmd_gui(args) -> int:
    from .gui import serve

    serve(host=args.host, port=args.port, open_browser=not args.no_browser,
          catalog=_catalog(args))
    return 0


def _cmd_list(args) -> int:
    cat = _catalog(args)
    needle = args.filter.lower()

    if args.what == "models":
        rows = []
        for m in cat.models:
            if needle and needle not in m.id.lower() and needle not in m.name.lower():
                continue
            if args.korean and not m.korean:
                continue
            rows.append([
                m.id, m.size_class, f"{m.params_b:.1f}",
                f"{m.active_params_b:.1f}" if m.active_params_b else "-",
                f"{m.n_layer}", f"{m.n_head}/{m.n_kv_head}", f"{m.n_vocab}",
                "O" if m.korean else "", "확인" if m.source != "unverified" else "미확인",
            ])
        print(report._table(
            ["id", "규모", "params B", "활성 B", "층", "head/kv", "vocab", "한국어", "출처"],
            rows, aligns="llrrrrrcl"))
        print(f"\n{len(rows)}개")
        return 0

    if args.what == "cpus":
        rows = []
        for c in cat.cpus:
            if needle and needle not in c.id.lower() and needle not in c.model.lower():
                continue
            amx = "AMX" if c.has("amx-bf16") else ("AVX512" if c.has("avx512") else "AVX2")
            rows.append([
                c.id, c.vendor, f"{c.cores}C/{c.threads}T",
                f"{c.all_core_turbo_ghz:.1f}", amx,
                f"{c.mem_channels}ch {c.ddr_gen}-{c.max_ddr_mts}",
                f"{c.peak_bandwidth_gbs:.0f}", f"{c.tdp_w}",
                "확인" if c.source != "unverified" else "미확인",
            ])
        print(report._table(
            ["id", "벤더", "코어", "전코어GHz", "ISA", "메모리", "GB/s", "TDP", "출처"],
            rows, aligns="llrrllrrl"))
        print(f"\n{len(rows)}개")
        return 0

    if args.what == "memory":
        rows = [[
            m.id, m.ddr_gen, f"{m.rated_mts}", f"{m.dimms_per_channel}",
            f"{m.effective_mts}", f"{m.dimm_gb}", m.kind,
        ] for m in cat.memory if not needle or needle in m.id.lower()]
        print(report._table(["id", "세대", "규격MT/s", "DPC", "실효MT/s", "DIMM GB", "종류"],
                            rows, aligns="llrrrrl"))
        print(f"\n{len(rows)}개")
        return 0

    if args.what == "coefficients":
        rows = [[
            c.id, c.kind, c.key, f"{c.value:g}",
            report.CONFIDENCE_LABEL.get(c.confidence, c.confidence), report.clip(c.notes),
        ] for c in cat.coefficients if not needle or needle in c.id.lower()]
        print(report._table(["id", "종류", "적용", "값", "근거", "비고"], rows, aligns="lllrll"))
        print(f"\n{len(rows)}개. 실측 근거가 없는 '추정' 행은 예측 오차 범위를 넓힌다.")
        return 0

    rows = [[q.id, f"{q.bits_per_weight:.2f}", q.quality, report.clip(q.notes)] for q in cat.quants]
    print(report._table(["id", "bpw", "품질", "비고"], rows, aligns="lrll"))
    return 0


def _cmd_recommend(args) -> int:
    cat = _catalog(args)
    model = cat.model(args.model)
    quant = cat.quant(args.quant)
    workload = _workload_from(args)
    eff = _efficiency(cat)

    candidates = sweep_cpus(
        cat, model, quant, workload,
        sockets=args.sockets, dimms_per_channel=args.dpc, cpu_ids=args.cpu,
    )
    if args.only_pass:
        candidates = [c for c in candidates if c.verdict != "fail"]
    if not candidates:
        print("조건을 만족하는 CPU 후보가 없다. --sockets 를 늘리거나 더 작은 모델을 시도하라.")
        return 1

    tier_map = tiers(candidates)
    print(report.render_console(candidates, tier_map, workload, eff, cat.unverified()))
    _emit_files(args, candidates, tier_map, workload, eff, cat,
                title=f"{model.name} 서버 스펙 산정 리포트")
    return 0


def _cmd_fit(args) -> int:
    cat = _catalog(args)
    cpu = cat.cpu(args.cpu)
    quant = cat.quant(args.quant)
    workload = _workload_from(args)
    eff = _efficiency(cat)

    candidates = sweep_models(
        cat, cpu, quant, workload, sockets=args.sockets, dimms_per_channel=args.dpc
    )
    installable = cpu.max_mem_gb * args.sockets
    rows = []
    for c in candidates:
        sim = c.sim_pessimistic or c.sim
        rows.append([
            c.model.id, f"{c.model.params_b:.1f}",
            f"{c.throughput.decode_tps_single:.1f}", f"{c.latency.total_s:.1f}",
            f"{sim.p95_steady_s or sim.p95_s:.1f}", f"{sim.storm_drain_s / 60:.1f}",
            f"{c.memory_gb}", report.VERDICT_LABEL.get(c.verdict, c.verdict),
        ])
    print(f"{cpu.vendor} {cpu.model} · {cpu.cores * args.sockets}코어 · "
          f"{cpu.mem_channels * args.sockets}ch {cpu.ddr_gen} · 최대 {installable}GB")
    print(f"부하  {report.describe_workload(workload)}\n")
    print(report._table(
        ["모델", "params B", "decode t/s", "1건 지연s", "평상시 p95", "스톰 분", "RAM GB", "판정"],
        rows, aligns="lrrrrrrl"))

    ok = [c for c in candidates if c.verdict == "pass"]
    if ok:
        biggest = max(ok, key=lambda c: c.model.params_b)
        print(f"\n이 서버가 감당하는 가장 큰 모델: {biggest.model.name} "
              f"({biggest.model.params_b:.1f}B, {biggest.quant.id}, RAM {biggest.memory_gb}GB)")
    else:
        print("\n통과하는 모델이 없다. 더 작은 모델이나 더 느슨한 SLA가 필요하다.")

    if args.html:
        args.html.write_text(
            report.render_html(candidates, tiers(candidates), workload, eff,
                               cat.unverified(),
                               title=f"{cpu.model} 탑재 가능 모델 산정"),
            encoding="utf-8")
        print(f"HTML 리포트: {args.html}")
    return 0


def _cmd_size(args) -> int:
    cat = _catalog(args)
    model = cat.model(args.model)
    quant = cat.quant(args.quant)
    cpu = cat.cpu(args.cpu)
    memory = cat.memory_for(cpu, args.dpc)
    workload = _workload_from(args)
    eff = _efficiency(cat)

    c = evaluate(model, quant, cpu, memory, eff, workload, args.sockets)
    t, sim = c.throughput, (c.sim_pessimistic or c.sim)

    print(f"모델    {model.name}  {model.params_b:.2f}B"
          + (f" (활성 {model.active_params_b:.2f}B MoE)" if model.active_params_b else "")
          + f"  {quant.id} {quant.bits_per_weight:.2f}bpw")
    print(f"서버    {cpu.vendor} {cpu.model}  {cpu.cores * args.sockets}코어 "
          f"{args.sockets}소켓  {cpu.mem_channels * args.sockets}ch "
          f"{memory.ddr_gen}-{memory.effective_mts}")
    print(f"부하    {report.describe_workload(workload)}")
    print()
    print("메모리 산정")
    print(report._table(["항목", "GiB"], [
        ["가중치", f"{c.ram.weights_gb:.2f}"],
        [f"KV 캐시 ({workload.slots} 슬롯)", f"{c.ram.kv_cache_gb:.2f}"],
        ["컴퓨트 버퍼", f"{c.ram.compute_buffer_gb:.2f}"],
        ["OS·런타임", f"{c.ram.runtime_os_gb:.2f}"],
        ["소계", f"{c.ram.subtotal_gb:.2f}"],
        [f"여유계수 x{c.ram.headroom_factor}", f"{c.ram.recommended_gb:.2f}"],
        ["장착 권장", f"{c.ram.provision_gb} GB"],
    ], aligns="lr"))
    print()
    print("처리량 예측")
    print(report._table(["항목", "값", "병목"], [
        ["실효 대역폭", f"{t.effective_bandwidth_gbs:.1f} GB/s", t.decode_bound_by],
        ["연산 능력", f"{t.peak_flops_tflops:.2f} TFLOP/s", ""],
        ["prefill", f"{t.prefill_tps:.1f} tok/s", t.prefill_bound_by],
        ["decode (1슬롯)", f"{t.decode_tps_single:.2f} tok/s", t.decode_bound_by],
        [f"decode ({workload.slots}슬롯 합계)", f"{t.decode_tps_aggregate:.2f} tok/s", ""],
        ["불확실도", f"±{t.uncertainty:.0%}", ""],
    ], aligns="llL"))
    print()
    print("알람 1건 지연 (대기 없음)")
    print(report._table(["단계", "초"], [
        [f"prefill {workload.tokens.billed_prefill_tokens} tok", f"{c.latency.ttft_s:.2f}"],
        [f"생성 {workload.tokens.output_tokens} tok", f"{c.latency.generate_s:.2f}"],
        ["Teams 전송", f"{c.latency.teams_s:.2f}"],
        ["합계", f"{c.latency.total_s:.2f}"],
    ], aligns="lr"))
    print()
    storm_alarms = sim.completed - sim.steady_completed
    print(f"하루 시뮬레이션 (불리한 추정 기준, 완료 {sim.completed}건 — "
          f"평상시 {sim.steady_completed}건 / 스톰 {storm_alarms}건)")
    print(report._table(["지표", "값", "판정 기준"], [
        ["평상시 p95", f"{sim.p95_steady_s:.1f} s",
         f"SLA {workload.sla_seconds:.0f}s {'충족' if sim.sla_met else '초과'}"],
        ["스톰 소진", f"{sim.storm_drain_s / 60:.2f} 분",
         f"목표 {workload.storm_drain_sla_s / 60:.0f}분 "
         f"{'충족' if sim.storm_sla_met else '초과'}"],
        ["전체 p50", f"{sim.p50_s:.1f} s", ""],
        ["전체 p95", f"{sim.p95_s:.1f} s", "스톰 대기 포함"],
        ["전체 p99", f"{sim.p99_s:.1f} s", "스톰 대기 포함"],
        ["최대", f"{sim.max_s:.1f} s", ""],
        ["최대 큐 깊이", f"{sim.max_queue_depth}", ""],
        ["슬롯 점유율", f"{sim.slot_utilisation:.1%}", ""],
    ], aligns="lrl"))
    print("  스톰 알람은 정의상 큐가 쌓이므로 지연 SLA가 아니라 소진 시간으로 판정한다.")
    print()
    print(f"판정  {report.VERDICT_LABEL.get(c.verdict, c.verdict)}")
    for r in c.reasons:
        print(f"  - {r}")
    for w in t.warnings:
        print(f"  ! {w}")
    return 0


def _cmd_gguf(args) -> int:
    info = read_gguf(args.path)
    spec = to_model_spec(info)
    print(report._table(["항목", "값"], [
        ["파일", info.path.name],
        ["아키텍처", info.architecture],
        ["이름", info.name],
        ["양자화", info.quant],
        ["파일 크기", f"{info.file_bytes / 1024**3:.2f} GiB"],
        ["파라미터", f"{info.param_count / 1e9:.3f} B"],
        ["실측 BPW", f"{info.measured_bpw:.3f}"],
        ["층", f"{spec.n_layer}"],
        ["임베딩 차원", f"{spec.n_embd}"],
        ["헤드 (전체/KV)", f"{spec.n_head} / {spec.n_kv_head}"],
        ["head_dim", f"{spec.kv_head_dim}"],
        ["vocab", f"{spec.n_vocab}"],
        ["학습 컨텍스트", f"{spec.ctx_train}"],
        ["텐서 수", f"{info.tensor_count}"],
    ], aligns="lr"))
    print("\n카탈로그 항목 형태:")
    row = {k: v for k, v in dataclasses.asdict(spec).items() if v not in (None, "", False)}
    import json as _json
    print(_json.dumps(row, indent=2, ensure_ascii=False))
    return 0


def _cmd_verify(args) -> int:
    """Check the memory model and the BPW table against real GGUF files."""
    cat = _catalog(args)
    paths = args.gguf or sorted((Path.home() / "models").glob("*.gguf"))
    if not paths:
        print("검증할 GGUF 파일이 없다. --gguf 로 경로를 지정하거나 ~/models 에 두어라.")
        return 1

    rows, failures = [], 0
    for path in paths:
        info = read_gguf(path)
        spec = to_model_spec(info)
        table_quant = next((q for q in cat.quants if q.id == info.quant), None)
        if not table_quant:
            rows.append([path.name, info.quant, "-", f"{info.measured_bpw:.3f}", "표에 없음"])
            continue

        predicted = weight_bytes(spec, table_quant)
        error = (predicted - info.file_bytes) / info.file_bytes
        ok = abs(error) <= 0.05
        failures += 0 if ok else 1
        rows.append([
            path.name, info.quant, f"{table_quant.bits_per_weight:.3f}",
            f"{info.measured_bpw:.3f}", f"{error:+.1%} {'OK' if ok else 'MISS'}",
        ])

    print("가중치 크기 모델 검증 — 표의 BPW로 계산한 파일 크기 vs 실제 파일 크기")
    print(report._table(["파일", "양자화", "표 bpw", "실측 bpw", "오차"], rows, aligns="llrrl"))

    print("\nKV 캐시·컴퓨트 버퍼 모델")
    for path in paths:
        spec = to_model_spec(read_gguf(path))
        for ctx in (2048, 8192):
            kv = kv_cache_bytes(spec, ctx, 1) / 1024**2
            cb = compute_buffer_bytes(spec, 1) / 1024**2
            print(f"  {path.name}  ctx {ctx}: KV {kv:.0f} MiB, compute {cb:.0f} MiB")
    print("\n  이 두 값은 llama-server 기동 로그의 'KV self size' / 'compute buffer size'와")
    print("  직접 대조할 수 있다. 5% 넘게 벌어지면 memory.py 의 ACTIVATION_TENSORS 를 조정하라.")

    if failures:
        print(f"\n{failures}개 파일에서 5% 초과 오차. quants.json 의 bits_per_weight 를 보정하라.")
        return 1
    print("\n전부 5% 이내.")
    return 0


def _passmark_outliers(cat: Catalog) -> tuple[list[str], float]:
    """Audit the transcribed specs against the one measured number available.

    PassMark's CPU Mark is a real benchmark result, while cores, clocks and
    memory channels are figures somebody typed in from a table. Dividing the
    score by cores x all-core-GHz gives a per-core-per-GHz throughput that
    should sit in a fairly narrow band across x86 server parts. A row far
    outside it usually means a spec was transcribed wrong -- so the measured
    number polices the unmeasurable ones.

    It is deliberately NOT used to predict prefill: CPU Mark is a mixed
    integer/compression/physics workload, not a GEMM, and it does not exercise
    AVX-512 or AMX the way llama.cpp does. Using it as a predictor would swap a
    documented estimate for an undocumented one.
    """
    scored = [
        (c, c.passmark_multithread / (c.cores * c.all_core_turbo_ghz))
        for c in cat.cpus
        if c.passmark_multithread and c.cores and c.all_core_turbo_ghz
    ]
    if len(scored) < 4:
        return [], 0.0

    ratios = sorted(r for _, r in scored)
    median = ratios[len(ratios) // 2]
    out = [
        f"{c.id}: CPU Mark {c.passmark_multithread}, 코어x클럭 "
        f"{c.cores * c.all_core_turbo_ghz:.1f} → 비율 {r:.0f} "
        f"(중위값 {median:.0f} 대비 {(r - median) / median:+.0%})"
        for c, r in sorted(scored, key=lambda t: t[1])
        if abs(r - median) / median > 0.35
    ]
    return out, median


def _cmd_catalog_validate(args) -> int:
    cat = _catalog(args)
    print(f"카탈로그 정합성 통과 — {cat.summary()}")

    outliers, median = _passmark_outliers(cat)
    scored = sum(1 for c in cat.cpus if c.passmark_multithread)
    if scored:
        print(f"\nPassMark 교차검증 — {scored}/{len(cat.cpus)}개 CPU에 실측 점수 있음, "
              f"코어x클럭당 비율 중위값 {median:.0f}")
        if outliers:
            print(f"  비율이 35% 넘게 벗어난 {len(outliers)}건 (스펙 전사 오류 의심, "
                  f"세대별 IPC 차이일 수도 있음):")
            for o in outliers:
                print(f"    - {o}")
        else:
            print("  이상치 없음 — 전사된 코어·클럭 값이 실측 점수와 정합적이다.")

    unmatched = []
    for cpu in cat.cpus:
        for dpc in (1, 2):
            try:
                cat.memory_for(cpu, dpc)
            except CatalogError as exc:
                unmatched.append(f"{cpu.id} @ {dpc}DPC: {exc}")
    if unmatched:
        print(f"\n메모리 옵션이 없는 CPU 조합 {len(unmatched)}건:")
        for u in unmatched[:20]:
            print(f"  - {u}")

    unverified = cat.unverified()
    if unverified:
        print(f"\n미확인 출처 {len(unverified)}건 / 전체 "
              f"{len(cat.models) + len(cat.cpus) + len(cat.memory)}건")
        for kind in ("model", "cpu", "memory"):
            ids = [i for k, i in unverified if k == kind]
            if ids:
                print(f"  {kind}: {', '.join(ids[:10])}"
                      + (f" 외 {len(ids) - 10}건" if len(ids) > 10 else ""))
    else:
        print("\n모든 항목이 출처 확인됨.")

    by_class: dict[str, int] = {}
    for m in cat.models:
        by_class[m.size_class] = by_class.get(m.size_class, 0) + 1
    print("\n모델 규모 분포: " + ", ".join(f"{k} {v}개" for k, v in sorted(by_class.items())))
    moe = [m.id for m in cat.models if m.active_params_b]
    print(f"MoE {len(moe)}개" + (f": {', '.join(moe)}" if moe else ""))
    return 0 if not unmatched else 1


def _cmd_bundle(args) -> int:
    import zipfile

    root = Path(__file__).resolve().parent.parent
    include = [
        *sorted((root / "svrspec").rglob("*.py")),
        *sorted((root / "svrspec" / "catalog").glob("*.json")),
        root / "pyproject.toml",
    ]
    readme = root / "README.md"
    if readme.exists():
        include.append(readme)

    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z:
        for path in include:
            if "__pycache__" in path.parts:
                continue
            z.writestr(str(path.relative_to(root)), path.read_bytes())
        z.writestr(
            "RUN.txt",
            "svrspec — CPU+RAM 전용. 표준 라이브러리만 쓰므로 설치가 필요 없다.\n\n"
            "  unzip svrspec-bundle.zip -d svrspec && cd svrspec\n"
            "  python3 -m svrspec.cli gui              # 브라우저 GUI\n"
            "  python3 -m svrspec.cli list cpus\n"
            "  python3 -m svrspec.cli recommend --model <모델id>\n\n"
            "순수 해석적 시뮬레이터라 모델 파일도, 벤치마크도 필요 없다. 실제 CPU에\n"
            "부하를 주지 않는다.\n",
        )
    size_kb = args.out.stat().st_size / 1024
    print(f"{args.out} ({size_kb:.0f} KB, {len(include) + 1}개 파일)")
    print("에어갭 서버에 반입 후 python3 만으로 실행된다 (외부 의존성 없음).")
    return 0


def _emit_files(args, candidates, tier_map, workload, eff, cat, title: str) -> None:
    if getattr(args, "html", None):
        args.html.write_text(
            report.render_html(candidates, tier_map, workload, eff, cat.unverified(), title=title),
            encoding="utf-8",
        )
        print(f"\nHTML 리포트: {args.html}")
    if getattr(args, "csv", None):
        report.write_csv(candidates, args.csv)
        print(f"CSV: {args.csv}")
    if getattr(args, "json", None):
        report.write_json(candidates, tier_map, workload, eff, args.json)
        print(f"JSON: {args.json}")


if __name__ == "__main__":
    raise SystemExit(main())
