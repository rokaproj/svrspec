"""svrspec command line.

    svrspec app                       데스크톱 창 실행 (서버 없음)
    svrspec gui                       브라우저 GUI 실행
    svrspec list cpus                 카탈로그 열람
    svrspec recommend --model X       모델 하나에 대한 CPU 후보 전체 산정
    svrspec fit --cpu Y               서버 하나에 어떤 모델까지 올라가는지
    svrspec size --model X --cpu Y    한 조합의 상세 내역
    svrspec timeline --model X --cpu Y  하루 리소스 시계열 (CPU·대역폭·연산·KV)
    svrspec capacity --model X --cpu Y  이 서버가 어디서 무너지는지
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


def _force_utf8() -> None:
    """Make Korean output survive a Windows console.

    On Windows, stdout defaults to the system ANSI code page (cp949 on a Korean
    install, cp1252 in CI), so printing any Korean label raises
    UnicodeEncodeError. Frozen builds do not inherit PYTHONUTF8 either. Every
    entry point goes through main(), so this belongs here rather than in the
    packaging shims.
    """
    if sys.platform == "win32":
        try:
            import ctypes

            # 65001 = CP_UTF8, so a real console renders what we emit.
            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
            ctypes.windll.kernel32.SetConsoleCP(65001)
        except Exception:
            pass
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def main(argv: list[str] | None = None) -> int:
    _force_utf8()
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

    # timeline -------------------------------------------------------------
    tl = sub.add_parser("timeline", help="하루 리소스 사용량 시계열 (CPU·대역폭·연산·KV)")
    tl.add_argument("--model", required=True)
    tl.add_argument("--cpu", required=True)
    _add_workload_args(tl)
    tl.add_argument("--sockets", type=int, default=1)
    tl.add_argument("--dpc", type=int, default=1, choices=[1, 2])
    tl.add_argument("--buckets", type=int, default=96,
                    help="하루를 몇 구간으로 나눌지 (기본 96 = 15분)")
    tl.add_argument("--pessimistic", action="store_true",
                    help="판정에 쓰는 불리한 추정으로 그린다 (기본은 명목 예측)")
    tl.add_argument("--csv", type=Path, default=None, help="버킷 시계열 CSV 출력")
    tl.add_argument("--host-csv", type=Path, default=None,
                    help="모니터링 에이전트 형태의 호스트 샘플 CSV 출력")
    tl.add_argument("--host-period", type=float, default=60.0,
                    help="호스트 샘플 주기(초). 기본 60 = 하루 1440행")
    tl.set_defaults(handler=_cmd_timeline)

    # capacity -------------------------------------------------------------
    cp = sub.add_parser("capacity", help="이 서버가 어디서 무너지는지 (과부하 지점)")
    cp.add_argument("--model", required=True)
    cp.add_argument("--cpu", required=True)
    _add_workload_args(cp)
    cp.add_argument("--sockets", type=int, default=1)
    cp.add_argument("--dpc", type=int, default=1, choices=[1, 2])
    cp.add_argument("--axis", default="all",
                    choices=["all", "alarms", "storm", "prompt", "output"],
                    help="부하를 올릴 축 (기본 all = 네 축 전부)")
    cp.add_argument("--curve", action="store_true", help="탐색 경로의 모든 지점을 표로 출력")
    cp.add_argument("--csv", type=Path, default=None, help="곡선 CSV 출력")
    cp.set_defaults(handler=_cmd_capacity)

    # mock -----------------------------------------------------------------
    mk = sub.add_parser("mock", help="실측 알람량을 재현한 목데이터 생성")
    mk.add_argument("--date", default="2026-06-01", help="생성할 날짜 (실측일이면 그날의 실제 건수)")
    mk.add_argument("--days", type=int, default=1, help="이 날짜부터 며칠치")
    mk.add_argument("--count", type=int, default=None, help="건수 직접 지정 (기본: 실측값)")
    mk.add_argument("--seed", type=int, default=20260730)
    mk.add_argument("--storm", default="40/30", help="스톰 크기/창(초)")
    mk.add_argument("--storms-per-day", type=int, default=2)
    mk.add_argument("--out", type=Path, default=None, help="저장 경로 (.jsonl 또는 .csv)")
    mk.add_argument("--show", type=int, default=5, help="표본 몇 건을 화면에 보일지")
    mk.set_defaults(handler=_cmd_mock)

    # serve ----------------------------------------------------------------
    sv = sub.add_parser(
        "serve",
        help="알람을 실제로 받아 처리하고 Teams로 내보내는 파이프라인 실행")
    sv.add_argument("--model", required=True)
    sv.add_argument("--cpu", required=True)
    _add_workload_args(sv)
    sv.add_argument("--sockets", type=int, default=1)
    sv.add_argument("--dpc", type=int, default=1, choices=[1, 2])
    sv.add_argument("--alarms", type=Path, default=None,
                    help="JSONL 알람 파일 (미지정 시 --date 로 생성)")
    sv.add_argument("--date", default="2026-06-01", help="목데이터를 생성할 날짜")
    sv.add_argument("--speed", type=float, default=0.0,
                    help="0=가상시간(즉시). 60=실시간 60배속 재생")
    sv.add_argument("--queue-limit", type=int, default=None,
                    help="큐 상한. 넘으면 새 알람을 버린다(백프레셔)")
    sv.add_argument("--trace", type=int, default=0, help="처리 로그를 앞에서 N건 보인다")
    sv.add_argument("--csv", type=Path, default=None, help="건별 전달 기록 CSV")
    sv.set_defaults(handler=_cmd_serve)

    # lab ------------------------------------------------------------------
    lb = sub.add_parser("lab", help="가상 서버 조립 — CPU·소켓·DIMM을 직접 고른다")
    lb_sub = lb.add_subparsers(dest="lab_command")

    lbb = lb_sub.add_parser("build", help="머신을 조립해 검증하고 저장")
    lbb.add_argument("--name", default="머신")
    lbb.add_argument("--cpu", required=True)
    lbb.add_argument("--sockets", type=int, default=1)
    lbb.add_argument("--dimm", type=int, required=True, help="DIMM 한 장의 용량(GB)")
    lbb.add_argument("--count", type=int, required=True, help="DIMM 총 장수")
    lbb.add_argument("--model", required=True)
    lbb.add_argument("--quant", default="Q4_K_M")
    lbb.add_argument("--slots", type=int, default=4)
    lbb.add_argument("--out", type=Path, default=None, help="머신 정의 저장 경로(.json)")
    lbb.set_defaults(handler=_cmd_lab_build)

    lbs = lb_sub.add_parser("show", help="저장된 머신의 구성과 검증 결과")
    lbs.add_argument("machine", type=Path)
    lbs.set_defaults(handler=_cmd_lab_show)

    lb.set_defaults(handler=_cmd_lab_help)

    # bench ----------------------------------------------------------------
    bn = sub.add_parser("bench", help="조립한 머신에 부하를 걸어 실행")
    bn.add_argument("--machine", type=Path, default=None, help="lab build 로 저장한 머신")
    bn.add_argument("--cpu", default=None, help="--machine 대신 즉석 조립")
    bn.add_argument("--model", default=None)
    bn.add_argument("--sockets", type=int, default=1)
    bn.add_argument("--dimm", type=int, default=None, help="DIMM 한 장의 용량(GB)")
    bn.add_argument("--dimm-count", dest="count", type=int, default=None,
                    help="DIMM 총 장수")
    bn.add_argument("--profile", default="replay",
                    choices=["replay", "ramp", "spike", "soak"])
    bn.add_argument("--date", default="2026-06-01", help="replay: 재생할 날짜")
    bn.add_argument("--from", dest="from_rate", type=int, default=100,
                    help="ramp: 시작 부하(건/일)")
    bn.add_argument("--to", dest="to_rate", type=int, default=2000,
                    help="ramp: 끝 부하(건/일)")
    bn.add_argument("--rate", type=int, default=300, help="soak/spike: 평시 부하")
    bn.add_argument("--peak", type=int, default=800, help="spike: 급증 부하")
    bn.add_argument("--hours", type=float, default=None, help="ramp/soak: 구간 길이")
    _add_workload_args(bn)
    bn.add_argument("--frames", type=int, default=600)
    bn.add_argument("--queue-limit", type=int, default=None)
    bn.add_argument("--live", action="store_true", help="프레임을 스파크라인으로 재생")
    bn.add_argument("--csv", type=Path, default=None, help="프레임 시계열 CSV")
    bn.set_defaults(handler=_cmd_bench)

    # calibrate ------------------------------------------------------------
    cb = sub.add_parser(
        "calibrate",
        help="실측 로그를 읽어 효율 계수를 실측으로 승격 (아무것도 실행하지 않는다)")
    cb.add_argument("log", type=Path, nargs="+",
                    help="llama-bench 출력(마크다운/JSON) 또는 llama-server 로그")
    cb.add_argument("--cpu", required=True, help="그 로그를 낸 CPU의 카탈로그 id")
    cb.add_argument("--model", required=True, help="그 로그가 돌린 모델의 카탈로그 id")
    cb.add_argument("--quant", default="Q4_K_M")
    cb.add_argument("--sockets", type=int, default=1)
    cb.add_argument("--dpc", type=int, default=1, choices=[1, 2])
    cb.add_argument("--confidence", default="measured",
                    choices=["measured", "derived"],
                    help="로그가 이 SKU의 것이 아니거나 모델을 손으로 맞췄으면 derived")
    cb.add_argument("--out", type=Path, default=None,
                    help="유도한 계수를 coefficients.json 형태로 저장")
    cb.set_defaults(handler=_cmd_calibrate)

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

    # selfcheck -------------------------------------------------------------
    sc = sub.add_parser(
        "selfcheck",
        help="이 설치본이 데스크톱 창을 띄울 수 있는지 검사 (아무것도 실행하지 않는다)")
    sc.set_defaults(handler=_cmd_selfcheck)

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
        print("조건을 만족하는 CPU 후보가 없다. --sockets 를 늘리거나 더 작은 모델을 시도해 보세요.")
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


# --------------------------------------------------------------------------
# timeline / capacity
#
# These two answer the questions `recommend` leaves open: what the machine is
# actually doing minute to minute, and how much more load it would take before
# it stops coping. Both are still analytical -- nothing here runs a model or
# loads this machine. See `_efficiency` for why that boundary matters.
# --------------------------------------------------------------------------

#: Eight levels of block, for drawing a series in one terminal row.
SPARK = " ▁▂▃▄▅▆▇█"


def _display_width(text: str) -> int:
    """Terminal columns a string occupies.

    Hangul is double-width, so `len()` misaligns every label column in this
    file. East Asian Wide and Fullwidth both take two cells.
    """
    import unicodedata

    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)


def _sparkline(values: list[float], ceiling: float | None = None) -> str:
    """One row per metric. A day of load has to be legible at a glance.

    Scaled to the series' own maximum unless a ceiling is given, because most
    of these series spend the day near zero and scaling everything to 100%
    would draw a flat line for all of them.
    """
    if not values:
        return ""
    top = ceiling if ceiling else max(values)
    if top <= 0:
        return SPARK[0] * len(values)
    out = []
    for v in values:
        level = int(round(max(0.0, min(v / top, 1.0)) * (len(SPARK) - 1)))
        out.append(SPARK[level])
    return "".join(out)


def _timeline_for(args, cat: Catalog, pessimistic: bool = False):
    """(candidate, trace, ceilings, timeline) for one build under one workload."""
    from .simulate import simulate
    from .sizing import decode_table
    from .timeline import build_timeline, ceilings_for

    model = cat.model(args.model)
    quant = cat.quant(args.quant)
    cpu = cat.cpu(args.cpu)
    memory = cat.memory_for(cpu, args.dpc)
    workload = _workload_from(args)
    eff = _efficiency(cat)

    candidate = evaluate(model, quant, cpu, memory, eff, workload, args.sockets)
    # The verdict runs on the derated prediction; the resource picture defaults
    # to the nominal one. Drawing the pessimistic day by default would show a
    # machine slower than the one being quoted.
    derate = max(1.0 - candidate.throughput.uncertainty, 0.05) if pessimistic else 1.0
    _, trace = simulate(
        workload,
        prefill_tps=candidate.throughput.prefill_tps * derate,
        decode_by_active=decode_table(
            model, quant, cpu, memory, workload, args.sockets, eff, derate=derate
        ),
    )
    ceilings = ceilings_for(
        model, quant, cpu, memory, eff, workload,
        candidate.throughput, args.sockets, candidate.memory_gb,
    )
    buckets = max(int(getattr(args, "buckets", 96)), 1)
    return candidate, trace, ceilings, build_timeline(trace, ceilings, buckets=buckets)


def _cmd_timeline(args) -> int:
    cat = _catalog(args)
    workload = _workload_from(args)
    candidate, trace, ceilings, tl = _timeline_for(args, cat, args.pessimistic)
    model, cpu = candidate.model, candidate.cpu
    b = tl.buckets

    print(f"모델    {model.name}  {model.params_b:.2f}B  {candidate.quant.id}")
    print(f"서버    {cpu.vendor} {cpu.model}  {cpu.cores * args.sockets}코어 "
          f"{args.sockets}소켓  {cpu.mem_channels * args.sockets}ch "
          f"{candidate.memory.ddr_gen}-{candidate.memory.effective_mts}  "
          f"RAM {candidate.memory_gb}GB")
    print(f"부하    {report.describe_workload(workload)}")
    print()

    minutes = (24 * 60) / len(b)
    basis = "불리한 추정" if args.pessimistic else "명목 예측"
    print(f"하루 리소스 시계열 ({minutes:.0f}분 x {len(b)}버킷, {basis} 기준)")
    print()
    rows = [
        ("CPU 가동", [x.cpu_pct for x in b], f"최대 {max(x.cpu_pct for x in b):.1f}%"),
        ("대역폭 평균", [x.bandwidth_avg_pct for x in b],
         f"최대 {max(x.bandwidth_avg_pct for x in b):.1f}% "
         f"(순간 피크 {tl.peak_bandwidth_pct:.0f}%)"),
        ("연산 평균", [x.compute_avg_pct for x in b],
         f"최대 {max(x.compute_avg_pct for x in b):.1f}% "
         f"(순간 피크 {tl.peak_compute_pct:.0f}%)"),
        ("KV 실사용", [x.kv_used_gb for x in b],
         f"최대 {tl.peak_kv_used_gb:.2f}GB / 예약 "
         f"{ceilings.kv_reserved_bytes / 1024 ** 3:.2f}GB"),
        ("큐 깊이", [float(x.queued) for x in b], f"최대 {tl.peak_queue}"),
        ("도착", [float(x.arrived) for x in b], f"합계 {sum(x.arrived for x in b)}건"),
    ]
    width = max(_display_width(name) for name, *_ in rows)
    for name, values, note in rows:
        pad = " " * (width - _display_width(name))
        print(f"  {name}{pad}  {_sparkline(values)}  {note}")
    axis_pad = " " * (width + 2)
    print(f"  {axis_pad}0h{' ' * max(len(b) - 5, 0)}24h")
    print("  각 행은 자기 계열의 최대값에 맞춰 그린다 — 절대값은 오른쪽 주석을 참고하세요.")
    print()

    label = {"bandwidth": "메모리 대역폭", "compute": "연산", "none": "없음"}
    print("병목 진단")
    print(f"  작업시간 배분   prefill {tl.prefill_share:.1%} / decode {tl.decode_share:.1%}")
    print(f"  천장에 붙은 시간 대역폭 {tl.seconds_bandwidth_bound / 60:.1f}분 · "
          f"연산 {tl.seconds_compute_bound / 60:.1f}분 "
          f"(총 가동 {tl.busy_seconds / 60:.1f}분, 하루의 {tl.busy_seconds / 864:.2f}%)")
    print(f"  →  {label.get(tl.binding_resource, tl.binding_resource)} 바운드")
    # The two splits are not the same question, and they can disagree. Batched
    # decode is 2*P flops per token like prefill is, so on a part with narrow
    # vector units the compute ceiling binds decode too -- the phase holding
    # the machine is then decode while the ceiling it sits on is compute.
    if (tl.decode_share > tl.prefill_share) != (
        tl.seconds_bandwidth_bound > tl.seconds_compute_bound
    ):
        print("  (decode가 더 오래 돌지만 대역폭이 아니라 연산 천장에 걸려 있다 — "
              "슬롯을 묶어 배치 decode를 하면 토큰당 연산량이 prefill과 같아지고, "
              "벡터유닛이 좁은 부품에서는 그쪽이 먼저 찬다)")
    print("  순간 포화는 과부하가 아니다 — decode 중인 요청은 정의상 어느 한 천장에 붙어 있다. "
          "과부하는 큐가 쌓이는 것이다.")
    print()

    busiest = sorted(b, key=lambda x: (-x.queued, -x.cpu_pct))[:5]
    print("가장 바쁜 구간 (상위 5)")
    print(report._table(
        ["시각", "CPU%", "대역폭 평균%", "대역폭 피크%", "연산 평균%", "연산 피크%",
         "KV GB", "RAM GB", "큐", "도착", "완료"],
        [[f"{x.t_s / 3600:.2f}h", f"{x.cpu_pct:.1f}",
          f"{x.bandwidth_avg_pct:.1f}", f"{x.bandwidth_peak_pct:.0f}",
          f"{x.compute_avg_pct:.1f}", f"{x.compute_peak_pct:.0f}",
          f"{x.kv_used_gb:.2f}", f"{x.ram_used_gb:.2f}",
          f"{x.queued}", f"{x.arrived}", f"{x.completed}"] for x in busiest],
        aligns="lrrrrrrrrrr"))
    print("  평균은 그 15분의 정직한 부하, 피크는 그 안에서 실제로 닿은 천장이다. "
          "평균만 보면 스톰이 사라지고 피크만 보면 종일 포화로 보인다.")
    for note in tl.notes:
        print(f"  ! {note}")

    if args.csv:
        _write_timeline_csv(args.csv, tl)
        print(f"\n버킷 시계열 CSV 저장: {args.csv}")

    if args.host_csv:
        from .hostsim import sample_host, to_csv

        host = sample_host(
            trace, ceilings, candidate.cpu,
            installed_gb=candidate.memory_gb,
            sockets=args.sockets,
            period_s=args.host_period,
        )
        # utf-8-sig: these land in Excel next to the other exports, and a
        # sibling file opening as mojibake is worse than a three-byte prefix.
        args.host_csv.write_text(to_csv(host), encoding="utf-8-sig")
        print(f"호스트 샘플 CSV 저장: {args.host_csv} "
              f"({len(host.samples)}행, {host.period_s:.0f}초 주기, "
              f"최대 RSS {host.peak_rss_gb:.2f}GB)")
        for note in host.notes:
            print(f"  ! {note}")
    return 0


def _write_timeline_csv(path: Path, tl) -> None:
    import csv

    fields = [
        "t_s", "cpu_pct", "bandwidth_avg_gbs", "bandwidth_avg_pct", "bandwidth_peak_pct",
        "compute_avg_tflops", "compute_avg_pct", "compute_peak_pct", "prefill_tps",
        "decode_tps", "kv_used_gb", "ram_used_gb", "ram_pct", "queued", "active",
        "arrived", "completed",
    ]
    # utf-8-sig so Excel opens the Korean headers of sibling exports correctly;
    # this file has none, but the exports must not disagree on encoding.
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(fields)
        for bucket in tl.buckets:
            writer.writerow([f"{getattr(bucket, f):.4f}"
                             if isinstance(getattr(bucket, f), float)
                             else getattr(bucket, f) for f in fields])


def _cmd_capacity(args) -> int:
    from .capacity import AXES, AXIS_LABEL, LIMITER_LABEL, find_knee, weakest_axis

    cat = _catalog(args)
    model = cat.model(args.model)
    quant = cat.quant(args.quant)
    cpu = cat.cpu(args.cpu)
    memory = cat.memory_for(cpu, args.dpc)
    workload = _workload_from(args)
    eff = _efficiency(cat)

    axes = AXES if args.axis == "all" else (args.axis,)
    curves = {
        axis: find_knee(model, quant, cpu, memory, eff, workload, axis, args.sockets)
        for axis in axes
    }

    print(f"모델    {model.name}  {model.params_b:.2f}B  {quant.id}")
    print(f"서버    {cpu.vendor} {cpu.model}  {cpu.cores * args.sockets}코어 "
          f"{args.sockets}소켓  {cpu.mem_channels * args.sockets}ch "
          f"{memory.ddr_gen}-{memory.effective_mts}")
    print(f"부하    {report.describe_workload(workload)}")
    print()
    print("과부하 지점 — 부하를 올려 SLA가 깨지는 지점을 찾는다 (불리한 추정 기준)")
    print()

    rows = []
    for axis, curve in curves.items():
        name, unit = AXIS_LABEL[axis]
        knee = f"{curve.knee.value:,}{unit}" if curve.knee else "없음"
        brk = f"{curve.breaks_at.value:,}{unit}" if curve.breaks_at else "—"
        headroom = "∞" if curve.hit_ceiling else (
            f"{curve.headroom:.1f}배" if curve.knee else "0배")
        rows.append([name, f"{curve.baseline:,}{unit}", knee, brk, headroom,
                     LIMITER_LABEL[curve.limiter]])
    print(report._table(["부하 축", "현재", "무릎(마지막 통과)", "붕괴", "여유", "한계 요인"],
                        rows, aligns="lrrrrl"))
    print()

    weakest = weakest_axis(curves)
    if weakest:
        curve = curves[weakest]
        name, unit = AXIS_LABEL[weakest]
        print(f"가장 먼저 무너지는 축: {name} — 지금 {curve.baseline:,}{unit}에서 "
              f"{curve.knee.value:,}{unit}까지가 한계다 (여유 {curve.headroom:.1f}배)")
        if curve.breaks_at:
            for reason in curve.breaks_at.reasons:
                print(f"  - {reason}")
    else:
        print("탐색 상한까지 올려도 무너지는 축이 없다 — 이 부하 범위에서는 하드웨어가 문제가 아니다")
    print()

    for axis, curve in curves.items():
        for note in curve.notes:
            print(f"  ! [{AXIS_LABEL[axis][0]}] {note}")

    if args.curve:
        for axis, curve in curves.items():
            name, unit = AXIS_LABEL[axis]
            print()
            print(f"{name} 탐색 경로 ({len(curve.points)}점)")
            print(report._table(
                ["부하", "판정", "평상시 p95", "스톰 소진", "최대 큐", "가동률", "병목"],
                [[f"{p.value:,}{unit}", report.VERDICT_LABEL.get(p.verdict, p.verdict),
                  f"{p.p95_steady_s:.1f}s", f"{p.storm_drain_s / 60:.1f}분",
                  f"{p.max_queue}", f"{p.busy_fraction:.1%}", p.binding_resource]
                 for p in curve.points],
                aligns="rlrrrrl"))

    if args.csv:
        _write_capacity_csv(args.csv, curves)
        print(f"\nCSV 저장: {args.csv}")
    return 0


def _write_capacity_csv(path: Path, curves: dict) -> None:
    import csv

    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(["axis", "value", "verdict", "p95_steady_s", "storm_drain_s",
                         "max_queue", "busy_fraction", "binding_resource",
                         "ram_needed_gb", "reasons"])
        for axis, curve in curves.items():
            for p in curve.points:
                writer.writerow([axis, p.value, p.verdict, f"{p.p95_steady_s:.3f}",
                                 f"{p.storm_drain_s:.3f}", p.max_queue,
                                 f"{p.busy_fraction:.4f}", p.binding_resource,
                                 p.ram_needed_gb, " | ".join(p.reasons)])


# --------------------------------------------------------------------------
# mock / serve
#
# `serve` is the closest this tool gets to running the thing it sizes: real
# alarm records go in, a queue forms, slots pick them up, cards come out the
# other end. What it does *not* do is run a model -- service times come from
# the same analytical prediction `recommend` uses. So it exercises the shape of
# the system (queueing, back-pressure, storm drain, per-alarm journeys) without
# loading this machine, which is the boundary the whole tool is built on.
# --------------------------------------------------------------------------


def _cmd_mock(args) -> int:
    from .mockdata import generate_range, to_csv, to_jsonl

    try:
        size_s, window_s = args.storm.split("/")
        storm_size, storm_window = int(size_s), float(window_s)
    except ValueError:
        raise SystemExit(f"--storm 형식은 '개수/초' 이다 (받은 값: {args.storm!r})")

    days = generate_range(
        args.date, max(args.days, 1), count=args.count, seed=args.seed,
        storm_size=storm_size, storms_per_day=args.storms_per_day,
        storm_window_s=storm_window,
    )
    total = sum(len(d.alarms) for d in days)
    print(f"목데이터 {len(days)}일치 · 총 {total:,}건")
    print(report._table(
        ["날짜", "건수", "스톰", "심각도 분포"],
        [[d.date, f"{len(d.alarms)}", f"{d.storms}회",
          " ".join(f"{s} {sum(1 for a in d.alarms if a.severity == s)}"
                   for s in ("critical", "major", "minor", "warning"))]
         for d in days], aligns="lrrl"))

    if args.show:
        print()
        print(f"표본 {args.show}건")
        first = days[0]
        for a in first.alarms[:args.show]:
            hh, mm, ss = int(a.at_s // 3600), int(a.at_s % 3600 // 60), int(a.at_s % 60)
            tag = f" [스톰{a.storm_id}]" if a.storm_id is not None else ""
            child = " ↳파생" if a.parent_id else ""
            print(f"  {hh:02d}:{mm:02d}:{ss:02d}  {a.severity:<8} {a.code:<12} "
                  f"{a.device:<16} {a.message}{tag}{child}")

    for note in days[0].notes:
        print(f"  ! {note}")

    if args.out:
        if args.out.suffix == ".csv":
            # CSV has no place for the day-level notes, and those notes are the
            # only record of which numbers are measured and which are assumed.
            # Writing it is fine; letting it leave silently is not.
            args.out.write_text(to_csv(days[0]), encoding="utf-8-sig")
            print(f"\n저장: {args.out} (1일치 {len(days[0].alarms):,}건, CSV)")
            print("  ! CSV에는 위 가정 기록(notes)이 담기지 않고 다시 읽을 수도 없다. "
                  "파이프라인에 넣거나 보관하려면 .jsonl 로 저장하세요.")
            if len(days) > 1:
                print(f"  ! CSV는 첫날만 저장했다 ({len(days)}일치를 요청했다).")
        else:
            args.out.write_text("".join(to_jsonl(d) for d in days), encoding="utf-8")
            print(f"\n저장: {args.out} ({len(days)}일치 {total:,}건, JSONL)")
    return 0


def _cmd_serve(args) -> int:
    from .mockdata import from_jsonl, generate_day
    from .pipeline import TeamsSink, build_service_model, run_pipeline

    cat = _catalog(args)
    model = cat.model(args.model)
    quant = cat.quant(args.quant)
    cpu = cat.cpu(args.cpu)
    memory = cat.memory_for(cpu, args.dpc)
    workload = _workload_from(args)
    eff = _efficiency(cat)

    if args.alarms:
        day = from_jsonl(args.alarms.read_text(encoding="utf-8"))
        source = str(args.alarms)
    else:
        day = generate_day(
            args.date, count=args.alarms_per_day if args.alarms_per_day != 150 else None,
            seed=args.seed, storm_size=workload.storm_size,
            storms_per_day=workload.storms_per_day,
            storm_window_s=workload.storm_window_s,
        )
        source = f"생성 ({day.date})"

    service = build_service_model(model, quant, cpu, memory, eff, workload, args.sockets)
    sink = TeamsSink()
    seen: list[dict] = []
    stats, deliveries = run_pipeline(
        day.alarms, service, workload, sink=sink, speed=args.speed,
        queue_limit=args.queue_limit,
        on_event=(lambda ev: seen.append(ev)) if args.trace else None,
    )

    mode = "가상시간" if args.speed <= 0 else f"실시간 {args.speed:g}배속"
    print(f"모델    {model.name}  {model.params_b:.2f}B  {quant.id}")
    print(f"서버    {cpu.vendor} {cpu.model}  {cpu.cores * args.sockets}코어 "
          f"{args.sockets}소켓  {cpu.mem_channels * args.sockets}ch "
          f"{memory.ddr_gen}-{memory.effective_mts}  ·  {workload.slots}슬롯")
    print(f"알람    {source} · {len(day.alarms):,}건 · 스톰 {day.storms}회")
    print(f"모드    {mode}"
          + (f" (실제 소요 {stats.wall_clock_s:.2f}초)" if stats.wall_clock_s > 0.01 else ""))
    print()

    if args.trace:
        print(f"처리 로그 (앞 {args.trace}건)")
        for ev in seen[:args.trace]:
            print(f"  t={ev['t']:>9.2f}s  {ev['phase']:<8} {ev['alarm_id']:<22} "
                  f"큐 {ev['queue']:>3}  처리중 {ev['active']}")
        print()

    print("실행 결과")
    print(report._table(["지표", "값", "판정 기준"], [
        ["수신", f"{stats.received:,}건", ""],
        ["Teams 전달", f"{stats.delivered:,}건", f"싱크 수신 {len(sink.sent):,}건"],
        ["버림", f"{stats.dropped:,}건", "큐 상한 초과" if stats.dropped else "없음"],
        ["평상시 p95", f"{stats.p95_steady_s:.1f} s",
         f"SLA {workload.sla_seconds:.0f}s {'충족' if stats.sla_met else '초과'}"],
        ["스톰 소진", f"{stats.storm_drain_s / 60:.2f} 분",
         f"목표 {workload.storm_drain_sla_s / 60:.0f}분 "
         f"{'충족' if stats.storm_sla_met else '초과'}"],
        ["전체 p50 / p99", f"{stats.p50_s:.1f} / {stats.p99_s:.1f} s", "스톰 대기 포함"],
        ["최대 / 평균 큐", f"{stats.max_queue} / {stats.mean_queue:.1f}", ""],
        ["가동률", f"{stats.busy_fraction:.2%}", "하루 중 일한 시간"],
        ["슬롯 점유율", f"{stats.slot_utilisation:.1%}", ""],
        ["처리 토큰", f"{stats.tokens_prefill:,} + {stats.tokens_generated:,}",
         "prefill + 생성"],
    ], aligns="lrl"))

    if deliveries:
        worst = sorted((d for d in deliveries if not d.dropped),
                       key=lambda d: -d.total_s)[:5]
        print()
        print("가장 오래 걸린 5건")
        # TTFT is measured from arrival, the way a client experiences it, so it
        # already contains the queue wait. Showing both beside a "합계" column
        # would read as a sum that does not add up, so the prompt-processing
        # time is broken back out instead.
        print(report._table(
            ["알람", "심각도", "큐 대기", "프롬프트", "생성", "전송", "합계", "스톰"],
            [[d.alarm_id, d.severity, f"{d.queue_wait_s:.1f}s",
              f"{d.ttft_s - d.queue_wait_s:.1f}s",
              f"{d.generate_s:.1f}s", f"{d.deliver_s:.2f}s", f"{d.total_s:.1f}s",
              str(d.storm_id) if d.storm_id is not None else "—"] for d in worst],
            aligns="llrrrrrl"))
        print("  큐 대기 + 프롬프트 + 생성 + 전송 = 합계. "
              "(첫 토큰까지의 시간 TTFT는 도착 기준이라 큐 대기를 포함한다)")

    if args.csv:
        _write_deliveries_csv(args.csv, deliveries)
        print(f"\n건별 기록 저장: {args.csv} ({len(deliveries):,}행)")
    return 0


def _write_deliveries_csv(path: Path, deliveries: list) -> None:
    import csv

    fields = ["alarm_id", "severity", "storm_id", "arrived_s", "started_s",
              "first_token_s", "generated_s", "delivered_s", "slot",
              "prompt_tokens", "output_tokens", "queue_wait_s", "ttft_s",
              "generate_s", "deliver_s", "total_s", "dropped"]
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(fields)
        for d in deliveries:
            writer.writerow([getattr(d, f) for f in fields])


# --------------------------------------------------------------------------
# lab / bench
#
# The lab is where a build stops being a row in a table and becomes a machine
# somebody could order: a specific CPU, a specific number of DIMMs of a
# specific size. That distinction matters because the most expensive mistake in
# this domain is invisible in a capacity column -- two 64 GB DIMMs in an
# eight-channel board have the right capacity and a quarter of the bandwidth.
# --------------------------------------------------------------------------

LEVEL_MARK = {"error": "✗", "warn": "!", "info": "·"}


def _print_assembly(asm) -> None:
    """The bill of materials, then everything wrong with it."""
    vm = asm.vm
    print(f"머신    {vm.name}")
    print(f"CPU     {asm.cpu.vendor} {asm.cpu.model}  "
          f"{asm.cpu.cores * vm.sockets}코어 {vm.sockets}소켓  "
          f"({asm.channels_total}채널 {asm.memory.ddr_gen})")
    print(f"DIMM    {vm.dimm_count} × {vm.dimm_gb}GB {asm.memory.ddr_gen}-"
          f"{asm.memory.effective_mts}  =  {asm.ram_total_gb}GB  "
          f"({asm.channels_populated}/{asm.channels_total}채널 장착, "
          f"채널당 {asm.dimms_per_channel}장)")
    print(f"모델    {asm.model.name}  {asm.quant.id}  {vm.slots}슬롯  "
          f"(실사용 {asm.ram_used_gb:.1f}GB)")
    print()

    lost = asm.bandwidth_full_gbs - asm.bandwidth_gbs
    print(report._table(["항목", "이 구성", "전 채널 장착", "차이"], [
        ["실효 대역폭", f"{asm.bandwidth_gbs:.1f} GB/s",
         f"{asm.bandwidth_full_gbs:.1f} GB/s",
         f"-{lost / asm.bandwidth_full_gbs:.0%}" if lost > 0.05 else "—"],
        ["decode (1슬롯)", f"{asm.decode_tps_single:.1f} tok/s",
         f"{asm.decode_tps_full:.1f} tok/s",
         f"-{1 - asm.decode_tps_single / asm.decode_tps_full:.0%}"
         if asm.decode_tps_full and asm.decode_tps_single < asm.decode_tps_full * 0.99
         else "—"],
        ["prefill", f"{asm.prefill_tps:.0f} tok/s", "", ""],
        ["불확실도", f"±{asm.uncertainty:.0%}", "", ""],
    ], aligns="lrrr"))

    if asm.findings:
        print()
        print("검증 결과")
        for f in asm.findings:
            print(f"  {LEVEL_MARK.get(f.level, '·')} {f.message}")
            if f.remedy:
                print(f"      → {f.remedy}")
    else:
        print("\n검증 결과  지적 사항 없음")


def _machine_from_args(cat, args):
    """A VirtualMachine from --machine, or assembled from the flags."""
    from .lab import VirtualMachine, assemble, load

    if getattr(args, "machine", None):
        return load(cat, args.machine)
    missing = [f for f in ("cpu", "model", "dimm", "count")
               if getattr(args, f, None) in (None, "")]
    if missing:
        raise SystemExit(
            "--machine 을 주거나 --cpu/--model/--dimm/--count 를 전부 줘라 "
            f"(빠진 것: {', '.join('--' + m for m in missing)})"
        )
    return assemble(cat, VirtualMachine(
        name=getattr(args, "name", None) or "즉석 조립",
        cpu_id=args.cpu, sockets=args.sockets,
        dimm_gb=args.dimm, dimm_count=args.count,
        model_id=args.model, quant_id=args.quant, slots=args.slots,
    ))


def _cmd_lab_help(args) -> int:
    print("사용법: svrspec lab build ... | svrspec lab show <머신.json>")
    return 2


def _cmd_lab_build(args) -> int:
    from .lab import VirtualMachine, assemble, save

    cat = _catalog(args)
    asm = assemble(cat, VirtualMachine(
        name=args.name, cpu_id=args.cpu, sockets=args.sockets,
        dimm_gb=args.dimm, dimm_count=args.count,
        model_id=args.model, quant_id=args.quant, slots=args.slots,
    ))
    _print_assembly(asm)
    if args.out:
        save(asm, args.out)
        print(f"\n저장: {args.out}")
    # A build with an error is still written -- the operator asked for it and
    # the findings say why it will not work. The exit code is what a script
    # branches on.
    return 0 if asm.ok else 1


def _cmd_lab_show(args) -> int:
    cat = _catalog(args)
    if not args.machine.exists():
        print(f"svrspec: 파일이 없다: {args.machine}", file=sys.stderr)
        return 1
    from .lab import load

    asm = load(cat, args.machine)
    _print_assembly(asm)
    return 0 if asm.ok else 1


def _cmd_bench(args) -> int:
    from .bench import frames_to_csv, run_bench
    from .loadgen import build_load

    cat = _catalog(args)
    asm = _machine_from_args(cat, args)
    workload = _workload_from(args)

    params: dict = {}
    if args.profile == "replay":
        params = dict(date=args.date, storm_size=workload.storm_size,
                      storms_per_day=workload.storms_per_day)
    elif args.profile == "ramp":
        params = dict(start_rate=args.from_rate, end_rate=args.to_rate,
                      hours=args.hours or 24.0)
    elif args.profile == "spike":
        params = dict(base_rate=args.rate, peak_rate=args.peak)
    else:  # soak
        params = dict(rate=args.rate, hours=args.hours or 72.0)

    alarms, profile = build_load(args.profile, seed=args.seed, **params)
    result = run_bench(cat, asm, alarms, profile, workload=workload,
                       frames=max(args.frames, 1), queue_limit=args.queue_limit)

    _print_assembly(asm)
    print()
    print(f"부하    {profile.label}  ·  {profile.total_alarms:,}건  "
          f"·  {profile.span_s / 3600:.0f}시간")
    for note in profile.notes:
        print(f"  ! {note}")
    print()

    if args.live:
        _replay_frames(result)

    s = result.stats
    print("실행 결과")
    print(report._table(["지표", "값", "판정 기준"], [
        ["수신 / 전달", f"{s.received:,} / {s.delivered:,}건",
         f"버림 {s.dropped:,}" if s.dropped else "유실 없음"],
        ["평상시 p95", f"{s.p95_steady_s:.1f} s",
         f"SLA {workload.sla_seconds:.0f}s {'충족' if s.sla_met else '초과'}"],
        ["스톰 소진", f"{s.storm_drain_s / 60:.2f} 분",
         f"목표 {workload.storm_drain_sla_s / 60:.0f}분 "
         f"{'충족' if s.storm_sla_met else '초과'}"],
        ["전체 p50 / p99", f"{s.p50_s:.1f} / {s.p99_s:.1f} s", "큐 대기 포함"],
        ["최대 / 평균 큐", f"{s.max_queue} / {s.mean_queue:.1f}", ""],
        ["가동률", f"{s.busy_fraction:.2%}", "구간 중 일한 시간"],
        ["처리 토큰", f"{s.tokens_prefill:,} + {s.tokens_generated:,}", "prefill + 생성"],
    ], aligns="lrl"))

    if result.breach:
        b = result.breach
        print()
        print(f"무너진 지점  {b['t_s'] / 3600:.1f}시간 지점, "
              f"부하 {b['offered_rate']:,.0f}건/일에서 p95 {b['p95_s']:.1f}초로 "
              f"SLA {workload.sla_seconds:.0f}초를 넘겼다")
    elif profile.kind == "ramp":
        print(f"\n무너진 지점  없음 — 최대 부하까지 SLA를 유지했다")

    if args.csv:
        args.csv.write_text(frames_to_csv(result), encoding="utf-8-sig")
        print(f"\n프레임 CSV 저장: {args.csv} ({len(result.frames):,}행)")
    return 0


def _replay_frames(result) -> None:
    """The run as four sparklines. The terminal's answer to the live gauges."""
    frames = result.frames
    if not frames:
        return
    rows = [
        ("CPU 가동", [f.cpu_pct for f in frames], "%"),
        ("대역폭", [f.bw_pct for f in frames], "%"),
        ("대기 큐", [float(f.queued) for f in frames], ""),
        ("도착", [float(f.arrived) for f in frames], "건"),
        ("p95 누적", [f.p95_so_far_s for f in frames], "s"),
    ]
    width = max(_display_width(n) for n, *_ in rows)
    span_h = (frames[-1].t_s + (frames[1].t_s - frames[0].t_s)) / 3600 if len(frames) > 1 else 0
    print(f"실행 재생 ({len(frames)}프레임 · {span_h:.0f}시간)")
    for name, values, unit in rows:
        pad = " " * (width - _display_width(name))
        top = max(values) if values else 0
        print(f"  {name}{pad}  {_sparkline(values)}  최대 {top:,.1f}{unit}")
    print(f"  {' ' * (width + 2)}0h{' ' * max(len(frames) - 5, 0)}{span_h:.0f}h")
    print()


# --------------------------------------------------------------------------
# calibrate
#
# The only path by which a measurement enters this tool. It reads a log
# somebody else produced; it never generates load. That is not squeamishness --
# the server being sized is not this machine, so benchmarking this one would
# answer a different question. See the module docstring of `measured.py`.
# --------------------------------------------------------------------------


def _cmd_calibrate(args) -> int:
    from .measured import (
        compare_to_prediction,
        derive_eta_bw,
        derive_eta_compute,
        parse_llama_bench,
        parse_memory,
        parse_server_log,
    )
    from .perf import predict_throughput

    cat = _catalog(args)
    model = cat.model(args.model)
    quant = cat.quant(args.quant)
    cpu = cat.cpu(args.cpu)
    memory = cat.memory_for(cpu, args.dpc)
    eff = _efficiency(cat)

    points, memories = [], []
    for path in args.log:
        if not path.exists():
            print(f"svrspec: 파일이 없다: {path}", file=sys.stderr)
            return 1
        text = path.read_text(encoding="utf-8", errors="replace")
        found = parse_llama_bench(text, source=str(path)) or parse_server_log(
            text, source=str(path))
        points.extend(found)
        mem = parse_memory(text, source=str(path))
        if any((mem.model_size_mib, mem.kv_self_mib, mem.compute_buffer_mib)):
            memories.append(mem)

    if not points:
        print("측정값을 하나도 읽지 못했다. llama-bench 표/JSON 또는 llama-server 로그인지 "
              "확인해 주세요.", file=sys.stderr)
        return 1

    print(f"모델    {model.name}  {model.params_b:.2f}B  {quant.id}")
    print(f"서버    {cpu.vendor} {cpu.model}  {cpu.cores * args.sockets}코어 "
          f"{args.sockets}소켓  {cpu.mem_channels * args.sockets}ch "
          f"{memory.ddr_gen}-{memory.effective_mts}")
    print()
    print(f"읽은 측정값 {len(points)}건")
    print(report._table(
        ["출처", "구간", "tok/s", "편차", "ctx", "스레드", "모델 라벨"],
        [[Path(p.source).name, p.kind, f"{p.tokens_per_s:.2f}",
          f"±{p.stddev:.2f}" if p.stddev else "—", str(p.n_ctx or "—"),
          str(p.n_threads or "—"), p.model_label[:28]] for p in points],
        aligns="llrrrrl"))
    print()

    # Predicted vs measured, before anything is changed. The error is the
    # reason to calibrate, so it has to be visible first.
    tokens = TokenProfile()
    prediction = predict_throughput(
        model, quant, cpu, memory, tokens, eff, slots=1, sockets=args.sockets)
    print(f"예측 대조 (현재 카탈로그 계수, 불확실도 ±{prediction.uncertainty:.0%})")
    rows = []
    for p in points:
        c = compare_to_prediction(p, prediction)
        rows.append([p.kind, f"{c['measured_tps']:.2f}", f"{c['predicted_tps']:.2f}",
                     f"{c['error_pct']:+.1f}%",
                     "오차범위 안" if c["within_uncertainty"] else "오차범위 밖",
                     c["verdict"]])
    print(report._table(["구간", "실측 tok/s", "예측 tok/s", "오차", "판정", ""],
                        rows, aligns="lrrrll"))
    print()

    derived = []
    for p in points:
        try:
            if p.kind == "tg":
                cal = derive_eta_bw(p, model, quant, cpu, memory,
                                    sockets=args.sockets,
                                    previous=_previous(eff, "eta_bw", memory.ddr_gen),
                                    confidence=args.confidence)
            else:
                from .perf import widest_isa

                cal = derive_eta_compute(p, model, cpu, sockets=args.sockets,
                                         previous=_previous(eff, "eta_compute",
                                                            widest_isa(cpu)),
                                         confidence=args.confidence)
        except ValueError as exc:
            print(f"  건너뜀 [{p.kind} {p.tokens_per_s:.2f} tok/s]: {exc}")
            continue
        derived.append(cal)

    if not derived:
        print("유도할 수 있는 계수가 없다.", file=sys.stderr)
        return 1

    print(f"유도한 계수 {len(derived)}건")
    print(report._table(
        ["계수", "기존", "실측", "변화", "근거 수준"],
        [[c.coefficient.id,
          f"{c.previous_value:.3f}" if c.previous_value is not None else "—",
          f"{c.coefficient.value:.3f}",
          f"{c.change_pct:+.1f}%" if c.change_pct is not None else "—",
          report.CONFIDENCE_LABEL.get(c.coefficient.confidence,
                                      c.coefficient.confidence)]
         for c in derived],
        aligns="lrrrl"))
    print()
    for c in derived:
        print(f"  {c.coefficient.id}: {c.basis}")

    if memories:
        print()
        print("로그가 보고한 메모리 (svrspec verify가 대조하는 그 줄들)")
        print(report._table(
            ["출처", "모델", "KV", "컴퓨트 버퍼", "n_ctx", "슬롯"],
            [[Path(m.source).name,
              f"{m.model_size_mib:.0f} MiB" if m.model_size_mib else "—",
              f"{m.kv_self_mib:.0f} MiB" if m.kv_self_mib else "—",
              f"{m.compute_buffer_mib:.0f} MiB" if m.compute_buffer_mib else "—",
              str(m.n_ctx or "—"), str(m.n_slots or "—")] for m in memories],
            aligns="lrrrrr"))

    if args.out:
        _write_coefficients(args.out, derived)
        print(f"\n계수 저장: {args.out}")
        print("  카탈로그에 반영하려면 이 파일의 항목을 "
              "svrspec/catalog/coefficients.json 에 병합하세요 — "
              "덮어쓰기는 되돌리기 어려우므로 자동으로 하지 않는다.")
    return 0


def _previous(eff: Efficiency, kind: str, key: str):
    """The catalogue value this derivation would replace, if there is one."""
    try:
        return eff.get(kind, key)
    except KeyError:
        return None


def _write_coefficients(path: Path, calibrations: list) -> None:
    """Write a catalog-shaped coefficients file. Never merges in place.

    Merging would rewrite the shipped catalogue from a single log, and one
    log is one machine on one day. The operator decides what to promote.
    """
    import json

    payload = {
        "schema": "coefficients/v1",
        "entries": [dataclasses.asdict(c.coefficient) for c in calibrations],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


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
    print("  직접 대조할 수 있다. 5% 넘게 벌어지면 memory.py 의 ACTIVATION_TENSORS 를 조정하세요.")

    if failures:
        print(f"\n{failures}개 파일에서 5% 초과 오차. quants.json 의 bits_per_weight 를 보정하세요.")
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


def _cmd_selfcheck(args) -> int:
    """Can this install open its own window? Answered from inside the install.

    Checking that `webview/js/api.js` exists at the path the build *put* it is
    not the same question as whether pywebview will *find* it: pywebview globs
    `Path(webview.__file__).parent / "js"`, and if the packager left the module
    inside library.zip that path points into the archive and matches nothing.
    A shipped build failed exactly there, and a build-time file listing passed
    it. So resolve the directory the way pywebview does, in the environment
    that will actually run it.
    """
    from . import __version__

    print(f"svrspec {__version__}")
    print(f"  실행 방식: {'패키징된 앱' if getattr(sys, 'frozen', False) else '소스'}")

    ok = True
    try:
        import webview
    except ImportError as exc:
        print(f"  pywebview: 없음 ({exc}) — 데스크톱 창을 띄울 수 없다")
        print("  서버 방식(svrspec gui)은 영향받지 않는다")
        return 1

    print(f"  pywebview: {getattr(webview, '__version__', '버전 불명')}")
    js_dir = Path(webview.__file__).parent / "js"
    found = sorted(p.name for p in js_dir.rglob("*.js"))
    print(f"  브리지 JS 경로: {js_dir}")
    if not found:
        print("  브리지 JS: 하나도 없음 — 창은 열리지만 아무것도 동작하지 않는다")
        ok = False
    else:
        missing = {"api.js", "finish.js"} - set(found)
        print(f"  브리지 JS: {len(found)}개 {found}")
        if missing:
            print(f"  빠진 파일: {sorted(missing)} — 브리지가 뜨지 않는다")
            ok = False

    try:
        cat = Catalog()
        print(f"  카탈로그: 모델 {len(cat.models)} · CPU {len(cat.cpus)} · "
              f"메모리 {len(cat.memory)} · 계수 {len(cat.coefficients)}")
    except CatalogError as exc:
        print(f"  카탈로그: 읽기 실패 ({exc})")
        ok = False

    if ok:
        print("\n정상 — 데스크톱 창이 뜬다.")
    else:
        print("\n문제 있음. 다만 창이 죽어도 앱은 로컬 서버로 자동 전환한다 — "
              "쓰지 못하게 되지는 않는다.")
    return 0 if ok else 1


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
