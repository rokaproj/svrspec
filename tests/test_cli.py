"""CLI smoke tests against the fixture catalogue."""

from pathlib import Path

import pytest

from svrspec.cli import main

DATA = Path(__file__).parent / "data"
BASE = ["--catalog-dir", str(DATA)]


def run(args: list[str]) -> int:
    return main(BASE + args)


def test_no_command_prints_help():
    assert main([]) == 2


def test_list_each_catalogue(capsys):
    for what in ("models", "cpus", "memory", "quants", "coefficients"):
        assert run(["list", what]) == 0
        out = capsys.readouterr().out
        assert out.strip()


def test_list_filters(capsys):
    assert run(["list", "models", "--korean"]) == 0
    out = capsys.readouterr().out
    assert "test-3b" in out
    assert "test-8b-gqa" not in out


def test_recommend_produces_a_table_and_tiers(capsys):
    assert run(["recommend", "--model", "test-3b"]) == 0
    out = capsys.readouterr().out
    assert "권장 스펙" in out
    assert "최소 스펙" in out
    assert "p95" in out


def test_recommend_writes_all_three_output_formats(tmp_path, capsys):
    html = tmp_path / "r.html"
    csv = tmp_path / "r.csv"
    js = tmp_path / "r.json"
    assert run([
        "recommend", "--model", "test-3b",
        "--html", str(html), "--csv", str(csv), "--json", str(js),
    ]) == 0
    capsys.readouterr()

    body = html.read_text(encoding="utf-8")
    assert body.startswith("<!doctype html>")
    assert "</html>" in body
    # Self-contained: no external fetches, which a CSP-restricted viewer or an
    # air-gapped reviewer would block anyway.
    for forbidden in ("http://", "https://", "<script src", "@import"):
        assert forbidden not in body

    body = csv.read_text(encoding="utf-8-sig")
    assert "verdict" in body
    assert "p95_steady_s" in body

    import json

    payload = json.loads(js.read_text(encoding="utf-8"))
    assert payload["candidates"]
    assert payload["coefficients"]
    assert set(payload["tiers"]) == {"minimum", "recommended", "comfortable"}


def test_size_shows_the_full_breakdown(capsys):
    assert run(["size", "--model", "test-8b-gqa", "--cpu", "test-amx-8ch"]) == 0
    out = capsys.readouterr().out
    for section in ("메모리 산정", "처리량 예측", "알람 1건 지연", "하루 시뮬레이션", "판정"):
        assert section in out


def test_fit_reports_the_largest_model_that_works(capsys):
    assert run(["fit", "--cpu", "test-amx-8ch"]) == 0
    out = capsys.readouterr().out
    assert "감당하는 가장 큰 모델" in out or "통과하는 모델이 없다" in out


def test_catalog_validate(capsys):
    assert run(["catalog", "validate"]) == 0
    out = capsys.readouterr().out
    assert "정합성 통과" in out
    assert "모델 규모 분포" in out


def test_unknown_model_id_is_an_error(capsys):
    assert run(["recommend", "--model", "does-not-exist"]) == 1
    assert "unknown model id" in capsys.readouterr().err


def test_bad_storm_format_is_rejected():
    with pytest.raises(SystemExit):
        run(["recommend", "--model", "test-3b", "--storm", "forty-in-thirty"])


def test_coefficients_are_listed_with_their_confidence(capsys):
    assert run(["list", "coefficients"]) == 0
    out = capsys.readouterr().out
    assert "eta_bw" in out and "eta_compute" in out
    # Provenance must be visible: an estimate cannot look like a measurement.
    assert "추정" in out and "실측" in out


def test_nothing_in_the_package_can_run_a_benchmark():
    """The simulator is analytical. Nothing here launches a process.

    Sizing a customer's server must never depend on flogging whatever machine
    this happens to run on -- the server being sized is somewhere else, so a
    local benchmark answers a different question.

    The original guard forbade the *string* "llama-bench" in cli.py. That was a
    proxy for "no benchmark runner", and it stopped working the moment
    `calibrate` had to name the tool whose log it reads. So the guard now
    checks what actually matters -- the ability to spawn a process -- and
    checks it across the whole package rather than one file.

    `desktop.py` and `gui.py` are exempt: launching a browser or a native
    window is what they are for.
    """
    import svrspec

    package = Path(svrspec.__file__).parent
    exempt = {"desktop.py", "gui.py", "update.py"}
    execution = ("subprocess", "os.system", "os.popen", "os.spawn",
                 "Popen", "pty.spawn", "multiprocessing")

    checked = 0
    for path in sorted(package.rglob("*.py")):
        if path.name in exempt:
            continue
        source = path.read_text(encoding="utf-8")
        for forbidden in execution:
            assert forbidden not in source, f"{path.name} can spawn: {forbidden}"
        checked += 1
    assert checked > 8, "the sweep must actually be reaching the modules"


def test_timeline_shows_resources_moving_over_the_day(capsys):
    """The headline fix: CPU, bandwidth, compute and KV are separate series."""
    assert run([
        "timeline", "--model", "test-8b-gqa", "--cpu", "test-amx-8ch",
        "--alarms-per-day", "300", "--alarm-tokens", "1200",
    ]) == 0
    out = capsys.readouterr().out
    for section in ("하루 리소스 시계열", "병목 진단", "가장 바쁜 구간"):
        assert section in out
    for series in ("CPU 가동", "대역폭 평균", "연산 평균", "KV 실사용", "큐 깊이"):
        assert series in out
    # The distinction the whole module exists to make.
    assert "평균" in out and "피크" in out


def test_timeline_bucket_count_is_honoured(capsys):
    assert run([
        "timeline", "--model", "test-3b", "--cpu", "test-amx-8ch", "--buckets", "24",
    ]) == 0
    assert "60분 x 24버킷" in capsys.readouterr().out


def test_timeline_writes_a_csv_with_every_series(tmp_path, capsys):
    csv_path = tmp_path / "t.csv"
    assert run([
        "timeline", "--model", "test-3b", "--cpu", "test-amx-8ch",
        "--buckets", "12", "--csv", str(csv_path),
    ]) == 0
    capsys.readouterr()

    body = csv_path.read_text(encoding="utf-8-sig")
    header = body.splitlines()[0]
    for column in ("cpu_pct", "bandwidth_avg_pct", "compute_avg_pct", "kv_used_gb",
                   "ram_used_gb", "queued", "arrived", "completed"):
        assert column in header
    assert len(body.strip().splitlines()) == 13  # header + 12 buckets


def test_timeline_pessimistic_differs_from_nominal(capsys):
    """The derated run is a different machine and must not print the same day."""
    run(["timeline", "--model", "test-8b-gqa", "--cpu", "test-amx-8ch",
         "--alarms-per-day", "300"])
    nominal = capsys.readouterr().out
    run(["timeline", "--model", "test-8b-gqa", "--cpu", "test-amx-8ch",
         "--alarms-per-day", "300", "--pessimistic"])
    pessimistic = capsys.readouterr().out
    assert "명목 예측 기준" in nominal
    assert "불리한 추정 기준" in pessimistic
    assert nominal != pessimistic


def test_capacity_reports_a_knee_and_what_broke(capsys):
    assert run([
        "capacity", "--model", "test-8b-gqa", "--cpu", "test-amx-8ch",
        "--axis", "storm", "--alarms-per-day", "150",
    ]) == 0
    out = capsys.readouterr().out
    assert "과부하 지점" in out
    assert "무릎" in out
    assert "한계 요인" in out
    # Either it found a breaking point or it said it could not.
    assert "가장 먼저 무너지는 축" in out or "무너지는 축이 없다" in out


def test_capacity_curve_lists_every_probe(capsys):
    assert run([
        "capacity", "--model", "test-8b-gqa", "--cpu", "test-amx-8ch",
        "--axis", "prompt", "--curve",
    ]) == 0
    out = capsys.readouterr().out
    assert "탐색 경로" in out
    assert "평상시 p95" in out


def test_capacity_writes_a_csv(tmp_path, capsys):
    csv_path = tmp_path / "c.csv"
    assert run([
        "capacity", "--model", "test-3b", "--cpu", "test-amx-8ch",
        "--axis", "storm", "--csv", str(csv_path),
    ]) == 0
    capsys.readouterr()

    body = csv_path.read_text(encoding="utf-8-sig")
    assert "axis,value,verdict" in body.splitlines()[0]
    assert len(body.strip().splitlines()) > 1


def test_hangul_labels_are_measured_in_terminal_columns():
    """Korean labels are double-width; len() would misalign every column."""
    from svrspec.cli import _display_width

    # "CPU" + space = 4 cells, "가동" = 2 double-width chars = 4 cells.
    assert _display_width("CPU 가동") == 8
    assert len("CPU 가동") == 6, "len() undercounts, which is the bug"
    assert _display_width("abc") == 3


def test_sparkline_scales_to_its_own_series():
    from svrspec.cli import SPARK, _sparkline

    assert _sparkline([]) == ""
    # An all-zero series must not divide by zero, and must draw as empty.
    assert _sparkline([0.0, 0.0, 0.0]) == SPARK[0] * 3
    drawn = _sparkline([0.0, 50.0, 100.0])
    assert drawn[0] == SPARK[0] and drawn[-1] == SPARK[-1]
    # Against an explicit ceiling the same series reads lower.
    assert _sparkline([0.0, 50.0, 100.0], ceiling=1000.0)[-1] != SPARK[-1]


def test_bundle_is_self_contained(tmp_path, capsys):
    out_zip = tmp_path / "b.zip"
    assert run(["bundle", "--out", str(out_zip)]) == 0
    capsys.readouterr()

    import zipfile

    with zipfile.ZipFile(out_zip) as z:
        names = z.namelist()
    assert "RUN.txt" in names
    assert any(n.endswith("svrspec/cli.py") for n in names)
    assert any(n.endswith("catalog/quants.json") for n in names)
    assert not any("__pycache__" in n for n in names)


def test_workload_flags_change_the_answer(capsys):
    run(["size", "--model", "test-8b-gqa", "--cpu", "test-desktop-2ch", "--sla", "30"])
    lenient = capsys.readouterr().out
    run(["size", "--model", "test-8b-gqa", "--cpu", "test-desktop-2ch", "--sla", "3"])
    strict = capsys.readouterr().out
    assert lenient != strict


def test_passmark_audit_finds_a_transcription_error(catalog):
    """The one measured number must police the typed-in ones.

    Cores, clocks and channels are transcribed by hand; CPU Mark is a real
    benchmark result. Halving a CPU's core count while leaving its score alone
    is exactly the sort of typo the audit exists to catch.
    """
    from dataclasses import replace

    from svrspec.cli import _passmark_outliers

    clean, median = _passmark_outliers(catalog)
    assert median > 0
    assert clean == []

    broken = replace(catalog.cpus[1], cores=catalog.cpus[1].cores // 4)
    catalog.cpus[1] = broken
    flagged, _ = _passmark_outliers(catalog)
    assert any(broken.id in f for f in flagged)


def test_passmark_audit_is_silent_without_enough_scores(catalog):
    from dataclasses import replace

    from svrspec.cli import _passmark_outliers

    catalog.cpus = [replace(c, passmark_multithread=None) for c in catalog.cpus]
    assert _passmark_outliers(catalog) == ([], 0.0)


def test_catalog_validate_reports_the_audit(capsys):
    assert run(["catalog", "validate"]) == 0
    out = capsys.readouterr().out
    assert "PassMark 교차검증" in out


def test_cli_survives_a_non_utf8_stdout(capsysbinary, monkeypatch):
    """Korean output must not crash a Windows console.

    The bundle command prints "N개 파일"; on a cp1252/cp949 stdout that raised
    UnicodeEncodeError and killed the release build. main() now forces UTF-8, so
    a stream that cannot represent Korean must still not take the process down.
    """
    import io
    import sys

    buffer = io.BytesIO()
    narrow = io.TextIOWrapper(buffer, encoding="cp1252", errors="strict")
    monkeypatch.setattr(sys, "stdout", narrow)

    assert run(["list", "quants"]) == 0
    narrow.flush()
    # reconfigure() switched it to UTF-8, so the Korean header is really there.
    assert "품질".encode() in buffer.getvalue()


def test_bundle_prints_without_encoding_errors(tmp_path, monkeypatch):
    import io
    import sys

    buffer = io.BytesIO()
    narrow = io.TextIOWrapper(buffer, encoding="cp1252", errors="strict")
    monkeypatch.setattr(sys, "stdout", narrow)

    assert run(["bundle", "--out", str(tmp_path / "b.zip")]) == 0
    narrow.flush()
    assert "개 파일".encode() in buffer.getvalue()


MEASURED = Path(__file__).parent / "data" / "measured"


def test_calibrate_reads_a_log_and_derives_coefficients(capsys):
    """The path that turns an estimate into a measurement."""
    assert run([
        "calibrate", str(MEASURED / "llama-bench-devbox.md"),
        "--cpu", "test-desktop-2ch", "--model", "test-8b-gqa",
    ]) == 0
    out = capsys.readouterr().out
    for section in ("읽은 측정값", "예측 대조", "유도한 계수"):
        assert section in out
    assert "근거 수준" in out


def test_calibrate_writes_a_catalog_shaped_file(tmp_path, capsys):
    out_path = tmp_path / "coef.json"
    assert run([
        "calibrate", str(MEASURED / "llama-bench-devbox.md"),
        "--cpu", "test-desktop-2ch", "--model", "test-8b-gqa",
        "--out", str(out_path),
    ]) == 0
    capsys.readouterr()

    import json

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "coefficients/v1"
    assert payload["entries"]
    for entry in payload["entries"]:
        assert entry["source"] == "measurement"
        # The loader refuses a non-unverified row without a source_url.
        assert entry["source_url"]
        assert entry["confidence"] in ("measured", "derived")


def test_calibrate_never_rewrites_the_shipped_catalogue(tmp_path, capsys):
    """One log is one machine on one day. Promotion is the operator's call."""
    shipped = Path("svrspec/catalog/coefficients.json")
    before = shipped.read_bytes()
    assert run([
        "calibrate", str(MEASURED / "llama-bench-devbox.md"),
        "--cpu", "test-desktop-2ch", "--model", "test-8b-gqa",
        "--out", str(tmp_path / "c.json"),
    ]) == 0
    capsys.readouterr()
    assert shipped.read_bytes() == before


def test_calibrate_rejects_a_file_that_is_not_a_log(capsys):
    assert run([
        "calibrate", str(MEASURED / "garbage.txt"),
        "--cpu", "test-desktop-2ch", "--model", "test-8b-gqa",
    ]) == 1
    assert "측정값을 하나도 읽지 못했다" in capsys.readouterr().err


def test_calibrate_reports_a_missing_file(tmp_path, capsys):
    assert run([
        "calibrate", str(tmp_path / "nope.log"),
        "--cpu", "test-desktop-2ch", "--model", "test-8b-gqa",
    ]) == 1
    assert "파일이 없다" in capsys.readouterr().err


def test_mock_reproduces_the_measured_day(capsys):
    """The generator must land on the real figure, not near it."""
    assert run(["mock", "--date", "2026-06-01", "--show", "3"]) == 0
    out = capsys.readouterr().out
    assert "359" in out
    # The assumptions have to travel with the data.
    assert "시간대 분포는 실측이 아니다" in out


def test_mock_writes_jsonl_that_round_trips(tmp_path, capsys):
    path = tmp_path / "alarms.jsonl"
    assert run(["mock", "--date", "2026-06-02", "--out", str(path)]) == 0
    capsys.readouterr()

    from svrspec.mockdata import from_jsonl

    day = from_jsonl(path.read_text(encoding="utf-8"))
    assert len(day.alarms) == 164          # 2026-06-02 measured
    assert day.notes


def test_mock_csv_warns_that_notes_are_lost(tmp_path, capsys):
    """CSV cannot carry the assumptions, so the CLI has to say so out loud."""
    path = tmp_path / "alarms.csv"
    assert run(["mock", "--date", "2026-06-01", "--days", "3", "--out", str(path)]) == 0
    out = capsys.readouterr().out
    assert "notes" in out or "가정 기록" in out
    assert "첫날만" in out
    assert path.read_text(encoding="utf-8-sig").count("\n") == 360  # header + 359


def test_serve_runs_the_whole_pipeline(capsys):
    assert run([
        "serve", "--model", "test-3b", "--cpu", "test-amx-8ch",
        "--date", "2026-06-21", "--slots", "4", "--trace", "3",
    ]) == 0
    out = capsys.readouterr().out
    for section in ("실행 결과", "수신", "Teams 전달", "처리 로그", "가장 오래 걸린"):
        assert section in out
    assert "가상시간" in out


def test_serve_conserves_every_alarm(tmp_path, capsys):
    csv_path = tmp_path / "d.csv"
    assert run([
        "serve", "--model", "test-3b", "--cpu", "test-amx-8ch",
        "--date", "2026-06-21", "--csv", str(csv_path),
    ]) == 0
    capsys.readouterr()

    body = csv_path.read_text(encoding="utf-8-sig").strip().splitlines()
    assert body[0].startswith("alarm_id,severity")
    assert len(body) == 27  # header + 26 alarms on 2026-06-21


def test_serve_accepts_an_alarm_file(tmp_path, capsys):
    src = tmp_path / "in.jsonl"
    assert run(["mock", "--date", "2026-06-21", "--out", str(src)]) == 0
    capsys.readouterr()

    assert run([
        "serve", "--model", "test-3b", "--cpu", "test-amx-8ch", "--alarms", str(src),
    ]) == 0
    out = capsys.readouterr().out
    assert str(src) in out
    assert "26건" in out
