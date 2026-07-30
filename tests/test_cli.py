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


def test_nothing_in_the_cli_loads_the_local_cpu():
    """The simulator is analytical. Nothing in the CLI runs a model.

    Guard against a benchmark path creeping back in: sizing a customer's server
    must never depend on flogging whatever machine this happens to run on.
    """
    import svrspec.cli as cli_module

    source = Path(cli_module.__file__).read_text(encoding="utf-8")
    for forbidden in ("llama-bench", "llama_bench", "subprocess"):
        assert forbidden not in source


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
