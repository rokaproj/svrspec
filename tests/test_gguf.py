"""GGUF header parsing.

Runs against whatever .gguf files exist on the machine; skips cleanly when there
are none, so the suite stays portable.
"""

from pathlib import Path

import pytest

from svrspec.gguf import GgufError, read_gguf, to_model_spec
from svrspec.memory import kv_bytes_per_token

MODEL_DIRS = [Path.home() / "models", Path("/mnt/models")]


def _find_gguf() -> Path | None:
    for d in MODEL_DIRS:
        if d.exists():
            for f in sorted(d.glob("*.gguf")):
                return f
    return None


def test_rejects_a_non_gguf_file(tmp_path):
    bogus = tmp_path / "not.gguf"
    bogus.write_bytes(b"NOPE" + b"\x00" * 64)
    with pytest.raises(GgufError, match="not a GGUF file"):
        read_gguf(bogus)


def test_missing_file_is_reported(tmp_path):
    with pytest.raises(GgufError, match="no such file"):
        read_gguf(tmp_path / "absent.gguf")


def test_reads_a_real_model_header():
    path = _find_gguf()
    if not path:
        pytest.skip("no .gguf available on this machine")

    info = read_gguf(path)
    assert info.architecture
    assert info.param_count > 1e8
    assert info.file_bytes > 1e8
    # A quantised model must land between 2 and 17 bits per weight; anything
    # outside that means the tensor walk went wrong.
    assert 2.0 < info.measured_bpw < 17.0


def test_derived_model_spec_is_self_consistent():
    path = _find_gguf()
    if not path:
        pytest.skip("no .gguf available on this machine")

    spec = to_model_spec(read_gguf(path))
    assert spec.n_layer > 0
    assert spec.n_head >= spec.n_kv_head
    assert spec.n_head % spec.n_kv_head == 0
    assert spec.n_vocab > 1000
    assert spec.kv_head_dim > 0
    assert kv_bytes_per_token(spec) > 0


def test_measured_bpw_agrees_with_the_quant_table():
    """The shipped bits_per_weight must match a real file within 5%.

    This is the check that keeps quants.json from drifting into fiction.
    """
    path = _find_gguf()
    if not path:
        pytest.skip("no .gguf available on this machine")

    from svrspec.catalog import load_quants

    info = read_gguf(path)
    table = {q.id: q for q in load_quants()}
    if info.quant not in table:
        pytest.skip(f"{info.quant} not in the quant table")

    nominal = table[info.quant].bits_per_weight
    error = abs(info.measured_bpw - nominal) / nominal
    assert error < 0.05, (
        f"{path.name}: table says {nominal:.2f} bpw, file measures "
        f"{info.measured_bpw:.2f} ({error:.1%} off)"
    )
