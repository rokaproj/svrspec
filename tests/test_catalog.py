"""Catalogue loading. A silently-dropped or mistyped row would corrupt a
delivery-grade sizing report, so every failure here must be loud."""

import json
from pathlib import Path

import pytest

from svrspec.catalog import Catalog, CatalogError, load_cpus, load_models

DATA = Path(__file__).parent / "data"


def _write(tmp_path: Path, name: str, doc: dict) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def test_loads_the_fixture_catalogue(catalog):
    assert len(catalog.models) == 3
    assert len(catalog.cpus) == 4
    assert "models" in catalog.summary()


def test_lookup_by_id(catalog):
    assert catalog.model("test-3b").params_b == 3.09
    assert catalog.cpu("test-amx-8ch").has("amx-bf16")


def test_unknown_id_suggests_near_matches(catalog):
    with pytest.raises(CatalogError, match="Did you mean"):
        catalog.model("test-3")


def test_rejects_unknown_field(tmp_path):
    path = _write(tmp_path, "models.json", {
        "schema": "models/v1",
        "entries": [{
            "id": "x", "name": "X", "family": "F", "params_b": 1.0, "n_layer": 2,
            "n_embd": 64, "n_head": 4, "n_kv_head": 4, "n_vocab": 100,
            "ctx_train": 512, "typo_field": 1,
        }],
    })
    with pytest.raises(CatalogError, match="unknown field"):
        load_models(path)


def test_rejects_missing_required_field(tmp_path):
    path = _write(tmp_path, "models.json", {
        "schema": "models/v1",
        "entries": [{"id": "x", "name": "X", "family": "F", "params_b": 1.0}],
    })
    with pytest.raises(CatalogError, match="missing required field"):
        load_models(path)


def test_rejects_wrong_field_type(tmp_path):
    path = _write(tmp_path, "models.json", {
        "schema": "models/v1",
        "entries": [{
            "id": "x", "name": "X", "family": "F", "params_b": "1.0", "n_layer": 2,
            "n_embd": 64, "n_head": 4, "n_kv_head": 4, "n_vocab": 100,
            "ctx_train": 512,
        }],
    })
    with pytest.raises(CatalogError, match="expected float"):
        load_models(path)


def test_rejects_duplicate_ids(tmp_path):
    row = {
        "id": "dup", "name": "X", "family": "F", "params_b": 1.0, "n_layer": 2,
        "n_embd": 64, "n_head": 4, "n_kv_head": 4, "n_vocab": 100, "ctx_train": 512,
    }
    path = _write(tmp_path, "models.json", {"schema": "models/v1", "entries": [row, dict(row)]})
    with pytest.raises(CatalogError, match="duplicate id"):
        load_models(path)


def test_rejects_wrong_schema_version(tmp_path):
    path = _write(tmp_path, "models.json", {"schema": "models/v2", "entries": [{}]})
    with pytest.raises(CatalogError, match="schema must be"):
        load_models(path)


def test_claimed_source_requires_a_url(tmp_path):
    path = _write(tmp_path, "models.json", {
        "schema": "models/v1",
        "entries": [{
            "id": "x", "name": "X", "family": "F", "params_b": 1.0, "n_layer": 2,
            "n_embd": 64, "n_head": 4, "n_kv_head": 4, "n_vocab": 100,
            "ctx_train": 512, "source": "model_card",
        }],
    })
    with pytest.raises(CatalogError, match="requires a source_url"):
        load_models(path)


def test_rejects_impossible_gqa(tmp_path):
    path = _write(tmp_path, "models.json", {
        "schema": "models/v1",
        "entries": [{
            "id": "x", "name": "X", "family": "F", "params_b": 1.0, "n_layer": 2,
            "n_embd": 64, "n_head": 4, "n_kv_head": 8, "n_vocab": 100, "ctx_train": 512,
        }],
    })
    with pytest.raises(CatalogError, match="n_kv_head"):
        load_models(path)


def test_rejects_unknown_isa_token(tmp_path):
    path = _write(tmp_path, "cpus.json", {
        "schema": "cpus/v1",
        "entries": [{
            "id": "x", "vendor": "Intel", "family": "F", "model": "M", "cores": 8,
            "threads": 16, "base_ghz": 2.0, "all_core_turbo_ghz": 2.5,
            "max_turbo_ghz": 3.0, "isa": ["avx2", "avx1024"], "mem_channels": 8,
            "ddr_gen": "DDR5", "max_ddr_mts": 4800, "max_mem_gb": 1024,
            "sockets_max": 2, "l3_mb": 32.0, "tdp_w": 200,
        }],
    })
    with pytest.raises(CatalogError, match="unknown isa"):
        load_cpus(path)


def test_rejects_inverted_clock_ordering(tmp_path):
    path = _write(tmp_path, "cpus.json", {
        "schema": "cpus/v1",
        "entries": [{
            "id": "x", "vendor": "Intel", "family": "F", "model": "M", "cores": 8,
            "threads": 16, "base_ghz": 3.0, "all_core_turbo_ghz": 2.5,
            "max_turbo_ghz": 3.0, "isa": ["avx2"], "mem_channels": 8,
            "ddr_gen": "DDR5", "max_ddr_mts": 4800, "max_mem_gb": 1024,
            "sockets_max": 2, "l3_mb": 32.0, "tdp_w": 200,
        }],
    })
    with pytest.raises(CatalogError, match="all_core_turbo"):
        load_cpus(path)


def test_memory_matching_picks_the_fastest_supported(catalog):
    cpu = catalog.cpu("test-avx512-8ch")
    one_dpc = catalog.memory_for(cpu, 1)
    two_dpc = catalog.memory_for(cpu, 2)
    assert one_dpc.effective_mts == 4800
    assert two_dpc.effective_mts == 4400  # the DDR5 2 DPC derate


def test_memory_matching_respects_the_cpu_ddr_limit(catalog):
    cpu = catalog.cpu("test-desktop-2ch")
    assert catalog.memory_for(cpu, 1).ddr_gen == "DDR4"


def test_unverified_rows_are_reported(catalog):
    unverified = catalog.unverified()
    assert unverified  # the fixtures are all unverified on purpose
    assert all(kind in ("model", "cpu", "memory") for kind, _ in unverified)


def test_shipped_catalogue_is_valid_if_present():
    """The real catalogue is filled in by the data owners; validate it once it
    exists rather than failing the suite while it is still being collected."""
    shipped = Path(__file__).resolve().parent.parent / "svrspec" / "catalog"
    if not (shipped / "models.json").exists() or not (shipped / "cpus.json").exists():
        pytest.skip("shipped catalogue not populated yet")
    cat = Catalog(shipped)
    assert cat.models and cat.cpus and cat.memory and cat.quants
    for cpu in cat.cpus:
        cat.memory_for(cpu, 1)  # every CPU must have a usable memory option


def test_the_catalogue_carries_the_small_dimms_that_fill_a_board_cheaply():
    """8 and 16 GB RDIMMs exist, and leaving them out distorted the advice.

    Bandwidth comes from filling every channel, not from capacity. With only
    32 and 64 GB modules catalogued, the only way to populate an eight-channel
    board was 256 GB -- so the tool told people to buy four times the memory
    they needed. 8x8GB fills the same board for 64 GB at the same speed.
    """
    from svrspec.catalog import Catalog

    catalog = Catalog()
    ddr5_1dpc = [
        m for m in catalog.memory
        if m.ddr_gen == "DDR5" and m.dimms_per_channel == 1
    ]
    sizes = {m.dimm_gb for m in ddr5_1dpc}
    assert {8, 16, 32, 64} <= sizes

    # Capacity must not change the clock at a given grade and population.
    by_grade: dict[int, set[int]] = {}
    for m in ddr5_1dpc:
        by_grade.setdefault(m.rated_mts, set()).add(m.effective_mts)
    for rated, effective in by_grade.items():
        assert len(effective) == 1, (
            f"DDR5-{rated} at 1DPC reports {effective} depending on module size; "
            f"capacity does not change the clock"
        )


def test_every_dimm_size_reaches_the_same_bandwidth_when_the_board_is_full():
    """The whole point: a full board is a full board, whatever the modules."""
    from svrspec.catalog import Catalog
    from svrspec.lab import VirtualMachine, assemble

    catalog = Catalog()
    reference = None
    for size in (8, 16, 32, 64):
        assembly = assemble(catalog, VirtualMachine(
            name="t", cpu_id="xeon-gold-6426y", sockets=1,
            dimm_gb=size, dimm_count=8,
            model_id="qwen2.5-7b-instruct", quant_id="Q4_K_M", slots=4,
        ))
        assert assembly.channels_populated == assembly.channels_total
        assert assembly.ram_total_gb == size * 8
        if reference is None:
            reference = assembly.bandwidth_gbs
        assert assembly.bandwidth_gbs == pytest.approx(reference)
        assert not [f for f in assembly.findings if f.code == "channels-underfilled"]


def test_the_underfilled_remedy_now_offers_a_same_capacity_fix():
    """Two 64 GB DIMMs should be told to become eight 16 GB ones.

    Before the small modules were catalogued this advice had to be "buy 256 GB",
    which is four times the memory for the same result.
    """
    from svrspec.catalog import Catalog
    from svrspec.lab import VirtualMachine, assemble

    catalog = Catalog()
    assembly = assemble(catalog, VirtualMachine(
        name="t", cpu_id="xeon-gold-6426y", sockets=1, dimm_gb=64, dimm_count=2,
        model_id="qwen2.5-7b-instruct", quant_id="Q4_K_M", slots=4,
    ))
    finding = next(f for f in assembly.findings if f.code == "channels-underfilled")
    assert "8" in finding.remedy and "16GB" in finding.remedy
    # The remedy keeps the capacity the operator asked for.
    assert "128GB" in finding.remedy
