"""Catalog loading + validation.

The catalogs are plain JSON so the whole tool stays stdlib-only and therefore
runs unmodified on an air-gapped CPU+RAM server. Each file is a JSON object:

    {"schema": "<name>/v1", "entries": [ {...}, {...} ]}

Every entry is validated against the frozen dataclasses in `types.py` before
anything downstream sees it. A malformed catalog is a hard failure, not a
warning: a sizing report built on a silently-dropped row is worse than no
report.
"""

from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from pathlib import Path
from types import UnionType
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints

from ..types import (
    VALID_CONFIDENCE,
    VALID_SOURCES,
    CoefficientSpec,
    CpuSpec,
    MemoryOption,
    ModelSpec,
    QuantSpec,
)

CATALOG_DIR = Path(__file__).parent

T = TypeVar("T")


class CatalogError(Exception):
    """Raised when a catalog file is malformed or internally inconsistent."""


def _coerce(cls: type[T], row: dict[str, Any], where: str) -> T:
    """Build one frozen dataclass from a JSON object, strictly.

    Unknown keys and missing required keys both fail loudly; a typo'd field
    name would otherwise silently fall back to a default and quietly change
    the sizing result.
    """
    if not is_dataclass(cls):  # pragma: no cover - programming error
        raise TypeError(f"{cls} is not a dataclass")

    spec = {f.name: f for f in fields(cls)}
    annotations = get_type_hints(cls)
    unknown = set(row) - set(spec)
    if unknown:
        raise CatalogError(f"{where}: unknown field(s) {sorted(unknown)}")

    missing = [name for name, f in spec.items() if name not in row and _is_required(f)]
    if missing:
        raise CatalogError(f"{where}: missing required field(s) {sorted(missing)}")

    kwargs: dict[str, Any] = {}
    for name, value in row.items():
        kwargs[name] = _convert(
            annotations.get(name, spec[name].type), value, f"{where}.{name}"
        )
    return cls(**kwargs)  # type: ignore[return-value]


def _is_required(f: Any) -> bool:
    import dataclasses

    return f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING


def _convert(annotation: Any, value: Any, where: str) -> Any:
    """Validate and minimally coerce the JSON shapes used by the catalogues."""
    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin in (Union, UnionType):
        if value is None and type(None) in args:
            return None
        for option in args:
            if option is type(None):
                continue
            try:
                return _convert(option, value, where)
            except CatalogError:
                continue
        raise CatalogError(f"{where}: value has an invalid type")

    if origin is list:
        if not isinstance(value, list):
            raise CatalogError(f"{where}: expected list, got {type(value).__name__}")
        item_type = args[0] if args else Any
        return [_convert(item_type, item, f"{where}[{i}]") for i, item in enumerate(value)]

    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise CatalogError(f"{where}: expected tuple, got {type(value).__name__}")
        if len(args) == 2 and args[1] is Ellipsis:
            types = [args[0]] * len(value)
        else:
            types = list(args)
            if len(types) != len(value):
                raise CatalogError(f"{where}: expected {len(types)} values, got {len(value)}")
        return tuple(
            _convert(kind, item, f"{where}[{i}]")
            for i, (kind, item) in enumerate(zip(types, value))
        )

    if annotation is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise CatalogError(f"{where}: expected int, got {type(value).__name__}")
        return value
    if annotation is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise CatalogError(f"{where}: expected float, got {type(value).__name__}")
        return float(value)
    if annotation is str:
        if not isinstance(value, str):
            raise CatalogError(f"{where}: expected str, got {type(value).__name__}")
        return value
    if annotation is bool:
        if not isinstance(value, bool):
            raise CatalogError(f"{where}: expected bool, got {type(value).__name__}")
        return value
    if annotation is Any:
        return value

    return value


def _load_file(path: Path, schema: str) -> list[dict[str, Any]]:
    if not path.exists():
        raise CatalogError(f"catalog file missing: {path}")
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CatalogError(f"{path.name}: invalid JSON at line {exc.lineno}: {exc.msg}") from exc

    if not isinstance(doc, dict) or "entries" not in doc:
        raise CatalogError(f"{path.name}: expected an object with an 'entries' array")
    if doc.get("schema") != schema:
        raise CatalogError(f"{path.name}: schema must be {schema!r}, got {doc.get('schema')!r}")
    entries = doc["entries"]
    if not isinstance(entries, list) or not entries:
        raise CatalogError(f"{path.name}: 'entries' must be a non-empty array")
    return entries


def _check_ids(items: list[Any], path: Path) -> None:
    seen: set[str] = set()
    for item in items:
        if item.id in seen:
            raise CatalogError(f"{path.name}: duplicate id {item.id!r}")
        seen.add(item.id)


def _check_sources(items: list[Any], path: Path) -> None:
    for item in items:
        if item.source not in VALID_SOURCES:
            raise CatalogError(
                f"{path.name}: {item.id}: source must be one of {VALID_SOURCES}, got {item.source!r}"
            )
        if item.source != "unverified" and not item.source_url:
            raise CatalogError(
                f"{path.name}: {item.id}: source={item.source} requires a source_url"
            )


def load_models(path: Path | None = None) -> list[ModelSpec]:
    path = path or CATALOG_DIR / "models.json"
    rows = _load_file(path, "models/v1")
    items = [_coerce(ModelSpec, r, f"{path.name}[{i}]") for i, r in enumerate(rows)]
    _check_ids(items, path)
    _check_sources(items, path)
    for m in items:
        if m.n_kv_head > m.n_head:
            raise CatalogError(f"{path.name}: {m.id}: n_kv_head {m.n_kv_head} > n_head {m.n_head}")
        if m.n_head % m.n_kv_head:
            raise CatalogError(
                f"{path.name}: {m.id}: n_head {m.n_head} must be a multiple of n_kv_head {m.n_kv_head}"
            )
        if not m.head_dim and m.n_embd % m.n_head:
            raise CatalogError(
                f"{path.name}: {m.id}: n_embd {m.n_embd} not divisible by n_head {m.n_head}; "
                "set head_dim explicitly"
            )
        if m.active_params_b and m.active_params_b > m.params_b:
            raise CatalogError(f"{path.name}: {m.id}: active_params_b exceeds params_b")
    return items


def load_quants(path: Path | None = None) -> list[QuantSpec]:
    path = path or CATALOG_DIR / "quants.json"
    rows = _load_file(path, "quants/v1")
    items = [_coerce(QuantSpec, r, f"{path.name}[{i}]") for i, r in enumerate(rows)]
    _check_ids(items, path)
    for q in items:
        if not 1.0 <= q.bits_per_weight <= 32.0:
            raise CatalogError(f"{path.name}: {q.id}: implausible bits_per_weight {q.bits_per_weight}")
    return items


def load_cpus(path: Path | None = None) -> list[CpuSpec]:
    path = path or CATALOG_DIR / "cpus.json"
    rows = _load_file(path, "cpus/v1")
    items = [_coerce(CpuSpec, r, f"{path.name}[{i}]") for i, r in enumerate(rows)]
    _check_ids(items, path)
    _check_sources(items, path)
    known_isa = {"avx2", "avx512", "avx512-vnni", "amx-bf16", "amx-int8", "avx512-bf16"}
    for c in items:
        bad = set(c.isa) - known_isa
        if bad:
            raise CatalogError(f"{path.name}: {c.id}: unknown isa token(s) {sorted(bad)}")
        if "avx2" not in c.isa:
            raise CatalogError(f"{path.name}: {c.id}: every supported x86 server CPU has avx2")
        if c.threads < c.cores:
            raise CatalogError(f"{path.name}: {c.id}: threads {c.threads} < cores {c.cores}")
        if not c.base_ghz <= c.all_core_turbo_ghz <= c.max_turbo_ghz:
            raise CatalogError(
                f"{path.name}: {c.id}: expected base <= all_core_turbo <= max_turbo, got "
                f"{c.base_ghz}/{c.all_core_turbo_ghz}/{c.max_turbo_ghz}"
            )
        if c.ddr_gen not in {"DDR4", "DDR5", "MRDIMM"}:
            raise CatalogError(f"{path.name}: {c.id}: unexpected ddr_gen {c.ddr_gen!r}")
    return items


def load_memory(path: Path | None = None) -> list[MemoryOption]:
    path = path or CATALOG_DIR / "memory.json"
    rows = _load_file(path, "memory/v1")
    items = [_coerce(MemoryOption, r, f"{path.name}[{i}]") for i, r in enumerate(rows)]
    _check_ids(items, path)
    _check_sources(items, path)
    for m in items:
        if m.effective_mts > m.rated_mts:
            raise CatalogError(
                f"{path.name}: {m.id}: effective_mts {m.effective_mts} exceeds rated {m.rated_mts}"
            )
        if m.dimms_per_channel < 1:
            raise CatalogError(f"{path.name}: {m.id}: dimms_per_channel must be >= 1")
    return items


def load_coefficients(path: Path | None = None) -> list[CoefficientSpec]:
    path = path or CATALOG_DIR / "coefficients.json"
    rows = _load_file(path, "coefficients/v1")
    items = [_coerce(CoefficientSpec, r, f"{path.name}[{i}]") for i, r in enumerate(rows)]
    _check_ids(items, path)
    _check_sources(items, path)

    known_kinds = {"eta_bw", "eta_compute", "per_core_bw_gbs", "dual_socket_efficiency"}
    for c in items:
        if c.kind not in known_kinds:
            raise CatalogError(f"{path.name}: {c.id}: unknown kind {c.kind!r}")
        if c.confidence not in VALID_CONFIDENCE:
            raise CatalogError(
                f"{path.name}: {c.id}: confidence must be one of {VALID_CONFIDENCE}, "
                f"got {c.confidence!r}"
            )
        if c.kind in ("eta_bw", "eta_compute", "dual_socket_efficiency") and not 0 < c.value <= 1:
            raise CatalogError(
                f"{path.name}: {c.id}: {c.kind} must be a fraction in (0, 1], got {c.value}"
            )
        if c.kind == "per_core_bw_gbs" and not 1 <= c.value <= 200:
            raise CatalogError(f"{path.name}: {c.id}: implausible per-core bandwidth {c.value}")

    for kind in ("per_core_bw_gbs", "dual_socket_efficiency"):
        if not any(c.kind == kind and c.key == "*" for c in items):
            raise CatalogError(f"{path.name}: missing global '{kind}' entry (key '*')")
    return items


class Catalog:
    """Everything the simulator can choose from, loaded once."""

    def __init__(self, directory: Path | None = None) -> None:
        d = directory or CATALOG_DIR
        self.models = load_models(d / "models.json")
        self.quants = load_quants(d / "quants.json")
        self.cpus = load_cpus(d / "cpus.json")
        self.memory = load_memory(d / "memory.json")
        self.coefficients = load_coefficients(d / "coefficients.json")

    def model(self, model_id: str) -> ModelSpec:
        return _pick(self.models, model_id, "model")

    def quant(self, quant_id: str) -> QuantSpec:
        return _pick(self.quants, quant_id, "quant")

    def cpu(self, cpu_id: str) -> CpuSpec:
        return _pick(self.cpus, cpu_id, "cpu")

    def memory_option(self, memory_id: str) -> MemoryOption:
        return _pick(self.memory, memory_id, "memory")

    def coefficient(self, kind: str, key: str = "*") -> CoefficientSpec:
        """The coefficient for this kind and key, falling back to the global."""
        for c in self.coefficients:
            if c.kind == kind and c.key == key:
                return c
        for c in self.coefficients:
            if c.kind == kind and c.key == "*":
                return c
        raise CatalogError(f"no {kind!r} coefficient for key {key!r}")

    def memory_for(self, cpu: CpuSpec, dimms_per_channel: int = 1) -> MemoryOption:
        """Best memory option this CPU can actually clock, at the given DPC."""
        usable = [
            m
            for m in self.memory
            if m.ddr_gen == cpu.ddr_gen
            and m.dimms_per_channel == dimms_per_channel
            and m.rated_mts <= cpu.max_ddr_mts
        ]
        if not usable:
            raise CatalogError(
                f"no {cpu.ddr_gen} memory option at {dimms_per_channel} DPC "
                f"within {cpu.id}'s {cpu.max_ddr_mts} MT/s limit"
            )
        return max(usable, key=lambda m: m.effective_mts)

    def unverified(self) -> list[tuple[str, str]]:
        """Rows still lacking a datasheet, for the report's caveat section."""
        out: list[tuple[str, str]] = []
        for kind, items in (
            ("model", self.models),
            ("cpu", self.cpus),
            ("memory", self.memory),
        ):
            out.extend((kind, i.id) for i in items if i.source == "unverified")
        return out

    def summary(self) -> str:
        return (
            f"{len(self.models)} models, {len(self.quants)} quants, "
            f"{len(self.cpus)} CPUs, {len(self.memory)} memory options, "
            f"{len(self.coefficients)} coefficients"
        )


def _pick(items: list[Any], wanted: str, kind: str) -> Any:
    for item in items:
        if item.id == wanted:
            return item
    near = [i.id for i in items if wanted.lower() in i.id.lower()]
    hint = f" Did you mean: {', '.join(near[:5])}?" if near else ""
    raise CatalogError(f"unknown {kind} id {wanted!r}.{hint}")
