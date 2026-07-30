"""Minimal GGUF header reader — stdlib only, never loads weights.

Two jobs. First, let someone size a model that is not in the catalogue by
pointing at a local .gguf. Second, and more useful: it counts parameters exactly
from the tensor table and reads the file size, which yields the *measured*
bits-per-weight. That is how `svrspec verify` keeps `quants.json` honest instead
of trusting a table of nominal figures.

Format reference (GGUF v2/v3): magic, version, tensor count, KV count, then the
metadata key/value pairs, then one info record per tensor.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

from .types import SOURCE_MODEL_CARD, ModelSpec

MAGIC = b"GGUF"

# GGUF metadata value type tags.
_UINT8, _INT8, _UINT16, _INT16, _UINT32, _INT32, _FLOAT32 = range(7)
_BOOL, _STRING, _ARRAY, _UINT64, _INT64, _FLOAT64 = range(7, 13)

_FIXED = {
    _UINT8: ("<B", 1),
    _INT8: ("<b", 1),
    _UINT16: ("<H", 2),
    _INT16: ("<h", 2),
    _UINT32: ("<I", 4),
    _INT32: ("<i", 4),
    _FLOAT32: ("<f", 4),
    _BOOL: ("<?", 1),
    _UINT64: ("<Q", 8),
    _INT64: ("<q", 8),
    _FLOAT64: ("<d", 8),
}

#: llama.cpp's LLAMA_FTYPE enum, for naming the quantisation.
_FTYPE = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    7: "Q8_0",
    8: "Q5_0",
    9: "Q5_1",
    10: "Q2_K",
    11: "Q3_K_S",
    12: "Q3_K_M",
    13: "Q3_K_L",
    14: "Q4_K_S",
    15: "Q4_K_M",
    16: "Q5_K_S",
    17: "Q5_K_M",
    18: "Q6_K",
    30: "IQ4_XS",
}


class GgufError(Exception):
    pass


@dataclass(frozen=True)
class GgufFile:
    path: Path
    architecture: str
    name: str
    quant: str
    file_bytes: int
    param_count: int
    embedding_params: int
    metadata: dict[str, object]
    tensor_count: int

    @property
    def measured_bpw(self) -> float:
        """Whole-file bits per weight. The ground truth for quants.json."""
        if not self.param_count:
            return 0.0
        return self.file_bytes * 8.0 / self.param_count


class _Reader:
    def __init__(self, fh) -> None:
        self.fh = fh

    def raw(self, n: int) -> bytes:
        data = self.fh.read(n)
        if len(data) != n:
            raise GgufError("unexpected end of file while reading header")
        return data

    def scalar(self, fmt: str, size: int):
        return struct.unpack(fmt, self.raw(size))[0]

    def u32(self) -> int:
        return self.scalar("<I", 4)

    def u64(self) -> int:
        return self.scalar("<Q", 8)

    def string(self) -> str:
        return self.raw(self.u64()).decode("utf-8", errors="replace")

    def skip_string(self) -> None:
        self.fh.seek(self.u64(), 1)

    def value(self, vtype: int):
        if vtype in _FIXED:
            fmt, size = _FIXED[vtype]
            return self.scalar(fmt, size)
        if vtype == _STRING:
            return self.string()
        if vtype == _ARRAY:
            elem = self.u32()
            count = self.u64()
            # Vocabularies are hundreds of thousands of strings. Record the
            # length and skip the payload rather than materialising it; the
            # vocab size we want comes from the token_embd tensor dims anyway.
            if elem == _STRING:
                for _ in range(count):
                    self.skip_string()
                return {"_array_len": count, "_elem": "string"}
            if elem == _ARRAY:
                raise GgufError("nested arrays are not supported")
            fmt, size = _FIXED[elem]
            if count > 4096:
                self.fh.seek(size * count, 1)
                return {"_array_len": count, "_elem": elem}
            return [self.scalar(fmt, size) for _ in range(count)]
        raise GgufError(f"unknown GGUF value type {vtype}")


def read_gguf(path: str | Path) -> GgufFile:
    path = Path(path)
    if not path.exists():
        raise GgufError(f"no such file: {path}")

    with path.open("rb") as fh:
        r = _Reader(fh)
        if r.raw(4) != MAGIC:
            raise GgufError(f"{path.name} is not a GGUF file")
        version = r.u32()
        if version not in (2, 3):
            raise GgufError(f"unsupported GGUF version {version}")
        tensor_count = r.u64()
        kv_count = r.u64()

        metadata: dict[str, object] = {}
        for _ in range(kv_count):
            key = r.string()
            metadata[key] = r.value(r.u32())

        params = 0
        embedding_params = 0
        for _ in range(tensor_count):
            name = r.string()
            n_dims = r.u32()
            dims = [r.u64() for _ in range(n_dims)]
            r.u32()  # ggml type
            r.u64()  # offset
            numel = 1
            for d in dims:
                numel *= d
            params += numel
            if name in ("token_embd.weight", "output.weight"):
                embedding_params += numel

    arch = str(metadata.get("general.architecture", "unknown"))
    ftype = metadata.get("general.file_type")
    quant = _FTYPE.get(int(ftype), f"ftype-{ftype}") if isinstance(ftype, int) else "unknown"

    return GgufFile(
        path=path,
        architecture=arch,
        name=str(metadata.get("general.name", path.stem)),
        quant=quant,
        file_bytes=path.stat().st_size,
        param_count=params,
        embedding_params=embedding_params,
        metadata=metadata,
        tensor_count=tensor_count,
    )


def to_model_spec(gguf: GgufFile, model_id: str | None = None) -> ModelSpec:
    """Build a ModelSpec from GGUF metadata."""
    arch = gguf.architecture
    md = gguf.metadata

    def need(suffix: str) -> int:
        key = f"{arch}.{suffix}"
        if key not in md:
            raise GgufError(f"{gguf.path.name}: missing required metadata {key!r}")
        return int(md[key])  # type: ignore[arg-type]

    def maybe(suffix: str, default: int | None = None) -> int | None:
        key = f"{arch}.{suffix}"
        value = md.get(key, default)
        return int(value) if isinstance(value, (int, float)) else default

    n_head = need("attention.head_count")
    n_embd = need("embedding_length")
    n_kv_head = maybe("attention.head_count_kv", n_head) or n_head
    head_dim = maybe("attention.key_length")
    if head_dim and head_dim == n_embd // n_head:
        head_dim = None  # the derived value already covers it

    experts = maybe("expert_count", 0) or 0
    experts_used = maybe("expert_used_count", 0) or 0
    active_b: float | None = None
    if experts and experts_used:
        # Attention and shared layers run for every token; only the routed
        # expert FFNs are sparse. Approximating the whole model's sparsity by
        # the expert ratio is close enough for bandwidth sizing.
        active_b = gguf.param_count / 1e9 * (experts_used / experts)

    vocab = _vocab_size(md, gguf)

    return ModelSpec(
        id=model_id or gguf.path.stem.lower(),
        name=gguf.name,
        family=arch,
        params_b=gguf.param_count / 1e9,
        n_layer=need("block_count"),
        n_embd=n_embd,
        n_head=n_head,
        n_kv_head=n_kv_head,
        n_vocab=vocab,
        ctx_train=maybe("context_length", 4096) or 4096,
        head_dim=head_dim,
        active_params_b=active_b,
        source=SOURCE_MODEL_CARD,
        source_url=str(gguf.path),
        notes=f"GGUF 헤더에서 직접 추출 (quant {gguf.quant}, {gguf.tensor_count} tensors)",
    )


def _vocab_size(md: dict[str, object], gguf: GgufFile) -> int:
    tokens = md.get("tokenizer.ggml.tokens")
    if isinstance(tokens, dict) and "_array_len" in tokens:
        return int(tokens["_array_len"])  # type: ignore[arg-type]
    if isinstance(tokens, list):
        return len(tokens)
    if gguf.embedding_params:
        # token_embd (+ output when untied) is n_vocab x n_embd.
        return 0
    raise GgufError(f"{gguf.path.name}: cannot determine vocabulary size")
