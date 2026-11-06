# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""INT8 row-scale Qwen4Exp PLE tables: the format, its builder and its checks.

A row stores its head as ``q`` int8 values and one float32 scale ``s``, with
``s = absmax(row) / 127``, ``q = round_half_even(x / s)`` clamped to +-127 and
``x ~= q * s``. At 164 B against BF16's 320 B, the table of Qwen3.8-Flash-Next is
48.9 GiB against 95.4 GiB; relative L2 error against BF16 is about 0.66 %.

The file is a safetensors file, with its header padded so the data starts on a
page, so a row sits at a computable offset and direct reads need no parsing::

    table                    U8  [rows, head_dim + 4]
    ngram_heads_offsets      I64 [heads]
    ngram_heads_vocab_sizes  I64 [heads]
    layer_multipliers        I64 [ngram_size]

The last three are copied from the checkpoint, so a boot can refuse a table built
for another hash layout. The same source gives the same bytes.

Build one from a checkpoint with a BF16 table, then pass it as ``int8_table_path``::

    python -m vllm.models.qwen4_exp.xpu.ple_int8 build <checkpoint> <out.safetensors>
    python -m vllm.models.qwen4_exp.xpu.ple_int8 verify <checkpoint> <out.safetensors>

``build`` needs the table's size in disk space and bounded RAM, and writes
``<out>.sha256``. ``verify`` reads the whole file against that checksum, and compares
a sample of rows with the checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import regex as re
import torch

from .ple_rows import PAGE, safetensors_header

FORMAT = "ple-int8-rowscale/v1"
LAYOUT_KEYS = ("ngram_heads_offsets", "ngram_heads_vocab_sizes", "layer_multipliers")
# A table whose rows are further than this from the checkpoint's is the wrong table;
# the format's own error is about a third of it.
MAX_REL_ERROR = 0.02
_SHARD_RE = re.compile(r".*\.ngram_embedding\.shard_(\d+)\.weight")


def quantize_rows(rows: torch.Tensor) -> torch.Tensor:
    """BF16 [n, head_dim] -> packed uint8 [n, head_dim + 4]."""
    values = rows.to(torch.float32)
    scales = values.abs().amax(dim=1, keepdim=True) / 127.0
    quantized = values / scales.clamp_min(1e-30)
    quantized = quantized.round().clamp_(-127, 127).to(torch.int8)
    packed = torch.empty(rows.shape[0], rows.shape[1] + 4, dtype=torch.uint8)
    packed[:, : rows.shape[1]] = quantized.view(torch.uint8)
    packed[:, rows.shape[1] :] = scales.contiguous().view(torch.uint8).reshape(-1, 4)
    return packed


def decode_rows(packed: torch.Tensor) -> torch.Tensor:
    """Packed uint8 [n, head_dim + 4] -> float32 [n, head_dim], on any device."""
    values = packed[..., :-4].view(torch.int8).to(torch.float32)
    return values * packed[..., -4:].contiguous().view(torch.float32)


def relative_l2(got: torch.Tensor, want: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(got - want)) / max(
        float(torch.linalg.vector_norm(want)), 1e-30
    )


def read_table_header(path: str) -> tuple[int, int, int, dict[str, np.ndarray]]:
    """(data start, rows, row bytes, hash layout) of a table file, or a refusal."""
    header = safetensors_header(path)
    metadata = header.get("__metadata__") or {}
    # Tables built before the format had this name carry a prefix before it.
    if not str(metadata.get("format", "")).endswith(FORMAT):
        raise ValueError(
            f"{path} has format {metadata.get('format')!r}, expected {FORMAT!r}; "
            "build one with python -m vllm.models.qwen4_exp.xpu.ple_int8"
        )
    table = header["table"]
    if table["dtype"] != "U8" or len(table["shape"]) != 2:
        raise ValueError(f"{path}: table is {table['dtype']} {table['shape']}")
    start = header[""]
    layout = {}
    with open(path, "rb") as handle:
        for key in LAYOUT_KEYS:
            low, high = header[key]["data_offsets"]
            handle.seek(start + low)
            layout[key] = np.frombuffer(handle.read(high - low), dtype="<i8").copy()
    return (
        start + int(table["data_offsets"][0]),
        int(table["shape"][0]),
        int(table["shape"][1]),
        layout,
    )


def _checkpoint_tensors(checkpoint: Path) -> dict[str, tuple[Path, dict, dict]]:
    """Every tensor's file, the file's header and the tensor's entry, by name."""
    index = checkpoint / "model.safetensors.index.json"
    if index.exists():
        files = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    else:
        files = sorted(path.name for path in checkpoint.glob("*.safetensors"))
    tensors = {}
    for name in files:
        path = checkpoint / name
        header = safetensors_header(str(path))
        for key, info in header.items():
            if key not in ("", "__metadata__"):
                tensors[key] = (path, header, info)
    return tensors


def _source_shards(
    tensors: dict[str, tuple[Path, dict, dict]],
) -> list[tuple[Path, int, int, int]]:
    """(file, data offset, rows, head_dim) of each table shard, in order."""
    found: dict[int, tuple[Path, dict, dict]] = {}
    for key, value in tensors.items():
        match = _SHARD_RE.fullmatch(key)
        if match is not None:
            found[int(match.group(1))] = value
    if not found:
        raise ValueError("The checkpoint ships no PLE table (ngram_embedding.shard_*)")
    if sorted(found) != list(range(len(found))):
        raise ValueError(f"PLE shards are not contiguous: {sorted(found)}")
    shards = []
    for index in range(len(found)):
        path, header, info = found[index]
        if info["dtype"] != "BF16":
            raise ValueError(
                f"The INT8 table is built from BF16 rows; shard {index} is "
                f"{info['dtype']}. Quantizing an already quantized table would "
                "compound the error."
            )
        rows, head_dim = info["shape"]
        shards.append((path, header[""] + int(info["data_offsets"][0]), rows, head_dim))
    return shards


def _layout(tensors: dict[str, tuple[Path, dict, dict]]) -> dict[str, np.ndarray]:
    layout = {}
    for key in LAYOUT_KEYS:
        names = [name for name in tensors if name.endswith(f".ple_embedding.{key}")]
        if len(names) != 1:
            raise ValueError(f"Expected one PLE {key} in the checkpoint, got {names}")
        path, header, info = tensors[names[0]]
        low, high = info["data_offsets"]
        with open(path, "rb") as handle:
            handle.seek(header[""] + low)
            layout[key] = np.frombuffer(handle.read(high - low), dtype="<i8").copy()
    return layout


def _header_bytes(rows: int, row_bytes: int, layout: dict[str, np.ndarray]) -> bytes:
    position = rows * row_bytes
    header: dict = {
        "__metadata__": {
            "format": FORMAT,
            "row_layout": "int8 q | float32 LE s; s=absmax/127, "
            "q=round_half_even(x/s) clamp +-127; x~=q*s",
        },
        "table": {
            "dtype": "U8",
            "shape": [rows, row_bytes],
            "data_offsets": [0, position],
        },
    }
    for key in LAYOUT_KEYS:
        size = layout[key].size * 8
        header[key] = {
            "dtype": "I64",
            "shape": list(layout[key].shape),
            "data_offsets": [position, position + size],
        }
        position += size
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    encoded += b" " * (-(8 + len(encoded)) % PAGE)
    return len(encoded).to_bytes(8, "little") + encoded


def build(checkpoint: Path, out: Path, chunk_rows: int = 1 << 20) -> str:
    """Write the table of ``checkpoint`` to ``out``; returns its sha256."""
    if out.exists():
        raise FileExistsError(f"{out} exists")
    tensors = _checkpoint_tensors(checkpoint)
    shards = _source_shards(tensors)
    layout = _layout(tensors)
    head_dim = shards[0][3]
    total_rows = sum(shard[2] for shard in shards)
    digest = hashlib.sha256()
    part = out.with_name(out.name + ".part")

    def write_all(fd: int, data: bytes) -> None:
        digest.update(data)
        view = memoryview(data)
        while view:
            view = view[os.write(fd, view) :]

    target = os.open(part, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        write_all(target, _header_bytes(total_rows, head_dim + 4, layout))
        for path, offset, rows, _ in shards:
            source = os.open(path, os.O_RDONLY)
            try:
                for first in range(0, rows, chunk_rows):
                    count = min(chunk_rows, rows - first)
                    position = offset + first * head_dim * 2
                    data = os.pread(source, count * head_dim * 2, position)
                    if len(data) != count * head_dim * 2:
                        raise ValueError(f"{path} is truncated at {position}")
                    os.posix_fadvise(
                        source, position, len(data), os.POSIX_FADV_DONTNEED
                    )
                    rows_bf16 = torch.frombuffer(
                        bytearray(data), dtype=torch.bfloat16
                    ).reshape(count, head_dim)
                    write_all(target, quantize_rows(rows_bf16).numpy().tobytes())
            finally:
                os.close(source)
        for key in LAYOUT_KEYS:
            write_all(target, layout[key].astype("<i8").tobytes())
        os.fdatasync(target)
    finally:
        os.close(target)
    os.rename(part, out)
    sha256 = digest.hexdigest()
    Path(f"{out}.sha256").write_text(f"{sha256}  {out.name}\n")
    return sha256


def verify(
    checkpoint: Path, table: Path, *, sample_rows: int = 262144, check_sha: bool = True
) -> dict:
    """Check a table against its checksum and against the checkpoint it came from."""
    start, rows, row_bytes, layout = read_table_header(str(table))
    tensors = _checkpoint_tensors(checkpoint)
    shards = _source_shards(tensors)
    expected = _layout(tensors)
    for key in LAYOUT_KEYS:
        if not np.array_equal(layout[key], expected[key]):
            raise ValueError(f"{table}: {key} differs from the checkpoint")
    head_dim = shards[0][3]
    if rows != sum(s[2] for s in shards) or row_bytes != head_dim + 4:
        raise ValueError(
            f"{table}: table [{rows}, {row_bytes}] does not fit the source"
        )
    result: dict = {"rows": rows, "row_bytes": row_bytes}
    if check_sha:
        sidecar = Path(f"{table}.sha256")
        if not sidecar.exists():
            raise ValueError(f"{sidecar} is missing")
        digest = hashlib.sha256()
        with open(table, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 24), b""):
                digest.update(block)
        result["sha256"] = digest.hexdigest()
        if result["sha256"] != sidecar.read_text().split()[0]:
            raise ValueError(f"{table} does not match {sidecar}")
    generator = np.random.default_rng(20260928)
    sample = np.unique(generator.integers(0, rows, min(sample_rows, rows)))
    starts = np.cumsum([0] + [s[2] for s in shards])
    got = np.memmap(
        table, dtype=np.uint8, mode="r", offset=start, shape=(rows, row_bytes)
    )
    picked = torch.from_numpy(np.array(got[sample]))
    original = torch.empty(sample.size, head_dim, dtype=torch.bfloat16)
    which = np.searchsorted(starts, sample, side="right") - 1
    for shard in np.unique(which).tolist():
        path, offset, count, _ = shards[shard]
        source = np.memmap(
            path, dtype=np.uint16, mode="r", offset=offset, shape=(count, head_dim)
        )
        selected = which == shard
        original[torch.from_numpy(selected)] = torch.from_numpy(
            np.array(source[sample[selected] - starts[shard]]).astype(np.int16)
        ).view(torch.bfloat16)
    if not torch.equal(quantize_rows(original), picked):
        raise ValueError(f"{table}: sampled rows differ from a fresh quantization")
    result["sampled_rows"] = int(sample.size)
    result["rel_l2"] = relative_l2(decode_rows(picked), original.to(torch.float32))
    if result["rel_l2"] > MAX_REL_ERROR:
        raise ValueError(f"{table}: relative L2 {result['rel_l2']:.4f} is too far")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("build", "verify"))
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("table", type=Path)
    args = parser.parse_args(argv)
    if args.mode == "build":
        print(build(args.checkpoint, args.table))
    else:
        print(json.dumps(verify(args.checkpoint, args.table)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
