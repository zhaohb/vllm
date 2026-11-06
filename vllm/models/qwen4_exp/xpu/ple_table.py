# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The XPU Qwen4Exp PLE tables behind ``--engram-config``.

Both tables hold each row on one TP rank, as a run of bytes in the format of the
table's source: BF16 or FP8 rows from the checkpoint, or the INT8 row-scale table
(``ple_int8.py``). A rank writes the rows it owns into a fixed device buffer of
``[tokens, heads, row_bytes]``, leaving zeros elsewhere; the forward sums that buffer
over TP, byte for byte, and decodes it.

``PinnedRowTable`` (``cpu_offload``) keeps the rows in pinned slabs, and a device
kernel inside the forward gathers them through UVA, so a step does no host work.
``DiskRowTable`` (``disk_offload``) keeps them in the files, behind a pinned row
cache filled before the forward with direct reads.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.distributed import get_tp_group
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

from .ple_int8 import decode_rows
from .ple_rows import (
    MAX_SLABS,
    PinnedRowSlabs,
    RowCache,
    RowExtent,
    RowLookahead,
    RowReader,
    RowServer,
    RowStats,
    current_chunk_keys,
    owned_heads,
    plan_next_chunks,
)

logger = init_logger(__name__)

# Scale bytes after the values of an INT8 row.
INT8_SCALE_BYTES = 4


class RowFormat:
    """How a row's bytes decode to a head's embedding."""

    def __init__(
        self, kind: str, head_dim: int, scale: torch.Tensor | None = None
    ) -> None:
        if kind not in ("bf16", "fp8", "int8"):
            raise ValueError(f"Unknown PLE row format {kind!r}")
        self.kind = kind
        self.head_dim = head_dim
        self.scale = scale
        self.row_bytes = {
            "bf16": head_dim * 2,
            "fp8": head_dim,
            "int8": head_dim + INT8_SCALE_BYTES,
        }[kind]

    def decode(self, packed: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """uint8 ``[..., row_bytes]`` -> ``[..., head_dim]`` of ``dtype``."""
        if self.kind == "bf16":
            return packed.view(torch.bfloat16).to(dtype)
        if self.kind == "fp8":
            assert self.scale is not None
            scale = self.scale.to(device=packed.device, dtype=dtype)
            return packed.view(torch.float8_e4m3fn).to(dtype) * scale
        # One rounding: to float32 for the product, then to the activation dtype.
        return decode_rows(packed).to(dtype)


@triton.jit
def _pick(values, slab, lo, hi, local, offsets, in_row, owned, ROW_BYTES: tl.constexpr):
    hit = owned & (local >= lo) & (local < hi)
    loaded = tl.load(
        slab + (local - lo) * ROW_BYTES + offsets, mask=in_row & hit, other=0
    )
    return tl.where(hit, loaded, values)


@triton.jit
def _gather_rows_from_slabs_kernel(
    ids_ptr,  # int64 [T, H], global row ids
    out_ptr,  # uint8 [T, H, ROW_BYTES]
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,  # uint8 slab views; unused ones repeat s0
    b1,
    b2,
    b3,
    b4,
    b5,
    b6,
    b7,
    b8,  # local row starts of slabs 1..7, then the end
    rank_start,
    rank_end,
    HEADS: tl.constexpr,
    ROW_BYTES: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_SLABS: tl.constexpr,
):
    """Copy this rank's rows out of its slabs; other slots get zeros."""
    token = tl.program_id(0)
    head = tl.program_id(1)
    row = tl.load(ids_ptr + token * HEADS + head)
    owned = (row >= rank_start) & (row < rank_end)
    local = tl.where(owned, row - rank_start, 0).to(tl.int64)
    offsets = tl.arange(0, BLOCK)
    in_row = offsets < ROW_BYTES
    values = tl.zeros([BLOCK], dtype=tl.uint8)
    values = _pick(values, s0, 0, b1, local, offsets, in_row, owned, ROW_BYTES)
    if NUM_SLABS > 1:
        values = _pick(values, s1, b1, b2, local, offsets, in_row, owned, ROW_BYTES)
    if NUM_SLABS > 2:
        values = _pick(values, s2, b2, b3, local, offsets, in_row, owned, ROW_BYTES)
    if NUM_SLABS > 3:
        values = _pick(values, s3, b3, b4, local, offsets, in_row, owned, ROW_BYTES)
    if NUM_SLABS > 4:
        values = _pick(values, s4, b4, b5, local, offsets, in_row, owned, ROW_BYTES)
    if NUM_SLABS > 5:
        values = _pick(values, s5, b5, b6, local, offsets, in_row, owned, ROW_BYTES)
    if NUM_SLABS > 6:
        values = _pick(values, s6, b6, b7, local, offsets, in_row, owned, ROW_BYTES)
    if NUM_SLABS > 7:
        values = _pick(values, s7, b7, b8, local, offsets, in_row, owned, ROW_BYTES)
    tl.store(
        out_ptr + (token * HEADS + head).to(tl.int64) * ROW_BYTES + offsets,
        values,
        mask=in_row,
    )


def gather_rows_from_slabs(
    views: Sequence[torch.Tensor],
    bounds: Sequence[int],
    rank_start: int,
    rank_end: int,
    row_bytes: int,
    ids: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """Write rows ``ids`` [T, H] of the rank's slabs to ``output`` [T, H, row_bytes]."""
    num_slabs = len(views)
    if not 1 <= num_slabs <= MAX_SLABS or len(bounds) != num_slabs + 1:
        raise ValueError(f"{num_slabs} PLE slabs, {len(bounds)} bounds")
    tokens, heads = ids.shape
    if tokens == 0:
        return
    # Starts of slabs 1.., then the end. An unused slot never matches: lo == hi.
    tail = [int(b) for b in bounds[1:]] + [int(bounds[-1])] * (MAX_SLABS - num_slabs)
    slabs = list(views) + [views[0]] * (MAX_SLABS - num_slabs)
    _gather_rows_from_slabs_kernel[(tokens, heads)](
        ids,
        output,
        *slabs,
        *tail,
        rank_start,
        rank_end,
        HEADS=heads,
        ROW_BYTES=row_bytes,
        BLOCK=triton.next_power_of_2(row_bytes),
        NUM_SLABS=num_slabs,
    )


class _RowTable:
    """A rank's share of the table, and the fixed buffer its rows land in."""

    def __init__(
        self,
        row_format: RowFormat,
        rank_range: tuple[int, int],
        max_tokens: int,
        heads: int,
        device: torch.device,
    ) -> None:
        self.format = row_format
        self.start, self.end = rank_range
        self.max_tokens = max_tokens
        self.heads = heads
        self.device = device
        self.pin = device.type == "xpu"
        self.group = get_tp_group()
        self.rank = self.group.rank_in_group
        self.packed = torch.zeros(
            max_tokens, heads, row_format.row_bytes, dtype=torch.uint8, device=device
        )

    def _reduce_and_decode(
        self, packed: torch.Tensor, dtype: torch.dtype
    ) -> torch.Tensor:
        """Sum the ranks' bytes (each row has one owner) and decode the rows."""
        if self.group.world_size > 1:
            reduced = self.group.all_reduce(packed.view(torch.int8))
            packed = reduced.view(torch.uint8)
        return self.format.decode(packed, dtype).flatten(-2)

    def _pinned(self, *shape: int) -> torch.Tensor:
        tensor = torch.empty(
            *shape, dtype=torch.uint8, device="cpu", pin_memory=self.pin
        )
        if self.pin and not tensor.is_pinned():
            size = tensor.numel() / 2**30
            raise RuntimeError(f"PLE table: {size:.2f} GiB failed to pin")
        return tensor


class PinnedRowTable(_RowTable):
    """Rows in pinned slabs, gathered by a device kernel inside the forward."""

    reads_rows_on_host = False

    def __init__(
        self,
        row_format: RowFormat,
        rank_range: tuple[int, int],
        extents: Sequence[RowExtent],
        *,
        max_tokens: int,
        heads: int,
        device: torch.device,
    ) -> None:
        super().__init__(row_format, rank_range, max_tokens, heads, device)
        self._tensors: list[torch.Tensor] = []

        def allocate_slab(rows: int, row_bytes: int) -> np.ndarray:
            self._tensors.append(self._pinned(rows, row_bytes))
            return self._tensors[-1].numpy()

        self.slabs = PinnedRowSlabs(
            self.start, self.end, row_format.row_bytes, alloc=allocate_slab
        )
        seconds, direct = self.slabs.load(extents)
        if row_format.kind == "int8":
            self.slabs.check_scales(row_format.head_dim)
        self.bounds = self.slabs.bounds.tolist()
        self.views = [get_accelerator_view_from_cpu_tensor(t) for t in self._tensors]
        logger.info(
            "XPU PLE rows in pinned host memory (rank %d): %s rows [%d, %d), %d B/row, "
            "%.2f GiB in %d slabs, read in %.1f s (%.2f GB/s%s), gathered in forward",
            self.rank,
            row_format.kind,
            self.start,
            self.end,
            row_format.row_bytes,
            self.slabs.nbytes / 2**30,
            len(self._tensors),
            seconds,
            self.slabs.nbytes / max(seconds, 1e-9) / 1e9,
            "" if direct else ", through the page cache",
        )

    def lookup(self, ids: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        packed = self.packed[: ids.shape[0]]
        gather_rows_from_slabs(
            self.views,
            self.bounds,
            self.start,
            self.end,
            self.format.row_bytes,
            ids,
            packed,
        )
        return self._reduce_and_decode(packed, dtype)


class DiskRowTable(_RowTable):
    """Rows in the table's files, behind a pinned row cache and read-ahead."""

    reads_rows_on_host = True

    def __init__(
        self,
        row_format: RowFormat,
        rank_range: tuple[int, int],
        extents: Sequence[RowExtent],
        *,
        max_tokens: int,
        heads: int,
        device: torch.device,
        engram_config: Any,
        ple_constants: dict[str, np.ndarray],
        eos_token_id: int,
        heads_per_ngram: int,
    ) -> None:
        super().__init__(row_format, rank_range, max_tokens, heads, device)
        row_bytes = row_format.row_bytes
        sizes = ple_constants["sizes"]
        offsets = ple_constants["offsets"]
        self.own_heads = owned_heads(sizes, offsets, self.start, self.end)
        cache_bytes = int(engram_config.disk_row_cache_gib * (1 << 30))
        capacity = cache_bytes // self.group.world_size // row_bytes
        needed = 2 * max_tokens * max(1, self.own_heads.shape[0])
        if capacity < needed:
            raise ValueError(
                f"disk_row_cache_gib={engram_config.disk_row_cache_gib} gives "
                f"{capacity} rows per rank, below two steps of {needed} rows"
            )
        self._slab = self._pinned(capacity, row_bytes)
        self._slab_np = self._slab.numpy()
        self._stage = self._pinned(max_tokens, heads, row_bytes)
        self._stage_np = self._stage.numpy()
        self._stage_np.fill(0)
        self._dirty = 0
        threads = engram_config.disk_read_threads
        reader = RowReader(extents, row_bytes, threads=threads)
        self._server = RowServer(
            start=self.start,
            end=self.end,
            multipliers=ple_constants["multipliers"],
            sizes=sizes,
            offsets=offsets,
            eos_token_id=eos_token_id,
            heads_per_ngram=heads_per_ngram,
            cache=RowCache(self.end - self.start, self._slab_np),
            reader=reader,
            stats=RowStats(self.rank, engram_config.disk_stats_interval_s),
        )
        self._context_len = ple_constants["multipliers"].shape[0] - 1
        self._eos = eos_token_id
        self._lookahead: RowLookahead | None = None
        if engram_config.disk_lookahead:
            self._lookahead = RowLookahead(
                self._server,
                RowReader(extents, row_bytes, threads=threads),
                max_rows=max_tokens * max(1, self.own_heads.shape[0]),
            )
        logger.info(
            "XPU PLE rows on disk (rank %d): %s rows [%d, %d), %d B/row, %s, "
            "row cache %d rows (%.2f GiB pinned), %d read threads, lookahead %s",
            self.rank,
            row_format.kind,
            self.start,
            self.end,
            row_bytes,
            "direct reads" if reader.direct else "reads through the page cache",
            capacity,
            capacity * row_bytes / 2**30,
            threads,
            "on" if self._lookahead is not None else "off",
        )

    def stage(
        self,
        ids: np.ndarray,
        num_padded: int,
        input_batch: Any = None,
        req_states: Any = None,
    ) -> None:
        """Copy this step's own rows to the device buffer; zeros elsewhere.

        ``ids`` [T, heads] are the row ids of the step's real tokens. The stream
        has been synced, so the previous step's copy from the stage has finished.
        """
        num_tokens = ids.shape[0]
        stage = self._stage_np
        stage[: max(self._dirty, num_padded)].fill(0)
        prefetched, wait_ms = self._take_lookahead(input_batch, req_states)
        if num_tokens:
            own_heads = self.own_heads
            slots, mask = self._server.resolve(
                ids[:, own_heads], prefetched=prefetched, wait_ms=wait_ms
            )
            token, column = np.nonzero(mask)
            if token.size:
                stage[token, own_heads[column]] = self._slab_np[slots[token, column]]
        self._dirty = num_tokens
        if num_padded:
            self.packed[:num_padded].copy_(self._stage[:num_padded], non_blocking=True)
        self._submit_lookahead(input_batch, req_states)

    def lookup(self, num_tokens: int, dtype: torch.dtype) -> torch.Tensor:
        return self._reduce_and_decode(self.packed[:num_tokens], dtype)

    def _take_lookahead(self, input_batch: Any, req_states: Any):
        if self._lookahead is None or input_batch is None:
            return None, 0.0
        requests = input_batch.idx_mapping_np[: input_batch.num_reqs]
        return self._lookahead.take(
            current_chunk_keys(
                requests,
                req_states.num_computed_prefill_tokens[requests],
                input_batch.num_scheduled_tokens,
                req_states.prefill_len.np[requests],
            )
        )

    def _submit_lookahead(self, input_batch: Any, req_states: Any) -> None:
        if self._lookahead is None or input_batch is None:
            return
        token_buffer = getattr(req_states.all_token_ids, "_uva_buf", None)
        if token_buffer is None:
            return  # The token history is on the device only: no read-ahead.
        requests = input_batch.idx_mapping_np[: input_batch.num_reqs]
        plan = plan_next_chunks(
            requests,
            req_states.num_computed_prefill_tokens[requests],
            input_batch.num_scheduled_tokens,
            req_states.prefill_len.np[requests],
            self.max_tokens,
        )
        if not plan:
            return
        history = token_buffer.np
        tokens, starts, context = [], [0], []
        for request, start, end in plan:
            tokens.append(history[request, start:end])
            starts.append(starts[-1] + end - start)
            context.append(
                [
                    int(history[request, start + shift])
                    if start + shift >= 0
                    else self._eos
                    for shift in range(-self._context_len, 0)
                ]
            )
        self._lookahead.submit(
            frozenset((request, start) for request, start, _ in plan),
            np.concatenate(tokens),
            np.asarray(starts, dtype=np.int64),
            np.asarray(context, dtype=np.int64),
        )


def build_row_table(
    vllm_config: VllmConfig,
    row_format: RowFormat,
    rank_range: tuple[int, int],
    extents: Sequence[RowExtent],
    *,
    max_tokens: int,
    heads: int,
    device: torch.device,
    ple_constants: dict[str, np.ndarray],
    eos_token_id: int,
    heads_per_ngram: int,
) -> PinnedRowTable | DiskRowTable:
    """The table ``--engram-config`` selects; ``disk_offload`` wins, as on CUDA."""
    engram_config = vllm_config.engram_config
    assert engram_config is not None
    if engram_config.embedding_across_dp or engram_config.dp_shared_memory:
        raise ValueError(
            "The XPU PLE tables keep one copy per DP replica: unset "
            "embedding_across_dp and set dp_shared_memory=False"
        )
    if engram_config.disk_offload:
        return DiskRowTable(
            row_format,
            rank_range,
            extents,
            max_tokens=max_tokens,
            heads=heads,
            device=device,
            engram_config=engram_config,
            ple_constants=ple_constants,
            eos_token_id=eos_token_id,
            heads_per_ngram=heads_per_ngram,
        )
    return PinnedRowTable(
        row_format,
        rank_range,
        extents,
        max_tokens=max_tokens,
        heads=heads,
        device=device,
    )
