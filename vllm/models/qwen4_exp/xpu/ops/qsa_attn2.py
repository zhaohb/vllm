# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse paged GQA attention for Qwen4Exp QSA, shaped for Xe2 (Intel XPU).

Same numerical contract as ``_qsa_sparse_paged_gqa_splitk_kernel``: BF16 or FP8
K/V, BF16 Q, FP32 scores, softmax and accumulation, FP32 partials merged by
log-sum-exp, BF16 rounding of the attention output before the FP32 sigmoid gate.

Why a second kernel: split-K holds a whole ``[BLOCK_M, HEAD_DIM]`` FP32
accumulator plus gathered K and V tiles in one program. At head_dim 256 that
alone is the whole 256-GRF file of a Xe2 hardware thread, so every config spills
11-46 KB per thread. More warps do not help: Triton's Intel backend lays a
chained (Q.K -> P.V) dot's warps along M, and M is only the 6 (padded 8) heads
of one KV group, so extra warps replicate the same work. The SYCL attention
kernels in vllm-xpu-kernels instead keep each subgroup's live state small and
get their parallelism from many short token partials merged afterwards.

This kernel does the same within Triton:

1. **One program = one row x one KV group x TILES_PER_PROG tiles of BLOCK_N
   selected tokens, one subgroup.** All 6 heads of the group are the dot's M
   (padded to 8), so each gathered K/V row serves every head.
2. **Each tile is finished in place**: Q.K over head_dim chunks (a runtime loop,
   so only one Q chunk and one K chunk are ever live), an exact softmax over the
   tile's BLOCK_N scores, then P.V chunk by chunk with each ``[BM, DC]`` output
   chunk stored as soon as it is computed. No full-head_dim accumulator exists.
3. **Each tile writes an FP32 partial** (normalised output + log2-sum-exp, the
   layout split-K uses), and ``_qsa_attn2_merge_kernel`` combines a row's tiles
   with an online log-sum-exp merge, rounds to BF16 and applies the gate. The
   merge reads the row's valid count, so tiles past it are never read.
4. **Rows go in chunks** (``row_chunk``) so the partial workspace stays bounded.
   It is a transient allocation like split-K's partials.
5. Every dot is M 8, N = BLOCK_N or DC >= 16, K = DC or BLOCK_N >= 16: DPAS
   eligible and legal under Triton's N/K >= 16 minimum on XPU.
"""

from typing import NamedTuple

import torch

from vllm.logger import init_logger
from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    TritonWarmupTensor,
    triton_scalar_specialization_rep,
)
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)


@triton.jit
def _attn2_dequant(tile):
    # An fp8 cache is an e4m3 view on XPU; anything else loads as it is.
    if tile.dtype == tl.float8e4nv:
        return tile.to(tl.float16)
    else:
        return tile


@triton.jit(do_not_specialize=["num_rows", "num_requests"])
def _qsa_attn2_tile_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    indices_ptr,
    block_table_ptr,
    token_to_req_ptr,
    partial_output_ptr,
    partial_lse_ptr,
    softmax_scale,
    output_scale,
    stride_q_row,
    stride_q_head,
    stride_k_block,
    stride_k_token,
    stride_k_head,
    stride_v_block,
    stride_v_token,
    stride_v_head,
    stride_indices_row,
    stride_table_req,
    num_rows,
    num_cache_blocks,
    num_requests,
    TOPK: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_DCHUNK: tl.constexpr,
    TILES_PER_PROG: tl.constexpr,
) -> None:
    DC: tl.constexpr = HEAD_DIM // NUM_DCHUNK
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    prog = tl.program_id(2)
    request = tl.load(token_to_req_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    request_ok = (request >= 0) & (request < num_requests)
    # Trailing count column of the packed selection buffer (see qsa.py).
    valid_count = tl.load(indices_ptr + row * stride_indices_row + TOPK)
    num_tiles = tl.cdiv(tl.minimum(valid_count, TOPK), BLOCK_N)

    head_offsets = tl.arange(0, BLOCK_M)
    head_mask = head_offsets < GROUP_SIZE
    dc_offsets = tl.arange(0, DC)
    column_offsets = tl.arange(0, BLOCK_N)
    first_head = kv_head * GROUP_SIZE
    q_rows = (
        q_ptr
        + row * stride_q_row
        + (first_head + head_offsets[:, None]) * stride_q_head
    )
    score_scale = softmax_scale * 1.4426950408889634

    for t in range(TILES_PER_PROG):
        tile = prog * TILES_PER_PROG + t
        if tile < num_tiles:
            columns = tile * BLOCK_N + column_offsets
            logical_token = tl.load(
                indices_ptr + row * stride_indices_row + columns,
                mask=columns < TOPK,
                other=-1,
            )
            safe_token = tl.maximum(logical_token, 0)
            logical_page = safe_token // PAGE_SIZE
            page_offset = safe_token % PAGE_SIZE
            valid = (
                request_ok & (logical_token >= 0) & (logical_page < PAGE_TABLE_WIDTH)
            )
            physical_page = tl.load(
                block_table_ptr
                + safe_request * stride_table_req
                + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1),
                mask=valid,
                other=-1,
            )
            valid &= (physical_page >= 0) & (physical_page < num_cache_blocks)
            safe_page = tl.maximum(physical_page, 0).to(tl.int64)
            # One int64 base per token; elements add a contiguous dim offset.
            k_base = (
                k_cache_ptr
                + safe_page * stride_k_block
                + page_offset * stride_k_token
                + kv_head * stride_k_head
            )
            v_base = (
                v_cache_ptr
                + safe_page * stride_v_block
                + page_offset * stride_v_token
                + kv_head * stride_v_head
            )

            # Q.K, head_dim chunk by chunk (runtime loop: one chunk live).
            scores = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for c in range(NUM_DCHUNK):
                q_chunk = tl.load(
                    q_rows + c * DC + dc_offsets[None, :],
                    mask=head_mask[:, None],
                    other=0.0,
                )
                k_chunk = tl.load(
                    k_base[None, :] + c * DC + dc_offsets[:, None],
                    mask=valid[None, :],
                    other=0.0,
                )
                scores = tl.dot(
                    q_chunk, _attn2_dequant(k_chunk).to(q_chunk.dtype), acc=scores
                )

            # Exact softmax over this tile.
            scores = tl.where(valid[None, :], scores * score_scale, -1.0e20)
            tile_max = tl.max(scores, axis=1)
            probabilities = tl.where(
                valid[None, :], tl.math.exp2(scores - tile_max[:, None]), 0.0
            )
            tile_sum = tl.sum(probabilities, axis=1)
            has_values = tile_sum > 0
            inv_sum = tl.where(
                has_values, output_scale / tl.maximum(tile_sum, 1.0e-20), 0.0
            )
            partial_row = (tile * num_rows + row).to(tl.int64)
            tl.store(
                partial_lse_ptr
                + partial_row * NUM_QUERY_HEADS
                + first_head
                + head_offsets,
                tl.where(
                    has_values,
                    tile_max + tl.math.log2(tl.maximum(tile_sum, 1.0e-20)),
                    -float("inf"),
                ),
                mask=head_mask,
            )
            part_rows = (
                partial_output_ptr
                + (partial_row * NUM_QUERY_HEADS + first_head + head_offsets[:, None])
                * HEAD_DIM
            )

            # P.V, chunk by chunk; each [BM, DC] output chunk is stored at once.
            for c in range(NUM_DCHUNK):
                v_chunk = _attn2_dequant(
                    tl.load(
                        v_base[:, None] + c * DC + dc_offsets[None, :],
                        mask=valid[:, None],
                        other=0.0,
                    )
                )
                out_chunk = tl.dot(probabilities.to(v_chunk.dtype), v_chunk)
                tl.store(
                    part_rows + c * DC + dc_offsets[None, :],
                    out_chunk * inv_sum[:, None],
                    mask=head_mask[:, None],
                )


@triton.jit(do_not_specialize=["num_rows"])
def _qsa_attn2_merge_kernel(
    partial_output_ptr,
    partial_lse_ptr,
    indices_ptr,
    output_ptr,
    output_gate_ptr,
    stride_indices_row,
    stride_output_row,
    stride_output_head,
    stride_output_gate_row,
    stride_output_gate_head,
    num_rows,
    TOPK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
) -> None:
    """Online log-sum-exp merge of one (row, head)'s tile partials + gate."""
    row = tl.program_id(0)
    head = tl.program_id(1)
    valid_count = tl.load(indices_ptr + row * stride_indices_row + TOPK)
    num_tiles = tl.cdiv(tl.minimum(valid_count, TOPK), BLOCK_N)
    dim_offsets = tl.arange(0, HEAD_DIM)
    running_max = -float("inf")
    denominator = 0.0
    merged = tl.zeros((HEAD_DIM,), dtype=tl.float32)
    for tile in range(num_tiles):
        partial_row = (tile * num_rows + row).to(tl.int64)
        lse = tl.load(partial_lse_ptr + partial_row * NUM_QUERY_HEADS + head)
        if lse > -float("inf"):
            next_max = tl.maximum(running_max, lse)
            alpha = tl.math.exp2(running_max - next_max)
            weight = tl.math.exp2(lse - next_max)
            part = tl.load(
                partial_output_ptr
                + (partial_row * NUM_QUERY_HEADS + head) * HEAD_DIM
                + dim_offsets
            )
            merged = merged * alpha + part * weight
            denominator = denominator * alpha + weight
            running_max = next_max
    merged = tl.where(denominator > 0, merged / denominator, 0.0)
    # Original contract: BF16 attention output, then the FP32 sigmoid gate.
    merged = merged.to(output_ptr.dtype.element_ty).to(tl.float32)
    gate = tl.load(
        output_gate_ptr
        + row * stride_output_gate_row
        + head * stride_output_gate_head
        + dim_offsets
    ).to(tl.float32)
    tl.store(
        output_ptr + row * stride_output_row + head * stride_output_head + dim_offsets,
        merged * tl.sigmoid(gate),
    )


# --------------------------------------------------------------------------------------
# dispatch
# --------------------------------------------------------------------------------------
class _Launch(NamedTuple):
    block_n: int
    num_dchunk: int
    row_chunk: int
    # Selection tiles one program walks. The kernels and the interpreter tests
    # cover more than one; the B70 configs below use one per program.
    tiles_per_prog: int = 1


# The fastest configs measured on a B70 (Triton 3.8.0+xpu, IGC 2.41.5,
# compute-runtime 26.35): BLOCK_N 32 x 8 head_dim chunks at 256 GRF, zero spill,
# for both regions. Decode 8 rows 54-56 us (32 rows 84-202, 64 rows 191-372, over
# 4K-131K of context); prefill 4,096 rows 9.3 ms at 32K, 12.6 ms at 131K. At 128
# GRF the same tile spills 192 B and runs ~17 % slower. Keep one subgroup: every
# BLOCK_N 32 or 64 config with num_warps 2 crashes IGC 2.41.5 with a SIGFPE.
_DECODE = _Launch(block_n=32, num_dchunk=8, row_chunk=4096)
_PREFILL = _Launch(block_n=32, num_dchunk=8, row_chunk=512)
_GRF_MODE = {"grf_mode": "256"}
# Rows at/above which a use_prefill_config call takes the prefill region.
_PREFILL_ROWS = 2048


def _select_launch(num_rows: int, use_prefill_config: bool) -> _Launch:
    """Launch config of one call: a pure function of shapes (graph-capture stable)."""
    if use_prefill_config and num_rows >= _PREFILL_ROWS:
        return _PREFILL
    return _DECODE


def qsa_sparse_paged_attention_attn2(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    use_prefill_config: bool,
    out: torch.Tensor,
    output_gate_view: torch.Tensor,
    softmax_scale: float,
    output_scale: float,
    block_m: int,
) -> torch.Tensor:
    """Launch the kernels; ``qsa_sparse_paged_attention`` validated the inputs."""
    logger.info_once("Qwen4Exp QSA sparse attention on XPU: the attn2 kernel.")
    head_dim = q.shape[2]
    num_kv_heads = k_cache.shape[2]
    group_size = q.shape[1] // num_kv_heads
    selection_width = logical_indices.shape[1] - 1
    launch = _select_launch(q.shape[0], use_prefill_config)
    dc = head_dim // launch.num_dchunk
    assert dc >= 16 and dc * launch.num_dchunk == head_dim, "attn2 head_dim chunking"
    rows_total = q.shape[0]
    row_chunk = max(1, min(launch.row_chunk, rows_total))
    num_tiles = triton.cdiv(selection_width, launch.block_n)
    tiles_per_prog = min(launch.tiles_per_prog, num_tiles)
    partial_output = torch.empty(
        (num_tiles, row_chunk, q.shape[1], head_dim),
        dtype=torch.float32,
        device=q.device,
    )
    partial_lse = torch.empty(
        (num_tiles, row_chunk, q.shape[1]), dtype=torch.float32, device=q.device
    )
    progs = triton.cdiv(num_tiles, tiles_per_prog)
    for r0 in range(0, rows_total, row_chunk):
        rows = min(row_chunk, rows_total - r0)
        _qsa_attn2_tile_kernel[(rows, num_kv_heads, progs)](
            q[r0 : r0 + rows],
            k_cache,
            v_cache,
            logical_indices[r0 : r0 + rows],
            block_table,
            token_to_req[r0 : r0 + rows],
            partial_output,
            partial_lse,
            softmax_scale,
            output_scale,
            q.stride(0),
            q.stride(1),
            k_cache.stride(0),
            k_cache.stride(1),
            k_cache.stride(2),
            v_cache.stride(0),
            v_cache.stride(1),
            v_cache.stride(2),
            logical_indices.stride(0),
            block_table.stride(0),
            rows,
            k_cache.shape[0],
            block_table.shape[0],
            TOPK=selection_width,
            PAGE_SIZE=k_cache.shape[1],
            PAGE_TABLE_WIDTH=block_table.shape[1],
            GROUP_SIZE=group_size,
            HEAD_DIM=head_dim,
            NUM_QUERY_HEADS=q.shape[1],
            BLOCK_M=block_m,
            BLOCK_N=launch.block_n,
            NUM_DCHUNK=launch.num_dchunk,
            TILES_PER_PROG=tiles_per_prog,
            num_warps=1,
            num_stages=1,
            **_GRF_MODE,
        )
        _qsa_attn2_merge_kernel[(rows, q.shape[1])](
            partial_output,
            partial_lse,
            logical_indices[r0 : r0 + rows],
            out[r0 : r0 + rows],
            output_gate_view[r0 : r0 + rows],
            logical_indices.stride(0),
            out.stride(0),
            out.stride(1),
            output_gate_view.stride(0),
            output_gate_view.stride(1),
            rows,
            TOPK=selection_width,
            BLOCK_N=launch.block_n,
            HEAD_DIM=head_dim,
            NUM_QUERY_HEADS=q.shape[1],
            num_warps=2,
            num_stages=1,
        )
    return out


def warmup_qsa_sparse_paged_attention_attn2(
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    *,
    num_query_heads: int,
    selection_width: int,
) -> tuple[tuple[int, int, int], ...]:
    """Compile every tile and merge specialization the launch can pick."""
    head_dim = kv_cache.shape[-1] // 2
    key_cache, value_cache = kv_cache.transpose(1, 2).split(head_dim, dim=-1)
    # An fp8 cache is allocated as uint8 storage and handed over as an e4m3 view.
    cache_dtype = (
        torch.float8_e4m3fn if kv_cache.dtype == torch.uint8 else key_cache.dtype
    )
    num_kv_heads = key_cache.shape[2]
    group_size = num_query_heads // num_kv_heads
    num_rows = 16
    row_stride = num_query_heads * head_dim
    q_ptr = TritonWarmupTensor(
        torch.bfloat16, shape=(num_rows, num_query_heads, head_dim)
    )
    k_cache_ptr = TritonWarmupTensor(
        cache_dtype, shape=tuple(key_cache.shape), strides=tuple(key_cache.stride())
    )
    v_cache_ptr = TritonWarmupTensor(
        cache_dtype,
        shape=tuple(value_cache.shape),
        strides=tuple(value_cache.stride()),
    )
    indices_ptr = TritonWarmupTensor(torch.int32, shape=(num_rows, selection_width + 1))
    block_table_ptr = TritonWarmupTensor(
        block_table.dtype,
        shape=tuple(block_table.shape),
        strides=tuple(block_table.stride()),
    )
    token_to_req_ptr = TritonWarmupTensor(torch.int32)
    output_ptr = TritonWarmupTensor(
        torch.bfloat16, shape=(num_rows, num_query_heads, head_dim)
    )
    output_gate_ptr = TritonWarmupTensor(
        torch.bfloat16, shape=(num_rows, num_query_heads, head_dim)
    )
    partial_output_ptr = TritonWarmupTensor(torch.float32)
    partial_lse_ptr = TritonWarmupTensor(torch.float32)
    warmed = []
    for launch in sorted({_DECODE, _PREFILL}):
        num_tiles = triton.cdiv(selection_width, launch.block_n)
        tiles_per_prog = min(launch.tiles_per_prog, num_tiles)
        _qsa_attn2_tile_kernel.warmup(
            q_ptr,
            k_cache_ptr,
            v_cache_ptr,
            indices_ptr,
            block_table_ptr,
            token_to_req_ptr,
            partial_output_ptr,
            partial_lse_ptr,
            1.0,
            1.0,
            row_stride,
            head_dim,
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            value_cache.stride(0),
            value_cache.stride(1),
            value_cache.stride(2),
            selection_width + 1,
            block_table.stride(0),
            num_rows,
            triton_scalar_specialization_rep(kv_cache.shape[0]),
            16,
            TOPK=selection_width,
            PAGE_SIZE=key_cache.shape[1],
            PAGE_TABLE_WIDTH=block_table.shape[1],
            GROUP_SIZE=group_size,
            HEAD_DIM=head_dim,
            NUM_QUERY_HEADS=num_query_heads,
            BLOCK_M=triton.next_power_of_2(group_size),
            BLOCK_N=launch.block_n,
            NUM_DCHUNK=launch.num_dchunk,
            TILES_PER_PROG=tiles_per_prog,
            num_warps=1,
            num_stages=1,
            **_GRF_MODE,
            grid=(num_rows, num_kv_heads, 1),
        )
        _qsa_attn2_merge_kernel.warmup(
            partial_output_ptr,
            partial_lse_ptr,
            indices_ptr,
            output_ptr,
            output_gate_ptr,
            selection_width + 1,
            row_stride,
            head_dim,
            row_stride,
            head_dim,
            num_rows,
            TOPK=selection_width,
            BLOCK_N=launch.block_n,
            HEAD_DIM=head_dim,
            NUM_QUERY_HEADS=num_query_heads,
            num_warps=2,
            num_stages=1,
            grid=(num_rows, num_query_heads),
        )
        warmed.append((launch.block_n, launch.num_dchunk, tiles_per_prog))
    return tuple(warmed)


__all__ = [
    "qsa_sparse_paged_attention_attn2",
    "warmup_qsa_sparse_paged_attention_attn2",
]
