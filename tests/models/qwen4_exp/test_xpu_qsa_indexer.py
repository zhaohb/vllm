# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The QSA prefill scorer's launch on XPU, in the Triton interpreter on CPU.

The interpreter reads bfloat16 operands of tl.dot as integers, so the kernel runs
on bf16-rounded values held in fp32. test_qsa_reference.py covers BF16 on a card.
"""

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.ops import qsa_indexer
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl

pytestmark = pytest.mark.skipif(not HAS_TRITON, reason="requires Triton")

GB300_BF16 = (64, 64, 16, 2, 4)
GB300_FP8 = (32, 64, 16, 2, 8)
XE2_BF16 = (32, 128, 16, 1, 16)

HEADS, HEAD_DIM, PAGE, COMPRESS = 4, 128, 64, 4
# Requests of 150, 37 and 200 tokens, the last split across two scoring chunks.
QUERY_LENS = (150, 37, 200)
VISIBLE_END = (1000, 130, 512)
TABLE_WIDTH = 16
TOPK = 32
CHUNK_ROWS = 300


def _bf16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).to(torch.float32)


def _inputs() -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    rows = sum(QUERY_LENS)
    q = torch.randn(rows, HEADS, HEAD_DIM)
    q = _bf16(q / q.norm(dim=-1, keepdim=True).clamp_min(1e-6) * HEAD_DIM**0.5)
    blocks = TABLE_WIDTH * len(QUERY_LENS)
    k = torch.randn(blocks, PAGE, 1, HEAD_DIM)
    k = _bf16(k / k.norm(dim=-1, keepdim=True).clamp_min(1e-6) * HEAD_DIM**0.5)
    table = torch.randperm(blocks, dtype=torch.int32).reshape(-1, TABLE_WIDTH)
    starts = [0]
    for query_len in QUERY_LENS:
        starts.append(starts[-1] + query_len)
    # A backing tensor's offset: the kernel subtracts the first entry.
    query_start_loc = torch.tensor(starts, dtype=torch.int32) + 7
    visible = torch.cat(
        [
            # Each row sees a growing causal prefix up to its request's end.
            ((torch.arange(query_len) * 5 + 3).clamp_max(end)).to(torch.int32)
            for query_len, end in zip(QUERY_LENS, VISIBLE_END, strict=True)
        ]
    )
    return {
        "q": q,
        "k_cache": k,
        "page_table": table,
        "query_start_loc": query_start_loc,
        "visible_blocks": visible,
    }


def _reference_row(inputs: dict[str, torch.Tensor], row: int, req: int) -> torch.Tensor:
    visible = int(inputs["visible_blocks"][row])
    columns = torch.arange(visible)
    pages = inputs["page_table"][req, columns // PAGE].long()
    keys = inputs["k_cache"][pages, columns % PAGE, 0].double()
    scores = inputs["q"][row].double() @ keys.T
    return scores.clamp_min(0.0).sum(0)


def _request_of_rows() -> list[int]:
    return [req for req, query_len in enumerate(QUERY_LENS) for _ in range(query_len)]


def _score_in_chunks(
    inputs: dict[str, torch.Tensor], width: int, launch_seen: list
) -> torch.Tensor:
    """Run the scorer as qsa_select_paged_prefill does, in chunks of CHUNK_ROWS."""
    rows = inputs["q"].shape[0]
    workspace = torch.empty(CHUNK_ROWS * width, dtype=torch.float32)
    out = torch.full((rows, width), float("nan"))
    launched = qsa_indexer._qsa_mqa_paged_prefill_kernel

    class Spy:
        def __getitem__(self, grid):
            call = launched[grid]

            def launch(*args, **kwargs):
                launch_seen.append(
                    (
                        kwargs["TILE_R"],
                        kwargs["BLOCK_N"],
                        kwargs["K_TILES"],
                        kwargs["STAGES"],
                        kwargs["num_warps"],
                    )
                )
                call(*args, **kwargs)

            return launch

    qsa_indexer._qsa_mqa_paged_prefill_kernel = Spy()
    for start in range(0, rows, CHUNK_ROWS):
        count = min(CHUNK_ROWS, rows - start)
        workspace.fill_(float("nan"))
        logits = qsa_indexer._prefill_logits(
            inputs["q"],
            inputs["k_cache"],
            inputs["page_table"],
            inputs["query_start_loc"],
            inputs["visible_blocks"],
            workspace,
            max(QUERY_LENS),
            width,
            query_offset=start,
            num_queries=count,
        )
        out[start : start + count] = logits
    qsa_indexer._qsa_mqa_paged_prefill_kernel = launched
    return out


def _run_interpreter(xpu: bool) -> torch.Tensor:
    from triton.runtime import interpreter

    builder = interpreter.InterpreterBuilder
    load, store = builder.create_masked_load, builder.create_masked_store

    # The interpreter builds a scalar & vector mask as an integer array, and then
    # refuses it in its own masked load and store.
    def masked_load(self, ptrs, mask, *args):
        mask = interpreter.TensorHandle(mask.data != 0, tl.int1)
        return load(self, ptrs, mask, *args)

    def masked_store(self, ptrs, value, mask, *args):
        mask = interpreter.TensorHandle(mask.data != 0, tl.int1)
        return store(self, ptrs, value, mask, *args)

    builder.create_masked_load = masked_load  # type: ignore[method-assign]
    builder.create_masked_store = masked_store  # type: ignore[method-assign]
    current_platform.is_xpu = lambda: xpu  # type: ignore[method-assign]

    inputs = _inputs()
    width = 1024
    seen: list = []
    out = _score_in_chunks(inputs, width, seen)
    assert set(seen) == {XE2_BF16 if xpu else GB300_BF16}
    return out


def _check_scores_and_selection(out_path: str) -> None:
    """Runs in a process with TRITON_INTERPRET=1, which triton reads at import."""
    logits = _run_interpreter(xpu=True)
    inputs = _inputs()
    requests = _request_of_rows()
    for row, req in enumerate(requests):
        visible = int(inputs["visible_blocks"][row])
        expected = _reference_row(inputs, row, req)
        actual = logits[row, :visible].double()
        assert not actual.isnan().any(), f"row {row}: an unwritten column"
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
        # Columns past a row's visible count stay unwritten.
        assert logits[row, visible:].isnan().all(), f"row {row}: wrote past visible"
    torch.save(logits, out_path)


def _check_stock_launch(out_path: str) -> None:
    torch.save(_run_interpreter(xpu=False), out_path)


def _in_interpreter(function: str, out_path: Path) -> torch.Tensor:
    root = Path(__file__).parents[3]
    code = (
        f"from tests.models.qwen4_exp.test_xpu_qsa_indexer import {function}\n"
        f"{function}({str(out_path)!r})\n"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        cwd=root,
        env={**os.environ, "TRITON_INTERPRET": "1"},
        timeout=280,
    )
    return torch.load(out_path)


def test_xe2_launch_scores_and_selects_like_the_gb300_launch(tmp_path) -> None:
    xe2 = _in_interpreter("_check_scores_and_selection", tmp_path / "xe2.pt")
    stock = _in_interpreter("_check_stock_launch", tmp_path / "stock.pt")
    inputs = _inputs()
    for row in range(xe2.shape[0]):
        visible = int(inputs["visible_blocks"][row])
        a, b = xe2[row, :visible], stock[row, :visible]
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)
        k = min(TOPK, visible)
        if not k:
            continue
        picked_a = set(a.topk(k).indices.tolist())
        picked_b = set(b.topk(k).indices.tolist())
        # Scores within a last place can swap a tie at the k-th block, no more.
        assert len(picked_a & picked_b) >= k - 1, f"row {row}"


@pytest.mark.parametrize(
    ("xpu", "dtype", "launch"),
    [
        pytest.param(True, torch.bfloat16, XE2_BF16, id="xpu_bf16"),
        pytest.param(True, torch.float8_e4m3fn, GB300_FP8, id="xpu_fp8"),
        pytest.param(False, torch.bfloat16, GB300_BF16, id="other_bf16"),
        pytest.param(False, torch.float8_e4m3fn, GB300_FP8, id="other_fp8"),
    ],
)
def test_launch_by_platform_and_dtype(monkeypatch, xpu, dtype, launch) -> None:
    kernel = MagicMock()
    monkeypatch.setattr(current_platform, "is_xpu", lambda: xpu)
    monkeypatch.setattr(qsa_indexer, "_qsa_mqa_paged_prefill_kernel", kernel)
    rows, width, requests = 300, 4096, 2
    qsa_indexer._prefill_logits(
        torch.zeros(rows, HEADS, HEAD_DIM, dtype=torch.bfloat16),
        torch.zeros(64, PAGE, 1, HEAD_DIM, dtype=dtype),
        torch.zeros(requests, TABLE_WIDTH * 4, dtype=torch.int32),
        torch.tensor([0, 100, rows], dtype=torch.int32),
        torch.zeros(rows, dtype=torch.int32),
        torch.empty(rows * width, dtype=torch.float32),
        200,
        width,
        query_offset=0,
        num_queries=rows,
    )
    tile_r, block_n, k_tiles, stages, num_warps = launch
    grid = kernel.__getitem__.call_args.args[0]
    assert grid == (requests, -(-200 // tile_r), -(-width // (block_n * k_tiles)))
    kwargs = kernel.__getitem__.return_value.call_args.kwargs
    assert kwargs["TILE_R"] == tile_r
    assert kwargs["BLOCK_N"] == block_n
    assert kwargs["K_TILES"] == k_tiles
    assert kwargs["STAGES"] == stages
    assert kwargs["num_warps"] == num_warps
    # Options of the Intel backend alone would break the CUDA launch.
    assert "grf_mode" not in kwargs
