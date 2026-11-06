# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""XPU attn2 kernels against a torch reference, in the Triton interpreter on CPU.

The interpreter reads bfloat16 operands of tl.dot as integers, so the kernels run
in float16 here. test_qsa_reference.py covers BF16 and FP8 on a card.
"""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops
from vllm.models.qwen4_exp.xpu.ops import qsa_attn2
from vllm.models.qwen4_exp.xpu.ops.qsa_attn2 import _Launch
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl

pytestmark = pytest.mark.skipif(not HAS_TRITON, reason="requires Triton")

ROWS, SELECTED, HEAD_DIM, HEADS, PAGE, BLOCKS, TABLE_WIDTH = 5, 150, 128, 6, 32, 40, 8


def _inputs() -> SimpleNamespace:
    """Two requests, holes, partial tiles, a bad page and an empty last row."""
    torch.manual_seed(0)
    q = torch.randn(ROWS, HEADS, HEAD_DIM).to(torch.float16)
    gate = torch.randn(ROWS, HEADS, HEAD_DIM).to(torch.float16)
    kv = torch.randn(BLOCKS, PAGE, 1, 2 * HEAD_DIM).to(torch.float16)
    k_cache, v_cache = kv.split(HEAD_DIM, dim=-1)
    table = torch.randint(0, BLOCKS, (2, TABLE_WIDTH), dtype=torch.int32)
    table[1, 3] = BLOCKS + 5
    token_to_req = torch.tensor([i % 2 for i in range(ROWS)], dtype=torch.int32)
    indices = torch.full((ROWS, SELECTED + 1), -1, dtype=torch.int32)
    for row in range(ROWS - 1):
        count = SELECTED - 7 * row
        tokens = torch.randperm(TABLE_WIDTH * PAGE)[:count].to(torch.int32)
        tokens[5] = -1
        indices[row, : tokens.numel()] = tokens
        indices[row, SELECTED] = tokens.numel()
    indices[ROWS - 1, SELECTED] = 0
    return SimpleNamespace(
        q=q,
        gate=gate,
        k_cache=k_cache,
        v_cache=v_cache,
        table=table,
        token_to_req=token_to_req,
        indices=indices,
    )


def _reference(inputs: SimpleNamespace) -> torch.Tensor:
    out = torch.zeros(ROWS, HEADS, HEAD_DIM)
    for row in range(ROWS):
        tokens = inputs.indices[row, : int(inputs.indices[row, SELECTED])].long()
        tokens = tokens[tokens >= 0]
        pages = inputs.table[int(inputs.token_to_req[row]), tokens // PAGE].long()
        valid = (pages >= 0) & (pages < BLOCKS)
        tokens, pages = tokens[valid], pages[valid]
        if not tokens.numel():
            continue
        keys = inputs.k_cache[pages, tokens % PAGE, 0].float()
        values = inputs.v_cache[pages, tokens % PAGE, 0].float()
        scores = inputs.q[row].float() @ keys.T * HEAD_DIM**-0.5
        attended = torch.softmax(scores, -1) @ values
        gate = torch.sigmoid(inputs.gate[row].float())
        out[row] = attended.to(torch.float16).float() * gate
    return out


def _check_against_reference(block_n, num_dchunk, row_chunk, tiles_per_prog) -> None:
    """Runs in a process with TRITON_INTERPRET=1, which triton reads at import."""
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
    qsa_attn2._DECODE = _Launch(block_n, num_dchunk, row_chunk, tiles_per_prog)
    # A launch option of the XPU backend only.
    qsa_attn2._GRF_MODE = {}
    inputs = _inputs()
    out = torch.empty_like(inputs.q)
    qsa_attn2.qsa_sparse_paged_attention_attn2(
        inputs.q,
        inputs.k_cache,
        inputs.v_cache,
        inputs.indices,
        inputs.table,
        inputs.token_to_req,
        False,
        out,
        inputs.gate,
        HEAD_DIM**-0.5,
        1.0,
        8,
    )
    torch.testing.assert_close(out.float(), _reference(inputs), rtol=0, atol=2e-2)


@pytest.mark.parametrize(
    "launch",
    [
        pytest.param((32, 8, 4096, 1), id="shipped_decode"),
        pytest.param((32, 8, 512, 1), id="shipped_prefill"),
        pytest.param((16, 4, 4096, 1), id="one_tile_per_program"),
        pytest.param((32, 2, 2, 3), id="row_chunks"),
        pytest.param((64, 8, 4096, 2), id="wide_tiles"),
    ],
)
def test_attn2_matches_reference(launch) -> None:
    root = Path(__file__).parents[3]
    code = (
        "from tests.models.qwen4_exp.test_qsa_attn2 import _check_against_reference\n"
        f"_check_against_reference{launch}\n"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        cwd=root,
        env={**os.environ, "TRITON_INTERPRET": "1"},
        timeout=300,
    )


def test_shipped_launches_fit_the_chunking_and_regions() -> None:
    for launch in (qsa_attn2._DECODE, qsa_attn2._PREFILL):
        assert 256 // launch.num_dchunk >= 16
        assert launch.block_n >= 16
    assert qsa_attn2._select_launch(4096, True) is qsa_attn2._PREFILL
    assert qsa_attn2._select_launch(4096, False) is qsa_attn2._DECODE
    assert qsa_attn2._select_launch(16, True) is qsa_attn2._DECODE


@pytest.mark.parametrize("xpu", [True, False])
def test_only_xpu_dispatches_to_attn2(monkeypatch, xpu) -> None:
    attn2 = MagicMock(return_value="attn2")
    splitk = MagicMock()
    monkeypatch.setattr(current_platform, "is_xpu", lambda: xpu)
    monkeypatch.setattr(qsa_attn2, "qsa_sparse_paged_attention_attn2", attn2)
    monkeypatch.setattr(qsa_ops, "_qsa_sparse_paged_gqa_splitk_kernel", splitk)
    monkeypatch.setattr(qsa_ops, "_select_config", lambda *_: (16, 1, 1, 1))
    inputs = _inputs()
    q = inputs.q.to(torch.bfloat16)
    k_cache = inputs.k_cache.to(torch.bfloat16)
    v_cache = inputs.v_cache.to(torch.bfloat16)
    qsa_ops.qsa_sparse_paged_attention(
        q,
        k_cache,
        v_cache,
        inputs.indices,
        inputs.table,
        inputs.token_to_req,
        False,
        output_gate=inputs.gate.to(torch.bfloat16),
    )
    assert attn2.called is xpu
    assert splitk.__getitem__.called is not xpu
