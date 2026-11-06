# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The XPU PLE hash and row gather on a card, bitwise, eagerly and in graph replay.

Needs one XPU device; skipped without one. Rows are synthetic unless
``PLE_INT8_TABLE`` names a real table, whose rank 0 share of ``PLE_TP_SIZE`` (default
4) is then read into real pinned slabs. ``PLE_ITERATIONS`` sets the rounds.

    ZE_AFFINITY_MASK=0 PLE_INT8_TABLE=<table> PLE_ITERATIONS=1200 \\
        pytest -s tests/models/qwen4_exp/test_xpu_ple_device.py
"""

import os

import numpy as np
import pytest
import torch

from vllm.models.qwen4_exp.common.ops.ple import ple_ngram_ids
from vllm.models.qwen4_exp.xpu.ple_int8 import read_table_header
from vllm.models.qwen4_exp.xpu.ple_rows import (
    PinnedRowSlabs,
    RowExtent,
    host_ngram_ids,
)
from vllm.models.qwen4_exp.xpu.ple_table import gather_rows_from_slabs
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

pytestmark = pytest.mark.skipif(
    not (hasattr(torch, "xpu") and torch.xpu.is_available()), reason="needs an XPU card"
)

ITERATIONS = int(os.environ.get("PLE_ITERATIONS", "300"))
EOS, HEADS_PER_NGRAM = 248044, 8
ROW_BYTES = 164


@pytest.fixture(scope="module")
def table() -> dict:
    tensors: list[torch.Tensor] = []

    def allocate(rows: int, row_bytes: int) -> np.ndarray:
        tensors.append(
            torch.empty((rows, row_bytes), dtype=torch.uint8, pin_memory=True)
        )
        assert tensors[-1].is_pinned()
        return tensors[-1].numpy()

    path = os.environ.get("PLE_INT8_TABLE")
    if path:
        start, rows, row_bytes, layout = read_table_header(path)
        share = rows // int(os.environ.get("PLE_TP_SIZE", "4"))
        slabs = PinnedRowSlabs(0, share, row_bytes, alloc=allocate)
        slabs.load([RowExtent(path, start, rows)])
        multipliers = layout["layer_multipliers"]
        sizes = layout["ngram_heads_vocab_sizes"]
        offsets = layout["ngram_heads_offsets"]
    else:
        # Six slabs, a row crossing no special boundary, as the real table's share.
        rows = 4 << 18
        slabs = PinnedRowSlabs(
            0,
            rows // 4,
            ROW_BYTES,
            alloc=allocate,
            max_bytes=1 << 25,
            tail_bytes=1 << 16,
        )
        generator = np.random.default_rng(3)
        for slab in slabs.slabs:
            slab[:] = generator.integers(0, 256, slab.shape, dtype=np.uint8)
        multipliers = np.array([1_000_003, 998_244_353, 2_147_483_647], dtype=np.int64)
        sizes = np.full(16, rows // 16, dtype=np.int64)
        offsets = np.arange(16, dtype=np.int64) * (rows // 16)
    return {
        "slabs": slabs,
        "views": [get_accelerator_view_from_cpu_tensor(t) for t in tensors],
        "multipliers": multipliers,
        "sizes": sizes,
        "offsets": offsets,
        "keepalive": tensors,
    }


def _batch(generator: np.random.Generator, kind: int):
    """Tokens, query_start_loc and context of one step shape, padding included."""
    lengths = {
        0: [4096],  # a full prefill chunk
        1: [8] * int(generator.integers(1, 17)),  # speculative decodes
        2: generator.integers(1, 64, int(generator.integers(2, 9))).tolist(),  # ragged
        3: [int(generator.integers(512, 2048))]
        + [1] * int(generator.integers(1, 8)),  # prefill and decode
        4: generator.integers(8, 256, 3).tolist(),  # many EOS
        5: generator.integers(16, 512, 2).tolist(),  # high ids overflow the products
    }[kind]
    total = int(sum(lengths))
    tokens = generator.integers(0, 248000, total, dtype=np.int64)
    if kind == 4:
        tokens[generator.random(total) < 0.15] = EOS
    if kind == 5:
        tokens = generator.integers(248000, 248320, total, dtype=np.int64)
    context = generator.integers(0, 248000, (len(lengths), 2), dtype=np.int64)
    qsl = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    return tokens, qsl, context


def _device_ids(table: dict, tokens, qsl, context) -> torch.Tensor:
    return ple_ngram_ids(
        torch.from_numpy(tokens.astype(np.int32)).to("xpu"),
        torch.from_numpy(qsl.astype(np.int32)).to("xpu"),
        torch.from_numpy(context.astype(np.int32)).to("xpu"),
        torch.from_numpy(table["multipliers"]).to("xpu"),
        torch.from_numpy(table["sizes"]).to("xpu"),
        torch.from_numpy(table["offsets"]).to("xpu"),
        EOS,
        HEADS_PER_NGRAM,
    )


def _host_ids(table: dict, tokens, qsl, context) -> np.ndarray:
    return host_ngram_ids(
        tokens,
        qsl,
        context,
        multipliers=table["multipliers"],
        sizes=table["sizes"],
        offsets=table["offsets"],
        eos_token_id=EOS,
        heads_per_ngram=HEADS_PER_NGRAM,
        heads=np.arange(table["sizes"].shape[0]),
    )


def _expected(slabs: PinnedRowSlabs, ids: np.ndarray) -> np.ndarray:
    out = np.zeros((*ids.shape, slabs.row_bytes), dtype=np.uint8)
    owned = (ids >= slabs.start) & (ids < slabs.end)
    out[owned] = slabs.rows(ids[owned])
    return out


def _gather(table: dict, ids: torch.Tensor, out: torch.Tensor) -> None:
    slabs = table["slabs"]
    gather_rows_from_slabs(
        table["views"],
        slabs.bounds.tolist(),
        slabs.start,
        slabs.end,
        slabs.row_bytes,
        ids,
        out,
    )


def test_hash_and_gather_match_the_host_eagerly(table) -> None:
    generator = np.random.default_rng(5)
    slabs = table["slabs"]
    bad_hash = bad_gather = 0
    slabs_hit: set[int] = set()
    for iteration in range(ITERATIONS):
        tokens, qsl, context = _batch(generator, iteration % 6)
        host = _host_ids(table, tokens, qsl, context)
        ids = _device_ids(table, tokens, qsl, context)
        bad_hash += not np.array_equal(ids.cpu().numpy(), host)
        out = torch.full(
            (*host.shape, slabs.row_bytes), 0xCD, dtype=torch.uint8, device="xpu"
        )
        _gather(table, ids, out)
        bad_gather += not np.array_equal(out.cpu().numpy(), _expected(slabs, host))
        local = host[(host >= slabs.start) & (host < slabs.end)] - slabs.start
        slabs_hit.update(
            np.unique(np.searchsorted(slabs.bounds, local, side="right") - 1).tolist()
        )
    print(f"eager: {ITERATIONS} rounds, slabs hit {sorted(slabs_hit)}")
    assert (bad_hash, bad_gather) == (0, 0)
    assert len(slabs_hit) == len(slabs.slabs)


@pytest.mark.parametrize("tokens", [8, 4096])
def test_gather_matches_the_host_in_graph_replay(table, tokens: int) -> None:
    """The gather as a captured forward does it: fixed buffers, new ids each replay."""
    generator = np.random.default_rng(6)
    slabs = table["slabs"]
    heads = table["sizes"].shape[0]
    ids = torch.zeros(tokens, heads, dtype=torch.int64, device="xpu")
    out = torch.zeros(tokens, heads, slabs.row_bytes, dtype=torch.uint8, device="xpu")
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph, stream=torch.xpu.current_stream()):
        _gather(table, ids, out)
    torch.xpu.synchronize()
    bad = 0
    for _ in range(ITERATIONS):
        host = np.stack(
            [
                generator.integers(offset, offset + size, tokens, dtype=np.int64)
                for offset, size in zip(table["offsets"], table["sizes"])
            ],
            axis=1,
        )
        ids.copy_(torch.from_numpy(host))
        graph.replay()
        bad += not np.array_equal(out.cpu().numpy(), _expected(slabs, host))
    print(f"replay: {ITERATIONS} rounds of {tokens} tokens")
    assert bad == 0
