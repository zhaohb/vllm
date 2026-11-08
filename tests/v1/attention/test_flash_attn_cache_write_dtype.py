# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The dtype string FlashAttention hands to the cache write op."""

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends import flash_attn


@pytest.mark.parametrize(
    ("configured", "passed"),
    [
        ("auto", "auto"),
        ("bfloat16", "auto"),
        ("fp8_e4m3", "fp8_e4m3"),
        ("fp8", "fp8"),
    ],
)
def test_flash_cache_write_gets_a_dtype_the_op_knows(
    monkeypatch: pytest.MonkeyPatch, configured: str, passed: str
) -> None:
    seen: list[str] = []
    monkeypatch.setattr(
        flash_attn,
        "reshape_and_cache_flash",
        lambda *args: seen.append(args[5]),
        raising=False,
    )
    impl = SimpleNamespace(
        attn_type=AttentionType.DECODER,
        head_size=8,
        kv_cache_dtype=configured,
    )
    key = torch.zeros(4, 1, 8, dtype=torch.bfloat16)
    kv_cache = torch.zeros(1, 1, 4, 16, dtype=torch.bfloat16)
    flash_attn.FlashAttentionImpl.do_kv_cache_update(
        impl,
        SimpleNamespace(_k_scale=None, _v_scale=None),
        key,
        key,
        kv_cache,
        torch.arange(4),
    )
    assert seen == [passed]
