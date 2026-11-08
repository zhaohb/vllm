# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of DFlash2 on the XPU Qwen4Exp target.

The decoder layers and the mixer are stand-ins, so the tests cover what the target
decides: which layers feed the drafter, with what, and that the target's own results
don't depend on it. The feature contract and the QSA ring are checked as functions.
"""

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from vllm.model_executor.models import config as config_module
from vllm.models.qwen4_exp.common.qsa_indexer import QSARawKeyCache
from vllm.models.qwen4_exp.xpu import model as model_module
from vllm.models.qwen4_exp.xpu.model import Qwen4ExpModel
from vllm.platforms import current_platform
from vllm.v1.kv_cache_interface import CircularBufferSpec, KVCacheSpec

HC, N_EMBD, TOKENS = 4, 8, 5


class _HC:
    def __init__(self, calls: list[int], index: int):
        self.calls, self.index = calls, index

    def combine(self, residual, block_output, injection):
        self.calls.append(self.index)
        weight = 2 * torch.sigmoid(injection / HC).unsqueeze(-1)
        return residual + block_output.unsqueeze(1) * weight


class _Layer:
    """The layer's outputs depend on its inputs and index, and the pending combine
    has to reach the next layer untouched."""

    def __init__(self, index: int, calls: list[int]):
        self.index = index
        self.mlp_hyper_connection = _HC(calls, index)

    def __call__(
        self, positions, hidden_states, residual, block_output, injection, **_
    ):
        if residual is None:
            residual = hidden_states.unsqueeze(1).expand(-1, HC, -1).contiguous()
        if block_output is not None:
            residual = residual + block_output.unsqueeze(1) * injection.unsqueeze(-1)
        scale = self.index + 1.0
        block = residual.sum(dim=1) * scale / 16
        return residual, block, residual[:, :, 0] / scale


class _Mixer:
    def combine_and_mix(self, residual, block_output, injection):
        wide = residual + block_output.unsqueeze(1) * injection.unsqueeze(-1)
        return wide, wide.mean(dim=1), None


def _run(aux_layers: tuple[int, ...], monkeypatch, start: int = 0, end: int = 8):
    monkeypatch.setattr(
        model_module,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    calls: list[int] = []
    torch.manual_seed(3)
    model: Any = SimpleNamespace(
        layers=[_Layer(i, calls) for i in range(end)],
        start_layer=start,
        end_layer=end,
        hyper_connection_mixer=_Mixer(),
        _mtp_hidden_buffer=None,
        aux_hidden_state_layers=aux_layers,
    )
    out = Qwen4ExpModel.forward(
        cast(Any, model),
        input_ids=torch.zeros(TOKENS, dtype=torch.int64),
        positions=torch.arange(TOKENS),
        inputs_embeds=torch.randn(TOKENS, N_EMBD),
    )
    return out, calls


def test_no_aux_layers_returns_the_tensor(monkeypatch) -> None:
    out, calls = _run((), monkeypatch)
    assert isinstance(out, torch.Tensor)
    assert calls == []


def test_aux_layers_are_the_completed_stream_means(monkeypatch) -> None:
    plain, _ = _run((), monkeypatch)
    (hidden, aux), calls = _run((2, 6), monkeypatch)

    # The target's results do not change, and boundary n reads layer n - 1.
    assert torch.equal(hidden, plain)
    assert calls == [1, 5]
    assert len(aux) == 2

    # The same features from a target stepped by hand.
    torch.manual_seed(3)
    embeds = torch.randn(TOKENS, N_EMBD)
    residual, block, injection = None, None, None
    expected = []
    recorder: list[int] = []
    for index in range(8):
        layer = _Layer(index, recorder)
        residual, block, injection = layer(None, embeds, residual, block, injection)
        if index + 1 in (2, 6):
            wide = layer.mlp_hyper_connection.combine(residual, block, injection)
            expected.append(wide.mean(dim=1))
    for actual, reference in zip(aux, expected, strict=True):
        assert actual.shape == (TOKENS, N_EMBD)
        assert torch.equal(actual, reference)


def test_aux_layers_count_from_the_rank_start(monkeypatch) -> None:
    (_, aux), calls = _run((2, 6), monkeypatch, start=2)
    assert calls == [5]
    assert len(aux) == 1


def _draft(**overrides) -> SimpleNamespace:
    values: dict[str, Any] = {
        "architectures": ["DFlash2DraftModel"],
        "dflash_config": {"target_layer_ids": [5, 15, 25, 35, 45], "block_size": 8},
        "hidden_size": 2560,
        "vocab_size": 248320,
    }
    return SimpleNamespace(**(values | overrides))


TEXT = SimpleNamespace(
    hidden_size=2560, vocab_size=248320, hc_count=4, num_hidden_layers=48
)


def _spec(draft: SimpleNamespace, num_speculative_tokens: int = 7) -> Any:
    return SimpleNamespace(
        num_speculative_tokens=num_speculative_tokens,
        draft_model_config=SimpleNamespace(hf_config=draft),
    )


@pytest.fixture
def on_xpu(monkeypatch) -> None:
    monkeypatch.setattr(current_platform, "is_xpu", lambda: True)


@pytest.mark.parametrize("num_speculative_tokens", [1, 3, 7])
def test_contract_accepts_flash_next(on_xpu, num_speculative_tokens) -> None:
    config_module._verify_qwen4_exp_dflash2(
        _spec(_draft(), num_speculative_tokens), TEXT
    )


def test_contract_follows_the_drafter_not_one_checkpoint(on_xpu) -> None:
    """Other layer choices and block sizes pass when the drafter declares them."""
    draft = _draft(dflash_config={"target_layer_ids": [1, 24, 48], "block_size": 16})
    config_module._verify_qwen4_exp_dflash2(_spec(draft, 15), TEXT)


def _layers(ids) -> SimpleNamespace:
    return _draft(dflash_config={"target_layer_ids": ids, "block_size": 8})


@pytest.mark.parametrize(
    ("draft", "num_speculative_tokens"),
    [
        (_draft(architectures=["DFlashQwen3ForCausalLM"]), 7),
        (_layers([]), 7),
        (_layers([0, 15]), 7),
        (_layers([5, 49]), 7),
        (_layers(None), 7),
        (_draft(dflash_config={"block_size": 8}), 7),
        (_draft(hidden_size=1024), 7),
        (_draft(vocab_size=1024), 7),
        (_draft(dflash_config=None), 7),
        (_draft(), 8),
        (_draft(), 0),
    ],
)
def test_contract_refuses_the_rest(on_xpu, draft, num_speculative_tokens) -> None:
    with pytest.raises(NotImplementedError, match="DFlash2"):
        config_module._verify_qwen4_exp_dflash2(
            _spec(draft, num_speculative_tokens), TEXT
        )


def test_contract_refuses_other_platforms(monkeypatch) -> None:
    monkeypatch.setattr(current_platform, "is_xpu", lambda: False)
    with pytest.raises(NotImplementedError, match="XPU"):
        config_module._verify_qwen4_exp_dflash2(_spec(_draft()), TEXT)


def _ring(num_speculative_tokens: int, block_size: int) -> KVCacheSpec:
    # The layer's own cache config is a copy from before the block size
    # resolved (a draft's is); the ring must widen against the resolved one.
    cache: Any = SimpleNamespace(
        compress_ratio=4,
        cache_config=SimpleNamespace(block_size=64),
        head_size=128,
        dtype=torch.bfloat16,
    )
    live = SimpleNamespace(
        num_speculative_tokens=num_speculative_tokens,
        cache_config=SimpleNamespace(block_size=block_size),
    )
    return QSARawKeyCache.get_kv_cache_spec(cache, cast(Any, live))


@pytest.mark.parametrize(
    ("num_speculative_tokens", "block_size", "capacity"),
    [
        # The minimum whenever it divides the block, as before.
        (3, 832, 8),
        (4, 832, 8),
        (7, 960, 12),
        # 12 doesn't divide 832 = 13 x 64: the next whole group that does.
        (7, 1024, 16),
        (5, 832, 16),
        (7, 832, 16),
        (8, 832, 16),
    ],
)
def test_ring_divides_the_block(num_speculative_tokens, block_size, capacity) -> None:
    spec = _ring(num_speculative_tokens, block_size)
    assert isinstance(spec, CircularBufferSpec)
    assert spec.block_size == capacity
    assert block_size % spec.block_size == 0
    assert spec.block_size >= 4 + num_speculative_tokens
