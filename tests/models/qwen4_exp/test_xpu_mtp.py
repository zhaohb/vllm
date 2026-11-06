# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of the XPU Qwen4Exp MTP layer: the sparse draft layer for QSA targets.

The decoder layer and the dense sub-modules are replaced by stand-ins, so the
tests cover what the MTP module itself decides: layer type, top-k buffer, weight
names and the top-k hooks the proposer calls.
"""

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from torch import nn
from transformers import Qwen4ExpTextConfig

from vllm.config import VllmConfig
from vllm.models.qwen4_exp.xpu import mtp as xpu_mtp

MAX_TOKENS = 8
INDEXER_BUDGET, COMPRESS_RATIO = 12, 4
BUFFER_WIDTH = INDEXER_BUDGET + COMPRESS_RATIO


class _Indexer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.skip_topk = False


class _Attention(nn.Module):
    def __init__(self, topk_indices_buffer: torch.Tensor) -> None:
        super().__init__()
        self.indexer = _Indexer()
        self.topk_indices_buffer = topk_indices_buffer


class _FakeDecoderLayer(nn.Module):
    """Records how the predictor built it; sparse layers own an indexer."""

    def __init__(
        self,
        vllm_config: object,
        layer_type: str,
        prefix: str = "",
        topk_indices_buffer: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.layer_type = layer_type
        self.prefix = prefix
        self.topk_indices_buffer = topk_indices_buffer
        if layer_type == "qwen_sparse_attention":
            assert topk_indices_buffer is not None
            self.self_attn = _Attention(topk_indices_buffer)
        else:
            self.self_attn = nn.Identity()


def _config(qsa: bool, num_mtp_layers: int = 1) -> Qwen4ExpTextConfig:
    values: dict[str, Any] = {
        "vocab_size": 64,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "layer_types": ["linear_attention", "full_attention"],
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 2,
        "linear_key_head_dim": 8,
        "linear_value_head_dim": 8,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "eos_token_id": 1,
        "hc_count": 2,
        "hc_lowrank": 4,
        "ple_layer_ids": [1],
        "mtp_num_hidden_layers": num_mtp_layers,
        "mtp": {"hybrid": True},
    }
    config = Qwen4ExpTextConfig(**values)
    if qsa:
        config.indexer_n_heads = 2
        config.indexer_budget = INDEXER_BUDGET
        config.indexer_compress_ratio = COMPRESS_RATIO
    else:
        config.indexer_n_heads = None
    return config


def _vllm_config(config: Qwen4ExpTextConfig) -> VllmConfig:
    return cast(
        VllmConfig,
        SimpleNamespace(
            model_config=SimpleNamespace(hf_text_config=config, dtype=torch.bfloat16),
            scheduler_config=SimpleNamespace(max_num_batched_tokens=MAX_TOKENS),
            quant_config=None,
        ),
    )


def _build_predictor(
    monkeypatch: pytest.MonkeyPatch, config: Qwen4ExpTextConfig
) -> xpu_mtp.Qwen4ExpMultiTokenPredictor:
    def stand_in(*_args: object, **_kwargs: object) -> nn.Module:
        return nn.Identity()

    monkeypatch.setattr(xpu_mtp, "Qwen4ExpDecoderLayer", _FakeDecoderLayer)
    for name in (
        "VocabParallelEmbedding",
        "HyperConnectionRMSNorm",
        "_WideRMSNorm",
        "ReplicatedLinear",
        "HyperConnection",
    ):
        monkeypatch.setattr(xpu_mtp, name, stand_in)
    return xpu_mtp.Qwen4ExpMultiTokenPredictor(
        vllm_config=_vllm_config(config), prefix="mtp"
    )


def _predictor_with_layers(
    num_layers: int, num_sparse: int
) -> xpu_mtp.Qwen4ExpMultiTokenPredictor:
    """A predictor shell whose first `num_sparse` layers own an indexer."""
    predictor = object.__new__(xpu_mtp.Qwen4ExpMultiTokenPredictor)
    nn.Module.__init__(predictor)
    layers = []
    for i in range(num_layers):
        sparse = i < num_sparse
        buffer = torch.zeros(MAX_TOKENS, BUFFER_WIDTH, dtype=torch.int32)
        layers.append(
            _FakeDecoderLayer(
                None,
                "qwen_sparse_attention" if sparse else "full_attention",
                topk_indices_buffer=buffer if sparse else None,
            )
        )
    predictor.layers = nn.ModuleList(layers)
    return predictor


def test_qsa_config_builds_sparse_draft_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictor = _build_predictor(monkeypatch, _config(qsa=True))

    (layer,) = predictor.layers
    assert layer.layer_type == "qwen_sparse_attention"
    buffer = layer.topk_indices_buffer
    assert buffer is not None
    assert buffer.shape == (MAX_TOKENS, BUFFER_WIDTH)
    assert buffer.dtype == torch.int32


def test_dense_config_keeps_full_attention_draft_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictor = _build_predictor(monkeypatch, _config(qsa=False))

    (layer,) = predictor.layers
    assert layer.layer_type == "full_attention"
    assert layer.topk_indices_buffer is None


CHECKPOINT_NAMES = [
    "mtp.layers.0.self_attn.indexer.wq.weight",
    "mtp.layers.0.self_attn.qkv_proj.weight",
    "mtp.fc_hidden.weight",
    "model.language_model.embed_tokens.weight",
    "model.language_model.layers.3.self_attn.indexer.wq.weight",
    "model.language_model.layers.3.self_attn.qkv_proj.weight",
    "lm_head.weight",
]


def _remapped_names(qsa: bool) -> list[str]:
    mtp = object.__new__(xpu_mtp.Qwen4ExpMTP)
    nn.Module.__init__(mtp)
    mtp.config = _config(qsa=qsa)
    seen: list[str] = []

    class _Loader:
        def __init__(self, _module: nn.Module) -> None: ...

        def load_weights(self, weights):
            seen.extend(name for name, _ in weights)
            return set(seen)

    stream = ((name, torch.zeros(1)) for name in CHECKPOINT_NAMES)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(xpu_mtp, "AutoWeightsLoader", _Loader)
        mtp.load_weights(stream)
    return seen


def test_weight_remap_keeps_indexer_for_qsa() -> None:
    assert _remapped_names(qsa=True) == [
        "model.layers.0.self_attn.indexer.wq.weight",
        "model.layers.0.self_attn.qkv_proj.weight",
        "model.fc_hidden.weight",
        "model.embed_tokens.weight",
        "lm_head.weight",
    ]


def test_weight_remap_drops_indexer_for_dense() -> None:
    names = _remapped_names(qsa=False)

    assert not any(".indexer." in name for name in names)
    assert names == [
        "model.layers.0.self_attn.qkv_proj.weight",
        "model.fc_hidden.weight",
        "model.embed_tokens.weight",
        "lm_head.weight",
    ]


@pytest.mark.parametrize("skip", [True, False])
def test_set_skip_topk_reaches_every_indexer(skip: bool) -> None:
    predictor = _predictor_with_layers(num_layers=3, num_sparse=2)
    for layer in predictor.layers:
        if layer.layer_type == "qwen_sparse_attention":
            layer.self_attn.indexer.skip_topk = not skip

    predictor.set_skip_topk(skip)

    flags = [
        layer.self_attn.indexer.skip_topk
        for layer in predictor.layers
        if layer.layer_type == "qwen_sparse_attention"
    ]
    assert flags == [skip, skip]
    # A dense layer owns no indexer and is left alone.
    assert not hasattr(predictor.layers[2].self_attn, "indexer")


def test_compact_topk_indices_keeps_selected_rows_in_order() -> None:
    predictor = _predictor_with_layers(num_layers=2, num_sparse=2)
    for i, layer in enumerate(predictor.layers):
        buffer = layer.self_attn.topk_indices_buffer
        buffer.copy_(
            torch.arange(MAX_TOKENS * BUFFER_WIDTH, dtype=torch.int32).reshape_as(
                buffer
            )
            + i * 1000
        )
    before = [layer.self_attn.topk_indices_buffer.clone() for layer in predictor.layers]
    rows = torch.tensor([6, 2, 5])

    predictor.compact_topk_indices(rows)

    for old, layer in zip(before, predictor.layers):
        new = layer.self_attn.topk_indices_buffer
        # Selected rows land at the front, in the order given.
        assert torch.equal(new[:3], old[rows])
        # Rows past the compacted prefix are untouched.
        assert torch.equal(new[3:], old[3:])
