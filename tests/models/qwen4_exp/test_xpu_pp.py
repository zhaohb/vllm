# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of the XPU Qwen4Exp model under pipeline parallelism.

The decoder layers and the embedding are stand-ins, so the tests cover what the
model decides per PP stage: which stages hold the token embedding, and which
MoE blocks an EPLB rebalance updates.
"""

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.config.compilation import CompilationMode
from vllm.model_executor.models.qwen3_next import Qwen3NextSparseMoeBlock
from vllm.model_executor.models.utils import PPMissingLayer
from vllm.models.qwen4_exp.xpu import model as xpu_model

NUM_EXPERTS = 4


class _Experts:
    def __init__(self) -> None:
        self.updates = 0

    def update_expert_map(self) -> None:
        self.updates += 1


def _moe_layer() -> xpu_model.Qwen4ExpDecoderLayer:
    layer = object.__new__(xpu_model.Qwen4ExpDecoderLayer)
    nn.Module.__init__(layer)
    moe = object.__new__(Qwen3NextSparseMoeBlock)
    nn.Module.__init__(moe)
    moe.experts = _Experts()
    moe.n_logical_experts = NUM_EXPERTS
    moe.n_physical_experts = NUM_EXPERTS
    moe.n_local_physical_experts = NUM_EXPERTS
    moe.n_routed_experts = NUM_EXPERTS
    moe.n_redundant_experts = 0
    layer.mlp = moe
    return layer


def _stage_layers() -> list[nn.Module]:
    """One PP stage's view: the other stages' layers are placeholders."""
    return [PPMissingLayer(), _moe_layer(), _moe_layer(), PPMissingLayer()]


def _check_rebalance(host: Any, layers: list[nn.Module]) -> None:
    host.set_moe_parameters()
    assert len(host.moe_layers) == 2
    host.update_physical_experts_metadata(
        num_physical_experts=NUM_EXPERTS + 2,
        num_local_physical_experts=NUM_EXPERTS,
    )
    assert host.num_redundant_experts == 2
    for layer in layers:
        if isinstance(layer, PPMissingLayer):
            continue
        moe = layer.mlp
        assert moe.n_physical_experts == NUM_EXPERTS + 2
        assert moe.n_redundant_experts == 2
        assert moe.experts.updates == 1


def test_eplb_rebalance_skips_the_layers_of_other_stages() -> None:
    host = object.__new__(xpu_model.Qwen4ExpForCausalLM)
    nn.Module.__init__(host)
    layers = _stage_layers()
    host.model = SimpleNamespace(layers=layers)
    _check_rebalance(host, layers)


def test_eplb_rebalance_skips_the_layers_of_other_stages_vl() -> None:
    # The wrapper's abstract interface methods are not needed here.
    host = object.__new__(cast(Any, xpu_model.Qwen4ExpForConditionalGeneration))
    nn.Module.__init__(host)
    layers = _stage_layers()
    host.language_model = SimpleNamespace(model=SimpleNamespace(layers=layers))
    _check_rebalance(host, layers)


class _Embedding(nn.Module):
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        super().__init__()


def _build_model(
    monkeypatch: pytest.MonkeyPatch, *, first: bool, last: bool, tied: bool
) -> xpu_model.Qwen4ExpModel:
    monkeypatch.setattr(xpu_model, "VocabParallelEmbedding", _Embedding)
    monkeypatch.setattr(
        xpu_model, "make_layers", lambda *_args, **_kwargs: (0, 0, nn.ModuleList())
    )
    monkeypatch.setattr(
        xpu_model, "HyperConnection", lambda *_args, **_kwargs: nn.Identity()
    )
    monkeypatch.setattr(
        xpu_model,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=first, is_last_rank=last),
    )
    config = SimpleNamespace(
        vocab_size=16,
        hidden_size=8,
        hc_count=2,
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
        indexer_n_heads=None,
        tie_word_embeddings=tied,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=config, max_model_len=64, dtype=torch.bfloat16
        ),
        quant_config=None,
        speculative_config=None,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8),
        compilation_config=SimpleNamespace(mode=CompilationMode.NONE),
    )
    return xpu_model.Qwen4ExpModel(vllm_config=cast(VllmConfig, vllm_config))


@pytest.mark.parametrize(
    ("first", "last", "tied", "embeds"),
    [
        pytest.param(True, False, False, True, id="first"),
        pytest.param(False, False, False, False, id="middle"),
        pytest.param(False, True, False, False, id="last"),
        pytest.param(False, True, True, True, id="last_tied"),
        pytest.param(False, False, True, False, id="middle_tied"),
        pytest.param(True, True, False, True, id="single"),
    ],
)
def test_token_embedding_only_where_a_stage_reads_it(
    monkeypatch: pytest.MonkeyPatch,
    first: bool,
    last: bool,
    tied: bool,
    embeds: bool,
) -> None:
    model = _build_model(monkeypatch, first=first, last=last, tied=tied)
    assert isinstance(model.embed_tokens, _Embedding) is embeds
    assert isinstance(model.embed_tokens, PPMissingLayer) is not embeds
