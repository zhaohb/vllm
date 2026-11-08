# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of the XPU Qwen4Exp PLE layer around its table: the packed key and
value projection, the prefill checkpoints of its short conv, and the refusal of
the V1 model runner."""

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from transformers import Qwen4ExpTextConfig

from vllm.config import VllmConfig
from vllm.config.engram import EngramConfig
from vllm.model_executor import parameter
from vllm.model_executor.layers import linear
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.models import config as model_configs
from vllm.models.qwen4_exp.xpu import model as xpu_model
from vllm.platforms import current_platform
from vllm.v1.attention.backends.short_conv_attn import PleShortConvAttentionMetadata
from vllm.v1.kv_cache_interface import MambaSpec

HC, HIDDEN, EMBED = 2, 8, 12
NUM_TOKENS = 5


@pytest.fixture(autouse=True)
def _no_tp(monkeypatch: pytest.MonkeyPatch) -> None:
    for module in (linear, parameter):
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 1)
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 0)


def _make_ple(mamba_cache_mode: str = "none") -> xpu_model.Qwen4ExpPLE:
    config = SimpleNamespace(
        hc_count=HC,
        hidden_size=HIDDEN,
        rms_norm_eps=1e-6,
        ngram_size=3,
        heads_per_ngram=2,
        ple_embed_dim=EMBED,
        ple_conv_kernel_size=4,
        eos_token_id=0,
        make_ngram_vocab_size_divisible_by=1,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(dtype=torch.bfloat16),
        cache_config=SimpleNamespace(mamba_cache_mode=mamba_cache_mode),
        num_speculative_tokens=0,
        compilation_config=SimpleNamespace(static_forward_context={}),
        load_config=SimpleNamespace(load_format="dummy"),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8),
        parallel_config=SimpleNamespace(data_parallel_rank=0),
        quant_config=None,
        engram_config=EngramConfig(cpu_offload=False),
    )
    return xpu_model.Qwen4ExpPLE(
        cast(Qwen4ExpTextConfig, config), cast(VllmConfig, vllm_config), prefix="ple"
    )


def test_key_and_value_load_into_one_projection() -> None:
    ple = _make_ple()
    key = torch.randn(HC * HIDDEN, EMBED)
    value = torch.randn(HIDDEN, EMBED)
    mapper = xpu_model.Qwen4ExpModel.hf_to_vllm_mapper
    loaded = dict(ple.named_parameters())
    for name, weight in (
        ("ple.key_proj.weight", key),
        ("ple.value_proj.weight", value),
    ):
        mapped = mapper._map_name_with_shard(f"model.layers.1.{name}")
        assert mapped is not None
        mapped_name, shard_id = mapped
        assert mapped_name == "model.layers.1.ple.kv_proj.weight"
        param = loaded["kv_proj.weight"]
        cast(Any, param).weight_loader(param, weight, shard_id)
    rows = torch.randn(NUM_TOKENS, EMBED)
    # The packed weight, applied as the layer's GEMM applies it.
    packed = torch.nn.functional.linear(rows, ple.kv_proj.weight)
    got_key, got_value = packed.split(ple.kv_proj.output_sizes, dim=-1)
    torch.testing.assert_close(got_key, rows @ key.t())
    torch.testing.assert_close(got_value, rows @ value.t())
    for cls in (
        xpu_model.Qwen4ExpForCausalLM,
        xpu_model.Qwen4ExpForConditionalGeneration,
    ):
        assert cls.packed_modules_mapping["kv_proj"] == ["key_proj", "value_proj"]


@pytest.mark.parametrize(("mode", "alignment"), [("align", 1), ("all", None)])
def test_checkpoint_alignment_follows_the_cache_mode(mode, alignment) -> None:
    assert _make_ple(mode).get_prefill_checkpoint_alignment() == alignment


def test_kv_cache_spec_takes_checkpoints_when_every_layer_can(monkeypatch) -> None:
    ple = _make_ple("align")
    spec = MambaSpec(shapes=((4,),), dtypes=(torch.bfloat16,), block_size=16)
    monkeypatch.setattr(MambaBase, "get_kv_cache_spec", lambda self, config: spec)
    monkeypatch.setattr(
        xpu_model, "get_mamba_prefill_checkpoint_grid", lambda config: 16
    )
    monkeypatch.setattr(
        xpu_model, "get_mamba_prefill_checkpoint_alignment", lambda config: None
    )
    assert ple.get_kv_cache_spec(cast(VllmConfig, None)) is spec
    monkeypatch.setattr(
        xpu_model, "get_mamba_prefill_checkpoint_alignment", lambda config: 1
    )
    exported = ple.get_kv_cache_spec(cast(VllmConfig, None))
    assert isinstance(exported, MambaSpec)
    assert exported.num_prefill_checkpoint_blocks == 1
    assert exported.prefill_checkpoint_alignment == 16


def test_prefill_exports_its_checkpoint(monkeypatch) -> None:
    ple = _make_ple("align")
    channels = HC * HIDDEN
    ple.kv_cache = (torch.zeros(2, channels, ple.hist + 1),)
    metadata = PleShortConvAttentionMetadata.__new__(PleShortConvAttentionMetadata)
    fields = dict(
        num_actual_tokens=NUM_TOKENS,
        spec_sequence_masks=None,
        num_prefills=1,
        num_decodes=0,
        num_decode_tokens=0,
        num_prefill_tokens=NUM_TOKENS,
        state_indices_tensor=torch.tensor([1]),
        query_start_loc_p=torch.tensor([0, NUM_TOKENS]),
        has_initial_states_p=torch.tensor([False]),
        has_initial_states_d=None,
        non_spec_query_start_loc=torch.tensor([0, NUM_TOKENS]),
        checkpoint=object(),
    )
    for name, value in fields.items():
        object.__setattr__(metadata, name, value)
    context = SimpleNamespace(attn_metadata={"ple": metadata})
    monkeypatch.setattr(xpu_model, "get_forward_context", lambda: context)
    monkeypatch.setattr(xpu_model, "is_conv_state_dim_first", lambda: True)
    modes: list[str] = []
    monkeypatch.setattr(
        xpu_model, "ple_conv", lambda *args, mode, **kwargs: modes.append(mode)
    )
    exports: list[dict[str, Any]] = []
    assert ple._checkpoint_exporter.state_len == ple.hist
    monkeypatch.setattr(
        ple,
        "_checkpoint_exporter",
        SimpleNamespace(
            export=lambda checkpoint, **kwargs: exports.append(
                dict(checkpoint=checkpoint, **kwargs)
            )
        ),
    )
    inputs = torch.randn(NUM_TOKENS, channels)
    ple.short_conv(inputs, torch.zeros_like(inputs), torch.zeros_like(inputs))
    assert modes == ["prefill"]
    assert len(exports) == 1
    assert exports[0]["checkpoint"] is metadata.checkpoint
    assert exports[0]["raw_qkv"].shape == inputs.shape
    assert exports[0]["conv_state"].shape[-1] == ple.hist
    assert exports[0]["cu_seqlens"] is metadata.non_spec_query_start_loc


def _verify(monkeypatch, *, xpu: bool, v2: bool, ple_layer_ids: list[int]) -> None:
    monkeypatch.setattr(current_platform, "is_xpu", lambda: xpu)
    monkeypatch.setattr(current_platform, "is_cuda_alike", lambda: not xpu)
    monkeypatch.setattr(
        model_configs.Qwen3_5ForConditionalGenerationConfig,
        "verify_and_update_config",
        staticmethod(lambda vllm_config: None),
    )
    text_config = SimpleNamespace(
        hc_count=2, ple_layer_ids=ple_layer_ids, indexer_n_heads=None
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=text_config, multimodal_config=None
        ),
        parallel_config=SimpleNamespace(
            use_sequence_parallel_moe=False,
            pipeline_parallel_size=1,
            enable_dbo=False,
            ubatch_size=1,
        ),
        speculative_config=None,
        use_v2_model_runner=v2,
    )
    model_configs.Qwen4ExpForConditionalGenerationConfig.verify_and_update_config(
        cast(VllmConfig, vllm_config)
    )


def test_v1_runner_is_refused_for_ple_on_xpu(monkeypatch) -> None:
    with pytest.raises(NotImplementedError, match="VLLM_USE_V2_MODEL_RUNNER=1"):
        _verify(monkeypatch, xpu=True, v2=False, ple_layer_ids=[2])


@pytest.mark.parametrize(
    ("xpu", "v2", "ple_layer_ids"),
    [
        pytest.param(True, True, [2], id="xpu_v2"),
        pytest.param(True, False, [], id="xpu_v1_no_ple"),
        pytest.param(False, False, [2], id="cuda_v1"),
    ],
)
def test_v1_runner_is_refused_only_for_ple_on_xpu(
    monkeypatch, xpu, v2, ple_layer_ids
) -> None:
    _verify(monkeypatch, xpu=xpu, v2=v2, ple_layer_ids=ple_layer_ids)
