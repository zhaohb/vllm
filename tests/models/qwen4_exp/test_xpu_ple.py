# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of the XPU Qwen4Exp PLE table: the kernels are replaced by torch."""

from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import Qwen4ExpTextConfig

import vllm.model_executor.layers.vocab_parallel_embedding as embedding_module
import vllm.model_executor.parameter as parameter_module
import vllm.models.qwen4_exp.common.ngram_embedding as common_embedding_module
import vllm.models.qwen4_exp.nvidia.ngram_embedding as nvidia_embedding_module
from vllm.config import VllmConfig
from vllm.config.engram import EngramConfig
from vllm.config.model import ModelConfig
from vllm.model_executor import parameter
from vllm.model_executor.layers import linear
from vllm.models.qwen4_exp.common.ngram_embedding import (
    Qwen4ExpPLEEmbeddingMethod,
    Qwen4ExpPLEPinnedHostEmbedding,
)
from vllm.models.qwen4_exp.nvidia.ngram_embedding import (
    Qwen4ExpNGramEmbedding,
    Qwen4ExpPLEDiskOffloadEmbedding,
)
from vllm.models.qwen4_exp.xpu import model as xpu_model
from vllm.models.qwen4_exp.xpu.model import Qwen4ExpPLE, stage_ple_rows
from vllm.models.qwen4_exp.xpu.model_state import Qwen4ExpModelState
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState

NUM_ROWS, ROW_DIM, NUM_HEADS = 24, 32, 2
# Row ids of three tokens, two heads each, crossing the 6-row checkpoint shards.
ROW_IDS = torch.tensor([[0, 13], [5, 23], [6, 6]], dtype=torch.int64)
TABLE = torch.linspace(-2, 2, NUM_ROWS * ROW_DIM).reshape(NUM_ROWS, ROW_DIM)
FP8_SCALE = 0.25


@pytest.fixture(autouse=True)
def _no_tp(monkeypatch: pytest.MonkeyPatch) -> None:
    # The PLE key/value projection is a replicated linear layer.
    for module in (linear, parameter):
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 1)
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 0)


def _expected_rows(fmt: str) -> torch.Tensor:
    if fmt == "fp8":
        rows = TABLE.to(torch.float8_e4m3fn).to(torch.bfloat16)
        rows = rows * torch.tensor(FP8_SCALE, dtype=torch.bfloat16)
    else:
        rows = TABLE.to(torch.bfloat16)
    return rows[ROW_IDS].flatten(-2)


@pytest.fixture
def checkpoint(tmp_path: Path, request: pytest.FixtureRequest) -> tuple[Path, str]:
    fmt = getattr(request, "param", "bf16")
    dtype = torch.float8_e4m3fn if fmt == "fp8" else torch.bfloat16
    tensors = {
        f"ngram_embedding.shard_{i}.weight": shard.clone()
        for i, shard in enumerate(TABLE.to(dtype).split(6))
    }
    if fmt == "fp8":
        tensors["ngram_embedding.weight_scale"] = torch.tensor([FP8_SCALE])
    path = tmp_path / "ple.safetensors"
    save_file(tensors, path)
    return path, fmt


def _make_ple(monkeypatch: pytest.MonkeyPatch, engram_config, fmt: str) -> Qwen4ExpPLE:
    def hash_ids(*, input_ids, output, **_):
        assert input_ids.numel() == ROW_IDS.shape[0]
        return output.copy_(ROW_IDS)

    monkeypatch.setattr(xpu_model, "ple_ngram_ids", hash_ids)
    group = SimpleNamespace(rank_in_group=0, world_size=1, all_reduce=lambda x: x)
    for module in (common_embedding_module, nvidia_embedding_module):
        monkeypatch.setattr(module, "get_etp_group", lambda: group)
        monkeypatch.setattr(module, "get_tp_group", lambda: group)
    for module in (embedding_module, parameter_module):
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 0)
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 1)
    config = SimpleNamespace(
        hc_count=2,
        hidden_size=8,
        rms_norm_eps=1e-6,
        ngram_size=2,
        heads_per_ngram=NUM_HEADS,
        ple_embed_dim=NUM_HEADS * ROW_DIM,
        ple_conv_kernel_size=2,
        eos_token_id=0,
        vocab_size=64,
        split_ngram_parts=4,
        ngram_vocab_size_base=8,
        make_ngram_vocab_size_divisible_by=1,
        ple_embedding_dtype="float8_e4m3fn" if fmt == "fp8" else None,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(dtype=torch.bfloat16),
        cache_config=SimpleNamespace(),
        num_speculative_tokens=0,
        compilation_config=SimpleNamespace(static_forward_context={}),
        load_config=SimpleNamespace(load_format="auto"),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8),
        parallel_config=SimpleNamespace(data_parallel_rank=0),
        quant_config=None,
        engram_config=engram_config,
    )
    monkeypatch.setattr(
        nvidia_embedding_module, "get_current_vllm_config", lambda: vllm_config
    )
    return Qwen4ExpPLE(cast(Qwen4ExpTextConfig, config), vllm_config, prefix="ple")


def _load_mapped(ple: Qwen4ExpPLE, path: Path) -> None:
    with safe_open(path, framework="pt") as checkpoint:
        for name in checkpoint.keys():  # noqa: SIM118
            leaf = name.removeprefix("ngram_embedding.")
            ple.accept_checkpoint_tensor(leaf, checkpoint.get_tensor(name))
    ple.ple_embedding.ngram_heads_offsets.data.copy_(torch.tensor([0, 11]))
    ple.ple_embedding.ngram_heads_vocab_sizes.data.copy_(torch.tensor([11, 13]))
    ple.finalize_table()


def _load_table(ple: Qwen4ExpPLE, path: Path) -> None:
    table = ple.ple_embedding
    assert isinstance(table, Qwen4ExpNGramEmbedding)
    with safe_open(path, framework="pt") as checkpoint:
        table.load_weights(
            (name, checkpoint.get_tensor(name))
            for name in checkpoint.keys()  # noqa: SIM118
        )
    embedding = table.ngram_embedding
    embedding.quant_method.process_weights_after_loading(embedding)


def _stage(ple: Qwen4ExpPLE) -> None:
    stage_ple_rows(
        [ple],
        torch.zeros(ROW_IDS.shape[0], dtype=torch.int32),
        torch.tensor([0, ROW_IDS.shape[0]], dtype=torch.int32),
        torch.zeros(1, 1, dtype=torch.int32),
    )


NO_OFFLOAD = SimpleNamespace(cpu_offload=False, disk_offload=False)


@pytest.mark.parametrize("checkpoint", ["bf16", "fp8"], indirect=True)
def test_mapped_table_is_read_before_the_forward(monkeypatch, checkpoint) -> None:
    path, fmt = checkpoint
    ple = _make_ple(monkeypatch, NO_OFFLOAD, fmt)
    _load_mapped(ple, path)
    _stage(ple)
    # The forward's read must not touch the table: a captured graph replays it.
    ple._ckpt_shards.clear()
    rows = ple.lookup(ROW_IDS.shape[0], torch.bfloat16)
    torch.testing.assert_close(rows, _expected_rows(fmt), rtol=0, atol=0)


@pytest.mark.parametrize("checkpoint", ["bf16", "fp8"], indirect=True)
def test_disk_offload_reads_the_shared_table(monkeypatch, checkpoint) -> None:
    path, fmt = checkpoint
    ple = _make_ple(
        monkeypatch, SimpleNamespace(cpu_offload=True, disk_offload=True), fmt
    )
    table = ple.ple_embedding
    assert isinstance(table, Qwen4ExpNGramEmbedding)
    assert isinstance(table.ngram_embedding, Qwen4ExpPLEDiskOffloadEmbedding)
    assert not hasattr(ple, "quant_method")
    _load_table(ple, path)
    _stage(ple)
    rows = ple.lookup(ROW_IDS.shape[0], torch.bfloat16)
    torch.testing.assert_close(rows, _expected_rows(fmt), rtol=0, atol=0)


@pytest.mark.parametrize("checkpoint", ["bf16", "fp8"], indirect=True)
def test_cpu_offload_looks_rows_up_without_a_host_read(monkeypatch, checkpoint) -> None:
    path, fmt = checkpoint

    def lookup_from_pinned(self, layer, ids, output) -> None:
        output.copy_(layer._uva_weight[ids].reshape(output.shape))

    monkeypatch.setattr(common_embedding_module, "is_uva_available", lambda: True)
    monkeypatch.setattr(
        common_embedding_module, "get_accelerator_view_from_cpu_tensor", lambda x: x
    )
    monkeypatch.setattr(
        Qwen4ExpPLEPinnedHostEmbedding,
        "allocate_embedding_weight",
        lambda self, rows, dim, dtype: torch.empty(rows, dim, dtype=dtype),
    )
    monkeypatch.setattr(
        Qwen4ExpPLEEmbeddingMethod, "lookup_from_pinned", lookup_from_pinned
    )
    ple = _make_ple(
        monkeypatch, SimpleNamespace(cpu_offload=True, disk_offload=False), fmt
    )
    table = ple.ple_embedding
    assert isinstance(table, Qwen4ExpNGramEmbedding)
    assert isinstance(table.ngram_embedding, Qwen4ExpPLEPinnedHostEmbedding)
    assert not ple.reads_rows_on_host
    _load_table(ple, path)
    _stage(ple)
    rows = ple.lookup(ROW_IDS.shape[0], torch.bfloat16)
    torch.testing.assert_close(rows, _expected_rows(fmt), rtol=0, atol=0)


def _qwen4_exp_model(architecture: str = "Qwen4ExpForCausalLM") -> ModelConfig:
    field = "ple_layer_ids" if "Qwen4Exp" in architecture else "engram_layer_ids"
    return cast(
        ModelConfig,
        SimpleNamespace(
            architecture=architecture,
            hf_text_config=SimpleNamespace(**{field: [2]}),
        ),
    )


@pytest.mark.parametrize(
    ("architecture", "xpu", "cpu_offload"),
    [
        pytest.param("Qwen4ExpForCausalLM", True, False, id="qwen4exp_xpu"),
        pytest.param("Qwen4ExpForCausalLM", False, True, id="qwen4exp_other"),
        pytest.param("DeepseekV41ForCausalLM", True, True, id="deepseek_xpu"),
    ],
)
def test_engram_default_only_changes_for_qwen4exp_on_xpu(
    monkeypatch, architecture, xpu, cpu_offload
) -> None:
    monkeypatch.setattr(current_platform, "is_xpu", lambda: xpu)
    config = SimpleNamespace(
        model_config=_qwen4_exp_model(architecture),
        speculative_config=None,
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            enable_elastic_ep=False,
            use_ubatching=False,
        ),
        engram_config=None,
    )
    VllmConfig._resolve_and_verify_engram_config(cast(VllmConfig, config))
    assert config.engram_config.cpu_offload is cpu_offload


def test_disk_offload_passes_the_platform_check_on_xpu(monkeypatch) -> None:
    monkeypatch.setattr(current_platform, "is_cuda", lambda: False)
    monkeypatch.setattr(current_platform, "is_xpu", lambda: True)
    config = EngramConfig(disk_offload=True)
    config.verify_model_config(_qwen4_exp_model())
    with pytest.raises(ValueError, match="disk_offload"):
        config.verify_model_config(_qwen4_exp_model("DeepseekV41ForCausalLM"))


def test_model_state_stages_rows_and_passes_no_ple_inputs() -> None:
    state = object.__new__(Qwen4ExpModelState)
    state.uses_ngram_embedding = True
    layer = object.__new__(Qwen4ExpPLE)
    state.ple_layers = [layer]
    state.ple_query_start_loc = torch.zeros(5, dtype=torch.int32)
    input_batch = SimpleNamespace(
        num_reqs_after_padding=2,
        query_start_loc=torch.tensor([0, 2, 3], dtype=torch.int32),
        input_ids=torch.arange(3),
    )
    context = torch.zeros(2, 1, dtype=torch.int32)
    with (
        patch.object(MambaHybridModelState, "prepare_inputs", return_value={"a": 1}),
        patch.object(
            Qwen4ExpModelState, "_prepare_ngram_context", return_value=context
        ),
        patch(
            "vllm.models.qwen4_exp.xpu.model_state.stage_ple_rows",
        ) as stage,
    ):
        inputs = state.prepare_inputs(input_batch, SimpleNamespace())
    assert inputs == {"a": 1}
    layers, input_ids, query_start_loc, ngram_context = stage.call_args.args
    assert layers == [layer]
    assert input_ids is input_batch.input_ids
    assert query_start_loc.tolist() == [0, 2, 3]
    assert ngram_context is context
