# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of the XPU Qwen4Exp PLE tables: the kernels are replaced by torch."""

import gc
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import numpy as np
import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import Qwen4ExpTextConfig

from vllm.config import VllmConfig
from vllm.config.engram import EngramConfig
from vllm.config.model import ModelConfig
from vllm.config.parallel import ParallelConfig
from vllm.model_executor import parameter
from vllm.model_executor.layers import linear
from vllm.models.qwen4_exp.xpu import model as xpu_model
from vllm.models.qwen4_exp.xpu import ple_int8, ple_table
from vllm.models.qwen4_exp.xpu.model import Qwen4ExpPLE, stage_ple_rows
from vllm.models.qwen4_exp.xpu.model_state import Qwen4ExpModelState
from vllm.models.qwen4_exp.xpu.ple_table import DiskRowTable, PinnedRowTable
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState

NUM_ROWS, ROW_DIM, NUM_HEADS = 24, 32, 2
# Row ids of three tokens, two heads each, crossing the 6-row checkpoint shards.
ROW_IDS = torch.tensor([[0, 13], [5, 23], [6, 6]], dtype=torch.int64)
TABLE = torch.linspace(-2, 2, NUM_ROWS * ROW_DIM).reshape(NUM_ROWS, ROW_DIM)
FP8_SCALE = 0.25
OFFSETS = torch.tensor([0, 11])
SIZES = torch.tensor([11, 13])
MULTIPLIERS = torch.tensor([1, 3])

PREFIX = "model.layers.1.ple.ple_embedding."


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
    elif fmt == "int8":
        packed = ple_int8.quantize_rows(TABLE.to(torch.bfloat16))
        rows = ple_int8.decode_rows(packed).to(torch.bfloat16)
    else:
        rows = TABLE.to(torch.bfloat16)
    return rows[ROW_IDS].flatten(-2)


def _save_checkpoint(folder: Path, dtype: torch.dtype, table=TABLE) -> Path:
    tensors = {
        f"{PREFIX}ngram_embedding.shard_{i}.weight": shard.clone()
        for i, shard in enumerate(table.to(dtype).split(6))
    }
    if dtype == torch.float8_e4m3fn:
        tensors[f"{PREFIX}ngram_embedding.weight_scale"] = torch.tensor([FP8_SCALE])
    tensors[f"{PREFIX}ngram_heads_offsets"] = OFFSETS.clone()
    tensors[f"{PREFIX}ngram_heads_vocab_sizes"] = SIZES.clone()
    tensors[f"{PREFIX}layer_multipliers"] = MULTIPLIERS.clone()
    folder.mkdir(exist_ok=True)
    save_file(tensors, folder / "model.safetensors")
    return folder


@pytest.fixture
def checkpoint(tmp_path: Path, request: pytest.FixtureRequest) -> tuple[Path, str]:
    fmt = getattr(request, "param", "bf16")
    dtype = torch.float8_e4m3fn if fmt == "fp8" else torch.bfloat16
    return _save_checkpoint(tmp_path / "ckpt", dtype), fmt


def _engram(fmt: str, folder: Path, **fields) -> EngramConfig:
    table = None
    if fmt == "int8":
        table = folder / "int8.safetensors"
        if not table.exists():
            ple_int8.build(folder, table)
    fields.setdefault("disk_row_cache_gib", 1e-3)
    fields.setdefault("disk_read_threads", 2)
    return EngramConfig(int8_table_path=None if table is None else str(table), **fields)


def _make_ple(
    monkeypatch: pytest.MonkeyPatch, engram_config, fmt: str, rank=0, world=1
) -> Qwen4ExpPLE:
    def hash_ids(*, input_ids, output, **_):
        assert input_ids.numel() == ROW_IDS.shape[0]
        return output.copy_(ROW_IDS)

    def gather(views, bounds, start, end, row_bytes, ids, output) -> None:
        rows = torch.cat(views)
        owned = (ids >= start) & (ids < end)
        output.zero_()
        output[owned] = rows[ids[owned] - start]

    monkeypatch.setattr(xpu_model, "ple_ngram_ids", hash_ids)
    monkeypatch.setattr(ple_table, "gather_rows_from_slabs", gather)
    monkeypatch.setattr(ple_table, "get_accelerator_view_from_cpu_tensor", lambda x: x)
    group = SimpleNamespace(
        rank_in_group=rank, world_size=world, all_reduce=lambda x: x
    )
    monkeypatch.setattr(xpu_model, "get_tp_group", lambda: group)
    monkeypatch.setattr(ple_table, "get_tp_group", lambda: group)
    config = SimpleNamespace(
        hc_count=2,
        hidden_size=8,
        rms_norm_eps=1e-6,
        ngram_size=2,
        heads_per_ngram=NUM_HEADS,
        ple_embed_dim=NUM_HEADS * ROW_DIM,
        ple_conv_kernel_size=2,
        eos_token_id=0,
        make_ngram_vocab_size_divisible_by=1,
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
    return Qwen4ExpPLE(cast(Qwen4ExpTextConfig, config), vllm_config, prefix="ple")


def _load(ple: Qwen4ExpPLE, folder: Path) -> None:
    constants = ple.ple_embedding
    constants.ngram_heads_offsets.data.copy_(OFFSETS)
    constants.ngram_heads_vocab_sizes.data.copy_(SIZES)
    constants.layer_multipliers = torch.nn.Parameter(
        MULTIPLIERS.clone(), requires_grad=False
    )
    with safe_open(folder / "model.safetensors", framework="pt") as source:
        for name in source.keys():  # noqa: SIM118
            if ".ngram_embedding." in name:
                leaf = name.rsplit("ngram_embedding.", 1)[1]
                ple.accept_checkpoint_tensor(leaf, source.get_tensor(name))
    ple.finalize_table()


def _stage(ple: Qwen4ExpPLE, num_tokens: int = ROW_IDS.shape[0]) -> None:
    stage_ple_rows(
        [ple],
        SimpleNamespace(
            input_ids=torch.zeros(num_tokens, dtype=torch.int32),
            num_tokens=num_tokens,
            idx_mapping_np=np.zeros(0, dtype=np.int64),
            num_reqs=0,
            num_scheduled_tokens=np.zeros(0, dtype=np.int32),
        ),
        SimpleNamespace(
            num_computed_prefill_tokens=np.zeros(0, dtype=np.int32),
            prefill_len=SimpleNamespace(np=np.zeros(0, dtype=np.int32)),
            all_token_ids=SimpleNamespace(),
        ),
        torch.tensor([0, num_tokens], dtype=torch.int32),
        torch.zeros(1, 1, dtype=torch.int32),
    )


NO_OFFLOAD = EngramConfig(cpu_offload=False)
FORMATS = ["bf16", "fp8", "int8"]


@pytest.mark.parametrize("checkpoint", ["bf16", "fp8"], indirect=True)
def test_mapped_table_is_read_before_the_forward(monkeypatch, checkpoint) -> None:
    folder, fmt = checkpoint
    ple = _make_ple(monkeypatch, NO_OFFLOAD, fmt)
    _load(ple, folder)
    assert ple.row_table is None
    _stage(ple)
    # The forward's read must not touch the table: a captured graph replays it.
    ple._ckpt_shards.clear()
    rows = ple.lookup(ROW_IDS.shape[0], torch.bfloat16)
    torch.testing.assert_close(rows, _expected_rows(fmt), rtol=0, atol=0)


@pytest.mark.parametrize("checkpoint", FORMATS, indirect=True)
def test_disk_offload_reads_rows_through_the_cache(monkeypatch, checkpoint) -> None:
    folder, fmt = checkpoint
    ple = _make_ple(monkeypatch, _engram(fmt, folder, disk_offload=True), fmt)
    _load(ple, folder)
    assert isinstance(ple.row_table, DiskRowTable)
    assert ple.reads_rows_on_host
    assert not ple._ckpt_shards
    _stage(ple)
    rows = ple.lookup(ROW_IDS.shape[0], torch.bfloat16)
    torch.testing.assert_close(rows, _expected_rows(fmt), rtol=0, atol=0)
    # The same rows again come from the cache.
    monkeypatch.setattr(ple.row_table._server.reader, "read_rows", None)
    _stage(ple)
    rows = ple.lookup(ROW_IDS.shape[0], torch.bfloat16)
    torch.testing.assert_close(rows, _expected_rows(fmt), rtol=0, atol=0)


@pytest.mark.parametrize("checkpoint", FORMATS, indirect=True)
def test_cpu_offload_gathers_rows_without_a_host_read(monkeypatch, checkpoint) -> None:
    folder, fmt = checkpoint
    ple = _make_ple(monkeypatch, _engram(fmt, folder, cpu_offload=True), fmt)
    _load(ple, folder)
    assert isinstance(ple.row_table, PinnedRowTable)
    assert not ple.reads_rows_on_host
    _stage(ple)
    rows = ple.lookup(ROW_IDS.shape[0], torch.bfloat16)
    torch.testing.assert_close(rows, _expected_rows(fmt), rtol=0, atol=0)


@pytest.mark.parametrize("checkpoint", FORMATS, indirect=True)
@pytest.mark.parametrize("placement", ["cpu_offload", "disk_offload"])
def test_ranks_sum_to_the_full_rows(monkeypatch, checkpoint, placement) -> None:
    folder, fmt = checkpoint
    packed = []
    for rank in range(2):
        config = _engram(fmt, folder, **{placement: True})
        ple = _make_ple(monkeypatch, config, fmt, rank=rank, world=2)
        _load(ple, folder)
        _stage(ple)
        ple.lookup(ROW_IDS.shape[0], torch.bfloat16)
        table = cast(DiskRowTable | PinnedRowTable, ple.row_table)
        assert table.end - table.start == NUM_ROWS // 2
        packed.append(table.packed[: ROW_IDS.shape[0]].clone())
    assert not ((packed[0] != 0) & (packed[1] != 0)).any()
    table = cast(DiskRowTable | PinnedRowTable, ple.row_table)
    rows = table.format.decode(packed[0] + packed[1], torch.bfloat16).flatten(-2)
    torch.testing.assert_close(rows, _expected_rows(fmt), rtol=0, atol=0)


def test_int8_table_of_another_checkpoint_is_refused(monkeypatch, tmp_path) -> None:
    other = _save_checkpoint(tmp_path / "other", torch.bfloat16, TABLE.flip(0) + 1)
    table = tmp_path / "other.safetensors"
    ple_int8.build(other, table)
    folder = _save_checkpoint(tmp_path / "ckpt", torch.bfloat16)
    config = EngramConfig(
        int8_table_path=str(table), disk_row_cache_gib=1e-3, disk_read_threads=2
    )
    ple = _make_ple(monkeypatch, config, "int8")
    with pytest.raises(ValueError, match="not this checkpoint's table"):
        _load(ple, folder)


def test_int8_table_of_another_hash_layout_is_refused(monkeypatch, tmp_path) -> None:
    folder = _save_checkpoint(tmp_path / "ckpt", torch.bfloat16)
    config = _engram("int8", folder, cpu_offload=True)
    ple = _make_ple(monkeypatch, config, "int8")
    ple.ple_embedding.ngram_heads_offsets.data.copy_(OFFSETS)
    ple.ple_embedding.ngram_heads_vocab_sizes.data.copy_(SIZES)
    ple.ple_embedding.layer_multipliers = torch.nn.Parameter(
        MULTIPLIERS + 2, requires_grad=False
    )
    with pytest.raises(ValueError, match="layer_multipliers differs"):
        ple._int8_table_extents(ple._ple_constants())


def test_small_row_cache_is_refused_at_boot(monkeypatch, tmp_path) -> None:
    folder = _save_checkpoint(tmp_path / "ckpt", torch.bfloat16)
    config = EngramConfig(disk_offload=True, disk_row_cache_gib=1e-6)
    ple = _make_ple(monkeypatch, config, "bf16")
    with pytest.raises(ValueError, match="disk_row_cache_gib"):
        _load(ple, folder)


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


@pytest.mark.parametrize(
    "fields",
    [
        {"int8_table_path": "/table"},
        {"disk_row_cache_gib": 4.0},
        {"disk_read_threads": 4},
        {"disk_lookahead": False},
        {"disk_stats_interval_s": 60.0},
    ],
)
def test_xpu_table_settings_are_refused_elsewhere(monkeypatch, fields) -> None:
    monkeypatch.setattr(current_platform, "is_xpu", lambda: True)
    EngramConfig(**fields).verify_model_config(_qwen4_exp_model())
    monkeypatch.setattr(current_platform, "is_xpu", lambda: False)
    with pytest.raises(ValueError, match="on XPU"):
        EngramConfig(**fields).verify_model_config(_qwen4_exp_model())


def test_int8_table_needs_a_placement() -> None:
    with pytest.raises(ValueError, match="cpu_offload or disk_offload"):
        EngramConfig(cpu_offload=False, int8_table_path="/table")


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
    req_states = SimpleNamespace()
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
        inputs = state.prepare_inputs(input_batch, req_states)
    assert inputs == {"a": 1}
    layers, batch, states, query_start_loc, ngram_context = stage.call_args.args
    assert layers == [layer]
    assert batch is input_batch
    assert states is req_states
    assert query_start_loc.tolist() == [0, 2, 3]
    assert ngram_context is context


def _disk_table(monkeypatch, tmp_path: Path) -> DiskRowTable:
    folder = _save_checkpoint(tmp_path / "ckpt", torch.bfloat16)
    ple = _make_ple(monkeypatch, _engram("bf16", folder, disk_offload=True), "bf16")
    _load(ple, folder)
    table = ple.row_table
    assert isinstance(table, DiskRowTable)
    return table


def test_disk_table_close_stops_the_lookahead_and_the_readers(
    monkeypatch, tmp_path
) -> None:
    table = _disk_table(monkeypatch, tmp_path)
    lookahead = table._lookahead
    assert lookahead is not None and lookahead._thread.is_alive()
    readers = [table._server.reader, lookahead.reader]
    assert all(reader._fds for reader in readers)
    table.close()
    assert not lookahead._thread.is_alive()
    assert not any(reader._fds for reader in readers)
    table.close()


def test_disk_table_closes_when_collected(monkeypatch, tmp_path) -> None:
    table = _disk_table(monkeypatch, tmp_path)
    lookahead = table._lookahead
    assert lookahead is not None
    thread = lookahead._thread
    del table
    gc.collect()
    assert not thread.is_alive()


def test_pinned_table_needs_uva(monkeypatch, tmp_path) -> None:
    folder = _save_checkpoint(tmp_path / "ckpt", torch.bfloat16)
    ple = _make_ple(monkeypatch, _engram("bf16", folder, cpu_offload=True), "bf16")
    monkeypatch.setattr(ple_table, "is_uva_available", lambda: False)
    with pytest.raises(RuntimeError, match="UVA"):
        _load(ple, folder)


@pytest.mark.parametrize(("xpu", "shared"), [(True, False), (False, True)])
def test_dp_shared_memory_defaults_off_on_xpu(monkeypatch, xpu, shared) -> None:
    monkeypatch.setattr(current_platform, "is_xpu", lambda: xpu)
    config = EngramConfig(cpu_offload=True)
    parallel = SimpleNamespace(data_parallel_size=2, enable_elastic_ep=False)
    config.resolve_dp_shared_memory(cast(ParallelConfig, parallel))
    assert config.dp_shared_memory is shared
