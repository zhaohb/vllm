# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Qwen4Exp draft's own KV dtype, on CPU tensors."""

import dataclasses
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.config import CacheConfig
from vllm.config.speculative import SpeculativeConfig
from vllm.models.qwen4_exp.common.qsa_indexer import (
    QSACompressedKeyCache,
    QSARawKeyCache,
    qsa_kv_cache_dtype,
)
from vllm.models.qwen4_exp.nvidia import (
    model as _qwen4_exp_model,  # noqa: F401  (import order: model before qsa)
)
from vllm.models.qwen4_exp.nvidia import qsa as qsa_module
from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops
from vllm.models.qwen4_exp.nvidia.qsa import Qwen4ExpQSAAttention
from vllm.models.qwen4_exp.nvidia.qsa_indexer import QSAIndexer
from vllm.models.qwen4_exp.xpu import model as xpu_model
from vllm.v1.kv_cache_interface import FullAttentionSpec, get_kv_quant_mode

HD = 256


# --------------------------------------------------------------------------
# The draft's KV dtype
# --------------------------------------------------------------------------
def test_draft_override_reaches_the_mtp_owner_only() -> None:
    target = CacheConfig(cache_dtype="fp8_e4m3")
    draft = dataclasses.replace(target, cache_dtype="bfloat16")
    assert qsa_kv_cache_dtype(target, "model.layers.3.self_attn") == "fp8_e4m3"
    assert qsa_kv_cache_dtype(draft, "mtp.layers.0.self_attn") == "bfloat16"
    fields = {f.name: f for f in dataclasses.fields(SpeculativeConfig)}
    assert fields["kv_cache_dtype"].default is None


def _page(dtype: torch.dtype, mode: str, draft: bool) -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=1600,
        num_kv_heads=1,
        head_size=HD,
        head_size_v=HD,
        dtype=dtype,
        kv_quant_mode=get_kv_quant_mode(mode),
        is_draft=draft,
    )


def test_bf16_draft_page_is_twice_the_fp8_target_page() -> None:
    target = _page(torch.uint8, "fp8_e4m3", False)
    draft = _page(torch.bfloat16, "auto", True)
    assert draft.page_size_bytes == 2 * target.page_size_bytes


class _Owner(Qwen4ExpQSAAttention):
    def __init__(self, dtype: str, name: str) -> None:
        nn.Module.__init__(self)
        self.kv_cache_dtype = dtype
        self.layer_name = name
        self.kv_cache = torch.zeros(1)
        self.num_heads = 6
        self.head_dim = HD


class _Indexer(QSAIndexer):
    output_width = 8

    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.compressed_key_cache = SimpleNamespace(
            kv_cache=torch.zeros(1), prefix="indexer"
        )
        self.index_n_heads = 2
        self.index_head_dim = 128


@pytest.mark.parametrize(
    "draft_dtype,warmed",
    [("bfloat16", ["fp8_e4m3", "bfloat16"]), ("fp8_e4m3", ["fp8_e4m3"])],
)
def test_warmup_covers_a_draft_with_its_own_dtype(
    draft_dtype: str, warmed: list[str], monkeypatch
) -> None:
    from vllm.model_executor.warmup import qwen4_exp_qsa_warmup as warmup
    from vllm.models.qwen4_exp.nvidia.ops import qsa_indexer as indexer_ops

    calls: list[str] = []
    owners = {"t": _Owner("fp8_e4m3", "t"), "d": _Owner(draft_dtype, "d")}

    def fake_attention_warmup(kv_cache, block_table, **kwargs):
        calls.append(str(block_table))
        return ()

    monkeypatch.setattr(
        qsa_ops, "warmup_qsa_sparse_paged_attention", fake_attention_warmup
    )
    monkeypatch.setattr(indexer_ops, "warmup_qsa_mqa_paged_decode", lambda *a, **k: ())
    target = nn.ModuleList([_Indexer(), owners["t"]])
    runner = SimpleNamespace(
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(layer_names=["indexer"]),
                SimpleNamespace(layer_names=["t"]),
                SimpleNamespace(layer_names=["d"]),
            ]
        ),
        input_batch=SimpleNamespace(
            block_table=[
                SimpleNamespace(get_device_tensor=lambda _n, g=g: f"table{g}")
                for g in range(3)
            ]
        ),
        max_num_reqs=4,
        max_num_tokens=16,
        uniform_decode_query_len=4,
        drafter=SimpleNamespace(model=nn.ModuleList([owners["d"]])),
    )
    worker = SimpleNamespace(
        get_model=lambda: target, model_runner=runner, use_v2_model_runner=False
    )
    warmup.qwen4_exp_qsa_triton_warmup(worker)
    want = ["table1"] + (["table2"] if len(warmed) == 2 else [])
    assert calls == want


# --------------------------------------------------------------------------
# A draft's cache config is a copy from before the block size resolves
# --------------------------------------------------------------------------
@pytest.mark.parametrize("draft_dtype", ["bfloat16", "auto"])
def test_draft_qsa_caches_use_the_resolved_block_size(draft_dtype: str) -> None:
    resolved = 1600
    built_with = CacheConfig(block_size=64, cache_dtype=draft_dtype)
    live = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=resolved),
        num_speculative_tokens=3,
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    compressed = QSACompressedKeyCache(
        head_size=128,
        dtype=torch.bfloat16,
        cache_config=built_with,
        prefix="mtp.layers.0.self_attn.indexer.compressed_key_cache",
        vllm_config=live,
        compress_ratio=4,
    )
    raw = QSARawKeyCache(
        head_size=140,
        dtype=torch.bfloat16,
        cache_config=built_with,
        prefix="mtp.layers.0.self_attn.indexer.raw_key_cache",
        vllm_config=live,
        compress_ratio=4,
    )
    assert compressed.get_kv_cache_spec(live).block_size == resolved
    assert raw.get_kv_cache_spec(live).block_size == 8


# --------------------------------------------------------------------------
# The unfused QSA cache write takes a plain dtype string as "auto"
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("cache_dtype", "op_dtype"),
    [("bfloat16", "auto"), ("auto", "auto"), ("fp8_e4m3", "fp8_e4m3")],
)
def test_qsa_cache_write_passes_the_op_a_dtype_it_knows(
    monkeypatch: pytest.MonkeyPatch, cache_dtype: str, op_dtype: str
) -> None:
    impl = object.__new__(qsa_module.Qwen4ExpQSAImpl)
    impl.head_size = 8
    impl.kv_cache_dtype = cache_dtype
    calls: list[str] = []
    monkeypatch.setattr(
        qsa_module.ops,
        "reshape_and_cache_flash",
        lambda *args: calls.append(args[5]),
    )
    key = torch.zeros(2, 1, 8, dtype=torch.bfloat16)
    kv_cache = torch.zeros(4, 1, 1, 16, dtype=torch.bfloat16)
    layer = SimpleNamespace(_k_scale=torch.ones(1), _v_scale=torch.ones(1))
    impl.do_kv_cache_update(layer, key, key.clone(), kv_cache, torch.tensor([0, 1]))
    assert calls == [op_dtype]


# --------------------------------------------------------------------------
# The XPU loader: serialized KV scales of QSA layers
# --------------------------------------------------------------------------
class _QSALayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = object.__new__(Qwen4ExpQSAAttention)
        nn.Module.__init__(self.self_attn)


def test_xpu_loader_moves_qsa_kv_scales_onto_the_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = object.__new__(xpu_model.Qwen4ExpModel)
    nn.Module.__init__(model)
    model.layers = nn.ModuleList([nn.Identity(), _QSALayer()])
    model.config = SimpleNamespace()
    model.max_model_len = 16
    seen: dict[str, object] = {}

    class _Loader:
        def __init__(self, _module: nn.Module, **kwargs: object) -> None:
            seen["kwargs"] = kwargs

        def load_weights(self, weights, mapper):
            seen["names"] = [name for name, _ in weights]
            return set()

    monkeypatch.setattr(xpu_model, "AutoWeightsLoader", _Loader)
    names = [
        "layers.1.self_attn.k_proj.k_scale",
        "layers.1.self_attn.attn.v_scale",
        "layers.0.self_attn.attn.k_scale",
        "layers.1.self_attn.qkv_proj.weight",
    ]
    model.load_weights((name, torch.ones(1)) for name in names)
    assert seen["names"] == [
        "layers.1.self_attn._k_scale",
        "layers.1.self_attn._v_scale",
        "layers.0.self_attn.attn.k_scale",
        "layers.1.self_attn.qkv_proj.weight",
    ]
    suffixes = seen["kwargs"]["ignore_unexpected_suffixes"]  # type: ignore[index]
    assert ".k_scale" in suffixes and "_input_scale" in suffixes
