# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""XPU fused hyper-connections on CPU, kernels in the Triton interpreter.

``_Unfused`` is the torch-only HC the fused one replaces. Each BF16 op there rounds
its own result, while the kernels hold fp32 and round once per materialized tensor,
and a kernel sums in its own order. So they are not bitwise, in BF16 or fp32; the
test bounds the difference in units of the last place at the tensor's largest value.
A card runs the real kernels.
"""

import os
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode

from vllm.model_executor import parameter
from vllm.model_executor.layers import linear
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.models.qwen4_exp.xpu import hyperconnection as hc_module
from vllm.models.qwen4_exp.xpu import model as model_module
from vllm.models.qwen4_exp.xpu import mtp as mtp_module
from vllm.models.qwen4_exp.xpu.hyperconnection import HyperConnection
from vllm.models.qwen4_exp.xpu.model import Qwen4ExpDecoderLayer, Qwen4ExpModel
from vllm.models.qwen4_exp.xpu.mtp import Qwen4ExpMultiTokenPredictor
from vllm.sequence import IntermediateTensors
from vllm.triton_utils import HAS_TRITON

pytestmark = pytest.mark.skipif(not HAS_TRITON, reason="requires Triton")

HC, N_EMBD, LOWRANK, EPS, TOKENS = 4, 32, 12, 1e-6, 5
HC_DIM = HC * N_EMBD
CONFIG = SimpleNamespace(
    hc_count=HC, hidden_size=N_EMBD, hc_lowrank=LOWRANK, rms_norm_eps=EPS
)

# Each kernel rounds once where the unfused ops round at every step, and the mix
# differs by more than one place where the mean of the streams cancels.
MAX_ULPS = 2.5


class _Unfused:
    """The HC this replaces: torch ops, each one rounding to the input dtype."""

    def __init__(self, norm, down, up, inject):
        self.norm, self.down, self.up, self.inject = norm, down, up, inject

    def mix(self, x):
        var = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        xn = x * torch.rsqrt(var + EPS).to(x.dtype)
        flat = xn.reshape(x.shape[0], HC_DIM)
        flat = flat * (1.0 + self.norm.float()).to(flat.dtype)
        lo = F.silu(F.linear(flat, self.down) / HC)
        gate = torch.sigmoid(F.linear(lo, self.up))
        gated = (flat * gate).reshape(x.shape[0], HC, N_EMBD)
        injection = None if self.inject is None else F.linear(flat, self.inject)
        return gated.mean(dim=1), injection

    def combine(self, residual, block_out, injection):
        weight = (2.0 * torch.sigmoid(injection / HC)).unsqueeze(-1)
        return residual + block_out.unsqueeze(1) * weight


def _weights(dtype: torch.dtype, seed: int = 0) -> SimpleNamespace:
    torch.manual_seed(seed)
    scale = HC_DIM**-0.5
    return SimpleNamespace(
        norm=(0.1 * torch.randn(HC_DIM)).to(dtype),
        down=(scale * torch.randn(LOWRANK, HC_DIM)).to(dtype),
        up=(scale * torch.randn(HC_DIM, LOWRANK)).to(dtype),
        inject=(scale * torch.randn(HC, HC_DIM)).to(dtype),
    )


def _merged_down(w: SimpleNamespace) -> torch.Tensor:
    pad = -(LOWRANK + HC) % 16
    return torch.cat((w.down, w.inject, torch.zeros(pad, HC_DIM, dtype=w.down.dtype)))


def _ulps(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """The largest difference, in last places at ``expected``'s largest value."""
    assert actual.dtype == expected.dtype
    digits = {torch.bfloat16: 8, torch.float32: 24}[actual.dtype]
    exponent = torch.floor(torch.log2(expected.float().abs().max()))
    ulp = 2.0 ** (float(exponent) - digits + 1)
    return float((actual.double() - expected.double()).abs().max()) / ulp


def _check_kernels_against_unfused() -> None:
    """Runs in a process with TRITON_INTERPRET=1, which triton reads at import."""
    for dtype in (torch.bfloat16, torch.float32):
        w = _weights(dtype)
        down = _merged_down(w)
        unfused = _Unfused(w.norm, w.down, w.up, w.inject)
        torch.manual_seed(1)
        residual = torch.randn(TOKENS, HC, N_EMBD).to(dtype)
        block_out = torch.randn(TOKENS, N_EMBD).to(dtype)
        flat = residual.reshape(TOKENS, HC_DIM)

        mixed, inject = hc_module.hc_mix(flat, w.norm, down, w.up, EPS, HC, LOWRANK)
        ref_mixed, ref_inject = unfused.mix(residual)
        combined = hc_module.hc_combine(flat, block_out, inject, HC)
        ref_combined = unfused.combine(residual, block_out, inject).flatten(1)
        for name, actual, expected in (
            ("mix", mixed, ref_mixed),
            ("injection", inject, ref_inject),
            ("combine", combined, ref_combined),
        ):
            ulps = _ulps(actual, expected)
            print(
                f"{dtype} {name}: {ulps:.2f} ulps at the largest value, "
                f"bitwise {torch.equal(actual, expected)}"
            )
            assert ulps <= MAX_ULPS, name

        # Fused against the same kernels unfused: the combine rounds at the same
        # point, so it is bitwise. The norm sums its squares by tile, so on a card
        # the mix may differ by the sum order; the interpreter sums them in order.
        fused, fused_mixed, fused_inject = hc_module.hc_combine_and_mix(
            flat, block_out, inject, w.norm, down, w.up, EPS, HC, LOWRANK
        )
        then_mixed, then_inject = hc_module.hc_mix(
            combined, w.norm, down, w.up, EPS, HC, LOWRANK
        )
        assert torch.equal(fused, combined)
        assert torch.equal(fused_mixed, then_mixed)
        assert torch.equal(fused_inject, then_inject)
        print(f"{dtype} fused: bitwise with the same kernels unfused")


def test_kernels_against_unfused() -> None:
    root = Path(__file__).parents[3]
    code = (
        "from tests.models.qwen4_exp.test_xpu_hc import "
        "_check_kernels_against_unfused\n_check_kernels_against_unfused()\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env={**os.environ, "TRITON_INTERPRET": "1"},
        timeout=300,
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    assert result.returncode == 0, result.stderr[-4000:] + result.stdout


@pytest.fixture(autouse=True)
def _no_tp(monkeypatch: pytest.MonkeyPatch) -> None:
    for module in (linear, parameter):
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 1)
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 0)


def _torch_ops(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace the kernels with torch, in the unfused op order, and log the calls."""
    calls: list[str] = []

    def norm(x, weight, eps, hc):
        calls.append("norm")
        grouped = x.unflatten(-1, (hc, -1)).float()
        var = grouped.square().mean(-1, keepdim=True)
        out = (grouped * torch.rsqrt(var + eps)).flatten(-2) * (1 + weight.float())
        return out.to(x.dtype)

    def silu(x, hc):
        return F.silu(x.float() / hc).to(x.dtype)

    def gate_mix(xn, gate, hc):
        gated = torch.sigmoid(gate.float()) * xn.float()
        return gated.unflatten(-1, (hc, -1)).mean(-2).to(xn.dtype)

    def combine(residual, block, injection, hc):
        calls.append("combine")
        wide = residual.unflatten(-1, (hc, -1)).float()
        weight = 2 * torch.sigmoid(injection.float() / hc).unsqueeze(-1)
        return (wide + block.float().unsqueeze(1) * weight).flatten(-2)

    def combine_norm(residual, block, injection, weight, eps, hc):
        calls.append("combine_norm")
        out = combine(residual, block, injection, hc)
        calls.pop(-1)
        return out, norm(out, weight, eps, hc)

    monkeypatch.setattr(hc_module, "grouped_gemma_rmsnorm", norm)
    monkeypatch.setattr(hc_module, "hc_silu", silu)
    monkeypatch.setattr(hc_module, "hc_gate_mix", gate_mix)
    monkeypatch.setattr(hc_module, "hc_combine_norm", combine_norm)
    monkeypatch.setattr(hc_module, "_combine_op", lambda *args: combine(*args))
    return calls


def _hc_with_weights(seed: int, has_inject: bool = True) -> HyperConnection:
    module = HyperConnection(CONFIG, has_inject=has_inject, prefix="hc")
    w = _weights(torch.float32, seed)
    module.hc_norm.weight.data.copy_(w.norm)
    module.input_mix_weight_up.weight.data.copy_(w.up)
    if has_inject:
        module.input_mix_weight_down_block_inject.weight.data.copy_(_merged_down(w))
    else:
        module.input_mix_weight_down.weight.data.copy_(w.down)
    return module.float()


class _Layer(nn.Module):
    """``Qwen4ExpDecoderLayer.forward`` over stub attention, MLP and PLE."""

    def __init__(self, seed: int, ple: bool = False):
        super().__init__()
        self.hc = HC
        self.layer_type = "linear_attention"
        self.attn_hyper_connection = _hc_with_weights(seed)
        self.mlp_hyper_connection = _hc_with_weights(seed + 1)
        self.ple = (lambda residual, *_: residual + 0.5) if ple else None
        self.linear_attn = lambda hidden_states: hidden_states * 0.5
        self.mlp = lambda hidden_states: hidden_states * 0.25

    forward = Qwen4ExpDecoderLayer.forward


def _unfused_layer(layer: _Layer, residual: torch.Tensor) -> torch.Tensor:
    """The order the fused layers must keep: combine after each block, HC by HC."""
    if layer.ple is not None:
        residual = layer.ple(residual)
    for module, block in (
        (layer.attn_hyper_connection, lambda x: layer.linear_attn(hidden_states=x)),
        (layer.mlp_hyper_connection, layer.mlp),
    ):
        mixed, inject = module.mix(residual)
        residual = module.combine(residual, block(mixed), inject)
    return residual


def test_decoder_layers_delay_each_combine(monkeypatch) -> None:
    calls = _torch_ops(monkeypatch)
    layers = [_Layer(0), _Layer(2, ple=True), _Layer(4)]
    torch.manual_seed(5)
    embedding = torch.randn(TOKENS, N_EMBD)

    wide = embedding.unsqueeze(1).expand(-1, HC, -1).contiguous()
    expected = wide
    for layer in layers:
        expected = _unfused_layer(layer, expected)
    calls.clear()

    residual, block_output, injection = None, None, None
    for layer in layers:
        residual, block_output, injection = layer(
            hidden_states=embedding,
            residual=residual,
            input_ids=torch.zeros(TOKENS, dtype=torch.int64),
            query_start_loc=torch.tensor([0, TOKENS]),
            ngram_context=torch.zeros(1, 2, dtype=torch.int64),
            block_output=block_output,
            injection=injection,
        )
    # Layer 0: mix, fused. Layer 1: PLE takes the pending combine, mix, fused.
    # Layer 2: fused, fused. Nothing else combines on its own.
    assert calls.count("combine") == 1
    assert calls.count("combine_norm") == 4

    mixer = _hc_with_weights(7, has_inject=False)
    final_wide, final, empty = mixer.combine_and_mix(residual, block_output, injection)
    assert empty.shape == (TOKENS, 0)
    torch.testing.assert_close(final_wide, expected)
    torch.testing.assert_close(final, mixer.mix(expected)[0])


def _pp_group(is_last_rank: bool) -> SimpleNamespace:
    return SimpleNamespace(is_first_rank=True, is_last_rank=is_last_rank)


@pytest.mark.parametrize("is_last_rank", [True, False])
def test_model_materializes_the_last_combine(monkeypatch, is_last_rank) -> None:
    _torch_ops(monkeypatch)
    monkeypatch.setattr(model_module, "get_pp_group", lambda: _pp_group(is_last_rank))
    layers = [_Layer(0), _Layer(2, ple=True)]
    mixer = _hc_with_weights(7, has_inject=False)
    torch.manual_seed(5)
    embedding = torch.randn(TOKENS, N_EMBD)
    buffer = torch.zeros(TOKENS + 3, HC_DIM)
    model = SimpleNamespace(
        layers=layers,
        start_layer=0,
        end_layer=len(layers),
        hyper_connection_mixer=mixer,
        _mtp_hidden_buffer=buffer,
    )

    expected = embedding.unsqueeze(1).expand(-1, HC, -1).contiguous()
    for layer in layers:
        expected = _unfused_layer(layer, expected)

    out = Qwen4ExpModel.forward(
        model,
        input_ids=torch.zeros(TOKENS, dtype=torch.int64),
        positions=torch.arange(TOKENS),
        inputs_embeds=embedding,
        query_start_loc=torch.tensor([0, TOKENS]),
        ngram_context=torch.zeros(1, 2, dtype=torch.int64),
    )
    if is_last_rank:
        torch.testing.assert_close(out, mixer.mix(expected)[0])
        torch.testing.assert_close(buffer[:TOKENS], expected.flatten(1))
        assert not buffer[TOKENS:].any()
    else:
        assert isinstance(out, IntermediateTensors)
        torch.testing.assert_close(out["residual"], expected)
        assert not buffer.any()


@pytest.mark.parametrize("is_last_rank", [True, False])
def test_mtp_materializes_the_wide_state(monkeypatch, is_last_rank) -> None:
    _torch_ops(monkeypatch)
    monkeypatch.setattr(mtp_module, "get_pp_group", lambda: _pp_group(is_last_rank))
    layer = _Layer(0)
    mixer = _hc_with_weights(7, has_inject=False)
    torch.manual_seed(5)
    embedding = torch.randn(TOKENS, N_EMBD)
    target_wide = torch.randn(TOKENS, HC_DIM)
    predictor = SimpleNamespace(
        hc=HC,
        n_embd=N_EMBD,
        num_mtp_layers=1,
        layers=[layer],
        hyper_connection_mixer=mixer,
        # The fc stage is the identity on the wide state here.
        _fc_stage=lambda embeds, wide: wide,
    )

    wide = target_wide.view(TOKENS, HC, N_EMBD)
    expected = _unfused_layer(layer, wide)

    out = Qwen4ExpMultiTokenPredictor.forward(
        predictor,
        input_ids=None,
        positions=torch.arange(TOKENS),
        hidden_states=target_wide,
        inputs_embeds=embedding,
    )
    if is_last_rank:
        assert isinstance(out, tuple)
        collapsed, wide_flat = out
        torch.testing.assert_close(collapsed, mixer.mix(expected)[0])
        torch.testing.assert_close(wide_flat, expected.flatten(1))
    else:
        assert isinstance(out, IntermediateTensors)
        torch.testing.assert_close(out["residual"], expected)


def test_merged_down_inject_gemm_is_bitwise_on_rows(monkeypatch) -> None:
    """One GEMM against the separate ones, in BF16, on a width the CPU GEMM blocks."""
    w = _weights(torch.bfloat16)
    xn = torch.randn(TOKENS, HC_DIM).to(torch.bfloat16)
    merged = F.linear(xn, _merged_down(w))
    assert torch.equal(merged[:, :LOWRANK], F.linear(xn, w.down))
    assert torch.equal(merged[:, LOWRANK : LOWRANK + HC], F.linear(xn, w.inject))


def test_mixer_mix_takes_no_injection(monkeypatch) -> None:
    _torch_ops(monkeypatch)
    mixed, empty = _hc_with_weights(0, has_inject=False).mix(
        torch.randn(TOKENS, HC, N_EMBD)
    )
    assert mixed.shape == (TOKENS, N_EMBD)
    assert empty.shape == (TOKENS, 0)


class _LoadTarget(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn_hyper_connection = HyperConnection(CONFIG, prefix="attn")
        self.hyper_connection_mixer = HyperConnection(CONFIG, has_inject=False)


def _checkpoint(w: SimpleNamespace) -> Iterable[tuple[str, torch.Tensor]]:
    yield "attn_hyper_connection.hc_norm.weight", w.norm
    yield "attn_hyper_connection.input_mix_weight_down.weight", w.down
    yield "attn_hyper_connection.block_inject_weight.weight", w.inject
    yield "attn_hyper_connection.input_mix_weight_up.weight", w.up
    yield "hyper_connection_mixer.hc_norm.weight", w.norm
    yield "hyper_connection_mixer.input_mix_weight_down.weight", w.down
    yield "hyper_connection_mixer.input_mix_weight_up.weight", w.up


def test_checkpoint_names_load_into_the_merged_weight() -> None:
    w = _weights(torch.float32)
    target = _LoadTarget().float()
    target.attn_hyper_connection.input_mix_weight_down_block_inject.weight.data.zero_()
    loaded = AutoWeightsLoader(target).load_weights(
        _checkpoint(w), mapper=Qwen4ExpModel.hf_to_vllm_mapper
    )
    expected = {name for name, _ in target.named_parameters()}
    assert expected <= loaded
    hc = target.attn_hyper_connection
    assert torch.equal(hc.input_mix_weight_down_block_inject.weight, _merged_down(w))
    mixer = target.hyper_connection_mixer
    assert torch.equal(mixer.input_mix_weight_down.weight, w.down)
    assert not hasattr(mixer, "input_mix_weight_down_block_inject")


def test_fake_impls_match_the_real_shapes(monkeypatch) -> None:
    w = _weights(torch.bfloat16)
    down = _merged_down(w)
    with FakeTensorMode() as mode:
        x = mode.from_tensor(torch.empty(TOKENS, HC_DIM, dtype=torch.bfloat16))
        block = mode.from_tensor(torch.empty(TOKENS, N_EMBD, dtype=torch.bfloat16))
        inject = mode.from_tensor(torch.empty(TOKENS, HC, dtype=torch.bfloat16))
        norm = mode.from_tensor(w.norm)
        fake_down = mode.from_tensor(down)
        up = mode.from_tensor(w.up)
        mixed, injection = hc_module._hc_mix_fake(
            x, norm, fake_down, up, EPS, HC, LOWRANK
        )
        assert mixed.shape == (TOKENS, N_EMBD)
        assert injection.shape == (TOKENS, HC)
        assert injection.stride() == (down.shape[0], 1)
        combined, mixed, injection = hc_module._hc_combine_and_mix_fake(
            x, block, inject, norm, fake_down, up, EPS, HC, LOWRANK
        )
        assert combined.shape == (TOKENS, HC_DIM)
        assert mixed.shape == (TOKENS, N_EMBD)
        assert injection.shape == (TOKENS, HC)
        assert hc_module._hc_combine_fake(x, block, inject, HC).shape == x.shape
