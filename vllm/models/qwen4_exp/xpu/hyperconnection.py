# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Running the delayed, fused hyper-connection on XPU follows wu1ff's
# B70-LLM-Controller, used here under the MIT License:
#
# Copyright (c) 2026 wu1ff
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Fused hyper-connections for the XPU Qwen4-Exp model.

The state between blocks is ``hc`` parallel residual streams ``[T, hc, n_embd]``.
Each combine is delayed to the next mix, where it fuses with that mix's grouped
RMSNorm, and the down and injection projections run as one GEMM. The Triton kernels
are the ones the NVIDIA model runs. They hold the intermediate values in fp32 and
round once at each materialized tensor.

The kernels sit behind custom ops, like the PLE gate, so they stay opaque to
``torch.compile`` and run at call time with the real row count. HC holds no
per-request state, so these ops do not split a piecewise graph.
"""

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Qwen4ExpTextConfig

from vllm.model_executor.layers.linear import MergedColumnParallelLinear
from vllm.utils.torch_utils import direct_register_custom_op, eager_bypass_custom_op

from ..nvidia.ops.hc import (
    grouped_gemma_rmsnorm,
    hc_combine,
    hc_combine_norm,
    hc_gate_mix,
    hc_silu,
)


class _WeightContainer(nn.Module):
    """Holds a single ``weight`` parameter (no bias, no forward).

    Used for the hyper-connection weights, which are plain BF16 tensors
    (excluded from quantization via ``re:.*hyper_connection.*``) loaded
    directly from the checkpoint. The container exists only so the weight
    sits at the exact ``<name>.weight`` path the checkpoint expects.
    """

    def __init__(self, shape: Sequence[int]):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(*shape))


def _gate_mix_and_inject(
    xn: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    hc: int,
    lowrank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``xn``: normalized ``[T, hc * n_embd]``. Returns the block input and the
    injection logits, which are empty when ``down_weight`` has no injection rows."""
    down = F.linear(xn, down_weight)
    gate = F.linear(hc_silu(down[:, :lowrank], hc), up_weight)
    return hc_gate_mix(xn, gate, hc), down[:, lowrank : lowrank + hc]


def hc_mix(
    hidden: torch.Tensor,
    norm_weight: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    eps: float,
    hc: int,
    lowrank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    xn = grouped_gemma_rmsnorm(hidden, norm_weight, eps, hc)
    return _gate_mix_and_inject(xn, down_weight, up_weight, hc, lowrank)


def hc_combine_and_mix(
    hidden: torch.Tensor,
    block_output: torch.Tensor,
    injection: torch.Tensor,
    norm_weight: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    eps: float,
    hc: int,
    lowrank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    combined, xn = hc_combine_norm(
        hidden, block_output, injection, norm_weight, eps, hc
    )
    block_input, new_injection = _gate_mix_and_inject(
        xn, down_weight, up_weight, hc, lowrank
    )
    return combined, block_input, new_injection


def _hc_mix_fake(
    hidden: torch.Tensor,
    norm_weight: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    eps: float,
    hc: int,
    lowrank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = hidden.shape[0]
    down = hidden.new_empty((tokens, down_weight.shape[0]))
    block_input = hidden.new_empty((tokens, hidden.shape[1] // hc))
    # The same strides as the real column slice of the down projection.
    return block_input, down[:, lowrank : lowrank + hc]


def _hc_combine_and_mix_fake(
    hidden: torch.Tensor,
    block_output: torch.Tensor,
    injection: torch.Tensor,
    norm_weight: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    eps: float,
    hc: int,
    lowrank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    block_input, new_injection = _hc_mix_fake(
        hidden, norm_weight, down_weight, up_weight, eps, hc, lowrank
    )
    return hidden.new_empty(hidden.shape), block_input, new_injection


def _hc_combine_fake(
    hidden: torch.Tensor,
    block_output: torch.Tensor,
    injection: torch.Tensor,
    hc: int,
) -> torch.Tensor:
    return hidden.new_empty(hidden.shape)


direct_register_custom_op(
    op_name="qwen4_exp_xpu_hc_mix",
    op_func=hc_mix,
    mutates_args=[],
    fake_impl=_hc_mix_fake,
)
direct_register_custom_op(
    op_name="qwen4_exp_xpu_hc_combine_and_mix",
    op_func=hc_combine_and_mix,
    mutates_args=[],
    fake_impl=_hc_combine_and_mix_fake,
)
direct_register_custom_op(
    op_name="qwen4_exp_xpu_hc_combine",
    op_func=hc_combine,
    mutates_args=[],
    fake_impl=_hc_combine_fake,
)
_mix_op = eager_bypass_custom_op("qwen4_exp_xpu_hc_mix", hc_mix)
_combine_and_mix_op = eager_bypass_custom_op(
    "qwen4_exp_xpu_hc_combine_and_mix", hc_combine_and_mix
)
_combine_op = eager_bypass_custom_op("qwen4_exp_xpu_hc_combine", hc_combine)


class HyperConnection(nn.Module):
    """Hyper-connection mix/combine block.

    Replaces the standard per-layer RMSNorm + residual add.

    ``mix`` reads one mixed ``[T, n_embd]`` view:
        grouped-RMSNorm (per stream) -> gamma (``hc_norm``) ->
        low-rank down -> ``silu(x / hc)`` -> up -> sigmoid gate ->
        ``x * gate`` -> mean-collapse the streams.
    It also returns the ``[T, hc]`` injection logits of this block's combine.

    ``combine`` writes a block output back into the wide state:
        ``residual + block_out * 2 * sigmoid(inject / hc)``.
    The ``2 * sigmoid`` centres the scatter weights on 1, so an
    untrained injection matrix reproduces the plain residual add. It depends only
    on the logits, so any block can apply a pending combine.

    ``combine_and_mix`` is ``combine`` followed by ``mix`` in one pass.

    ``has_inject=False`` is used for the top-level ``hyper_connection_mixer``
    (the final output norm), which has no ``block_inject_weight``.
    """

    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        has_inject: bool = True,
        prefix: str = "",
    ):
        super().__init__()
        self.hc = config.hc_count
        self.n_embd = config.hidden_size
        self.hc_dim = self.hc * self.n_embd
        self.lowrank = config.hc_lowrank
        self.eps = config.rms_norm_eps
        self.has_inject = has_inject

        self.hc_norm = _WeightContainer([self.hc_dim])
        self.input_mix_weight_up = _WeightContainer([self.hc_dim, self.lowrank])
        if has_inject:
            # The checkpoint's down and injection weights share one GEMM, with its
            # row count padded to 16 like the NVIDIA model.
            pad = -(self.lowrank + self.hc) % 16
            self.input_mix_weight_down_block_inject = MergedColumnParallelLinear(
                self.hc_dim,
                [self.lowrank, self.hc] + ([pad] if pad else []),
                bias=False,
                quant_config=None,
                prefix=f"{prefix}.input_mix_weight_down_block_inject",
                return_bias=False,
                disable_tp=True,
            )
        else:
            self.input_mix_weight_down = _WeightContainer([self.lowrank, self.hc_dim])

    @property
    def _down_weight(self) -> torch.Tensor:
        if self.has_inject:
            return self.input_mix_weight_down_block_inject.weight
        return self.input_mix_weight_down.weight

    def _flat(self, wide: torch.Tensor) -> torch.Tensor:
        return wide.reshape(wide.shape[0], self.hc_dim)

    def _wide(self, flat: torch.Tensor) -> torch.Tensor:
        return flat.view(flat.shape[0], self.hc, self.n_embd)

    def mix(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [T, hc, n_embd]
        return _mix_op(
            self._flat(x),
            self.hc_norm.weight,
            self._down_weight,
            self.input_mix_weight_up.weight,
            self.eps,
            self.hc,
            self.lowrank,
        )

    def combine_and_mix(
        self,
        residual: torch.Tensor,
        block_out: torch.Tensor,
        inject: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # residual: [T, hc, n_embd]; block_out: [T, n_embd]; inject: [T, hc]
        combined, mixed, new_inject = _combine_and_mix_op(
            self._flat(residual),
            block_out,
            inject,
            self.hc_norm.weight,
            self._down_weight,
            self.input_mix_weight_up.weight,
            self.eps,
            self.hc,
            self.lowrank,
        )
        return self._wide(combined), mixed, new_inject

    def combine(
        self,
        residual: torch.Tensor,
        block_out: torch.Tensor,
        inject: torch.Tensor,
    ) -> torch.Tensor:
        # residual: [T, hc, n_embd]; block_out: [T, n_embd]; inject: [T, hc]
        return self._wide(_combine_op(self._flat(residual), block_out, inject, self.hc))
