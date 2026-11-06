# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of the XPU Qwen4Exp vision tower input.

With ``mm_device_do_normalize`` on (the default) the processor leaves the
pixels raw and the tower's input norm rescales and normalises them on the
device. The tower is replaced by a stand-in here, so the tests cover what the
model decides: which input norm the tower is built with, and that the norm
turns raw pixels into what the processor would have produced.
"""

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch
from torch import nn
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fusion import mm_input_norm
from vllm.model_executor.layers.fusion.mm_input_norm import (
    FusedMMInputNorm,
    IdentityInputNorm,
    NormParams,
)
from vllm.models.qwen4_exp.xpu import model as xpu_model

# Qwen3-VL image processor values.
MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]
RESCALE = 1 / 255
PATCHES, PATCH_PIXELS = 6, 3 * 16 * 16


class _Tower(nn.Module):
    """Records the input norm the model builds the tower with; like the real
    tower, falls back to the identity norm when none is passed."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        norm = kwargs.get("input_norm")
        self.input_norm = norm if norm is not None else IdentityInputNorm()


class _TextModel(nn.Module):
    def __init__(self, **_: Any) -> None:
        super().__init__()
        self.make_empty_intermediate_tensors = None


class _Marked:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> None:
        return None


def _config() -> VllmConfig:
    return VllmConfig(device_config=DeviceConfig(device="cpu"))


def _build(do_normalize: bool) -> Any:
    """Run the XPU wrapper's __init__ with the tower and text model stubbed."""
    config = SimpleNamespace(
        vision_config=SimpleNamespace(deepstack_visual_indexes=[], out_hidden_size=8),
        rms_norm_eps=1e-6,
    )
    model_config = SimpleNamespace(
        hf_config=config,
        model="stub",
        revision=None,
        multimodal_config=SimpleNamespace(
            mm_encoder_tp_mode="weights", mm_device_do_normalize=do_normalize
        ),
    )
    vllm_config = SimpleNamespace(model_config=model_config, quant_config=None)
    cls = xpu_model.Qwen4ExpForConditionalGeneration
    params = NormParams(image_mean=MEAN, image_std=STD, rescale_factor=RESCALE)
    with (
        patch.object(xpu_model, "Qwen3_VisionTransformer", _Tower),
        patch.object(xpu_model, "Qwen4ExpForCausalLM", _TextModel),
        patch.object(xpu_model, "cached_tokenizer_from_config", lambda _: None),
        patch.object(mm_input_norm, "_load_norm_params", lambda _: params),
        patch.object(cls, "_init_video_pruning", lambda *_: None),
        patch.object(cls, "set_moe_parameters", lambda *_: None),
        patch.object(cls, "_mark_tower_model", lambda *_, **__: _Marked()),
        patch.object(cls, "_mark_language_model", lambda *_, **__: _Marked()),
        set_current_vllm_config(_config()),
    ):
        return cls(vllm_config=vllm_config)


def test_tower_gets_input_norm_when_device_normalize_is_on() -> None:
    model = _build(do_normalize=True)
    assert isinstance(model.visual.input_norm, FusedMMInputNorm)
    assert model.visual.input_norm.input_dtype == torch.uint8


def test_tower_input_norm_is_identity_when_device_normalize_is_off() -> None:
    model = _build(do_normalize=False)
    assert model.visual.input_norm.input_dtype is None


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_raw_pixels_match_processor_normalised_pixels(dtype: torch.dtype) -> None:
    model = _build(do_normalize=True)
    raw = torch.randint(0, 256, (PATCHES, PATCH_PIXELS), dtype=torch.uint8)
    # What the HF processor produces when it rescales and normalises on the host.
    processed = ((raw.float() * RESCALE) - 0.5) / 0.5

    # The platform-independent reference path; the kernels are checked on cards.
    tower_input = model.visual.input_norm.forward_native(raw, dtype)

    assert tower_input.dtype == dtype
    torch.testing.assert_close(
        tower_input.float(), processed.to(dtype).float(), atol=1e-2, rtol=1e-2
    )
    # Raw 0..255 pixels reaching the tower unchanged is the bug.
    assert tower_input.float().abs().max() <= 1.0 + 1e-2
