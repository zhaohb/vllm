# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import lru_cache
from typing import Any

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

_OPS_REGISTERED = False


def _host_cpu_lacks_avx512() -> bool:
    """True on an x86 host without the AVX-512 subset ARK is built with.

    The ARK XPU library carries host code compiled for AVX-512 (F, BW, DQ and
    VL, e.g. ``vpmullq`` on XMM registers). On a CPU without it, such as AMD
    Zen 2/3 (EPYC 7002/7003) or Intel client parts, the first call traps with
    an invalid opcode. ``torch.cpu._is_avx512_supported`` checks that subset.
    """
    is_avx512_supported = getattr(torch.cpu, "_is_avx512_supported", None)
    if is_avx512_supported is None:
        return False
    return not is_avx512_supported()


@lru_cache(maxsize=1)
def get_ark_state() -> tuple[bool, str | None, Any | None, Any | None]:
    """Return ARK availability, error details, cached module, and QuantLinear."""
    # Only the "auto" backend steps aside: an explicit "ark" still loads ARK.
    if (
        current_platform.is_xpu()
        and envs.VLLM_XPU_INC_WNA16_BACKEND == "auto"
        and _host_cpu_lacks_avx512()
    ):
        logger.info(
            "The host CPU lacks AVX-512 (F/BW/DQ/VL), which auto_round_kernel "
            "requires; VLLM_XPU_INC_WNA16_BACKEND=auto falls back to the "
            "w4a16 backend. Set VLLM_XPU_INC_WNA16_BACKEND=ark to force ARK."
        )
        return False, "the host CPU lacks AVX-512", None, None

    try:
        import auto_round_kernel as ark
        from auto_round_kernel.qlinear import QuantLinear

        logger.info("Successfully imported auto_round_kernel.")
    except ImportError as error:
        return False, str(error), None, None

    if getattr(ark, "cpu_lib", None) is None and getattr(ark, "xpu_lib", None) is None:
        return (
            False,
            "No ARK backend library is available.",
            None,
            None,
        )

    logger.info("Successfully loaded auto_round_kernel backend library.")
    return True, None, ark, QuantLinear


def _inc_ark_woq_linear_impl(
    x: torch.Tensor,
    qweight: torch.Tensor,
    bias: torch.Tensor | None,
    out_features: int,
    in_features: int,
    group_size: int,
    compute_type: str,
    weight_type: str,
    scale_type: str,
    asym: bool,
) -> torch.Tensor:
    ark = get_ark_state()[2]
    assert ark is not None

    return ark.woqgemm_linear(
        x,
        qweight,
        bias,
        out_features,
        in_features,
        group_size,
        compute_type,
        weight_type,
        scale_type,
        asym,
    )


def _inc_ark_woq_linear_fake(
    x: torch.Tensor,
    qweight: torch.Tensor,
    bias: torch.Tensor | None,
    out_features: int,
    in_features: int,
    group_size: int,
    compute_type: str,
    weight_type: str,
    scale_type: str,
    asym: bool,
) -> torch.Tensor:
    del qweight
    del bias
    del in_features
    del group_size
    del compute_type
    del weight_type
    del scale_type
    del asym
    return torch.empty(
        (*x.shape[:-1], out_features),
        dtype=x.dtype,
        device=x.device,
    )


class ark_ops:
    @staticmethod
    def register_ops_once() -> None:
        global _OPS_REGISTERED
        if _OPS_REGISTERED:
            return

        is_available, error_str, _, _ = get_ark_state()
        if not is_available:
            logger.debug(
                "Skip registering ark op because ARK is unavailable: %s",
                error_str or "unknown error",
            )
            return

        direct_register_custom_op(
            op_name="inc_ark_woq_linear",
            op_func=_inc_ark_woq_linear_impl,
            fake_impl=_inc_ark_woq_linear_fake,
            dispatch_key=current_platform.dispatch_key,
        )
        _OPS_REGISTERED = True


ark_ops.register_ops_once()

__all__ = ["get_ark_state"]
