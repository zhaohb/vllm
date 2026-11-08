# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Where prefix-cache hits land on a dense FP8 target with a BF16 draft.

The two dtypes give the draft twice the target's page at one block size. The planner
evens the pages by doubling the target's block size, so the scheduler's block, and the
grain of every hit, is twice the state block. Prompts below it cannot hit; others do.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_planning import get_kv_cache_groups
from vllm.v1.core.kv_cache_utils import (
    get_request_block_hasher,
    init_none_hash,
    resolve_kv_cache_block_sizes,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheLayout,
    MambaSpec,
)
from vllm.v1.request import Request

pytestmark = pytest.mark.cpu_test

STATE_BLOCK = 1600
MAX_CHUNK = 4096
LOOKAHEAD = 1


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        attention_config=SimpleNamespace(hisparse_config=None),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        kv_transfer_config=None,
        cache_config=SimpleNamespace(
            block_size=STATE_BLOCK,
            enable_prefix_caching=True,
            prefix_match_unit=None,
            get_resolved_kv_cache_layout=lambda: KVCacheLayout.LBHNC,
        ),
    )


def _plan(draft_dtype: torch.dtype) -> tuple[KVCacheConfig, int, int]:
    def attention(dtype: torch.dtype, **kwargs: object) -> FullAttentionSpec:
        return FullAttentionSpec(
            block_size=STATE_BLOCK,
            num_kv_heads=1,
            head_size=1,
            dtype=dtype,
            **kwargs,
        )

    target_page = attention(torch.uint8).page_size_bytes
    specs = {
        "target.0": attention(torch.uint8),
        "target.1": attention(torch.uint8),
        "state.0": MambaSpec(
            block_size=STATE_BLOCK,
            shapes=((1, 1),),
            dtypes=(torch.float32,),
            mamba_cache_mode="align",
            page_size_padded=target_page,
            num_speculative_blocks=3,
        ),
        "state.1": MambaSpec(
            block_size=STATE_BLOCK,
            shapes=((1, 1),),
            dtypes=(torch.float32,),
            mamba_cache_mode="align",
            page_size_padded=target_page,
            num_speculative_blocks=3,
        ),
        "draft": attention(draft_dtype, is_draft=True),
    }
    vllm_config = _config()
    groups = get_kv_cache_groups(vllm_config, specs)
    for group in groups:
        group.is_eagle_group = "draft" in group.layer_names
    kv_cache_config = KVCacheConfig(
        num_blocks=4000, kv_cache_tensors=[], kv_cache_groups=groups
    )
    return kv_cache_config, *resolve_kv_cache_block_sizes(kv_cache_config, vllm_config)


def _manager(draft_dtype: torch.dtype) -> KVCacheManager:
    kv_cache_config, scheduler_block, hash_block = _plan(draft_dtype)
    return KVCacheManager(
        kv_cache_config,
        max_model_len=131072,
        scheduler_block_size=scheduler_block,
        hash_block_size=hash_block,
        enable_caching=True,
        use_eagle=False,
        num_prefill_lookahead=LOOKAHEAD,
        num_hash_lookahead_tokens=LOOKAHEAD,
    )


def _request(manager: KVCacheManager, name: str, tokens: list[int]) -> Request:
    return Request(
        request_id=name,
        prompt_token_ids=tokens,
        sampling_params=SamplingParams(max_tokens=2, ignore_eos=True),
        pooling_params=None,
        block_hasher=get_request_block_hasher(
            manager.coordinator.block_pool.hash_block_size, sha256, LOOKAHEAD
        ),
    )


def _split(
    manager: KVCacheManager, request: Request, num_new: int, num_local: int
) -> int:
    """The real chunk split, on the fields it reads from the scheduler."""
    stub = SimpleNamespace(
        block_size=manager.coordinator.scheduler_block_size,
        cache_config=SimpleNamespace(block_size=STATE_BLOCK),
        use_eagle_block_drop=False,
        num_hash_lookahead_tokens=LOOKAHEAD,
        max_num_scheduled_tokens=MAX_CHUNK,
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
        mamba_partial_cache_hit=False,
        mamba_shared_prefix_checkpoint=False,
        hash_block_size=manager.coordinator.block_pool.hash_block_size,
        mamba_has_prefill_checkpoint_blocks=False,
        mamba_prefill_checkpoint_alignment=None,
    )
    return Scheduler._mamba_block_aligned_split(stub, request, num_new, num_local)


def _serve(manager: KVCacheManager, name: str, tokens: list[int]) -> int:
    """Look the prompt up, prefill what is left and free it. Returns the hit."""
    init_none_hash(sha256)
    request = _request(manager, name, tokens)
    blocks, hit, _ = manager.get_computed_blocks(request)
    computed = hit
    new_blocks, new_hit = blocks, hit
    while computed < request.num_tokens:
        budget = min(request.num_tokens - computed, MAX_CHUNK)
        num_new = _split(manager, request, budget, new_hit)
        assert num_new > 0
        allocated = manager.allocate_slots(
            request,
            num_new,
            num_new_computed_tokens=new_hit,
            new_computed_blocks=new_blocks,
            num_lookahead_tokens=3,
        )
        assert allocated is not None
        computed += num_new
        request.num_computed_tokens = computed
        new_blocks, new_hit = None, 0
        manager.remove_skipped_blocks(
            request.request_id, computed, request.num_prompt_tokens
        )
        manager.new_step_starts()
    manager.free(request)
    return hit


def _prompt(length: int, seed: int = 0) -> list[int]:
    return [(i * 2654435761 + seed) % 50000 + 10 for i in range(length)]


def test_bf16_draft_doubles_the_fp8_target_block() -> None:
    kv_cache_config, scheduler_block, hash_block = _plan(torch.bfloat16)
    sizes = {
        tuple(group.layer_names): group.kv_cache_spec.block_size
        for group in kv_cache_config.kv_cache_groups
    }
    assert sizes[("target.0",)] == sizes[("target.1",)] == 2 * STATE_BLOCK
    assert (
        sizes[("draft",)] == sizes[("state.0",)] == sizes[("state.1",)] == STATE_BLOCK
    )
    assert (scheduler_block, hash_block) == (2 * STATE_BLOCK, STATE_BLOCK)


def test_fp8_draft_keeps_the_state_block() -> None:
    kv_cache_config, scheduler_block, hash_block = _plan(torch.uint8)
    assert {
        group.kv_cache_spec.block_size for group in kv_cache_config.kv_cache_groups
    } == {STATE_BLOCK}
    assert (scheduler_block, hash_block) == (STATE_BLOCK, STATE_BLOCK)


@pytest.mark.parametrize(
    ("draft_dtype", "length", "expected"),
    [
        # The smoke gate's prompt. It is shorter than the 3200-token block.
        (torch.bfloat16, 3053, 0),
        (torch.uint8, 3053, 1600),
        (torch.bfloat16, 8192, 6400),
        (torch.uint8, 8192, 8000),
        (torch.bfloat16, 42000, 41600),
        (torch.uint8, 42000, 41600),
    ],
)
def test_resent_prompt_hits_on_the_scheduler_block(
    draft_dtype: torch.dtype, length: int, expected: int
) -> None:
    manager = _manager(draft_dtype)
    tokens = _prompt(length)
    assert _serve(manager, "cold", tokens) == 0
    assert _serve(manager, "warm", tokens) == expected


def test_a_grown_session_reuses_its_earlier_prefill_on_a_bf16_draft() -> None:
    manager = _manager(torch.bfloat16)
    turn = _prompt(30000)
    assert _serve(manager, "turn1", turn) == 0
    next_turn = turn + _prompt(1500, seed=1)
    assert _serve(manager, "turn2", next_turn) == 28800
