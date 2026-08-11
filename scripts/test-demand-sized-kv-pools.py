#!/usr/bin/env python3
"""CPU-only checks for the default-enabled demand-sized KV pool patch."""

from types import SimpleNamespace

import torch

from vllm.v1.core.kv_cache_coordinator import get_kv_cache_coordinator
from vllm.v1.core.kv_cache_utils import (
    _get_kv_cache_config_demand_sized,
    get_max_concurrency_for_kv_cache_config,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)


def config(max_model_len: int = 4096, max_num_batched_tokens: int = 512):
    return SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=max_model_len),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=max_num_batched_tokens
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
        cache_config=SimpleNamespace(num_gpu_blocks_override=None),
        kv_transfer_config=None,
    )


def test_planner() -> None:
    cfg = config()
    full_layers = {
        f"full.{i}": MLAAttentionSpec(
            block_size=256,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.uint8,
            cache_dtype_str="nvfp4_ds_mla",
            alignment=256,
            model_version="deepseek_v4",
        )
        for i in range(2)
    }
    window_layers = {
        f"window.{i}": SlidingWindowMLASpec(
            block_size=256,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.uint8,
            cache_dtype_str="nvfp4_ds_mla",
            alignment=256,
            compress_ratio=4,
            sliding_window=128,
            model_version="deepseek_v4",
        )
        for i in range(3)
    }
    full = UniformTypeKVCacheSpecs.from_specs(full_layers)
    window = UniformTypeKVCacheSpecs.from_specs(window_layers)
    assert full is not None and window is not None
    groups = [
        KVCacheGroupSpec(list(full_layers), full),
        KVCacheGroupSpec(list(window_layers), window),
    ]

    demands = tuple(
        group.kv_cache_spec.max_memory_usage_pages(cfg) for group in groups
    )
    strides = tuple(group.kv_cache_spec.page_size_bytes for group in groups)
    bytes_per_request = sum(d * s for d, s in zip(demands, strides))
    available = bytes_per_request * 3 + strides[0] // 2

    legacy, tensors, counts, planned_strides = (
        _get_kv_cache_config_demand_sized(cfg, groups, available)
    )
    assert legacy == max(counts)
    assert planned_strides == strides
    assert sum(n * s for n, s in zip(counts, strides)) <= available
    assert max(counts[i] / demands[i] for i in range(2)) - min(
        counts[i] / demands[i] for i in range(2)
    ) <= max(1 / demands[i] for i in range(2))
    assert {tensor.pool_id for tensor in tensors} == {0, 1}
    for tensor in tensors:
        assert tensor.size == counts[tensor.pool_id] * strides[tensor.pool_id]

    planned = KVCacheConfig(
        num_blocks=legacy,
        kv_cache_tensors=tensors,
        kv_cache_groups=groups,
        pool_num_blocks=counts,
        group_pool_ids=(0, 1),
        pool_byte_strides=strides,
    )
    reported = get_max_concurrency_for_kv_cache_config(cfg, planned)
    expected = min(counts[i] / demands[i] for i in range(2))
    assert reported == expected


def test_coordinator() -> None:
    full = FullAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.bfloat16,
    )
    window = SlidingWindowSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.bfloat16,
        sliding_window=128,
    )
    groups = [
        KVCacheGroupSpec(["full"], full),
        KVCacheGroupSpec(["window"], window),
    ]
    cfg = KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=groups,
        pool_num_blocks=(8, 5),
        group_pool_ids=(0, 1),
        pool_byte_strides=(full.page_size_bytes, window.page_size_bytes),
    )
    coordinator = get_kv_cache_coordinator(
        kv_cache_config=cfg,
        max_model_len=4096,
        max_num_batched_tokens=512,
        use_eagle=False,
        enable_caching=False,
        enable_kv_cache_events=False,
        dcp_world_size=1,
        pcp_world_size=1,
        scheduler_block_size=256,
        hash_block_size=256,
    )
    assert len(coordinator.block_pools) == 2
    assert (
        coordinator.single_type_managers[0].block_pool
        is coordinator.block_pools[0]
    )
    assert (
        coordinator.single_type_managers[1].block_pool
        is coordinator.block_pools[1]
    )

    initial_free = tuple(pool.get_num_free_blocks() for pool in coordinator.block_pools)
    needed = coordinator.get_num_blocks_to_allocate_by_pool(
        request_id="r",
        num_tokens=256,
        new_computed_blocks=((), ()),
        num_encoder_tokens=0,
        total_computed_tokens=0,
        num_tokens_main_model=256,
    )
    assert needed == (1, 1)
    assert coordinator.can_allocate(needed, (0, 0))

    blocks = coordinator.allocate_new_blocks("r", 256, 256)
    assert [group[0].block_id for group in blocks] == [1, 1]
    popped = coordinator.pop_blocks_for_free("r")
    coordinator.free_blocks(list(reversed(popped)))
    assert tuple(
        pool.get_num_free_blocks() for pool in coordinator.block_pools
    ) == initial_free


if __name__ == "__main__":
    test_planner()
    test_coordinator()
    print("demand-sized KV pool CPU tests: ok")
