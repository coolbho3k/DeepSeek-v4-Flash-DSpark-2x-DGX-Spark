#!/usr/bin/env python3
"""CPU storage-alias checks for the vLLM V2 cache allocator."""

import torch

from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)
from vllm.v1.worker.gpu.attn_utils import _allocate_kv_cache


spec = FullAttentionSpec(
    block_size=16,
    num_kv_heads=1,
    head_size=8,
    dtype=torch.bfloat16,
)
cfg = KVCacheConfig(
    num_blocks=3,
    kv_cache_tensors=[
        KVCacheTensor(32, ["a"], 0, 16, 0),
        KVCacheTensor(32, ["b"], 8, 16, 0),
        KVCacheTensor(48, ["c"], 0, 24, 1),
        KVCacheTensor(48, ["d"], 8, 24, 1),
    ],
    kv_cache_groups=[
        KVCacheGroupSpec(["a", "b"], spec),
        KVCacheGroupSpec(["c", "d"], spec),
    ],
    pool_num_blocks=(2, 2),
    group_pool_ids=(0, 1),
    pool_byte_strides=(16, 24),
)

raw = _allocate_kv_cache(cfg, {}, torch.device("cpu"))
ptr = {name: tensor.untyped_storage().data_ptr() for name, tensor in raw.items()}
assert ptr["a"] == ptr["b"]
assert ptr["c"] == ptr["d"]
assert ptr["a"] != ptr["c"]
assert raw["a"].numel() == 32
assert raw["c"].numel() == 48
print("demand-sized KV V2 allocator tests: ok")
