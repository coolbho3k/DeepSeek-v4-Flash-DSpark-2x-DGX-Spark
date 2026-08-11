#!/usr/bin/env python3
"""Check that demand pools remove only cross-group page padding."""

import os
from types import SimpleNamespace

import torch

from vllm.v1.core.kv_cache_utils import get_kv_cache_groups
from vllm.v1.kv_cache_interface import (
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
)


def make_specs():
    specs = {}
    for i in range(2):
        specs[f"full.{i}"] = MLAAttentionSpec(
            block_size=256,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.uint8,
            page_size_padded=27072,
            cache_dtype_str="nvfp4_ds_mla",
            compress_ratio=4,
            model_version="deepseek_v4",
        )
        specs[f"state.{i}"] = SlidingWindowMLASpec(
            block_size=256,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.uint8,
            page_size_padded=32832,
            cache_dtype_str="nvfp4_ds_mla",
            compress_ratio=4,
            sliding_window=128,
            model_version="deepseek_v4",
        )
    return specs


cfg = SimpleNamespace(
    scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
    speculative_config=None,
)

os.environ["VLLM_DSV4_DEMAND_SIZED_KV_POOLS"] = "1"
demand_groups = get_kv_cache_groups(cfg, make_specs())
assert len(demand_groups) == 2
assert all(
    isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
    for group in demand_groups
)
assert demand_groups[0].kv_cache_spec.get_page_sizes() == [27072]
assert demand_groups[1].kv_cache_spec.get_page_sizes() == [32832]

os.environ["VLLM_DSV4_DEMAND_SIZED_KV_POOLS"] = "0"
shared_groups = get_kv_cache_groups(cfg, make_specs())
assert shared_groups[0].kv_cache_spec.get_page_sizes() == [32832]
assert shared_groups[1].kv_cache_spec.get_page_sizes() == [32832]

print("demand-sized KV grouping tests: ok")
