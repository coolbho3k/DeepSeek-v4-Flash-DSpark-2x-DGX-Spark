"""Install opt-in demand-sized DeepSeek V4 KV cache pools.

This accuracy-neutral allocator keeps every per-layer dtype and kernel layout,
but gives each DeepSeek cache group an independent local block-ID namespace
and a physical backing sized for that group's worst-case request demand.

Enable with VLLM_DSV4_DEMAND_SIZED_KV_POOLS=1.
"""

from pathlib import Path


ROOT = Path("/usr/local/lib/python3.12/dist-packages/vllm")


def replace(path: str, old: str, new: str) -> None:
    target = ROOT / path
    source = target.read_text()
    if new in source:
        return
    if old not in source:
        raise SystemExit(f"missing demand-pool patch anchor in {target}: {old!r}")
    target.write_text(source.replace(old, new, 1))


# Explicit metadata keeps every old constructor and the shared-pool default
# backward compatible.
replace(
    "v1/kv_cache_interface.py",
    """    block_stride: int = 0  # total bytes per block in a packed layout (0 = not packed)
""",
    """    block_stride: int = 0  # total bytes per block in a packed layout (0 = not packed)
    pool_id: int = 0  # packed backing allocation / local block-ID namespace
""",
)

replace(
    "v1/kv_cache_interface.py",
    """    kv_cache_groups: list[KVCacheGroupSpec]
    \"\"\"
    The kv cache groups of the model.
""",
    """    kv_cache_groups: list[KVCacheGroupSpec]
    pool_num_blocks: tuple[int, ...] | None = None
    \"\"\"Per-pool block counts for independent demand-sized pools.\"\"\"
    group_pool_ids: tuple[int, ...] | None = None
    \"\"\"Map each KV cache group to its local block pool.\"\"\"
    pool_byte_strides: tuple[int, ...] | None = None
    \"\"\"Physical bytes consumed by one block in each pool.\"\"\"
    \"\"\"
    The kv cache groups of the model.
""",
)


# Planner feature gate.
replace(
    "v1/core/kv_cache_utils.py",
    """logger = init_logger(__name__)

# The hash seed for the first block of any prefix block sequence.
""",
    """logger = init_logger(__name__)


def _demand_sized_kv_pools_enabled() -> bool:
    return os.environ.get("VLLM_DSV4_DEMAND_SIZED_KV_POOLS", "0") == "1"


def _use_demand_sized_kv_pools(
    kv_cache_groups: list[KVCacheGroupSpec],
) -> bool:
    return (
        _demand_sized_kv_pools_enabled()
        and len(kv_cache_groups) > 1
        and all(
            isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
            for group in kv_cache_groups
        )
    )


# The hash seed for the first block of any prefix block sequence.
""",
)

# Independent groups no longer need page-size compatibility with one another.
replace(
    "v1/core/kv_cache_utils.py",
    """    max_full_page_size = max(all_page_sizes)
    max_sm_page_size = max(
        (max(sm_spec.get_page_sizes()) for sm_spec in swa_mla_specs),
        default=max_full_page_size,
    )
    if max_sm_page_size > max_full_page_size:
        for layer_spec in full_mla_spec.kv_cache_specs.values():
            if layer_spec.page_size_bytes == max_full_page_size:
                object.__setattr__(
                    layer_spec, "page_size_padded", max_sm_page_size
                )
        all_page_sizes = full_mla_spec.get_page_sizes()
""",
    """    if not _demand_sized_kv_pools_enabled():
        max_full_page_size = max(all_page_sizes)
        max_sm_page_size = max(
            (max(sm_spec.get_page_sizes()) for sm_spec in swa_mla_specs),
            default=max_full_page_size,
        )
        if max_sm_page_size > max_full_page_size:
            for layer_spec in full_mla_spec.kv_cache_specs.values():
                if layer_spec.page_size_bytes == max_full_page_size:
                    object.__setattr__(
                        layer_spec, "page_size_padded", max_sm_page_size
                    )
            all_page_sizes = full_mla_spec.get_page_sizes()
""",
)

replace(
    "v1/core/kv_cache_utils.py",
    """        sm_page_sizes = sm_spec.get_page_sizes()
        layers_per_size: dict[int, list[str]] = defaultdict(list)
        assert max(sm_page_sizes) <= max(all_page_sizes)

        # Unify page size by padding layers' page_size to the nearest larger page_size.
        # Compute candidate (nearest larger page_size) for each unique page size.
        size_to_candidate: dict[int, int] = {}
        for ps in sm_page_sizes:
            size_to_candidate[ps] = min(x for x in all_page_sizes if x >= ps)
        # Pad and collect layer names per page size.
        for layer_name, layer_spec in sm_spec.kv_cache_specs.items():
            current_size = layer_spec.page_size_bytes
            candidate = size_to_candidate[current_size]
            if current_size < candidate:
                object.__setattr__(layer_spec, "page_size_padded", candidate)
            layers_per_size[candidate].append(layer_name)
""",
    """        sm_page_sizes = sm_spec.get_page_sizes()
        layers_per_size: dict[int, list[str]] = defaultdict(list)

        if _demand_sized_kv_pools_enabled():
            for layer_name, layer_spec in sm_spec.kv_cache_specs.items():
                layers_per_size[layer_spec.page_size_bytes].append(layer_name)
        else:
            assert max(sm_page_sizes) <= max(all_page_sizes)

            # Pad to the nearest larger page in the full-MLA tuple.
            size_to_candidate: dict[int, int] = {}
            for ps in sm_page_sizes:
                size_to_candidate[ps] = min(x for x in all_page_sizes if x >= ps)
            for layer_name, layer_spec in sm_spec.kv_cache_specs.items():
                current_size = layer_spec.page_size_bytes
                candidate = size_to_candidate[current_size]
                if current_size < candidate:
                    object.__setattr__(layer_spec, "page_size_padded", candidate)
                layers_per_size[candidate].append(layer_name)
""",
)

# Physical split: equalize the concurrency supported by every pool according
# to the same maximum-block-demand formula used by admission.
replace(
    "v1/core/kv_cache_utils.py",
    """_get_kv_cache_config_deepseek_v4 = _get_kv_cache_config_packed
""",
    """def _get_kv_cache_config_demand_sized(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> tuple[int, list[KVCacheTensor], tuple[int, ...], tuple[int, ...]]:
    if vllm_config.kv_transfer_config is not None:
        raise ValueError(
            "VLLM_DSV4_DEMAND_SIZED_KV_POOLS does not yet support KV "
            "transfer connectors"
        )

    pool_strides = tuple(
        group.kv_cache_spec.page_size_bytes for group in kv_cache_groups
    )
    pool_demands = tuple(
        cdiv(
            group.kv_cache_spec.max_memory_usage_bytes(vllm_config),
            stride,
        )
        for group, stride in zip(kv_cache_groups, pool_strides)
    )
    override = vllm_config.cache_config.num_gpu_blocks_override
    if override is not None:
        # vLLM uses this internally for its minimal graph-profiling cache.
        pool_num_blocks = [override] * len(kv_cache_groups)
    else:
        bytes_per_request = sum(
            demand * stride for demand, stride in zip(pool_demands, pool_strides)
        )
        concurrency = available_memory / bytes_per_request
        pool_num_blocks = [
            max(1, int(concurrency * demand)) for demand in pool_demands
        ]

        # Down-rounding leaves less than one block per pool. Spend residue only
        # on the current bottlenecks.
        remaining = available_memory - sum(
            count * stride
            for count, stride in zip(pool_num_blocks, pool_strides)
        )
        for pool_id in sorted(
            range(len(pool_num_blocks)),
            key=lambda i: pool_num_blocks[i] / pool_demands[i],
        ):
            if pool_strides[pool_id] <= remaining:
                pool_num_blocks[pool_id] += 1
                remaining -= pool_strides[pool_id]

    kv_cache_tensors: list[KVCacheTensor] = []
    for pool_id, (group, stride, num_blocks) in enumerate(
        zip(kv_cache_groups, pool_strides, pool_num_blocks)
    ):
        total_size = stride * num_blocks
        byte_offset = 0
        group_spec = cast(UniformTypeKVCacheSpecs, group.kv_cache_spec)
        for layer_name in group.layer_names:
            kv_cache_tensors.append(
                KVCacheTensor(
                    size=total_size,
                    shared_by=[layer_name],
                    offset=byte_offset,
                    block_stride=stride,
                    pool_id=pool_id,
                )
            )
            byte_offset += group_spec.kv_cache_specs[layer_name].page_size_bytes
        assert byte_offset == stride

    logger.info(
        "Demand-sized DeepSeek KV pools: strides=%s, request_demands=%s, "
        "blocks=%s, allocated=%.2f GiB",
        pool_strides,
        pool_demands,
        tuple(pool_num_blocks),
        sum(n * s for n, s in zip(pool_num_blocks, pool_strides)) / (1 << 30),
    )
    return (
        max(pool_num_blocks),
        kv_cache_tensors,
        tuple(pool_num_blocks),
        pool_strides,
    )


_get_kv_cache_config_deepseek_v4 = _get_kv_cache_config_packed
""",
)

replace(
    "v1/core/kv_cache_utils.py",
    """    elif _use_packed_kv_cache_config(vllm_config, kv_cache_groups):
        # DeepSeek V4 uses the packed layout by default. Other multi-group
        # layouts can opt in with --enable-cross-layers.
        num_blocks, kv_cache_tensors = _get_kv_cache_config_packed(
            vllm_config, kv_cache_groups, available_memory
        )
""",
    """    elif _use_demand_sized_kv_pools(kv_cache_groups):
        (
            num_blocks,
            kv_cache_tensors,
            pool_num_blocks,
            pool_byte_strides,
        ) = _get_kv_cache_config_demand_sized(
            vllm_config, kv_cache_groups, available_memory
        )
    elif _use_packed_kv_cache_config(vllm_config, kv_cache_groups):
        # DeepSeek V4 uses the packed layout by default. Other multi-group
        # layouts can opt in with --enable-cross-layers.
        num_blocks, kv_cache_tensors = _get_kv_cache_config_packed(
            vllm_config, kv_cache_groups, available_memory
        )
""",
)

replace(
    "v1/core/kv_cache_utils.py",
    """    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=kv_cache_tensors,
        kv_cache_groups=kv_cache_groups,
    )


def unify_hybrid_kv_cache_specs""",
    """    demand_sized = _use_demand_sized_kv_pools(kv_cache_groups)
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=kv_cache_tensors,
        kv_cache_groups=kv_cache_groups,
        pool_num_blocks=pool_num_blocks if demand_sized else None,
        group_pool_ids=(tuple(range(len(kv_cache_groups))) if demand_sized else None),
        pool_byte_strides=pool_byte_strides if demand_sized else None,
    )


def unify_hybrid_kv_cache_specs""",
)

# Capacity must be the minimum supported concurrency across independent pools.
replace(
    "v1/core/kv_cache_utils.py",
    """    if _use_packed_kv_cache_config(
        vllm_config, kv_cache_config.kv_cache_groups
    ):
""",
    """    if kv_cache_config.pool_num_blocks is not None:
        demands = (
            cdiv(
                group.kv_cache_spec.max_memory_usage_bytes(vllm_config),
                group.kv_cache_spec.page_size_bytes,
            )
            for group in kv_cache_config.kv_cache_groups
        )
        return min(
            count / demand
            for count, demand in zip(kv_cache_config.pool_num_blocks, demands)
        )

    if _use_packed_kv_cache_config(
        vllm_config, kv_cache_config.kv_cache_groups
    ):
""",
)

# The internal graph-profiling override means N local IDs in every pool.
replace(
    "v1/core/kv_cache_utils.py",
    """    if len(kv_cache_groups) == 1 and isinstance(
        kv_cache_groups[0].kv_cache_spec, UniformTypeKVCacheSpecs
    ):
        return kv_cache_groups[0].kv_cache_spec.page_size_bytes
    if _use_packed_kv_cache_config(vllm_config, kv_cache_groups):
""",
    """    if len(kv_cache_groups) == 1 and isinstance(
        kv_cache_groups[0].kv_cache_spec, UniformTypeKVCacheSpecs
    ):
        return kv_cache_groups[0].kv_cache_spec.page_size_bytes
    if _use_demand_sized_kv_pools(kv_cache_groups):
        return sum(group.kv_cache_spec.page_size_bytes for group in kv_cache_groups)
    if _use_packed_kv_cache_config(vllm_config, kv_cache_groups):
""",
)

replace(
    "v1/core/kv_cache_utils.py",
    """def generate_scheduler_kv_cache_config(
    kv_cache_configs: list[KVCacheConfig],
) -> KVCacheConfig:
    \"\"\"
    Generate the KV cache configuration for the scheduler.
    \"\"\"
    assert all(
        [cfg.num_blocks == kv_cache_configs[0].num_blocks for cfg in kv_cache_configs]
    )
""",
    """def generate_scheduler_kv_cache_config(
    kv_cache_configs: list[KVCacheConfig],
) -> KVCacheConfig:
    \"\"\"
    Generate the KV cache configuration for the scheduler.
    \"\"\"
    assert all(
        [cfg.num_blocks == kv_cache_configs[0].num_blocks for cfg in kv_cache_configs]
    )
    assert all(
        cfg.pool_num_blocks == kv_cache_configs[0].pool_num_blocks
        for cfg in kv_cache_configs
    )
""",
)

# A single full sequence occupies the sum of the independent group demands;
# it no longer pays the shared tuple's cross-group padding.
replace(
    "v1/core/kv_cache_utils.py",
    """    if not kv_cache_groups:
        return 0

    if len(kv_cache_groups) == 1 and isinstance(
""",
    """    if not kv_cache_groups:
        return 0

    if _use_demand_sized_kv_pools(kv_cache_groups):
        return sum(
            group.kv_cache_spec.max_memory_usage_bytes(vllm_config)
            for group in kv_cache_groups
        )

    if len(kv_cache_groups) == 1 and isinstance(
""",
)

# TP ranks may profile slightly different free memory. Normalize every local
# namespace independently, then resize only its backing.
replace(
    "v1/core/kv_cache_utils.py",
    """    # Change the num_blocks of each rank to the smallest among all ranks.
    # We also need to shrink the tensor size proportionally to avoid
    # allocating unused memory.
    min_num_blocks = min(
        kv_cache_config.num_blocks for kv_cache_config in kv_cache_configs
    )
    for kv_cache_config in kv_cache_configs:
        num_blocks_old = kv_cache_config.num_blocks
        kv_cache_config.num_blocks = min_num_blocks

        # Shrink tensor size proportionally
        for tensor in kv_cache_config.kv_cache_tensors:
            assert tensor.size % num_blocks_old == 0
            tensor.size = tensor.size // num_blocks_old * min_num_blocks
""",
    """    if kv_cache_configs[0].pool_num_blocks is not None:
        first_pool_counts = kv_cache_configs[0].pool_num_blocks
        assert first_pool_counts is not None
        min_pool_num_blocks = tuple(
            min(cfg.pool_num_blocks[i] for cfg in kv_cache_configs)
            for i in range(len(first_pool_counts))
        )
        min_num_blocks = max(min_pool_num_blocks)
    else:
        min_pool_num_blocks = None
        min_num_blocks = min(
            kv_cache_config.num_blocks for kv_cache_config in kv_cache_configs
        )

    for kv_cache_config in kv_cache_configs:
        num_blocks_old = kv_cache_config.num_blocks
        old_pool_counts = kv_cache_config.pool_num_blocks
        kv_cache_config.num_blocks = min_num_blocks
        kv_cache_config.pool_num_blocks = min_pool_num_blocks

        for tensor in kv_cache_config.kv_cache_tensors:
            if min_pool_num_blocks is not None:
                assert old_pool_counts is not None
                old_count = old_pool_counts[tensor.pool_id]
                tensor.size = (
                    tensor.size
                    // old_count
                    * min_pool_num_blocks[tensor.pool_id]
                )
            else:
                assert tensor.size % num_blocks_old == 0
                tensor.size = tensor.size // num_blocks_old * min_num_blocks
""",
)


# GPU worker: one physical backing tensor per local block-ID namespace.
replace(
    "v1/worker/gpu_model_runner.py",
    """        kv_cache_raw_tensors: dict[str, torch.Tensor] = {}
        packed_backing: torch.Tensor | None = None
        for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
            if kv_cache_tensor.block_stride > 0:
                # Allocate once; all packed tensors alias the same backing.
                if packed_backing is None:
                    packed_backing = torch.zeros(
                        kv_cache_tensor.size,
                        dtype=torch.int8,
                        device=self.device,
                    )
                tensor = packed_backing
""",
    """        kv_cache_raw_tensors: dict[str, torch.Tensor] = {}
        packed_backings: dict[int, torch.Tensor] = {}
        for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
            if kv_cache_tensor.block_stride > 0:
                tensor = packed_backings.get(kv_cache_tensor.pool_id)
                if tensor is None:
                    tensor = torch.zeros(
                        kv_cache_tensor.size,
                        dtype=torch.int8,
                        device=self.device,
                    )
                    packed_backings[kv_cache_tensor.pool_id] = tensor
""",
)


# Scheduler coordinator: local block pools, component-wise admission, byte
# weighted usage, group-scoped prefix lookup, and ownership-aware deferred free.
replace(
    "v1/core/kv_cache_coordinator.py",
    """        self.block_pool = BlockPool(
            num_gpu_blocks=kv_cache_config.num_blocks,
            enable_caching=enable_caching,
            hash_block_size=hash_block_size,
            enable_kv_cache_events=enable_kv_cache_events,
            metrics_collector=metrics_collector,
        )

        # KV cache group indices that get the EAGLE last-block drop.
""",
    """        pool_num_blocks = kv_cache_config.pool_num_blocks or (
            kv_cache_config.num_blocks,
        )
        self.group_pool_ids = kv_cache_config.group_pool_ids or tuple(
            0 for _ in kv_cache_config.kv_cache_groups
        )
        self.block_pools = tuple(
            BlockPool(
                num_gpu_blocks=num_blocks,
                enable_caching=enable_caching,
                hash_block_size=hash_block_size,
                enable_kv_cache_events=enable_kv_cache_events,
                metrics_collector=metrics_collector,
            )
            for num_blocks in pool_num_blocks
        )
        self.block_pool = self.block_pools[0]
        self.pool_byte_strides = kv_cache_config.pool_byte_strides or (1,)
        self._block_owner = {
            id(block): pool
            for pool in self.block_pools
            for block in pool.blocks
        }

        # KV cache group indices that get the EAGLE last-block drop.
""",
)

replace(
    "v1/core/kv_cache_coordinator.py",
    """                block_pool=self.block_pool,
                enable_caching=enable_caching,
                kv_cache_group_id=i,
""",
    """                block_pool=self.block_pools[self.group_pool_ids[i]],
                enable_caching=enable_caching,
                kv_cache_group_id=i,
""",
)

replace(
    "v1/core/kv_cache_coordinator.py",
    """        return num_blocks_to_allocate

    def allocate_new_computed_blocks(
""",
    """        return num_blocks_to_allocate

    def get_num_blocks_to_allocate_by_pool(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: tuple[Sequence[KVCacheBlock], ...],
        num_encoder_tokens: int,
        total_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
    ) -> tuple[int, ...]:
        counts = [0] * len(self.block_pools)
        for i, manager in enumerate(self.single_type_managers):
            if isinstance(manager, CrossAttentionManager):
                needed = manager.get_num_blocks_to_allocate(
                    request_id,
                    num_encoder_tokens,
                    [],
                    0,
                    num_encoder_tokens,
                    apply_admission_cap=apply_admission_cap,
                )
            else:
                needed = manager.get_num_blocks_to_allocate(
                    request_id,
                    num_tokens,
                    new_computed_blocks[i],
                    total_computed_tokens,
                    num_tokens_main_model,
                    apply_admission_cap=apply_admission_cap,
                )
            counts[self.group_pool_ids[i]] += needed
        return tuple(counts)

    def can_allocate(
        self,
        required_by_pool: Sequence[int],
        watermark_by_pool: Sequence[int],
        reserved_blocks: int = 0,
    ) -> bool:
        if len(self.block_pools) > 1 and reserved_blocks:
            raise ValueError(
                "scalar KV connector reservations are unsupported with "
                "demand-sized KV pools"
            )
        return all(
            required + watermark
            <= pool.get_num_free_blocks()
            - (reserved_blocks if pool_id == 0 else 0)
            for pool_id, (required, watermark, pool) in enumerate(
                zip(required_by_pool, watermark_by_pool, self.block_pools)
            )
        )

    def get_usage(self) -> float:
        used_bytes = sum(
            (pool.num_gpu_blocks - pool.get_num_free_blocks()) * stride
            for pool, stride in zip(self.block_pools, self.pool_byte_strides)
        )
        total_bytes = sum(
            pool.num_gpu_blocks * stride
            for pool, stride in zip(self.block_pools, self.pool_byte_strides)
        )
        return used_bytes / total_bytes

    def reset_prefix_cache(self) -> bool:
        results = [pool.reset_prefix_cache() for pool in self.block_pools]
        return all(results)

    def take_events(self):
        return [event for pool in self.block_pools for event in pool.take_events()]

    def free_blocks(self, blocks: Sequence[KVCacheBlock]) -> None:
        by_pool: dict[BlockPool, list[KVCacheBlock]] = {}
        for block in blocks:
            pool = self._block_owner[id(block)]
            by_pool.setdefault(pool, []).append(block)
        for pool, owned_blocks in by_pool.items():
            pool.free_blocks(owned_blocks)

    def allocate_new_computed_blocks(
""",
)

replace(
    "v1/core/kv_cache_coordinator.py",
    """            for idx, group in enumerate(self.attention_groups):
                if group.spec == spec:
""",
    """            for idx, group in enumerate(self.attention_groups):
                if (
                    group.spec == spec
                    and self.group_pool_ids[group.group_ids[0]]
                    == self.group_pool_ids[i]
                ):
""",
)

replace(
    "v1/core/kv_cache_coordinator.py",
    """                    kv_cache_group_ids=group_ids,
                    block_pool=self.block_pool,
                    kv_cache_spec=spec,
""",
    """                    kv_cache_group_ids=group_ids,
                    block_pool=self.block_pools[self.group_pool_ids[group_ids[0]]],
                    kv_cache_spec=spec,
""",
)

replace(
    "v1/core/kv_cache_coordinator.py",
    """                kv_cache_group_ids=group_ids,
                block_pool=self.block_pool,
                kv_cache_spec=spec,
""",
    """                kv_cache_group_ids=group_ids,
                block_pool=self.block_pools[self.group_pool_ids[group_ids[0]]],
                kv_cache_spec=spec,
""",
)


replace(
    "v1/core/kv_cache_manager.py",
    """        self.watermark_blocks = int(watermark * kv_cache_config.num_blocks)
""",
    """        self.watermark_blocks = int(watermark * kv_cache_config.num_blocks)
        pool_counts = kv_cache_config.pool_num_blocks or (kv_cache_config.num_blocks,)
        self.watermark_blocks_by_pool = tuple(
            int(watermark * count) for count in pool_counts
        )
""",
)

replace(
    "v1/core/kv_cache_manager.py",
    """        return self.block_pool.get_usage()
""",
    """        return self.coordinator.get_usage()
""",
)

replace(
    "v1/core/kv_cache_manager.py",
    """            num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
                request_id=request.request_id,
                num_tokens=full_num_tokens,
                new_computed_blocks=new_computed_block_list,
                num_encoder_tokens=num_encoder_tokens,
                total_computed_tokens=total_computed_tokens,
                num_tokens_main_model=full_num_tokens,
                apply_admission_cap=True,
            )
            required_blocks = num_blocks_to_allocate + watermark_blocks
            if required_blocks > self.block_pool.get_num_free_blocks():
                return None
""",
    """            required_by_pool = self.coordinator.get_num_blocks_to_allocate_by_pool(
                request_id=request.request_id,
                num_tokens=full_num_tokens,
                new_computed_blocks=new_computed_block_list,
                num_encoder_tokens=num_encoder_tokens,
                total_computed_tokens=total_computed_tokens,
                num_tokens_main_model=full_num_tokens,
                apply_admission_cap=True,
            )
            watermark_by_pool = (
                self.watermark_blocks_by_pool
                if watermark_blocks
                else (0,) * len(self.coordinator.block_pools)
            )
            if not self.coordinator.can_allocate(
                required_by_pool, watermark_by_pool
            ):
                return None
""",
)

replace(
    "v1/core/kv_cache_manager.py",
    """        num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=num_tokens_need_slot,
            new_computed_blocks=new_computed_block_list,
            num_encoder_tokens=num_encoder_tokens,
            total_computed_tokens=num_local_computed_tokens
            + num_external_computed_tokens,
            num_tokens_main_model=num_tokens_main_model,
        )
""",
    """        required_by_pool = self.coordinator.get_num_blocks_to_allocate_by_pool(
            request_id=request.request_id,
            num_tokens=num_tokens_need_slot,
            new_computed_blocks=new_computed_block_list,
            num_encoder_tokens=num_encoder_tokens,
            total_computed_tokens=num_local_computed_tokens
            + num_external_computed_tokens,
            num_tokens_main_model=num_tokens_main_model,
        )
""",
)

replace(
    "v1/core/kv_cache_manager.py",
    """        available_blocks = self.block_pool.get_num_free_blocks() - reserved_blocks
        required_blocks = num_blocks_to_allocate + watermark_blocks
        if required_blocks > available_blocks:
            # Cannot allocate new blocks
            return None
""",
    """        watermark_by_pool = (
            self.watermark_blocks_by_pool
            if watermark_blocks
            else (0,) * len(self.coordinator.block_pools)
        )
        if not self.coordinator.can_allocate(
            required_by_pool, watermark_by_pool, reserved_blocks
        ):
            return None
""",
)

replace(
    "v1/core/kv_cache_manager.py",
    """        self.block_pool.evict_blocks(block_ids)
""",
    """        if len(self.coordinator.block_pools) != 1:
            raise ValueError(
                "external block-ID eviction is unsupported with demand-sized KV pools"
            )
        self.block_pool.evict_blocks(block_ids)
""",
)

replace(
    "v1/core/kv_cache_manager.py",
    """        if not self.block_pool.reset_prefix_cache():
""",
    """        if not self.coordinator.reset_prefix_cache():
""",
)

replace(
    "v1/core/kv_cache_manager.py",
    """        events = self.block_pool.take_events()
""",
    """        events = self.coordinator.take_events()
""",
)

replace(
    "v1/core/sched/scheduler.py",
    """            self.kv_cache_manager.block_pool.free_blocks(reversed(blocks))
""",
    """            self.kv_cache_manager.coordinator.free_blocks(
                list(reversed(blocks))
            )
""",
)


# Register the flag so vLLM's environment validator recognizes it.
replace(
    "envs.py",
    """    VLLM_PREFIX_CACHE_RETENTION_INTERVAL: int | None = None
""",
    """    VLLM_PREFIX_CACHE_RETENTION_INTERVAL: int | None = None
    VLLM_DSV4_DEMAND_SIZED_KV_POOLS: bool = False
""",
)

replace(
    "envs.py",
    """    "VLLM_PREFIX_CACHE_RETENTION_INTERVAL": lambda: (
        int(os.environ["VLLM_PREFIX_CACHE_RETENTION_INTERVAL"])
        if "VLLM_PREFIX_CACHE_RETENTION_INTERVAL" in os.environ
        else None
    ),
""",
    """    "VLLM_PREFIX_CACHE_RETENTION_INTERVAL": lambda: (
        int(os.environ["VLLM_PREFIX_CACHE_RETENTION_INTERVAL"])
        if "VLLM_PREFIX_CACHE_RETENTION_INTERVAL" in os.environ
        else None
    ),
    "VLLM_DSV4_DEMAND_SIZED_KV_POOLS": lambda: os.getenv(
        "VLLM_DSV4_DEMAND_SIZED_KV_POOLS", "0"
    ) == "1",
""",
)


# V2 model runner has a parallel allocation helper.
replace(
    "v1/worker/gpu/attn_utils.py",
    """    packed_backing: torch.Tensor | None = None
    for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
        if kv_cache_tensor.block_stride > 0:
            # Allocate once; all packed tensors alias the same backing.
            if packed_backing is None:
                packed_backing = torch.zeros(
                    kv_cache_tensor.size, dtype=torch.int8, device=device
                )
            tensor = packed_backing
""",
    """    packed_backings: dict[int, torch.Tensor] = {}
    for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
        if kv_cache_tensor.block_stride > 0:
            tensor = packed_backings.get(kv_cache_tensor.pool_id)
            if tensor is None:
                tensor = torch.zeros(
                    kv_cache_tensor.size, dtype=torch.int8, device=device
                )
                packed_backings[kv_cache_tensor.pool_id] = tensor
""",
)


print("Opt-in DeepSeek V4 demand-sized KV pool patches applied")
