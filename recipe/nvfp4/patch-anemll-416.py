"""Install the true DeepSeek V4 416-byte NVFP4 cache in Anemll vLLM 0.25."""

from pathlib import Path


ROOT = Path("/usr/local/lib/python3.12/dist-packages/vllm")


def replace(path: str, old: str, new: str) -> None:
    target = ROOT / path
    source = target.read_text()
    if new in source:
        return
    if old not in source:
        raise SystemExit(f"missing Anemll 416 patch anchor in {target}: {old!r}")
    target.write_text(source.replace(old, new, 1))


# Allocate the compact layout and use its writers.
replace(
    "models/deepseek_v4/attention.py",
    '''                prefix=f"{prefix}.compressor",
                k_cache_prefix=self.prefix,
            )
''',
    '''                prefix=f"{prefix}.compressor",
                k_cache_prefix=self.prefix,
                use_fp4_cache=self.kv_cache_dtype == "nvfp4_ds_mla",
            )
''',
)

replace(
    "models/deepseek_v4/attention.py",
    '''        if cache_dtype == torch.uint8:
            # fp8_ds_mla UE8M0 paged path. Horizontally fused:
''',
    '''        if cache_dtype == torch.uint8:
            if self.kv_cache_dtype == "nvfp4_ds_mla":
                from .nvidia.nvfp4_cache import (
                    qnorm_rope_store_swa_nvfp4_416,
                )

                return qnorm_rope_store_swa_nvfp4_416(
                    q,
                    kv,
                    swa_kv_cache,
                    swa_metadata.slot_mapping,
                    positions,
                    cos_sin_cache,
                    padded_heads=self.padded_heads,
                    rms_norm_eps=self.eps,
                    cache_block_size=swa_metadata.block_size,
                )

            # fp8_ds_mla UE8M0 paged path. Horizontally fused:
''',
)

replace(
    "models/deepseek_v4/attention.py",
    '''        alignment = (
            584
            if self.kv_cache_dtype == "nvfp4_ds_mla"
            else 576
''',
    '''        alignment = (
            416
            if self.kv_cache_dtype == "nvfp4_ds_mla"
            else 576
''',
)

replace(
    "models/deepseek_v4/compressor.py",
    '''        if self.head_dim == 512:
            assert not use_fp4_cache, (
                "MXFP4 cache is only supported for indexer (head=128)"
            )
            self._quant_block = 64
            self._token_stride = self.nope_head_dim + self.rope_head_dim * 2
            self._scale_dim = self.nope_head_dim // 64 + 1  # 7 real + 1 pad
''',
    '''        if self.head_dim == 512:
            if use_fp4_cache:
                self._quant_block = 16
                self._token_stride = 416
                self._scale_dim = 32
            else:
                self._quant_block = 64
                self._token_stride = self.nope_head_dim + self.rope_head_dim * 2
                self._scale_dim = self.nope_head_dim // 64 + 1  # 7 real + 1 pad
''',
)

replace(
    "models/deepseek_v4/compressor.py",
    '''        if current_platform.is_cuda() and self.head_dim == 512:
            from .nvidia.ops.sparse_attn_compress_cutedsl import (
                compress_norm_rope_store_cutedsl,
            )
''',
    '''        if self.use_fp4_cache:
            from .nvidia.nvfp4_cache import (
                compress_norm_rope_store_nvfp4_416,
            )

            compress_norm_rope_store_fn = compress_norm_rope_store_nvfp4_416
            extra_kwargs = {}
        elif current_platform.is_cuda() and self.head_dim == 512:
            from .nvidia.ops.sparse_attn_compress_cutedsl import (
                compress_norm_rope_store_cutedsl,
            )
''',
)

# DSpark V2 owns a separate context-KV insertion path for its draft layers.
# Those layers inherit the target cache dtype, so a true 416-byte allocation is
# still ``torch.uint8``. The upstream dtype-only branch would otherwise call
# the 584-byte FP8/UE8M0 writer into a 416-byte NVFP4 page, corrupting the
# drafter's context cache and collapsing target/draft agreement.
replace(
    "models/deepseek_v4/nvidia/dspark.py",
    '''    if cache_dtype == torch.uint8:
        # fp8_ds_mla UE8M0 paged layout
        swa_2d = swa_cache.view(swa_cache.shape[0], -1)
        torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
            dummy_q,
            kv,
            swa_2d,
            slot_mapping,
            positions,
            cos_sin_cache,
            attn.padded_heads,
            attn.eps,
            block_size,
        )
''',
    '''    if cache_dtype == torch.uint8:
        if attn.kv_cache_dtype == "nvfp4_ds_mla":
            from .nvfp4_cache import qnorm_rope_store_swa_nvfp4_416

            qnorm_rope_store_swa_nvfp4_416(
                dummy_q,
                kv,
                swa_cache,
                slot_mapping,
                positions,
                cos_sin_cache,
                padded_heads=attn.padded_heads,
                rms_norm_eps=attn.eps,
                cache_block_size=block_size,
            )
            return

        # fp8_ds_mla UE8M0 paged layout
        swa_2d = swa_cache.view(swa_cache.shape[0], -1)
        torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
            dummy_q,
            kv,
            swa_2d,
            slot_mapping,
            positions,
            cos_sin_cache,
            attn.padded_heads,
            attn.eps,
            block_size,
        )
''',
)

# Main MLA and SWA allocation contracts.
replace(
    "models/deepseek_v4/sparse_mla.py",
    '''        if cache_dtype_str in ("fp8_ds_mla", "nvfp4_ds_mla"):
            # DeepseekV4 main MLA: 584B per token (448 NoPE + 128 RoPE + 8 fp8 scale).
            # head_size passed in is the semantic head_dim (512).
            return (num_blocks, block_size, 584)
''',
    '''        if cache_dtype_str == "nvfp4_ds_mla":
            return (num_blocks, block_size, 416)
        if cache_dtype_str == "fp8_ds_mla":
            # DeepseekV4 main MLA: 584B per token (448 NoPE + 128 RoPE + 8 fp8 scale).
            # head_size passed in is the semantic head_dim (512).
            return (num_blocks, block_size, 584)
''',
)

replace(
    "v1/attention/backends/mla/sparse_swa.py",
    '''            alignment=(
                584
                if uses_nvfp4_ds_mla_layout
                else 576
''',
    '''            alignment=(
                416
                if uses_nvfp4_ds_mla_layout
                else 576
''',
)

replace(
    "v1/attention/backends/mla/sparse_swa.py",
    '''        if cache_dtype_str in ("fp8_ds_mla", "nvfp4_ds_mla"):
            # DeepseekV4 SWA: 584B per token (448 NoPE + 128 RoPE + 8 fp8 scale).
            # head_size passed in is the semantic head_dim (512).
            return (num_blocks, block_size, 584)
''',
    '''        if cache_dtype_str == "nvfp4_ds_mla":
            return (num_blocks, block_size, 416)
        if cache_dtype_str == "fp8_ds_mla":
            # DeepseekV4 SWA: 584B per token (448 NoPE + 128 RoPE + 8 fp8 scale).
            # head_size passed in is the semantic head_dim (512).
            return (num_blocks, block_size, 584)
''',
)

replace(
    "v1/kv_cache_interface.py",
    '''        if self.cache_dtype_str in ("fp8_ds_mla", "nvfp4_ds_mla"):
            if self.model_version == "deepseek_v4":
                # DeepseekV4 uses the padded 584-byte sparse-MLA envelope for
                # both fp8_ds_mla and nvfp4_ds_mla. head_size stays semantic
                # (512); bytes are determined by the backend layout here.
                return self.storage_block_size * 584
''',
    '''        if self.cache_dtype_str in ("fp8_ds_mla", "nvfp4_ds_mla"):
            if self.model_version == "deepseek_v4":
                if self.cache_dtype_str == "nvfp4_ds_mla":
                    return self.storage_block_size * 416
                # DeepseekV4 fp8_ds_mla uses a 584-byte sparse-MLA envelope.
                return self.storage_block_size * 584
''',
)

replace(
    "v1/kv_cache_interface.py",
    '''        if self.model_version == "deepseek_v4" and self.cache_dtype_str in (
            "fp8_ds_mla",
            "nvfp4_ds_mla",
        ):
            # DeepseekV4 FlashMLA: 448B NoPE + 128B RoPE + 8B fp8 scale = 584B
            # per token. FlashInfer's contiguous bf16/fp8 cache falls through to
            # the element-size formula below.
            return self.storage_block_size * 584
''',
    '''        if self.model_version == "deepseek_v4" and self.cache_dtype_str in (
            "fp8_ds_mla",
            "nvfp4_ds_mla",
        ):
            if self.cache_dtype_str == "nvfp4_ds_mla":
                return self.storage_block_size * 416
            # DeepseekV4 fp8_ds_mla: 448B NoPE + 128B RoPE + 8B scale.
            return self.storage_block_size * 584
''',
)

# The fixed FP32 compressor-state page can be larger than a compact C4 page.
replace(
    "v1/core/kv_cache_utils.py",
    '''    all_page_sizes = full_mla_spec.get_page_sizes()
    swa_mla_groups = []
    for sm_spec in swa_mla_specs:
''',
    '''    all_page_sizes = full_mla_spec.get_page_sizes()

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

    swa_mla_groups = []
    for sm_spec in swa_mla_specs:
''',
)

# Route the SM120 class around FlashInfer's 584-byte-only reader while keeping
# the fast Anemll model, MoE, projection, scheduler, and graph implementations.
replace(
    "models/deepseek_v4/nvidia/flashinfer_sparse.py",
    '''        "fp8_e4m3",
        "fp8_ds_mla",
    ]
''',
    '''        "fp8_e4m3",
        "fp8_ds_mla",
        "nvfp4_ds_mla",
    ]
''',
)

replace(
    "models/deepseek_v4/nvidia/flashinfer_sparse.py",
    '''        q = self._prepare_query(q, output)
        swa_cache = self._as_sparse_cache(self.swa_cache_layer.kv_cache)
        extra_cache = self._as_sparse_cache(kv_cache) if kv_cache is not None else None
''',
    '''        q = self._prepare_query(q, output)
        if self.kv_cache_dtype == "nvfp4_ds_mla":
            from .nvfp4_cache import sparse_attention_nvfp4_416

            sparse_attention_nvfp4_416(
                q=q,
                swa_cache=self.swa_cache_layer.kv_cache,
                swa_indices=swa_indices,
                swa_lengths=swa_lens,
                swa_page_size=swa_metadata.block_size,
                indexed_cache=kv_cache,
                indexed_indices=extra_sparse_indices,
                indexed_lengths=extra_sparse_lengths,
                indexed_page_size=(
                    attn_metadata.block_size // self.compress_ratio
                    if kv_cache is not None and attn_metadata is not None
                    else None
                ),
                sm_scale=self.scale,
                attn_sink=self.attn_sink,
                output=output,
            )
            return

        swa_cache = self._as_sparse_cache(self.swa_cache_layer.kv_cache)
        extra_cache = self._as_sparse_cache(kv_cache) if kv_cache is not None else None
''',
)

replace(
    "models/deepseek_v4/nvidia/flashinfer_sparse.py",
    '''            flashinfer_trtllm_batch_decode_sparse_mla_dsv4(
                query=q_chunk,
                swa_kv_cache=swa_kv_paged,
''',
    '''            if self.kv_cache_dtype == "nvfp4_ds_mla":
                from .nvfp4_cache import sparse_attention_nvfp4_416

                sparse_attention_nvfp4_416(
                    q=q_chunk,
                    swa_cache=swa_k_cache,
                    swa_indices=swa_indices_chunk,
                    swa_lengths=swa_lens_chunk,
                    swa_page_size=swa_metadata.block_size,
                    indexed_cache=compressed_k_cache,
                    indexed_indices=extra_sparse_indices_chunk,
                    indexed_lengths=extra_sparse_lengths_chunk,
                    indexed_page_size=(
                        attn_metadata.block_size // self.compress_ratio
                        if compressed_k_cache is not None
                        and attn_metadata is not None
                        else None
                    ),
                    sm_scale=self.scale,
                    attn_sink=self.attn_sink,
                    output=output[query_start:query_end],
                )
                continue

            flashinfer_trtllm_batch_decode_sparse_mla_dsv4(
                query=q_chunk,
                swa_kv_cache=swa_kv_paged,
''',
)

print("Anemll vLLM 0.25 true 416-byte DeepSeek V4 cache patches applied")
