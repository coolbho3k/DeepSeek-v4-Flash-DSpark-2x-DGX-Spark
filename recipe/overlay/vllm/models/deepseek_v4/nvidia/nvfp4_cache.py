# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental 416-byte NVFP4 cache support for DeepSeek V4 sparse MLA.

The record is deliberately token-major so a physical cache slot has one
self-contained representation::

    [0:256]   512 E2M1 values, two values per byte
    [256:288] 32 E4M3 block scales, one per 16 values
    [288:416] 64 BF16 RoPE values (the authoritative RoPE copy)

The packed FP4 copy includes the RoPE dimensions to retain FlashInfer's
standard ``9 * head_dim / 16`` NVFP4 payload geometry. Attention uses FP4 for
the first 448 NoPE dimensions and the BF16 copy for the final 64 dimensions.

This module is a correctness-first bridge. Prefill gathers/dequantizes directly
to the existing BF16 workspace. Decode gathers the selected sparse rows and
uses PyTorch tensor-core attention; a fused sparse reader should replace that
fallback after the record contract is validated end to end.
"""

from __future__ import annotations

import os

import torch

from vllm.triton_utils import tl, triton

HEAD_DIM = 512
NOPE_DIM = 448
ROPE_DIM = 64
FP4_DATA_BYTES = HEAD_DIM // 2
SCALE_GROUP_SIZE = 16
SCALE_BYTES = HEAD_DIM // SCALE_GROUP_SIZE
ROPE_BYTES = ROPE_DIM * 2
RECORD_BYTES = FP4_DATA_BYTES + SCALE_BYTES + ROPE_BYTES

assert RECORD_BYTES == 416


@triton.jit
def _fp32x2_to_fp4x2(x_lo, x_hi):
    # PTX writes the first argument to the high nibble, so pass hi then lo.
    return tl.inline_asm_elementwise(
        """
        {
            .reg .b8 tmp;
            cvt.rn.satfinite.e2m1x2.f32 tmp, $1, $2;
            cvt.u32.u8 $0, tmp;
        }
        """,
        constraints="=r,f,f",
        args=[x_hi, x_lo],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    ).to(tl.uint8)


@triton.jit
def _insert_nvfp4_416_kernel(
    values_ptr,
    values_stride,
    positions_ptr,
    slots_ptr,
    cache_ptr,
    cache_block_stride,
    cache_block_size: tl.constexpr,
    compress_ratio: tl.constexpr,
    head_dim: tl.constexpr,
    nope_dim: tl.constexpr,
    fp4_data_bytes: tl.constexpr,
    scale_group_size: tl.constexpr,
    scale_bytes: tl.constexpr,
    record_bytes: tl.constexpr,
):
    row = tl.program_id(0)
    slot = tl.load(slots_ptr + row)
    position = tl.load(positions_ptr + row)
    if slot < 0 or (position + 1) % compress_ratio != 0:
        return

    offsets = tl.arange(0, head_dim)
    values = tl.load(values_ptr + row * values_stride + offsets).to(tl.float32)
    grouped = tl.reshape(values, (scale_bytes, scale_group_size))
    amax = tl.max(tl.abs(grouped), axis=1)

    # Match FlashInfer nvfp4_kv_quantize with global_scale == 1.0:
    # block scale = E4M3(amax / E2M1_MAX), then quantize by its reciprocal.
    scale_f32 = amax * (1.0 / 6.0)
    scale_fp8 = scale_f32.to(tl.float8e4nv)
    rounded_scale = scale_fp8.to(tl.float32)
    inv_scale = tl.where((amax != 0.0) & (rounded_scale != 0.0), 1.0 / rounded_scale, 0.0)
    scaled = grouped * tl.reshape(inv_scale, (scale_bytes, 1))
    scaled_flat = tl.reshape(scaled, (head_dim,))
    pairs = tl.reshape(scaled_flat, (fp4_data_bytes, 2))
    lo, hi = tl.split(pairs)
    packed = _fp32x2_to_fp4x2(lo, hi)

    block = slot // cache_block_size
    token = slot % cache_block_size
    record = (
        cache_ptr
        + block.to(tl.int64) * cache_block_stride
        + token * record_bytes
    )
    tl.store(record + tl.arange(0, fp4_data_bytes), packed)
    tl.store(
        record + fp4_data_bytes + tl.arange(0, scale_bytes),
        scale_fp8.to(tl.uint8, bitcast=True),
    )

    rope_offsets = tl.arange(0, head_dim - nope_dim)
    rope_out = (record + fp4_data_bytes + scale_bytes).to(
        tl.pointer_type(tl.bfloat16)
    )
    rope_values = tl.load(
        values_ptr + row * values_stride + nope_dim + rope_offsets
    )
    tl.store(rope_out + rope_offsets, rope_values.to(tl.bfloat16))


def insert_nvfp4_416(
    values: torch.Tensor,
    positions: torch.Tensor,
    slots: torch.Tensor,
    cache: torch.Tensor,
    *,
    cache_block_size: int,
    compress_ratio: int,
) -> None:
    """Quantize and insert already-normalized, already-rotated 512-d rows."""
    if values.ndim != 2 or values.shape[1] != HEAD_DIM:
        raise ValueError(f"values must be [N, {HEAD_DIM}], got {tuple(values.shape)}")
    if cache.ndim != 3 or cache.shape[-1] != RECORD_BYTES:
        raise ValueError(
            f"cache must be [pages, page_size, {RECORD_BYTES}], got {tuple(cache.shape)}"
        )
    _insert_nvfp4_416_kernel[(values.shape[0],)](
        values,
        values.stride(0),
        positions,
        slots,
        cache,
        cache.stride(0),
        cache_block_size=cache_block_size,
        compress_ratio=compress_ratio,
        head_dim=HEAD_DIM,
        nope_dim=NOPE_DIM,
        fp4_data_bytes=FP4_DATA_BYTES,
        scale_group_size=SCALE_GROUP_SIZE,
        scale_bytes=SCALE_BYTES,
        record_bytes=RECORD_BYTES,
        num_warps=4,
    )


def _apply_gptj_rope(
    values: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Rotate the final 64 dimensions as interleaved GPT-J pairs."""
    rows = values.shape[0]
    half = ROPE_DIM // 2
    pairs = values[..., NOPE_DIM:].reshape(rows, *values.shape[1:-1], half, 2)
    even, odd = pairs.unbind(dim=-1)
    while cos.ndim < even.ndim:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    rotated = torch.stack(
        (even * cos - odd * sin, odd * cos + even * sin), dim=-1
    ).reshape(rows, *values.shape[1:-1], ROPE_DIM)
    return torch.cat((values[..., :NOPE_DIM], rotated), dim=-1)


@triton.jit
def _qnorm_rope_store_swa_nvfp4_416_kernel(
    q_ptr,
    q_stride0,
    q_stride1,
    kv_ptr,
    kv_stride0,
    output_ptr,
    output_stride0,
    output_stride1,
    cache_ptr,
    cache_block_stride,
    slots_ptr,
    positions_ptr,
    cos_sin_ptr,
    cos_sin_stride0,
    cos_sin_stride1,
    num_heads,
    rms_norm_eps,
    cache_block_size: tl.constexpr,
    head_dim: tl.constexpr,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    fp4_data_bytes: tl.constexpr,
    scale_group_size: tl.constexpr,
    scale_bytes: tl.constexpr,
    record_bytes: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    dims = tl.arange(0, head_dim)
    valid_head = head < num_heads
    q_values = tl.load(
        q_ptr + row * q_stride0 + head * q_stride1 + dims,
        mask=valid_head,
        other=0.0,
    ).to(tl.float32)
    q_inv_rms = tl.rsqrt(tl.sum(q_values * q_values, axis=0) / head_dim + rms_norm_eps)

    position = tl.load(positions_ptr + row)
    rope_dims = dims - nope_dim
    pair = tl.maximum(rope_dims // 2, 0)
    cos = tl.load(
        cos_sin_ptr + position.to(tl.int64) * cos_sin_stride0 + pair * cos_sin_stride1
    ).to(tl.float32)
    sin = tl.load(
        cos_sin_ptr
        + position.to(tl.int64) * cos_sin_stride0
        + (rope_dim // 2 + pair) * cos_sin_stride1
    ).to(tl.float32)
    even_dim = nope_dim + pair * 2
    odd_dim = even_dim + 1
    q_even = tl.load(
        q_ptr + row * q_stride0 + head * q_stride1 + even_dim,
        mask=valid_head,
        other=0.0,
    ).to(tl.float32) * q_inv_rms
    q_odd = tl.load(
        q_ptr + row * q_stride0 + head * q_stride1 + odd_dim,
        mask=valid_head,
        other=0.0,
    ).to(tl.float32) * q_inv_rms
    q_rope = tl.where(
        (rope_dims & 1) == 0,
        q_even * cos - q_odd * sin,
        q_odd * cos + q_even * sin,
    )
    q_out = tl.where(dims < nope_dim, q_values * q_inv_rms, q_rope)
    tl.store(
        output_ptr + row * output_stride0 + head * output_stride1 + dims,
        q_out.to(tl.bfloat16),
    )

    # Head zero also owns the single shared KV record for this token.
    slot = tl.load(slots_ptr + row)
    if head == 0 and slot >= 0:
        kv_values = tl.load(kv_ptr + row * kv_stride0 + dims).to(tl.float32)
        kv_even = tl.load(kv_ptr + row * kv_stride0 + even_dim).to(tl.float32)
        kv_odd = tl.load(kv_ptr + row * kv_stride0 + odd_dim).to(tl.float32)
        kv_rope = tl.where(
            (rope_dims & 1) == 0,
            kv_even * cos - kv_odd * sin,
            kv_odd * cos + kv_even * sin,
        )
        kv_rotated = tl.where(dims < nope_dim, kv_values, kv_rope)
        kv_rotated = kv_rotated.to(tl.bfloat16).to(tl.float32)
        grouped = tl.reshape(kv_rotated, (scale_bytes, scale_group_size))
        amax = tl.max(tl.abs(grouped), axis=1)
        scale_fp8 = (amax * (1.0 / 6.0)).to(tl.float8e4nv)
        rounded_scale = scale_fp8.to(tl.float32)
        inv_scale = tl.where(
            (amax != 0.0) & (rounded_scale != 0.0),
            1.0 / rounded_scale,
            0.0,
        )
        scaled = grouped * tl.reshape(inv_scale, (scale_bytes, 1))
        pairs = tl.reshape(tl.reshape(scaled, (head_dim,)), (fp4_data_bytes, 2))
        lo, hi = tl.split(pairs)
        packed = _fp32x2_to_fp4x2(lo, hi)

        block = slot // cache_block_size
        token = slot % cache_block_size
        record = (
            cache_ptr
            + block.to(tl.int64) * cache_block_stride
            + token * record_bytes
        )
        tl.store(record + tl.arange(0, fp4_data_bytes), packed)
        tl.store(
            record + fp4_data_bytes + tl.arange(0, scale_bytes),
            scale_fp8.to(tl.uint8, bitcast=True),
        )
        rope_out = (record + fp4_data_bytes + scale_bytes).to(
            tl.pointer_type(tl.bfloat16)
        )
        rope_offsets = tl.arange(0, rope_dim)
        rope_pairs = rope_offsets // 2
        rope_cos = tl.load(
            cos_sin_ptr
            + position.to(tl.int64) * cos_sin_stride0
            + rope_pairs * cos_sin_stride1
        ).to(tl.float32)
        rope_sin = tl.load(
            cos_sin_ptr
            + position.to(tl.int64) * cos_sin_stride0
            + (rope_dim // 2 + rope_pairs) * cos_sin_stride1
        ).to(tl.float32)
        rope_even = tl.load(
            kv_ptr + row * kv_stride0 + nope_dim + rope_pairs * 2
        ).to(tl.float32)
        rope_odd = tl.load(
            kv_ptr + row * kv_stride0 + nope_dim + rope_pairs * 2 + 1
        ).to(tl.float32)
        rope_store = tl.where(
            (rope_offsets & 1) == 0,
            rope_even * rope_cos - rope_odd * rope_sin,
            rope_odd * rope_cos + rope_even * rope_sin,
        )
        tl.store(
            rope_out + rope_offsets,
            rope_store.to(tl.bfloat16),
        )


def qnorm_rope_store_swa_nvfp4_416(
    q: torch.Tensor,
    kv: torch.Tensor,
    cache: torch.Tensor,
    slots: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    *,
    padded_heads: int,
    rms_norm_eps: float,
    cache_block_size: int,
) -> torch.Tensor:
    """Normalize/rotate Q and rotate/store KV in the 416-byte SWA cache."""
    if q.ndim != 3 or q.shape[-1] != HEAD_DIM:
        raise ValueError(f"q must be [N, H, {HEAD_DIM}], got {tuple(q.shape)}")
    if kv.ndim != 2 or kv.shape[-1] != HEAD_DIM or kv.shape[0] != q.shape[0]:
        raise ValueError(f"kv must be [N, {HEAD_DIM}], got {tuple(kv.shape)}")
    if padded_heads < q.shape[1]:
        raise ValueError("padded_heads cannot be smaller than the Q head count")

    output = torch.empty(
        (q.shape[0], padded_heads, HEAD_DIM), dtype=q.dtype, device=q.device
    )
    _qnorm_rope_store_swa_nvfp4_416_kernel[(q.shape[0], padded_heads)](
        q,
        q.stride(0),
        q.stride(1),
        kv,
        kv.stride(0),
        output,
        output.stride(0),
        output.stride(1),
        cache,
        cache.stride(0),
        slots,
        positions,
        cos_sin_cache,
        cos_sin_cache.stride(0),
        cos_sin_cache.stride(1),
        q.shape[1],
        rms_norm_eps,
        cache_block_size=cache_block_size,
        head_dim=HEAD_DIM,
        nope_dim=NOPE_DIM,
        rope_dim=ROPE_DIM,
        fp4_data_bytes=FP4_DATA_BYTES,
        scale_group_size=SCALE_GROUP_SIZE,
        scale_bytes=SCALE_BYTES,
        record_bytes=RECORD_BYTES,
        num_warps=8,
    )
    return output


@triton.jit
def _norm_rope_store_nvfp4_416_kernel(
    values_ptr,
    values_stride0,
    positions_ptr,
    slots_ptr,
    norm_weight_ptr,
    cos_sin_ptr,
    cos_sin_stride0,
    cos_sin_stride1,
    cache_ptr,
    cache_block_stride,
    rms_norm_eps,
    cache_block_size: tl.constexpr,
    compress_ratio: tl.constexpr,
    head_dim: tl.constexpr,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    fp4_data_bytes: tl.constexpr,
    scale_group_size: tl.constexpr,
    scale_bytes: tl.constexpr,
    record_bytes: tl.constexpr,
):
    row = tl.program_id(0)
    position = tl.load(positions_ptr + row)
    slot = tl.load(slots_ptr + row)
    if slot < 0 or (position + 1) % compress_ratio != 0:
        return

    dims = tl.arange(0, head_dim)
    values = tl.load(values_ptr + row * values_stride0 + dims).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(values * values, axis=0) / head_dim + rms_norm_eps)
    normalized = values * inv_rms * tl.load(norm_weight_ptr + dims).to(tl.float32)

    compressed_position = (position // compress_ratio) * compress_ratio
    rope_dims = dims - nope_dim
    pair = tl.maximum(rope_dims // 2, 0)
    cos = tl.load(
        cos_sin_ptr
        + compressed_position.to(tl.int64) * cos_sin_stride0
        + pair * cos_sin_stride1
    ).to(tl.float32)
    sin = tl.load(
        cos_sin_ptr
        + compressed_position.to(tl.int64) * cos_sin_stride0
        + (rope_dim // 2 + pair) * cos_sin_stride1
    ).to(tl.float32)
    even_dim = nope_dim + pair * 2
    odd_dim = even_dim + 1
    even = tl.load(values_ptr + row * values_stride0 + even_dim).to(tl.float32)
    odd = tl.load(values_ptr + row * values_stride0 + odd_dim).to(tl.float32)
    even = even * inv_rms * tl.load(norm_weight_ptr + even_dim).to(tl.float32)
    odd = odd * inv_rms * tl.load(norm_weight_ptr + odd_dim).to(tl.float32)
    rope = tl.where(
        (rope_dims & 1) == 0,
        even * cos - odd * sin,
        odd * cos + even * sin,
    )
    rotated = tl.where(dims < nope_dim, normalized, rope)
    rotated = rotated.to(tl.bfloat16).to(tl.float32)

    grouped = tl.reshape(rotated, (scale_bytes, scale_group_size))
    amax = tl.max(tl.abs(grouped), axis=1)
    scale_fp8 = (amax * (1.0 / 6.0)).to(tl.float8e4nv)
    rounded_scale = scale_fp8.to(tl.float32)
    inv_scale = tl.where(
        (amax != 0.0) & (rounded_scale != 0.0), 1.0 / rounded_scale, 0.0
    )
    scaled = grouped * tl.reshape(inv_scale, (scale_bytes, 1))
    pairs = tl.reshape(tl.reshape(scaled, (head_dim,)), (fp4_data_bytes, 2))
    lo, hi = tl.split(pairs)
    packed = _fp32x2_to_fp4x2(lo, hi)

    block = slot // cache_block_size
    token = slot % cache_block_size
    record = (
        cache_ptr
        + block.to(tl.int64) * cache_block_stride
        + token * record_bytes
    )
    tl.store(record + tl.arange(0, fp4_data_bytes), packed)
    tl.store(
        record + fp4_data_bytes + tl.arange(0, scale_bytes),
        scale_fp8.to(tl.uint8, bitcast=True),
    )
    rope_out = (record + fp4_data_bytes + scale_bytes).to(
        tl.pointer_type(tl.bfloat16)
    )
    rope_offsets = tl.arange(0, rope_dim)
    rope_pairs = rope_offsets // 2
    rope_cos = tl.load(
        cos_sin_ptr
        + compressed_position.to(tl.int64) * cos_sin_stride0
        + rope_pairs * cos_sin_stride1
    ).to(tl.float32)
    rope_sin = tl.load(
        cos_sin_ptr
        + compressed_position.to(tl.int64) * cos_sin_stride0
        + (rope_dim // 2 + rope_pairs) * cos_sin_stride1
    ).to(tl.float32)
    rope_even_dim = nope_dim + rope_pairs * 2
    rope_odd_dim = rope_even_dim + 1
    rope_even = tl.load(values_ptr + row * values_stride0 + rope_even_dim).to(
        tl.float32
    )
    rope_odd = tl.load(values_ptr + row * values_stride0 + rope_odd_dim).to(
        tl.float32
    )
    rope_even = (
        rope_even
        * inv_rms
        * tl.load(norm_weight_ptr + rope_even_dim).to(tl.float32)
    )
    rope_odd = (
        rope_odd
        * inv_rms
        * tl.load(norm_weight_ptr + rope_odd_dim).to(tl.float32)
    )
    rope_store = tl.where(
        (rope_offsets & 1) == 0,
        rope_even * rope_cos - rope_odd * rope_sin,
        rope_odd * rope_cos + rope_even * rope_sin,
    )
    tl.store(
        rope_out + rope_offsets,
        rope_store.to(tl.bfloat16),
    )


def norm_rope_store_nvfp4_416(
    values: torch.Tensor,
    positions: torch.Tensor,
    slots: torch.Tensor,
    norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    cache: torch.Tensor,
    *,
    rms_norm_eps: float,
    cache_block_size: int,
    compress_ratio: int,
) -> None:
    """Fused weighted RMSNorm, RoPE, NVFP4 quantization, and cache store."""
    if values.ndim != 2 or values.shape[1] != HEAD_DIM:
        raise ValueError(f"values must be [N, {HEAD_DIM}], got {tuple(values.shape)}")
    if norm_weight.shape != (HEAD_DIM,):
        raise ValueError(f"norm_weight must be [{HEAD_DIM}]")
    if cache.ndim != 3 or cache.shape[-1] != RECORD_BYTES:
        raise ValueError(f"cache must contain {RECORD_BYTES}-byte records")
    _norm_rope_store_nvfp4_416_kernel[(values.shape[0],)](
        values,
        values.stride(0),
        positions,
        slots,
        norm_weight,
        cos_sin_cache,
        cos_sin_cache.stride(0),
        cos_sin_cache.stride(1),
        cache,
        cache.stride(0),
        rms_norm_eps,
        cache_block_size=cache_block_size,
        compress_ratio=compress_ratio,
        head_dim=HEAD_DIM,
        nope_dim=NOPE_DIM,
        rope_dim=ROPE_DIM,
        fp4_data_bytes=FP4_DATA_BYTES,
        scale_group_size=SCALE_GROUP_SIZE,
        scale_bytes=SCALE_BYTES,
        record_bytes=RECORD_BYTES,
        num_warps=8,
    )


@triton.jit
def _compress_kv_c4_nvfp4_416_kernel(
    state_cache_ptr,
    state_cache_stride0,
    state_cache_stride1,
    token_to_req_indices_ptr,
    positions_ptr,
    slot_mapping_ptr,
    block_table_ptr,
    block_table_stride0,
    compressed_ptr,
    compressed_stride0,
    state_block_size,
    state_width: tl.constexpr,
    head_dim: tl.constexpr,
    compress_ratio: tl.constexpr,
):
    """Compression-only C4A kernel missing from the Anemll vLLM 0.25 tree."""
    row = tl.program_id(0)
    position = tl.load(positions_ptr + row)
    slot = tl.load(slot_mapping_ptr + row)
    if slot < 0 or (position + 1) % compress_ratio != 0:
        return

    request = tl.load(token_to_req_indices_ptr + row)
    dims = tl.arange(0, head_dim)
    running_max = tl.full((head_dim,), -float("inf"), tl.float32)
    running_sum = tl.zeros((head_dim,), tl.float32)
    running_product = tl.zeros((head_dim,), tl.float32)
    start = position - (2 * compress_ratio - 1)

    # C4A's overlapping window has eight rows. The first four use the first
    # 512-wide state segment and the second four use the second segment.
    for window_row in range(2 * compress_ratio):
        logical = start + window_row
        valid = logical >= 0
        table_column = logical // state_block_size
        physical = tl.load(
            block_table_ptr + request * block_table_stride0 + table_column,
            mask=valid,
            other=0,
        )
        block_offset = logical % state_block_size
        segment = (window_row // compress_ratio) * head_dim
        state_row = (
            state_cache_ptr
            + physical.to(tl.int64) * state_cache_stride0
            + block_offset * state_cache_stride1
            + segment
        )
        values = tl.load(state_row + dims, mask=valid, other=0.0).to(tl.float32)
        scores = tl.load(
            state_row + state_width + dims,
            mask=valid,
            other=-float("inf"),
        ).to(tl.float32)

        new_max = tl.maximum(running_max, scores)
        old_scale = tl.where(
            running_max == -float("inf"),
            0.0,
            tl.exp2((running_max - new_max) * 1.4426950408889634),
        )
        new_scale = tl.where(
            valid,
            tl.exp2((scores - new_max) * 1.4426950408889634),
            0.0,
        )
        running_sum = running_sum * old_scale + new_scale
        running_product = running_product * old_scale + values * new_scale
        running_max = new_max

    tl.store(
        compressed_ptr + row * compressed_stride0 + dims,
        running_product / running_sum,
    )


def compress_kv_nvfp4_416(
    state_cache: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    compressed: torch.Tensor,
    *,
    state_width: int,
    compress_ratio: int,
    overlap: bool,
) -> None:
    """Run the best compression-only kernel exposed by the installed vLLM."""
    try:
        from vllm.models.deepseek_v4.nvidia.ops.sparse_attn_compress_cutedsl import (
            compress_kv_sparse_attn_cutedsl,
        )
    except ImportError:
        compress_kv_sparse_attn_cutedsl = None

    if compress_kv_sparse_attn_cutedsl is not None:
        compress_kv_sparse_attn_cutedsl(
            state_cache,
            token_to_req_indices,
            positions,
            slot_mapping,
            block_table,
            block_size,
            compressed,
            head_size=HEAD_DIM,
            state_width=state_width,
            compress_ratio=compress_ratio,
            overlap=overlap,
        )
        return

    if compress_ratio == 4:
        if not overlap or state_width != 2 * HEAD_DIM:
            raise ValueError("the C4A compressor requires overlap and state_width=1024")
        _compress_kv_c4_nvfp4_416_kernel[(compressed.shape[0],)](
            state_cache,
            state_cache.stride(0),
            state_cache.stride(1),
            token_to_req_indices,
            positions,
            slot_mapping,
            block_table,
            block_table.stride(0),
            compressed,
            compressed.stride(0),
            block_size,
            state_width=state_width,
            head_dim=HEAD_DIM,
            compress_ratio=compress_ratio,
            num_warps=8,
        )
        return

    if compress_ratio == 128:
        if overlap or state_width != HEAD_DIM or block_size != 8:
            raise ValueError(
                "the C128 compressor requires no overlap, state_width=512, "
                "and block_size=8"
            )
        from vllm.models.deepseek_v4.nvidia.ops.sparse_attn_compress_cutedsl import (
            SparseAttnCompressC128Block8Kernel,
        )

        compiled = SparseAttnCompressC128Block8Kernel.compile(
            head_size=HEAD_DIM,
            state_width=state_width,
        )
        compiled(
            state_cache,
            token_to_req_indices,
            positions,
            slot_mapping,
            block_table,
            compressed,
        )
        return

    raise ValueError(f"unsupported DeepSeek V4 compression ratio: {compress_ratio}")


def compress_norm_rope_store_nvfp4_416(
    state_cache: torch.Tensor,
    num_actual: int,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    state_width: int,
    cos_sin_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    k_cache_metadata,
    pdl_kwargs: dict,
    head_dim: int,
    rope_head_dim: int,
    compress_ratio: int,
    overlap: bool,
    use_fp4_cache: bool,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    quant_block: int,
    token_stride: int,
    scale_dim: int,
) -> None:
    """Correctness-first compressor bridge for the true 416-byte cache."""
    del pdl_kwargs, quant_block, token_stride, scale_dim
    if not use_fp4_cache or head_dim != HEAD_DIM or rope_head_dim != ROPE_DIM:
        raise ValueError("the 416-byte writer requires the DSV4 512/64 FP4 contract")

    compressed = torch.empty(
        (num_actual, HEAD_DIM), dtype=torch.float32, device=state_cache.device
    )
    compress_kv_nvfp4_416(
        state_cache,
        token_to_req_indices,
        positions,
        slot_mapping,
        block_table,
        block_size,
        compressed,
        state_width=state_width,
        compress_ratio=compress_ratio,
        overlap=overlap,
    )

    norm_rope_store_nvfp4_416(
        compressed,
        positions,
        k_cache_metadata.slot_mapping,
        rms_norm_weight,
        cos_sin_cache,
        kv_cache,
        rms_norm_eps=rms_norm_eps,
        cache_block_size=kv_cache.shape[1],
        compress_ratio=compress_ratio,
    )


@triton.jit
def _gather_paged_nvfp4_416_kernel(
    out_ptr,
    out_stride0,
    out_stride1,
    cache_ptr,
    seq_lens_ptr,
    gather_lens_ptr,
    block_table_ptr,
    block_table_stride,
    offset,
    cache_block_stride,
    cache_block_size: tl.constexpr,
    head_dim: tl.constexpr,
    nope_dim: tl.constexpr,
    fp4_data_bytes: tl.constexpr,
    scale_group_size: tl.constexpr,
    scale_bytes: tl.constexpr,
    record_bytes: tl.constexpr,
):
    req = tl.program_id(0)
    worker = tl.program_id(1)
    workers = tl.num_programs(1)
    seq_len = tl.load(seq_lens_ptr + req)
    gather_len = seq_len
    if gather_lens_ptr is not None:
        gather_len = tl.load(gather_lens_ptr + req)
    start = seq_len - gather_len

    dims = tl.arange(0, head_dim)
    for i in range(worker, gather_len, workers):
        logical = start + i
        table_block = logical // cache_block_size
        token = logical % cache_block_size
        physical = tl.load(
            block_table_ptr + req * block_table_stride + table_block
        )
        record = (
            cache_ptr
            + physical.to(tl.int64) * cache_block_stride
            + token * record_bytes
        )

        packed = tl.load(record + dims // 2)
        shift = (dims & 1) * 4
        nibble = (packed >> shift) & 0xF
        magnitude = nibble & 0x7
        fp4 = tl.where(
            magnitude <= 4,
            magnitude.to(tl.float32) * 0.5,
            tl.where(magnitude == 5, 3.0, tl.where(magnitude == 6, 4.0, 6.0)),
        )
        fp4 = tl.where((nibble & 0x8) != 0, -fp4, fp4)
        scale_u8 = tl.load(record + fp4_data_bytes + dims // scale_group_size)
        scale = scale_u8.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        dequant = fp4 * scale

        rope_ptr = (record + fp4_data_bytes + scale_bytes).to(
            tl.pointer_type(tl.bfloat16)
        )
        rope_dim = dims - nope_dim
        rope = tl.load(rope_ptr + rope_dim, mask=dims >= nope_dim, other=0.0)
        values = tl.where(dims < nope_dim, dequant, rope.to(tl.float32))
        out_row = out_ptr + req * out_stride0 + (offset + i) * out_stride1
        tl.store(out_row + dims, values.to(tl.bfloat16))


def dequantize_and_gather_nvfp4_416(
    out: torch.Tensor,
    cache: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor | None,
    block_table: torch.Tensor,
    block_size: int,
    offset: int,
) -> None:
    if cache.shape[-1] != RECORD_BYTES:
        raise ValueError(f"expected a {RECORD_BYTES}-byte cache record")
    workers = 128
    _gather_paged_nvfp4_416_kernel[(seq_lens.shape[0], workers)](
        out,
        out.stride(0),
        out.stride(1),
        cache,
        seq_lens,
        gather_lens,
        block_table,
        block_table.stride(0),
        offset,
        cache.stride(0),
        cache_block_size=block_size,
        head_dim=HEAD_DIM,
        nope_dim=NOPE_DIM,
        fp4_data_bytes=FP4_DATA_BYTES,
        scale_group_size=SCALE_GROUP_SIZE,
        scale_bytes=SCALE_BYTES,
        record_bytes=RECORD_BYTES,
    )


@triton.jit
def _gather_selected_kernel(
    out_ptr,
    out_stride0,
    out_stride1,
    cache_ptr,
    cache_block_stride,
    indices_ptr,
    indices_stride,
    lengths_ptr,
    output_column_offset,
    page_size: tl.constexpr,
    is_nvfp4: tl.constexpr,
    head_dim: tl.constexpr,
    nope_dim: tl.constexpr,
    fp4_data_bytes: tl.constexpr,
    scale_group_size: tl.constexpr,
    scale_bytes: tl.constexpr,
    record_bytes: tl.constexpr,
):
    row = tl.program_id(0)
    column = tl.program_id(1)
    length = tl.load(lengths_ptr + row)
    dims = tl.arange(0, head_dim)
    out_row = (
        out_ptr
        + row * out_stride0
        + (output_column_offset + column) * out_stride1
    )
    if column >= length:
        tl.store(out_row + dims, tl.zeros((head_dim,), tl.bfloat16))
        return

    slot = tl.load(indices_ptr + row * indices_stride + column)
    if slot < 0:
        tl.store(out_row + dims, tl.zeros((head_dim,), tl.bfloat16))
        return
    block = slot // page_size
    token = slot % page_size
    page = cache_ptr + block.to(tl.int64) * cache_block_stride

    if is_nvfp4:
        record = page + token * record_bytes
        packed = tl.load(record + dims // 2)
        shift = (dims & 1) * 4
        nibble = (packed >> shift) & 0xF
        magnitude = nibble & 0x7
        values = tl.where(
            magnitude <= 4,
            magnitude.to(tl.float32) * 0.5,
            tl.where(magnitude == 5, 3.0, tl.where(magnitude == 6, 4.0, 6.0)),
        )
        values = tl.where((nibble & 0x8) != 0, -values, values)
        scale_u8 = tl.load(record + fp4_data_bytes + dims // scale_group_size)
        scale = scale_u8.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        values *= scale
        rope_ptr = (record + fp4_data_bytes + scale_bytes).to(
            tl.pointer_type(tl.bfloat16)
        )
        rope = tl.load(
            rope_ptr + (dims - nope_dim), mask=dims >= nope_dim, other=0.0
        )
        values = tl.where(dims < nope_dim, values, rope.to(tl.float32))
    else:
        # Existing FP8 DSV4 page: all 576-byte payloads followed by 8-byte
        # per-token UE8M0 scale records.
        payload = page + token * 576
        fp8_u8 = tl.load(payload + dims, mask=dims < nope_dim, other=0)
        fp8 = fp8_u8.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        scale_u8 = tl.load(
            page + page_size * 576 + token * 8 + dims // 64,
            mask=dims < nope_dim,
            other=127,
        )
        scale = tl.exp2(scale_u8.to(tl.float32) - 127.0)
        values = fp8 * scale
        rope_ptr = (payload + nope_dim).to(tl.pointer_type(tl.bfloat16))
        rope = tl.load(
            rope_ptr + (dims - nope_dim), mask=dims >= nope_dim, other=0.0
        )
        values = tl.where(dims < nope_dim, values, rope.to(tl.float32))
    tl.store(out_row + dims, values.to(tl.bfloat16))


def _index_matrix(indices: torch.Tensor) -> torch.Tensor:
    if indices.ndim == 3:
        if indices.shape[1] != 1:
            raise ValueError(f"expected singleton KV-head axis, got {indices.shape}")
        indices = indices[:, 0]
    if indices.ndim != 2:
        raise ValueError(f"indices must be rank 2/3, got {indices.shape}")
    return indices


@triton.jit
def _load_nvfp4_416_record(
    cache_ptr,
    cache_block_stride,
    slot,
    valid,
    dims,
    page_size: tl.constexpr,
    nope_dim: tl.constexpr,
    fp4_data_bytes: tl.constexpr,
    scale_group_size: tl.constexpr,
    scale_bytes: tl.constexpr,
    record_bytes: tl.constexpr,
):
    safe_slot = tl.where(valid, slot, 0)
    block = safe_slot // page_size
    token = safe_slot % page_size
    record = (
        cache_ptr
        + block.to(tl.int64) * cache_block_stride
        + token * record_bytes
    )

    packed = tl.load(record + dims // 2, mask=valid, other=0)
    shift = (dims & 1) * 4
    nibble = (packed >> shift) & 0xF
    magnitude = nibble & 0x7
    values = tl.where(
        magnitude <= 4,
        magnitude.to(tl.float32) * 0.5,
        tl.where(magnitude == 5, 3.0, tl.where(magnitude == 6, 4.0, 6.0)),
    )
    values = tl.where((nibble & 0x8) != 0, -values, values)
    scale_u8 = tl.load(
        record + fp4_data_bytes + dims // scale_group_size,
        mask=valid,
        other=0,
    )
    scale = scale_u8.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    values *= scale

    rope_ptr = (record + fp4_data_bytes + scale_bytes).to(
        tl.pointer_type(tl.bfloat16)
    )
    rope_dim = dims - nope_dim
    rope = tl.load(rope_ptr + rope_dim, mask=valid & (dims >= nope_dim), other=0.0)
    return tl.where(dims < nope_dim, values, rope.to(tl.float32))


@triton.jit
def _sparse_attention_nvfp4_416_split_kernel(
    q_ptr,
    q_stride0,
    q_stride1,
    cache_ptr,
    cache_block_stride,
    indices_ptr,
    indices_stride0,
    indices_stride1,
    lengths_ptr,
    lengths_stride,
    mid_out_ptr,
    mid_out_stride0,
    mid_out_stride1,
    mid_out_stride2,
    mid_out_stride3,
    mid_lse_ptr,
    mid_lse_stride0,
    mid_lse_stride1,
    mid_lse_stride2,
    split_offset,
    index_width,
    sm_scale,
    page_size: tl.constexpr,
    head_dim: tl.constexpr,
    nope_dim: tl.constexpr,
    fp4_data_bytes: tl.constexpr,
    scale_group_size: tl.constexpr,
    scale_bytes: tl.constexpr,
    record_bytes: tl.constexpr,
    split_tile: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    local_split = tl.program_id(2)
    out_split = split_offset + local_split
    dims = tl.arange(0, head_dim)
    q = tl.load(q_ptr + row * q_stride0 + head * q_stride1 + dims).to(
        tl.float32
    )

    length = tl.minimum(tl.load(lengths_ptr + row * lengths_stride), index_width)
    first_col = local_split * split_tile
    running_max = -float("inf")
    running_sum = 0.0
    accumulator = tl.zeros((head_dim,), tl.float32)

    for key_offset in tl.static_range(0, split_tile):
        column = first_col + key_offset
        valid = column < length
        slot = tl.load(
            indices_ptr + row * indices_stride0 + column * indices_stride1,
            mask=valid,
            other=-1,
        )
        valid = valid & (slot >= 0)
        if valid:
            key = _load_nvfp4_416_record(
                cache_ptr,
                cache_block_stride,
                slot,
                valid,
                dims,
                page_size=page_size,
                nope_dim=nope_dim,
                fp4_data_bytes=fp4_data_bytes,
                scale_group_size=scale_group_size,
                scale_bytes=scale_bytes,
                record_bytes=record_bytes,
            )
            score = tl.sum(q * key, axis=0) * sm_scale
            new_max = tl.maximum(running_max, score)
            old_scale = tl.exp2((running_max - new_max) * 1.4426950408889634)
            new_scale = tl.exp2((score - new_max) * 1.4426950408889634)
            accumulator = accumulator * old_scale + key * new_scale
            running_sum = running_sum * old_scale + new_scale
            running_max = new_max

    inverse_sum = tl.where(running_sum > 0.0, 1.0 / running_sum, 0.0)
    partial = accumulator * inverse_sum
    lse = tl.where(
        running_sum > 0.0,
        running_max + tl.log2(running_sum) * 0.6931471805599453,
        -float("inf"),
    )
    mid_out = (
        mid_out_ptr
        + row * mid_out_stride0
        + head * mid_out_stride1
        + out_split * mid_out_stride2
        + dims * mid_out_stride3
    )
    tl.store(mid_out, partial.to(tl.bfloat16))
    tl.store(
        mid_lse_ptr
        + row * mid_lse_stride0
        + head * mid_lse_stride1
        + out_split * mid_lse_stride2,
        lse,
    )


@triton.jit
def _sparse_attention_nvfp4_416_split_mma_kernel(
    q_ptr,
    q_stride0,
    q_stride1,
    cache_ptr,
    cache_block_stride,
    indices_ptr,
    indices_stride0,
    indices_stride1,
    lengths_ptr,
    lengths_stride,
    mid_out_ptr,
    mid_out_stride0,
    mid_out_stride1,
    mid_out_stride2,
    mid_out_stride3,
    mid_lse_ptr,
    mid_lse_stride0,
    mid_lse_stride1,
    mid_lse_stride2,
    split_offset,
    index_width,
    num_heads,
    sm_scale,
    page_size: tl.constexpr,
    head_dim: tl.constexpr,
    nope_dim: tl.constexpr,
    fp4_data_bytes: tl.constexpr,
    scale_group_size: tl.constexpr,
    scale_bytes: tl.constexpr,
    record_bytes: tl.constexpr,
    split_tile: tl.constexpr,
    key_tile: tl.constexpr,
    head_tile: tl.constexpr,
):
    """One CTA reuses one record tile across all padded query heads.

    Records are dequantized only into the CTA's on-chip tile.  QK and PV use
    BF16 tensor-core dot operations; neither KV nor scores are written to
    global memory.
    """
    row = tl.program_id(0)
    local_split = tl.program_id(1)
    out_split = split_offset + local_split
    dims = tl.arange(0, head_dim)[None, :]
    columns = local_split * split_tile + tl.arange(0, split_tile)
    length = tl.minimum(tl.load(lengths_ptr + row * lengths_stride), index_width)
    valid = columns < length
    slots = tl.load(
        indices_ptr + row * indices_stride0 + columns * indices_stride1,
        mask=valid,
        other=-1,
    )
    valid = valid & (slots >= 0)
    key = _load_nvfp4_416_record(
        cache_ptr,
        cache_block_stride,
        slots[:, None],
        valid[:, None],
        dims,
        page_size=page_size,
        nope_dim=nope_dim,
        fp4_data_bytes=fp4_data_bytes,
        scale_group_size=scale_group_size,
        scale_bytes=scale_bytes,
        record_bytes=record_bytes,
    ).to(tl.bfloat16)

    for head_start in tl.range(0, num_heads, head_tile):
        heads = (head_start + tl.arange(0, head_tile))[:, None]
        q = tl.load(
            q_ptr + row * q_stride0 + heads * q_stride1 + dims,
            mask=heads < num_heads,
            other=0.0,
        ).to(tl.bfloat16)
        scores = tl.dot(q, key.T, out_dtype=tl.float32) * sm_scale
        scores = tl.where(valid[None, :], scores, -1.0e30)
        row_max = tl.max(scores, axis=1)
        weights = tl.exp2((scores - row_max[:, None]) * 1.4426950408889634)
        weights = tl.where(valid[None, :], weights, 0.0)
        row_sum = tl.sum(weights, axis=1)
        denominator = tl.where(row_sum > 0.0, row_sum, 1.0)
        partial = tl.dot(weights.to(tl.bfloat16), key, out_dtype=tl.float32)
        partial /= denominator[:, None]
        lse = tl.where(
            row_sum > 0.0,
            row_max + tl.log2(row_sum) * 0.6931471805599453,
            -float("inf"),
        )

        mid_out = (
            mid_out_ptr
            + row * mid_out_stride0
            + heads * mid_out_stride1
            + out_split * mid_out_stride2
            + dims * mid_out_stride3
        )
        tl.store(mid_out, partial.to(tl.bfloat16), mask=heads < num_heads)
        head_offsets = head_start + tl.arange(0, head_tile)
        tl.store(
            mid_lse_ptr
            + row * mid_lse_stride0
            + head_offsets * mid_lse_stride1
            + out_split * mid_lse_stride2,
            lse,
            mask=head_offsets < num_heads,
        )


@triton.jit
def _sparse_attention_nvfp4_416_grouped_mma_kernel(
    q_ptr,
    q_stride0,
    q_stride1,
    cache_ptr,
    cache_block_stride,
    indices_ptr,
    indices_stride0,
    indices_stride1,
    lengths_ptr,
    lengths_stride,
    mid_out_ptr,
    mid_out_stride0,
    mid_out_stride1,
    mid_out_stride2,
    mid_out_stride3,
    mid_lse_ptr,
    mid_lse_stride0,
    mid_lse_stride1,
    mid_lse_stride2,
    split_offset,
    index_width,
    num_heads,
    sm_scale,
    page_size: tl.constexpr,
    head_dim: tl.constexpr,
    nope_dim: tl.constexpr,
    fp4_data_bytes: tl.constexpr,
    scale_group_size: tl.constexpr,
    scale_bytes: tl.constexpr,
    record_bytes: tl.constexpr,
    split_tile: tl.constexpr,
    key_tile: tl.constexpr,
    head_tile: tl.constexpr,
):
    """Group several 32-record MMA tiles in one CTA.

    Keeping only one key tile resident stays within GB10 shared memory. Each
    head tile carries its online-softmax accumulator across key tiles, reducing
    CTA count for the fixed-width C128A index buffer.
    """
    row = tl.program_id(0)
    local_split = tl.program_id(1)
    out_split = split_offset + local_split
    dims = tl.arange(0, head_dim)[None, :]
    length = tl.minimum(tl.load(lengths_ptr + row * lengths_stride), index_width)

    for head_start in tl.range(0, num_heads, head_tile):
        heads = (head_start + tl.arange(0, head_tile))[:, None]
        q = tl.load(
            q_ptr + row * q_stride0 + heads * q_stride1 + dims,
            mask=heads < num_heads,
            other=0.0,
        ).to(tl.bfloat16)
        running_max = tl.full((head_tile,), -float("inf"), tl.float32)
        running_sum = tl.zeros((head_tile,), tl.float32)
        accumulator = tl.zeros((head_tile, head_dim), tl.float32)

        for key_start in tl.range(0, split_tile, key_tile):
            columns = (
                local_split * split_tile
                + key_start
                + tl.arange(0, key_tile)
            )
            valid = columns < length
            slots = tl.load(
                indices_ptr
                + row * indices_stride0
                + columns * indices_stride1,
                mask=valid,
                other=-1,
            )
            valid = valid & (slots >= 0)
            key = _load_nvfp4_416_record(
                cache_ptr,
                cache_block_stride,
                slots[:, None],
                valid[:, None],
                dims,
                page_size=page_size,
                nope_dim=nope_dim,
                fp4_data_bytes=fp4_data_bytes,
                scale_group_size=scale_group_size,
                scale_bytes=scale_bytes,
                record_bytes=record_bytes,
            ).to(tl.bfloat16)
            scores = tl.dot(q, key.T, out_dtype=tl.float32) * sm_scale
            scores = tl.where(valid[None, :], scores, -1.0e30)
            tile_max = tl.max(scores, axis=1)
            new_max = tl.maximum(running_max, tile_max)
            old_scale = tl.exp2(
                (running_max - new_max) * 1.4426950408889634
            )
            weights = tl.exp2(
                (scores - new_max[:, None]) * 1.4426950408889634
            )
            weights = tl.where(valid[None, :], weights, 0.0)
            tile_sum = tl.sum(weights, axis=1)
            tile_product = tl.dot(
                weights.to(tl.bfloat16), key, out_dtype=tl.float32
            )
            accumulator = accumulator * old_scale[:, None] + tile_product
            running_sum = running_sum * old_scale + tile_sum
            running_max = new_max

        denominator = tl.where(running_sum > 0.0, running_sum, 1.0)
        partial = accumulator / denominator[:, None]
        lse = tl.where(
            running_sum > 0.0,
            running_max + tl.log2(running_sum) * 0.6931471805599453,
            -float("inf"),
        )
        mid_out = (
            mid_out_ptr
            + row * mid_out_stride0
            + heads * mid_out_stride1
            + out_split * mid_out_stride2
            + dims * mid_out_stride3
        )
        tl.store(mid_out, partial.to(tl.bfloat16), mask=heads < num_heads)
        head_offsets = head_start + tl.arange(0, head_tile)
        tl.store(
            mid_lse_ptr
            + row * mid_lse_stride0
            + head_offsets * mid_lse_stride1
            + out_split * mid_lse_stride2,
            lse,
            mask=head_offsets < num_heads,
        )


@triton.jit
def _sparse_attention_nvfp4_416_merge_kernel(
    mid_out_ptr,
    mid_out_stride0,
    mid_out_stride1,
    mid_out_stride2,
    mid_out_stride3,
    mid_lse_ptr,
    mid_lse_stride0,
    mid_lse_stride1,
    mid_lse_stride2,
    sink_ptr,
    sink_stride,
    output_ptr,
    output_stride0,
    output_stride1,
    total_splits,
    head_dim: tl.constexpr,
    has_sink: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    dims = tl.arange(0, head_dim)
    running_max = -float("inf")
    running_sum = 0.0
    accumulator = tl.zeros((head_dim,), tl.float32)

    if has_sink:
        running_max = tl.load(sink_ptr + head * sink_stride).to(tl.float32)
        running_sum = 1.0

    for split in tl.range(0, total_splits):
        lse = tl.load(
            mid_lse_ptr
            + row * mid_lse_stride0
            + head * mid_lse_stride1
            + split * mid_lse_stride2
        )
        if lse != -float("inf"):
            partial = tl.load(
                mid_out_ptr
                + row * mid_out_stride0
                + head * mid_out_stride1
                + split * mid_out_stride2
                + dims * mid_out_stride3
            ).to(tl.float32)
            new_max = tl.maximum(running_max, lse)
            old_scale = tl.exp2((running_max - new_max) * 1.4426950408889634)
            new_scale = tl.exp2((lse - new_max) * 1.4426950408889634)
            accumulator = accumulator * old_scale + partial * new_scale
            running_sum = running_sum * old_scale + new_scale
            running_max = new_max

    result = accumulator * tl.where(running_sum > 0.0, 1.0 / running_sum, 0.0)
    tl.store(
        output_ptr + row * output_stride0 + head * output_stride1 + dims,
        result.to(tl.bfloat16),
    )


@triton.jit
def _sparse_attention_nvfp4_416_onepass_kernel(
    q_ptr,
    q_stride0,
    q_stride1,
    swa_cache_ptr,
    swa_cache_block_stride,
    swa_indices_ptr,
    swa_indices_stride0,
    swa_indices_stride1,
    swa_lengths_ptr,
    swa_lengths_stride,
    swa_width,
    indexed_cache_ptr,
    indexed_cache_block_stride,
    indexed_indices_ptr,
    indexed_indices_stride0,
    indexed_indices_stride1,
    indexed_lengths_ptr,
    indexed_lengths_stride,
    indexed_width,
    sink_ptr,
    sink_stride,
    output_ptr,
    output_stride0,
    output_stride1,
    sm_scale,
    swa_page_size: tl.constexpr,
    indexed_page_size: tl.constexpr,
    head_dim: tl.constexpr,
    nope_dim: tl.constexpr,
    fp4_data_bytes: tl.constexpr,
    scale_group_size: tl.constexpr,
    scale_bytes: tl.constexpr,
    record_bytes: tl.constexpr,
    has_indexed: tl.constexpr,
    has_sink: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    dims = tl.arange(0, head_dim)
    q = tl.load(q_ptr + row * q_stride0 + head * q_stride1 + dims).to(
        tl.float32
    )
    running_max = -float("inf")
    running_sum = 0.0
    accumulator = tl.zeros((head_dim,), tl.float32)

    if has_sink:
        running_max = tl.load(sink_ptr + head * sink_stride).to(tl.float32)
        running_sum = 1.0

    swa_length = tl.minimum(
        tl.load(swa_lengths_ptr + row * swa_lengths_stride), swa_width
    )
    for column in tl.range(0, swa_length):
        slot = tl.load(
            swa_indices_ptr
            + row * swa_indices_stride0
            + column * swa_indices_stride1
        )
        valid = slot >= 0
        if valid:
            key = _load_nvfp4_416_record(
                swa_cache_ptr,
                swa_cache_block_stride,
                slot,
                valid,
                dims,
                page_size=swa_page_size,
                nope_dim=nope_dim,
                fp4_data_bytes=fp4_data_bytes,
                scale_group_size=scale_group_size,
                scale_bytes=scale_bytes,
                record_bytes=record_bytes,
            )
            score = tl.sum(q * key, axis=0) * sm_scale
            new_max = tl.maximum(running_max, score)
            old_scale = tl.exp2((running_max - new_max) * 1.4426950408889634)
            new_scale = tl.exp2((score - new_max) * 1.4426950408889634)
            accumulator = accumulator * old_scale + key * new_scale
            running_sum = running_sum * old_scale + new_scale
            running_max = new_max

    if has_indexed:
        indexed_length = tl.minimum(
            tl.load(indexed_lengths_ptr + row * indexed_lengths_stride),
            indexed_width,
        )
        for column in tl.range(0, indexed_length):
            slot = tl.load(
                indexed_indices_ptr
                + row * indexed_indices_stride0
                + column * indexed_indices_stride1
            )
            valid = slot >= 0
            if valid:
                key = _load_nvfp4_416_record(
                    indexed_cache_ptr,
                    indexed_cache_block_stride,
                    slot,
                    valid,
                    dims,
                    page_size=indexed_page_size,
                    nope_dim=nope_dim,
                    fp4_data_bytes=fp4_data_bytes,
                    scale_group_size=scale_group_size,
                    scale_bytes=scale_bytes,
                    record_bytes=record_bytes,
                )
                score = tl.sum(q * key, axis=0) * sm_scale
                new_max = tl.maximum(running_max, score)
                old_scale = tl.exp2((running_max - new_max) * 1.4426950408889634)
                new_scale = tl.exp2((score - new_max) * 1.4426950408889634)
                accumulator = accumulator * old_scale + key * new_scale
                running_sum = running_sum * old_scale + new_scale
                running_max = new_max

    result = accumulator * tl.where(running_sum > 0.0, 1.0 / running_sum, 0.0)
    tl.store(
        output_ptr + row * output_stride0 + head * output_stride1 + dims,
        result.to(tl.bfloat16),
    )


@triton.jit
def _sparse_attention_nvfp4_416_direct_mma_kernel(
    q_ptr,
    q_stride0,
    q_stride1,
    swa_cache_ptr,
    swa_cache_block_stride,
    swa_indices_ptr,
    swa_indices_stride0,
    swa_indices_stride1,
    swa_lengths_ptr,
    swa_lengths_stride,
    swa_width,
    indexed_cache_ptr,
    indexed_cache_block_stride,
    indexed_indices_ptr,
    indexed_indices_stride0,
    indexed_indices_stride1,
    indexed_lengths_ptr,
    indexed_lengths_stride,
    indexed_width,
    sink_ptr,
    sink_stride,
    output_ptr,
    output_stride0,
    output_stride1,
    num_heads,
    sm_scale,
    swa_page_size: tl.constexpr,
    indexed_page_size: tl.constexpr,
    head_dim: tl.constexpr,
    nope_dim: tl.constexpr,
    fp4_data_bytes: tl.constexpr,
    scale_group_size: tl.constexpr,
    scale_bytes: tl.constexpr,
    record_bytes: tl.constexpr,
    has_indexed: tl.constexpr,
    has_sink: tl.constexpr,
    key_tile: tl.constexpr,
    head_tile: tl.constexpr,
):
    """Direct online-softmax attention for one 16-head tile.

    This avoids the global split-K output and LSE workspace. Compact records
    are reread once per head tile, while QK/PV remain BF16 tensor-core MMA.
    """
    row = tl.program_id(0)
    head_block = tl.program_id(1)
    heads = (head_block * head_tile + tl.arange(0, head_tile))[:, None]
    dims = tl.arange(0, head_dim)[None, :]
    q = tl.load(
        q_ptr + row * q_stride0 + heads * q_stride1 + dims,
        mask=heads < num_heads,
        other=0.0,
    ).to(tl.bfloat16)
    running_max = tl.full((head_tile,), -float("inf"), tl.float32)
    running_sum = tl.zeros((head_tile,), tl.float32)
    accumulator = tl.zeros((head_tile, head_dim), tl.float32)
    if has_sink:
        head_offsets = head_block * head_tile + tl.arange(0, head_tile)
        running_max = tl.load(
            sink_ptr + head_offsets * sink_stride,
            mask=head_offsets < num_heads,
            other=-float("inf"),
        ).to(tl.float32)
        running_sum = tl.where(head_offsets < num_heads, 1.0, 0.0)

    swa_length = tl.minimum(
        tl.load(swa_lengths_ptr + row * swa_lengths_stride), swa_width
    )
    # Both index tensors are capacity-padded (C128 can be 8192 columns wide),
    # while the per-row lengths contain the actual selected-key count. Bound
    # the dynamic loops by those lengths so short-context decode does not run
    # hundreds of fully masked MMA tiles.
    for key_start in tl.range(0, swa_length, key_tile):
        columns = key_start + tl.arange(0, key_tile)
        valid = columns < swa_length
        slots = tl.load(
            swa_indices_ptr
            + row * swa_indices_stride0
            + columns * swa_indices_stride1,
            mask=valid,
            other=-1,
        )
        valid = valid & (slots >= 0)
        key = _load_nvfp4_416_record(
            swa_cache_ptr,
            swa_cache_block_stride,
            slots[:, None],
            valid[:, None],
            dims,
            page_size=swa_page_size,
            nope_dim=nope_dim,
            fp4_data_bytes=fp4_data_bytes,
            scale_group_size=scale_group_size,
            scale_bytes=scale_bytes,
            record_bytes=record_bytes,
        ).to(tl.bfloat16)
        scores = tl.dot(q, key.T, out_dtype=tl.float32) * sm_scale
        scores = tl.where(valid[None, :], scores, -1.0e30)
        tile_max = tl.max(scores, axis=1)
        new_max = tl.maximum(running_max, tile_max)
        old_scale = tl.exp2((running_max - new_max) * 1.4426950408889634)
        weights = tl.exp2(
            (scores - new_max[:, None]) * 1.4426950408889634
        )
        weights = tl.where(valid[None, :], weights, 0.0)
        accumulator = accumulator * old_scale[:, None] + tl.dot(
            weights.to(tl.bfloat16), key, out_dtype=tl.float32
        )
        running_sum = running_sum * old_scale + tl.sum(weights, axis=1)
        running_max = new_max

    if has_indexed:
        indexed_length = tl.minimum(
            tl.load(indexed_lengths_ptr + row * indexed_lengths_stride),
            indexed_width,
        )
        for key_start in tl.range(0, indexed_length, key_tile):
            columns = key_start + tl.arange(0, key_tile)
            valid = columns < indexed_length
            slots = tl.load(
                indexed_indices_ptr
                + row * indexed_indices_stride0
                + columns * indexed_indices_stride1,
                mask=valid,
                other=-1,
            )
            valid = valid & (slots >= 0)
            key = _load_nvfp4_416_record(
                indexed_cache_ptr,
                indexed_cache_block_stride,
                slots[:, None],
                valid[:, None],
                dims,
                page_size=indexed_page_size,
                nope_dim=nope_dim,
                fp4_data_bytes=fp4_data_bytes,
                scale_group_size=scale_group_size,
                scale_bytes=scale_bytes,
                record_bytes=record_bytes,
            ).to(tl.bfloat16)
            scores = tl.dot(q, key.T, out_dtype=tl.float32) * sm_scale
            scores = tl.where(valid[None, :], scores, -1.0e30)
            tile_max = tl.max(scores, axis=1)
            new_max = tl.maximum(running_max, tile_max)
            old_scale = tl.exp2(
                (running_max - new_max) * 1.4426950408889634
            )
            weights = tl.exp2(
                (scores - new_max[:, None]) * 1.4426950408889634
            )
            weights = tl.where(valid[None, :], weights, 0.0)
            accumulator = accumulator * old_scale[:, None] + tl.dot(
                weights.to(tl.bfloat16), key, out_dtype=tl.float32
            )
            running_sum = running_sum * old_scale + tl.sum(weights, axis=1)
            running_max = new_max

    result = accumulator / tl.where(running_sum > 0.0, running_sum, 1.0)[:, None]
    tl.store(
        output_ptr
        + row * output_stride0
        + heads * output_stride1
        + dims,
        result.to(tl.bfloat16),
        mask=heads < num_heads,
    )


def sparse_attention_nvfp4_416(
    *,
    q: torch.Tensor,
    swa_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_lengths: torch.Tensor,
    swa_page_size: int,
    indexed_cache: torch.Tensor | None,
    indexed_indices: torch.Tensor | None,
    indexed_lengths: torch.Tensor | None,
    indexed_page_size: int | None,
    sm_scale: float,
    attn_sink: torch.Tensor | None,
    output: torch.Tensor,
    mid_out: torch.Tensor | None = None,
    mid_lse: torch.Tensor | None = None,
    split_tile: int = 32,
) -> None:
    """Fused sparse attention that consumes true 416-byte records directly.

    Decode uses caller-owned split-K scratch and a fused LSE merge. Large
    prefill chunks use a one-pass online softmax, avoiding any BF16 KV or FP32
    score materialization in both cases.
    """
    swa_indices = _index_matrix(swa_indices)
    if indexed_indices is not None:
        indexed_indices = _index_matrix(indexed_indices)
    if q.ndim != 3 or q.shape[-1] != HEAD_DIM:
        raise ValueError(f"q must be [tokens, heads, {HEAD_DIM}]")
    if output.shape != q.shape:
        raise ValueError(f"output must match q shape, got {output.shape} vs {q.shape}")
    if swa_cache.shape[-1] != RECORD_BYTES:
        raise ValueError("SWA cache is not a true 416-byte NVFP4 cache")
    has_indexed = indexed_cache is not None
    if has_indexed != (indexed_indices is not None):
        raise ValueError("indexed_cache and indexed_indices must be passed together")
    if has_indexed and (indexed_lengths is None or indexed_page_size is None):
        raise ValueError("indexed lengths/page size are required with indexed cache")
    if has_indexed and indexed_cache.shape[-1] != RECORD_BYTES:
        raise ValueError("indexed cache is not a true 416-byte NVFP4 cache")
    if (mid_out is None) != (mid_lse is None):
        raise ValueError("mid_out and mid_lse must be provided together")
    if split_tile not in (32, 64, 128, 256, 512):
        raise ValueError(f"unsupported split_tile={split_tile}")

    rows, heads, _ = q.shape
    # The split-MMA path wins on compact C4/SWA selections, but its global
    # partial workspace is larger than the gathered bridge once a 128-head
    # query uses the fixed 8192-wide C128A index buffer. The bridge is fully
    # CUDA-graph capturable now that cache writing is fused, so dispatch by
    # static shape and keep the faster implementation for each regime.
    attention_mode = os.getenv("DSV4_NVFP4_ATTENTION_MODE", "auto")
    if attention_mode not in ("auto", "direct", "reference"):
        raise ValueError(
            "DSV4_NVFP4_ATTENTION_MODE must be auto, direct, or reference; "
            f"got {attention_mode!r}"
        )
    use_reference = attention_mode == "reference" or (
        attention_mode == "auto"
        and heads > 32
        and (
            mid_out is None
            or (
                indexed_indices is not None
                and indexed_indices.shape[1] >= 4096
            )
        )
    )
    if use_reference:
        sparse_decode_nvfp4_416_reference(
            q=q,
            swa_cache=swa_cache,
            swa_indices=swa_indices,
            swa_lengths=swa_lengths,
            swa_page_size=swa_page_size,
            indexed_cache=indexed_cache,
            indexed_indices=indexed_indices,
            indexed_lengths=indexed_lengths,
            indexed_page_size=indexed_page_size,
            sm_scale=sm_scale,
            attn_sink=attn_sink,
            output=output,
        )
        return
    sink_arg = attn_sink if attn_sink is not None else output
    indexed_cache_arg = indexed_cache if indexed_cache is not None else swa_cache
    indexed_indices_arg = (
        indexed_indices if indexed_indices is not None else swa_indices
    )
    indexed_lengths_arg = (
        indexed_lengths if indexed_lengths is not None else swa_lengths
    )
    indexed_page_size_arg = (
        int(indexed_page_size) if indexed_page_size is not None else int(swa_page_size)
    )

    if mid_out is None:
        _sparse_attention_nvfp4_416_direct_mma_kernel[
            (rows, (heads + 15) // 16)
        ](
            q,
            q.stride(0),
            q.stride(1),
            swa_cache,
            swa_cache.stride(0),
            swa_indices,
            swa_indices.stride(0),
            swa_indices.stride(1),
            swa_lengths,
            swa_lengths.stride(0),
            swa_indices.shape[1],
            indexed_cache_arg,
            indexed_cache_arg.stride(0),
            indexed_indices_arg,
            indexed_indices_arg.stride(0),
            indexed_indices_arg.stride(1),
            indexed_lengths_arg,
            indexed_lengths_arg.stride(0),
            indexed_indices_arg.shape[1],
            sink_arg,
            sink_arg.stride(0),
            output,
            output.stride(0),
            output.stride(1),
            heads,
            sm_scale,
            swa_page_size=int(swa_page_size),
            indexed_page_size=indexed_page_size_arg,
            head_dim=HEAD_DIM,
            nope_dim=NOPE_DIM,
            fp4_data_bytes=FP4_DATA_BYTES,
            scale_group_size=SCALE_GROUP_SIZE,
            scale_bytes=SCALE_BYTES,
            record_bytes=RECORD_BYTES,
            has_indexed=has_indexed,
            has_sink=attn_sink is not None,
            key_tile=32,
            head_tile=16,
            num_warps=8,
            num_stages=1,
        )
        return

    swa_splits = (swa_indices.shape[1] + split_tile - 1) // split_tile
    indexed_splits = (
        (indexed_indices.shape[1] + split_tile - 1) // split_tile
        if indexed_indices is not None
        else 0
    )
    total_splits = swa_splits + indexed_splits
    if mid_out.shape[:3] != (rows, heads, total_splits):
        raise ValueError(
            f"mid_out must start with {(rows, heads, total_splits)}, got {mid_out.shape}"
        )
    if mid_lse.shape != (rows, heads, total_splits):
        raise ValueError(
            f"mid_lse must be {(rows, heads, total_splits)}, got {mid_lse.shape}"
        )

    def launch_split(
        cache: torch.Tensor,
        indices: torch.Tensor,
        lengths: torch.Tensor,
        page_size: int,
        split_offset: int,
        num_splits: int,
    ) -> None:
        split_kernel = (
            _sparse_attention_nvfp4_416_split_mma_kernel
            if split_tile == 32
            else _sparse_attention_nvfp4_416_grouped_mma_kernel
        )
        split_kernel[(rows, num_splits)](
            q,
            q.stride(0),
            q.stride(1),
            cache,
            cache.stride(0),
            indices,
            indices.stride(0),
            indices.stride(1),
            lengths,
            lengths.stride(0),
            mid_out,
            mid_out.stride(0),
            mid_out.stride(1),
            mid_out.stride(2),
            mid_out.stride(3),
            mid_lse,
            mid_lse.stride(0),
            mid_lse.stride(1),
            mid_lse.stride(2),
            split_offset,
            indices.shape[1],
            heads,
            sm_scale,
            page_size=int(page_size),
            head_dim=HEAD_DIM,
            nope_dim=NOPE_DIM,
            fp4_data_bytes=FP4_DATA_BYTES,
            scale_group_size=SCALE_GROUP_SIZE,
            scale_bytes=SCALE_BYTES,
            record_bytes=RECORD_BYTES,
            split_tile=split_tile,
            key_tile=32,
            head_tile=16,
            num_warps=8,
            num_stages=1,
        )

    launch_split(swa_cache, swa_indices, swa_lengths, swa_page_size, 0, swa_splits)
    if indexed_cache is not None:
        assert indexed_indices is not None
        assert indexed_lengths is not None
        assert indexed_page_size is not None
        launch_split(
            indexed_cache,
            indexed_indices,
            indexed_lengths,
            indexed_page_size,
            swa_splits,
            indexed_splits,
        )

    _sparse_attention_nvfp4_416_merge_kernel[(rows, heads)](
        mid_out,
        mid_out.stride(0),
        mid_out.stride(1),
        mid_out.stride(2),
        mid_out.stride(3),
        mid_lse,
        mid_lse.stride(0),
        mid_lse.stride(1),
        mid_lse.stride(2),
        sink_arg,
        sink_arg.stride(0),
        output,
        output.stride(0),
        output.stride(1),
        total_splits,
        head_dim=HEAD_DIM,
        has_sink=attn_sink is not None,
        num_warps=8,
    )


def sparse_decode_nvfp4_416_reference(
    *,
    q: torch.Tensor,
    swa_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_lengths: torch.Tensor,
    swa_page_size: int,
    indexed_cache: torch.Tensor | None,
    indexed_indices: torch.Tensor | None,
    indexed_lengths: torch.Tensor | None,
    indexed_page_size: int | None,
    sm_scale: float,
    attn_sink: torch.Tensor | None,
    output: torch.Tensor,
) -> None:
    """Correctness bridge for sparse attention until a fused FP4 reader exists.

    The SM120 FlashInfer sparse MLA kernels currently interpret these pages as
    the 584-byte FP8 layout. Keep query chunks small here so prefill does not
    materialize an unbounded [tokens, selected_kv, 512] tensor.
    """
    swa_indices = _index_matrix(swa_indices)
    if indexed_indices is not None:
        indexed_indices = _index_matrix(indexed_indices)
    if (indexed_cache is None) != (indexed_indices is None):
        raise ValueError("indexed_cache and indexed_indices must be passed together")
    if indexed_cache is not None and (
        indexed_lengths is None or indexed_page_size is None
    ):
        raise ValueError(
            "indexed_lengths and indexed_page_size are required with indexed_cache"
        )

    rows = q.shape[0]
    swa_width = swa_indices.shape[1]
    indexed_width = 0 if indexed_indices is None else indexed_indices.shape[1]
    total_width = swa_width + indexed_width
    if total_width == 0:
        output.zero_()
        return

    # Bound gathered KV plus FP32 score storage to roughly 256 MiB per call.
    bytes_per_row = total_width * (HEAD_DIM * 2 + q.shape[1] * 4)
    rows_per_chunk = max(1, min(rows, (256 * 1024 * 1024) // bytes_per_row))
    swa_columns = torch.arange(swa_width, device=q.device)
    indexed_columns = torch.arange(indexed_width, device=q.device)

    for start in range(0, rows, rows_per_chunk):
        end = min(start + rows_per_chunk, rows)
        chunk_rows = end - start
        gathered = torch.empty(
            (chunk_rows, total_width, HEAD_DIM),
            dtype=torch.bfloat16,
            device=q.device,
        )
        _gather_selected_kernel[(chunk_rows, swa_width)](
            gathered,
            gathered.stride(0),
            gathered.stride(1),
            swa_cache,
            swa_cache.stride(0),
            swa_indices[start:end],
            swa_indices.stride(0),
            swa_lengths[start:end],
            0,
            page_size=swa_page_size,
            is_nvfp4=swa_cache.shape[-1] == RECORD_BYTES,
            head_dim=HEAD_DIM,
            nope_dim=NOPE_DIM,
            fp4_data_bytes=FP4_DATA_BYTES,
            scale_group_size=SCALE_GROUP_SIZE,
            scale_bytes=SCALE_BYTES,
            record_bytes=RECORD_BYTES,
        )
        if indexed_cache is not None:
            assert indexed_indices is not None
            assert indexed_lengths is not None
            assert indexed_page_size is not None
            _gather_selected_kernel[(chunk_rows, indexed_width)](
                gathered,
                gathered.stride(0),
                gathered.stride(1),
                indexed_cache,
                indexed_cache.stride(0),
                indexed_indices[start:end],
                indexed_indices.stride(0),
                indexed_lengths[start:end],
                swa_width,
                page_size=indexed_page_size,
                is_nvfp4=True,
                head_dim=HEAD_DIM,
                nope_dim=NOPE_DIM,
                fp4_data_bytes=FP4_DATA_BYTES,
                scale_group_size=SCALE_GROUP_SIZE,
                scale_bytes=SCALE_BYTES,
                record_bytes=RECORD_BYTES,
            )

        scores = (
            torch.einsum("thd,tkd->thk", q[start:end], gathered).float()
            * sm_scale
        )
        valid_parts = [
            swa_columns[None, :] < swa_lengths[start:end, None],
        ]
        if indexed_lengths is not None:
            valid_parts.append(
                indexed_columns[None, :] < indexed_lengths[start:end, None]
            )
        valid = torch.cat(valid_parts, dim=1)
        scores.masked_fill_(~valid[:, None, :], float("-inf"))

        max_score = scores.amax(dim=-1)
        if attn_sink is not None:
            max_score = torch.maximum(max_score, attn_sink[None, :])
        weights = torch.exp(scores - max_score[:, :, None])
        denominator = weights.sum(dim=-1)
        if attn_sink is not None:
            denominator += torch.exp(attn_sink[None, :] - max_score)
        weights /= denominator[:, :, None]
        result = torch.einsum("thk,tkd->thd", weights.to(q.dtype), gathered)
        output[start:end].copy_(result)


def pack_reference(values: torch.Tensor, *, page_size: int) -> torch.Tensor:
    """Portable reference packer used by layout tests (global scale is 1)."""
    if values.ndim != 2 or values.shape[1] != HEAD_DIM:
        raise ValueError(f"values must be [N, {HEAD_DIM}]")
    rows = values.shape[0]
    pages = (rows + page_size - 1) // page_size
    cache = torch.zeros(
        (pages, page_size, RECORD_BYTES), dtype=torch.uint8, device=values.device
    )
    grouped = values.float().reshape(rows, SCALE_BYTES, SCALE_GROUP_SIZE)
    scales = (grouped.abs().amax(dim=-1) / 6.0).to(torch.float8_e4m3fn)
    scaled = grouped / scales.float().unsqueeze(-1)
    lut = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        device=values.device,
    )
    distances = (scaled.abs().unsqueeze(-1) - lut).abs()
    codes = distances.argmin(dim=-1).to(torch.uint8)
    codes |= (scaled < 0).to(torch.uint8) << 3
    codes = codes.reshape(rows, HEAD_DIM)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    flat = cache.view(-1, RECORD_BYTES)[:rows]
    flat[:, :FP4_DATA_BYTES] = packed
    flat[:, FP4_DATA_BYTES : FP4_DATA_BYTES + SCALE_BYTES] = scales.view(
        torch.uint8
    )
    flat[:, FP4_DATA_BYTES + SCALE_BYTES :] = (
        values[:, NOPE_DIM:].to(torch.bfloat16).contiguous().view(torch.uint8).view(rows, -1)
    )
    return cache


def unpack_reference(cache: torch.Tensor, *, rows: int) -> torch.Tensor:
    """Portable reference unpacker; authoritative RoPE comes from BF16."""
    flat = cache.view(-1, RECORD_BYTES)[:rows]
    packed = flat[:, :FP4_DATA_BYTES]
    codes = torch.empty((rows, HEAD_DIM), dtype=torch.uint8, device=cache.device)
    codes[:, 0::2] = packed & 0xF
    codes[:, 1::2] = packed >> 4
    lut = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        device=cache.device,
    )
    magnitude = lut[(codes & 0x7).long()]
    fp4 = torch.where((codes & 0x8) != 0, -magnitude, magnitude)
    scales = flat[:, FP4_DATA_BYTES : FP4_DATA_BYTES + SCALE_BYTES].contiguous()
    scales = scales.view(torch.float8_e4m3fn).float().reshape(rows, SCALE_BYTES, 1)
    result = (fp4.reshape(rows, SCALE_BYTES, SCALE_GROUP_SIZE) * scales).reshape(
        rows, HEAD_DIM
    )
    rope = flat[:, FP4_DATA_BYTES + SCALE_BYTES :].contiguous()
    result[:, NOPE_DIM:] = rope.view(torch.bfloat16).float().reshape(rows, ROPE_DIM)
    return result


__all__ = [
    "RECORD_BYTES",
    "compress_norm_rope_store_nvfp4_416",
    "dequantize_and_gather_nvfp4_416",
    "insert_nvfp4_416",
    "norm_rope_store_nvfp4_416",
    "pack_reference",
    "qnorm_rope_store_swa_nvfp4_416",
    "sparse_attention_nvfp4_416",
    "sparse_decode_nvfp4_416_reference",
    "unpack_reference",
]
