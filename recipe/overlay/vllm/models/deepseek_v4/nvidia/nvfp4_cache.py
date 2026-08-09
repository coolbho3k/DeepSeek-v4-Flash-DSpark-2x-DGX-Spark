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

    cos_sin = cos_sin_cache.index_select(0, positions.to(torch.int64))
    half = ROPE_DIM // 2
    cos = cos_sin[:, :half].float()
    sin = cos_sin[:, half:ROPE_DIM].float()

    q_float = q.float()
    q_float = q_float * torch.rsqrt(
        q_float.square().mean(dim=-1, keepdim=True) + rms_norm_eps
    )
    q_rotated = _apply_gptj_rope(q_float, cos, sin).to(q.dtype)
    if padded_heads > q.shape[1]:
        q_rotated = torch.nn.functional.pad(
            q_rotated, (0, 0, 0, padded_heads - q.shape[1]), value=0.0
        )

    kv_rotated = _apply_gptj_rope(kv.float(), cos, sin).to(torch.bfloat16)
    insert_nvfp4_416(
        kv_rotated,
        positions,
        slots,
        cache,
        cache_block_size=cache_block_size,
        compress_ratio=1,
    )
    return q_rotated


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

    from vllm.models.deepseek_v4.nvidia.ops.sparse_attn_compress_cutedsl import (
        compress_kv_sparse_attn_cutedsl,
    )

    compressed = torch.zeros(
        (num_actual, HEAD_DIM), dtype=torch.float32, device=state_cache.device
    )
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

    variance = compressed.square().mean(dim=-1, keepdim=True)
    normalized = compressed * torch.rsqrt(variance + rms_norm_eps)
    normalized = normalized * rms_norm_weight.float()

    # GPT-J/interleaved-pair RoPE on the final 64 dimensions. The compressor
    # uses the beginning of the compression interval, matching the FP8 writer.
    compressed_positions = (positions // compress_ratio) * compress_ratio
    cos_sin = cos_sin_cache.index_select(0, compressed_positions.to(torch.int64))
    half = ROPE_DIM // 2
    cos = cos_sin[:, :half]
    sin = cos_sin[:, half:ROPE_DIM]
    rope_pairs = normalized[:, NOPE_DIM:].reshape(num_actual, half, 2)
    even, odd = rope_pairs.unbind(dim=-1)
    rotated = torch.stack(
        (even * cos - odd * sin, odd * cos + even * sin), dim=-1
    ).reshape(num_actual, ROPE_DIM)
    normalized = torch.cat((normalized[:, :NOPE_DIM], rotated), dim=-1).to(
        torch.bfloat16
    )

    insert_nvfp4_416(
        normalized,
        positions,
        k_cache_metadata.slot_mapping,
        kv_cache,
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
    "pack_reference",
    "qnorm_rope_store_swa_nvfp4_416",
    "sparse_decode_nvfp4_416_reference",
    "unpack_reference",
]
