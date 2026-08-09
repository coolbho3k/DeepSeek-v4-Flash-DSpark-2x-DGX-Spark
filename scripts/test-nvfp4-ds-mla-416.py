#!/usr/bin/env python3
"""Focused layout, boundary, and GPU round-trip checks for DSV4 NVFP4 KV."""

from __future__ import annotations

import argparse

import torch

from vllm.models.deepseek_v4.nvidia.nvfp4_cache import (
    FP4_DATA_BYTES,
    HEAD_DIM,
    NOPE_DIM,
    RECORD_BYTES,
    ROPE_DIM,
    SCALE_BYTES,
    compress_kv_nvfp4_416,
    dequantize_and_gather_nvfp4_416,
    insert_nvfp4_416,
    norm_rope_store_nvfp4_416,
    pack_reference,
    qnorm_rope_store_swa_nvfp4_416,
    sparse_attention_nvfp4_416,
    unpack_reference,
)


def make_values(rows: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(416)
    values = torch.randn(
        (rows, HEAD_DIM), generator=generator, dtype=torch.bfloat16, device=device
    )
    # Exercise exact zero blocks and both ends of E2M1's range.
    values[0, :16] = 0
    values[-1, 16:32] *= 8
    return values


def cpu_reference_test() -> None:
    values = make_values(67, torch.device("cpu"))
    cache = pack_reference(values, page_size=64)
    assert cache.shape == (2, 64, RECORD_BYTES)
    assert RECORD_BYTES == FP4_DATA_BYTES + SCALE_BYTES + ROPE_DIM * 2 == 416
    restored = unpack_reference(cache, rows=values.shape[0])
    torch.testing.assert_close(
        restored[:, NOPE_DIM:], values[:, NOPE_DIM:].float(), rtol=0, atol=0
    )
    assert torch.isfinite(restored).all()
    assert (restored[:, :NOPE_DIM] - values[:, :NOPE_DIM].float()).abs().mean() < 0.09
    print("cpu reference layout: ok")


def hybrid_allocator_geometry_test() -> None:
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWABackend
    from vllm.v1.core.kv_cache_utils import _get_kv_cache_groups_uniform_groups
    from vllm.v1.kv_cache_interface import (
        KVQuantMode,
        MLAAttentionSpec,
        SlidingWindowMLASpec,
        UniformTypeKVCacheSpecs,
    )

    def full(compress_ratio: int) -> MLAAttentionSpec:
        return MLAAttentionSpec(
            block_size=256,
            num_kv_heads=1,
            head_size=HEAD_DIM,
            dtype=torch.uint8,
            kv_quant_mode=KVQuantMode.NVFP4,
            cache_dtype_str="nvfp4_ds_mla",
            alignment=576,
            compress_ratio=compress_ratio,
            model_version="deepseek_v4",
        )

    def sliding(
        *, block_size: int, head_size: int, dtype: torch.dtype, window: int
    ) -> SlidingWindowMLASpec:
        return SlidingWindowMLASpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=head_size,
            dtype=dtype,
            sliding_window=window,
            alignment=576,
        )

    full_specs = UniformTypeKVCacheSpecs.from_specs(
        {
            "c4.0": full(4),
            "c4.1": full(4),
            "c128.0": full(128),
            "c128.1": full(128),
        }
    )
    swa_specs = UniformTypeKVCacheSpecs.from_specs(
        {
            f"swa.{i}": SlidingWindowMLASpec(
                block_size=64,
                num_kv_heads=1,
                head_size=HEAD_DIM,
                dtype=torch.uint8,
                kv_quant_mode=KVQuantMode.NVFP4,
                cache_dtype_str="nvfp4_ds_mla",
                sliding_window=128,
                alignment=576,
                model_version="deepseek_v4",
            )
            for i in range(2)
        }
    )
    state_c4_specs = UniformTypeKVCacheSpecs.from_specs(
        {f"state4.{i}": sliding(block_size=4, head_size=2048, dtype=torch.float32, window=8) for i in range(2)}
    )
    state_c128_specs = UniformTypeKVCacheSpecs.from_specs(
        {f"state128.{i}": sliding(block_size=8, head_size=1024, dtype=torch.float32, window=128) for i in range(2)}
    )
    assert all(
        spec is not None
        for spec in (full_specs, swa_specs, state_c4_specs, state_c128_specs)
    )
    grouped = _get_kv_cache_groups_uniform_groups(
        [full_specs, swa_specs, state_c4_specs, state_c128_specs]  # type: ignore[list-item]
    )
    assert grouped
    assert max(full_specs.get_page_sizes()) == 32832  # type: ignore[union-attr]
    assert DeepseekSparseSWABackend.get_kv_cache_shape(
        2, 64, 1, HEAD_DIM, "nvfp4_ds_mla"
    ) == (2, 64, RECORD_BYTES)
    print("hybrid allocator 416-byte geometry: ok")


def gpu_kernel_test() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for --gpu")

    device = torch.device("cuda")
    values = make_values(8, device)

    # Anemll vLLM 0.25 fuses C4 compression into its legacy cache writer and
    # no longer exports a compression-only helper. Verify our focused fallback
    # against the same per-dimension softmax-weighted state reduction.
    generator = torch.Generator(device=device).manual_seed(415)
    state_block_size = 4
    state_width = 2 * HEAD_DIM
    block_table = torch.tensor(
        [[2, 0, 4, 1, 3]], dtype=torch.int32, device=device
    )
    state_cache = torch.randn(
        (5, state_block_size, 2 * state_width),
        generator=generator,
        dtype=torch.float32,
        device=device,
    )
    compression_positions = torch.tensor(
        [3, 7, 11, 15], dtype=torch.int64, device=device
    )
    compression_slots = torch.arange(4, dtype=torch.int64, device=device)
    request_indices = torch.zeros(4, dtype=torch.int32, device=device)
    compressed = torch.empty((4, HEAD_DIM), dtype=torch.float32, device=device)
    compress_kv_nvfp4_416(
        state_cache,
        request_indices,
        compression_positions,
        compression_slots,
        block_table,
        state_block_size,
        compressed,
        state_width=state_width,
        compress_ratio=4,
        overlap=True,
    )
    compression_reference = []
    for position in compression_positions.cpu().tolist():
        start = position - 7
        row_values = []
        row_scores = []
        for window_row in range(8):
            logical = start + window_row
            if logical < 0:
                continue
            physical = int(block_table[0, logical // state_block_size])
            block_offset = logical % state_block_size
            segment = (window_row // 4) * HEAD_DIM
            state_row = state_cache[physical, block_offset]
            row_values.append(state_row[segment : segment + HEAD_DIM])
            row_scores.append(
                state_row[
                    state_width + segment : state_width + segment + HEAD_DIM
                ]
            )
        stacked_values = torch.stack(row_values)
        stacked_scores = torch.stack(row_scores)
        compression_reference.append(
            (stacked_values * stacked_scores.softmax(dim=0)).sum(dim=0)
        )
    torch.testing.assert_close(
        compressed,
        torch.stack(compression_reference),
        rtol=2e-5,
        atol=2e-5,
    )
    print("gpu C4 compression-only fallback: ok")

    # Cross page boundaries and leave sentinel rows untouched.
    slots = torch.tensor([0, 63, 64, 65, 127, 128, 191, 192], device=device)
    positions = torch.arange(values.shape[0], dtype=torch.int64, device=device)
    cache = torch.full((4, 64, RECORD_BYTES), 0xA5, dtype=torch.uint8, device=device)
    insert_nvfp4_416(
        values,
        positions,
        slots,
        cache,
        cache_block_size=64,
        compress_ratio=1,
    )
    torch.cuda.synchronize()

    flat = cache.view(-1, RECORD_BYTES)
    untouched = torch.ones(flat.shape[0], dtype=torch.bool, device=device)
    untouched[slots] = False
    assert torch.all(flat[untouched] == 0xA5), "writer crossed a 416-byte record"

    # Validate through the 416-byte format itself. The portable torch packer
    # and the serving Triton writer can make different choices at exact FP8/
    # E2M1 rounding ties while both remain valid NVFP4 encodings.
    selected = flat.index_select(0, slots)
    selected_dequant = unpack_reference(
        selected.view(1, values.shape[0], RECORD_BYTES), rows=values.shape[0]
    )
    assert (
        selected_dequant[:, :NOPE_DIM] - values[:, :NOPE_DIM].float()
    ).abs().mean() < 0.09
    expected_rope = (
        values[:, NOPE_DIM:]
        .contiguous()
        .view(torch.uint8)
        .view(-1, ROPE_DIM * 2)
    )
    torch.testing.assert_close(
        selected[:, -ROPE_DIM * 2 :], expected_rope, rtol=0, atol=0
    )

    block_table = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([193], dtype=torch.int32, device=device)
    gather_lens = torch.tensor([193], dtype=torch.int32, device=device)
    gathered = torch.empty((1, 193, HEAD_DIM), dtype=torch.bfloat16, device=device)
    dequantize_and_gather_nvfp4_416(
        gathered, cache, seq_lens, gather_lens, block_table, 64, 0
    )
    torch.cuda.synchronize()

    expected_dequant = selected_dequant
    actual = gathered[0].index_select(0, slots)
    torch.testing.assert_close(
        actual[:, :NOPE_DIM],
        expected_dequant[:, :NOPE_DIM].to(actual.dtype),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        actual[:, NOPE_DIM:], values[:, NOPE_DIM:], rtol=0, atol=0
    )
    print("gpu writer/gather/page-boundary round trip: ok")

    # Exercise the separate SWA prologue: weightless Q RMSNorm, GPT-J RoPE,
    # padded Q heads, and one 416-byte KV record per uncompressed token.
    generator = torch.Generator(device=device).manual_seed(417)
    q = torch.randn(
        (values.shape[0], 3, HEAD_DIM),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    kv = torch.randn(
        values.shape,
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    angles = torch.randn(
        (values.shape[0], ROPE_DIM // 2), generator=generator, device=device
    )
    cos_sin_cache = torch.cat((angles.cos(), angles.sin()), dim=-1)
    swa_cache = torch.full_like(cache, 0xA5)
    q_out = qnorm_rope_store_swa_nvfp4_416(
        q,
        kv,
        swa_cache,
        slots,
        positions,
        cos_sin_cache,
        padded_heads=32,
        rms_norm_eps=1e-6,
        cache_block_size=64,
    )
    torch.cuda.synchronize()

    cos = cos_sin_cache[:, : ROPE_DIM // 2]
    sin = cos_sin_cache[:, ROPE_DIM // 2 :]

    def rotate(x: torch.Tensor) -> torch.Tensor:
        pairs = x[..., NOPE_DIM:].reshape(
            x.shape[0], *x.shape[1:-1], ROPE_DIM // 2, 2
        )
        even, odd = pairs.unbind(dim=-1)
        local_cos, local_sin = cos, sin
        while local_cos.ndim < even.ndim:
            local_cos = local_cos.unsqueeze(1)
            local_sin = local_sin.unsqueeze(1)
        rope = torch.stack(
            (
                even * local_cos - odd * local_sin,
                odd * local_cos + even * local_sin,
            ),
            dim=-1,
        ).reshape(x.shape[0], *x.shape[1:-1], ROPE_DIM)
        return torch.cat((x[..., :NOPE_DIM], rope), dim=-1)

    q_float = q.float()
    q_reference = q_float * torch.rsqrt(
        q_float.square().mean(dim=-1, keepdim=True) + 1e-6
    )
    q_reference = torch.nn.functional.pad(
        rotate(q_reference).to(torch.bfloat16), (0, 0, 0, 29)
    )
    torch.testing.assert_close(q_out, q_reference, rtol=0, atol=0)

    kv_reference = rotate(kv.float()).to(torch.bfloat16)
    selected = swa_cache.view(-1, RECORD_BYTES).index_select(0, slots)
    swa_restored = unpack_reference(
        selected.view(1, values.shape[0], RECORD_BYTES), rows=values.shape[0]
    )
    assert (
        swa_restored[:, :NOPE_DIM] - kv_reference[:, :NOPE_DIM].float()
    ).abs().mean() < 0.09
    assert torch.all(swa_cache.view(-1, RECORD_BYTES)[untouched] == 0xA5)
    print("gpu SWA qnorm/rope/writer: ok")

    # Main-cache post-compression writer: weighted RMSNorm, compressed-position
    # RoPE, BF16 rounding, and NVFP4 record store are all one Triton launch.
    main_values = torch.randn(
        (values.shape[0], HEAD_DIM), generator=generator, device=device
    )
    norm_weight = torch.randn(
        (HEAD_DIM,), generator=generator, dtype=torch.bfloat16, device=device
    )
    main_positions = torch.arange(
        3, 3 + values.shape[0] * 4, 4, dtype=torch.int64, device=device
    )
    main_angles = torch.randn(
        (int(main_positions[-1]) + 1, ROPE_DIM // 2),
        generator=generator,
        device=device,
    )
    main_cos_sin_cache = torch.cat((main_angles.cos(), main_angles.sin()), dim=-1)
    main_cache = torch.full_like(cache, 0xA5)
    norm_rope_store_nvfp4_416(
        main_values,
        main_positions,
        slots,
        norm_weight,
        main_cos_sin_cache,
        main_cache,
        rms_norm_eps=1e-6,
        cache_block_size=64,
        compress_ratio=4,
    )
    torch.cuda.synchronize()

    main_float = main_values.float()
    main_normalized = main_float * torch.rsqrt(
        main_float.square().mean(dim=-1, keepdim=True) + 1e-6
    )
    main_normalized *= norm_weight.float()
    main_cos_sin = main_cos_sin_cache.index_select(0, (main_positions // 4) * 4)
    main_cos = main_cos_sin[:, : ROPE_DIM // 2]
    main_sin = main_cos_sin[:, ROPE_DIM // 2 :]
    main_pairs = main_normalized[:, NOPE_DIM:].reshape(-1, ROPE_DIM // 2, 2)
    main_even, main_odd = main_pairs.unbind(dim=-1)
    main_rope = torch.stack(
        (
            main_even * main_cos - main_odd * main_sin,
            main_odd * main_cos + main_even * main_sin,
        ),
        dim=-1,
    ).reshape(-1, ROPE_DIM)
    main_reference = torch.cat(
        (main_normalized[:, :NOPE_DIM], main_rope), dim=-1
    ).to(torch.bfloat16)
    main_selected = main_cache.view(-1, RECORD_BYTES).index_select(0, slots)
    main_writer_restored = unpack_reference(
        main_selected.view(1, values.shape[0], RECORD_BYTES),
        rows=values.shape[0],
    )
    assert (
        main_writer_restored[:, :NOPE_DIM] - main_reference[:, :NOPE_DIM].float()
    ).abs().mean() < 0.09
    torch.testing.assert_close(
        main_selected[:, -ROPE_DIM * 2 :],
        main_reference[:, NOPE_DIM:]
        .contiguous()
        .view(torch.uint8)
        .view(-1, ROPE_DIM * 2),
        rtol=0,
        atol=0,
    )
    assert torch.all(main_cache.view(-1, RECORD_BYTES)[untouched] == 0xA5)
    print("gpu main norm/rope/writer: ok")

    # Regression for the serving bridge: FlashInfer's SM120 sparse wrapper
    # assumes 584-byte FP8 pages, so both SWA-only and combined FP4 attention
    # must gather/dequantize before attention.
    test_rows = 3
    swa_indices = torch.tensor(
        [[0, 63, -1], [64, 65, -1], [127, 128, 191]],
        dtype=torch.int32,
        device=device,
    )
    swa_lengths = torch.tensor([2, 1, 3], dtype=torch.int32, device=device)
    indexed_indices = torch.tensor(
        [[64, 65], [127, -1], [0, 192]],
        dtype=torch.int32,
        device=device,
    )
    indexed_lengths = torch.tensor([2, 1, 2], dtype=torch.int32, device=device)
    q_attention = q_out[:test_rows]
    sink = torch.linspace(-0.5, 0.5, q_attention.shape[1], device=device)

    swa_restored[:, NOPE_DIM:] = kv_reference[:, NOPE_DIM:]
    main_restored = expected_dequant.clone()
    main_restored[:, NOPE_DIM:] = values[:, NOPE_DIM:]
    slot_to_row = {int(slot): row for row, slot in enumerate(slots.cpu().tolist())}

    def attention_reference(
        include_indexed: bool,
    ) -> torch.Tensor:
        result = torch.empty_like(q_attention)
        for row in range(test_rows):
            swa_rows = [
                slot_to_row[int(slot)]
                for slot in swa_indices[row, : swa_lengths[row]].cpu().tolist()
            ]
            keys = [swa_restored[swa_rows]]
            if include_indexed:
                indexed_rows = [
                    slot_to_row[int(slot)]
                    for slot in indexed_indices[
                        row, : indexed_lengths[row]
                    ].cpu().tolist()
                ]
                keys.append(main_restored[indexed_rows])
            selected_keys = torch.cat(keys).to(torch.bfloat16)
            scores = (
                torch.einsum("hd,kd->hk", q_attention[row], selected_keys).float()
                * (HEAD_DIM**-0.5)
            )
            max_score = torch.maximum(scores.amax(dim=-1), sink)
            weights = torch.exp(scores - max_score[:, None])
            denominator = weights.sum(dim=-1) + torch.exp(sink - max_score)
            weights /= denominator[:, None]
            result[row] = torch.einsum(
                "hk,kd->hd", weights.to(torch.bfloat16), selected_keys
            )
        return result

    combined_output = torch.empty_like(q_attention)
    combined_splits = (
        (swa_indices.shape[1] + 31) // 32
        + (indexed_indices.shape[1] + 31) // 32
    )
    combined_mid_out = torch.empty(
        (test_rows, q_attention.shape[1], combined_splits, HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    )
    combined_mid_lse = torch.empty(
        (test_rows, q_attention.shape[1], combined_splits),
        dtype=torch.float32,
        device=device,
    )
    sparse_attention_nvfp4_416(
        q=q_attention,
        swa_cache=swa_cache,
        swa_indices=swa_indices,
        swa_lengths=swa_lengths,
        swa_page_size=64,
        indexed_cache=cache,
        indexed_indices=indexed_indices,
        indexed_lengths=indexed_lengths,
        indexed_page_size=64,
        sm_scale=HEAD_DIM**-0.5,
        attn_sink=sink,
        output=combined_output,
        mid_out=combined_mid_out,
        mid_lse=combined_mid_lse,
    )
    torch.testing.assert_close(
        combined_output, attention_reference(True), rtol=0.02, atol=0.015625
    )

    swa_only_output = torch.empty_like(q_attention)
    swa_only_splits = (swa_indices.shape[1] + 31) // 32
    swa_mid_out = torch.empty(
        (test_rows, q_attention.shape[1], swa_only_splits, HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    )
    swa_mid_lse = torch.empty(
        (test_rows, q_attention.shape[1], swa_only_splits),
        dtype=torch.float32,
        device=device,
    )
    sparse_attention_nvfp4_416(
        q=q_attention,
        swa_cache=swa_cache,
        swa_indices=swa_indices,
        swa_lengths=swa_lengths,
        swa_page_size=64,
        indexed_cache=None,
        indexed_indices=None,
        indexed_lengths=None,
        indexed_page_size=None,
        sm_scale=HEAD_DIM**-0.5,
        attn_sink=sink,
        output=swa_only_output,
        mid_out=swa_mid_out,
        mid_lse=swa_mid_lse,
    )
    torch.testing.assert_close(
        swa_only_output, attention_reference(False), rtol=0.02, atol=0.015625
    )
    onepass_output = torch.empty_like(q_attention)
    sparse_attention_nvfp4_416(
        q=q_attention,
        swa_cache=swa_cache,
        swa_indices=swa_indices,
        swa_lengths=swa_lengths,
        swa_page_size=64,
        indexed_cache=cache,
        indexed_indices=indexed_indices,
        indexed_lengths=indexed_lengths,
        indexed_page_size=64,
        sm_scale=HEAD_DIM**-0.5,
        attn_sink=sink,
        output=onepass_output,
    )
    torch.testing.assert_close(
        onepass_output, attention_reference(True), rtol=0.02, atol=0.015625
    )

    graph_swa_cache = torch.full_like(cache, 0xA5)
    graph_main_cache = torch.full_like(cache, 0xA5)
    graph_output = torch.empty_like(q_attention)
    graph_mid_out = torch.empty_like(combined_mid_out)
    graph_mid_lse = torch.empty_like(combined_mid_lse)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_q = qnorm_rope_store_swa_nvfp4_416(
            q,
            kv,
            graph_swa_cache,
            slots,
            positions,
            cos_sin_cache,
            padded_heads=32,
            rms_norm_eps=1e-6,
            cache_block_size=64,
        )
        norm_rope_store_nvfp4_416(
            main_values,
            main_positions,
            slots,
            norm_weight,
            main_cos_sin_cache,
            graph_main_cache,
            rms_norm_eps=1e-6,
            cache_block_size=64,
            compress_ratio=4,
        )
        sparse_attention_nvfp4_416(
            q=graph_q[:test_rows],
            swa_cache=graph_swa_cache,
            swa_indices=swa_indices,
            swa_lengths=swa_lengths,
            swa_page_size=64,
            indexed_cache=cache,
            indexed_indices=indexed_indices,
            indexed_lengths=indexed_lengths,
            indexed_page_size=64,
            sm_scale=HEAD_DIM**-0.5,
            attn_sink=sink,
            output=graph_output,
            mid_out=graph_mid_out,
            mid_lse=graph_mid_lse,
        )
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(graph_q, q_reference, rtol=0, atol=0)
    torch.testing.assert_close(
        graph_output, attention_reference(True), rtol=0.02, atol=0.015625
    )
    torch.testing.assert_close(graph_swa_cache, swa_cache, rtol=0, atol=0)
    torch.testing.assert_close(graph_main_cache, main_cache, rtol=0, atol=0)
    print("gpu CUDA graph capture/replay (writers + attention): ok")
    print("gpu fused FP4 sparse attention (split + one-pass): ok")

    # Match serving's widest decode geometry (SWA=128, C128A=8192) and replay
    # long enough to catch graph/workspace lifetime and length-boundary bugs.
    stress_rows = 2
    # Exercise more than one 16-head CTA tile. The serving model uses many
    # padded query heads; limiting this test to one tile missed silent output
    # corruption in every head above 15.
    stress_heads = 128
    stress_swa_width = 128
    stress_indexed_width = 8192
    stress_values = torch.randn(
        (stress_indexed_width, HEAD_DIM),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    stress_cache = pack_reference(stress_values, page_size=64)
    stress_q = torch.randn(
        (stress_rows, stress_heads, HEAD_DIM),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    stress_swa_indices = torch.arange(
        stress_swa_width, dtype=torch.int32, device=device
    ).repeat(stress_rows, 1)
    stress_indexed_indices = torch.arange(
        stress_indexed_width, dtype=torch.int32, device=device
    ).repeat(stress_rows, 1)
    stress_swa_lengths = torch.tensor(
        [1, stress_swa_width], dtype=torch.int32, device=device
    )
    stress_indexed_lengths = torch.tensor(
        [1, 512], dtype=torch.int32, device=device
    )
    stress_splits = (stress_swa_width + 31) // 32 + (
        stress_indexed_width + 31
    ) // 32
    stress_output = torch.empty_like(stress_q)
    stress_mid_out = torch.empty(
        (stress_rows, stress_heads, stress_splits, HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    )
    stress_mid_lse = torch.empty(
        (stress_rows, stress_heads, stress_splits),
        dtype=torch.float32,
        device=device,
    )
    sparse_attention_nvfp4_416(
        q=stress_q,
        swa_cache=stress_cache,
        swa_indices=stress_swa_indices,
        swa_lengths=stress_swa_lengths,
        swa_page_size=64,
        indexed_cache=stress_cache,
        indexed_indices=stress_indexed_indices,
        indexed_lengths=stress_indexed_lengths,
        indexed_page_size=64,
        sm_scale=HEAD_DIM**-0.5,
        attn_sink=None,
        output=stress_output,
        mid_out=stress_mid_out,
        mid_lse=stress_mid_lse,
    )
    torch.cuda.synchronize()
    stress_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(stress_graph):
        sparse_attention_nvfp4_416(
            q=stress_q,
            swa_cache=stress_cache,
            swa_indices=stress_swa_indices,
            swa_lengths=stress_swa_lengths,
            swa_page_size=64,
            indexed_cache=stress_cache,
            indexed_indices=stress_indexed_indices,
            indexed_lengths=stress_indexed_lengths,
            indexed_page_size=64,
            sm_scale=HEAD_DIM**-0.5,
            attn_sink=None,
            output=stress_output,
            mid_out=stress_mid_out,
            mid_lse=stress_mid_lse,
        )
    for replay in range(256):
        stress_swa_lengths[0] = 1 + replay % stress_swa_width
        stress_indexed_lengths[0] = 1 + (replay * 31) % stress_indexed_width
        stress_graph.replay()
    torch.cuda.synchronize()
    assert torch.isfinite(stress_output).all()
    print("gpu exact-width CUDA graph replay stress (256 iterations): ok")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    cpu_reference_test()
    hybrid_allocator_geometry_test()
    if args.gpu:
        gpu_kernel_test()


if __name__ == "__main__":
    main()
