#!/usr/bin/env python3
"""Bounded DSV4 sparse-attention comparison: B12X vs FlashInfer SM120."""

from __future__ import annotations

import argparse
import statistics

import torch
import vllm._C_stable_libtorch  # noqa: F401  # Register the fused cache writer.

from b12x.attention.mla.compressed_api import compressed_mla_decode_forward
from b12x.attention.workspace import B12XAttentionWorkspace
from flashinfer.mla._sparse_mla_sm120 import (
    _MODEL_TYPE_DSV4,
    get_sparse_mla_sm120_module,
)
from vllm.models.deepseek_v4.nvidia.nvfp4_cache import (
    sparse_decode_nvfp4_416_reference,
)


HEAD_DIM = 512
PAGE_SIZE = 64
FP8_PAYLOAD_BYTES = 576
FP8_SCALE_BYTES = 8
FP8_PAGE_BYTES = PAGE_SIZE * (FP8_PAYLOAD_BYTES + FP8_SCALE_BYTES)


def measure_ms(fn, warmup: int, repetitions: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def fp8_cache(slots: int, device: torch.device, seed: int) -> torch.Tensor:
    pages = (slots + PAGE_SIZE - 1) // PAGE_SIZE
    cache = torch.empty(
        (pages, PAGE_SIZE, FP8_PAYLOAD_BYTES + FP8_SCALE_BYTES),
        dtype=torch.uint8,
        device=device,
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(
        (slots, 16, HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    kv = torch.randn(
        (slots, HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    positions = torch.arange(slots, dtype=torch.int64, device=device)
    cos_sin = torch.cat(
        (
            torch.ones((slots, 32), dtype=torch.float32, device=device),
            torch.zeros((slots, 32), dtype=torch.float32, device=device),
        ),
        dim=-1,
    )
    torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
        q,
        kv,
        cache.view(pages, FP8_PAGE_BYTES),
        positions,
        positions,
        cos_sin,
        16,
        1e-6,
        PAGE_SIZE,
    )
    return cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 8, 65])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=50)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    device = torch.device("cuda")
    heads = 32
    swa_topk = 128
    indexed_topk = 512
    total_topk = swa_topk + indexed_topk
    splits = (total_topk + 63) // 64
    max_rows = max(args.rows)

    swa_cache = fp8_cache(swa_topk, device, 120)
    indexed_cache = fp8_cache(indexed_topk, device, 121)
    generator = torch.Generator(device=device).manual_seed(122)
    q_all = torch.randn(
        (max_rows, heads, HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    swa_indices_all = torch.arange(
        swa_topk, dtype=torch.int32, device=device
    ).repeat(max_rows, 1)
    indexed_indices_all = torch.arange(
        indexed_topk, dtype=torch.int32, device=device
    ).repeat(max_rows, 1)
    swa_lengths_all = torch.full(
        (max_rows,), swa_topk, dtype=torch.int32, device=device
    )
    indexed_lengths_all = torch.full(
        (max_rows,), indexed_topk, dtype=torch.int32, device=device
    )
    sink = torch.linspace(-0.25, 0.25, heads, dtype=torch.float32, device=device)

    workspace = B12XAttentionWorkspace(
        mode="decode",
        device=device,
        dtype=torch.bfloat16,
        kv_dtype=torch.uint8,
        num_q_heads=heads,
        head_dim=HEAD_DIM,
        v_head_dim=HEAD_DIM,
        topk=total_topk,
        max_total_q=max_rows,
        max_batch=max_rows,
        max_page_table_width=total_topk,
        max_paged_q_rows=max_rows,
        page_size=PAGE_SIZE,
        padded_heads=heads,
        max_chunks_per_row=splits,
    )
    workspace.kv_chunk_size_ptr = torch.empty(
        (1,), dtype=torch.int32, device=device
    )
    workspace.num_chunks_ptr = torch.empty((1,), dtype=torch.int32, device=device)
    flashinfer = get_sparse_mla_sm120_module()

    print(
        "rows  flashinfer_ms  b12x_ms  b12x_to_flashinfer  "
        "fi_ref_error  b12x_ref_error"
    )
    for rows in args.rows:
        q = q_all[:rows]
        swa_indices = swa_indices_all[:rows]
        indexed_indices = indexed_indices_all[:rows]
        swa_lengths = swa_lengths_all[:rows]
        indexed_lengths = indexed_lengths_all[:rows]
        flashinfer_out = torch.empty_like(q)
        flashinfer_lse = torch.empty(
            (rows, heads), dtype=torch.float32, device=device
        )
        b12x_out = torch.empty_like(q)
        reference_out = torch.empty_like(q)
        mid_out = torch.empty(
            (rows, heads, splits, HEAD_DIM),
            dtype=torch.bfloat16,
            device=device,
        )
        mid_lse = torch.empty(
            (rows, heads, splits), dtype=torch.float32, device=device
        )
        workspace.tmp_output = mid_out
        workspace.tmp_lse = mid_lse
        workspace.output_buffer = b12x_out

        def run_flashinfer() -> None:
            flashinfer.paged_attention(
                q,
                swa_cache,
                swa_indices,
                flashinfer_out,
                flashinfer_lse,
                HEAD_DIM**-0.5,
                HEAD_DIM,
                _MODEL_TYPE_DSV4,
                swa_lengths,
                sink,
                indexed_cache,
                indexed_indices,
                indexed_lengths,
                mid_out,
                mid_lse,
            )

        def run_b12x() -> None:
            result = compressed_mla_decode_forward(
                q_all=q,
                swa_k_cache=swa_cache,
                swa_indices=swa_indices,
                swa_topk_lengths=swa_lengths,
                workspace=workspace,
                sm_scale=HEAD_DIM**-0.5,
                swa_page_size=PAGE_SIZE,
                indexed_k_cache=indexed_cache,
                indexed_indices=indexed_indices,
                indexed_topk_lengths=indexed_lengths,
                indexed_page_size=PAGE_SIZE,
                attn_sink=sink,
                expected_num_q_heads=heads,
                backend="sm120_unified",
            )
            if result.data_ptr() != b12x_out.data_ptr():
                b12x_out.copy_(result)

        run_flashinfer()
        run_b12x()
        sparse_decode_nvfp4_416_reference(
            q=q,
            swa_cache=swa_cache,
            swa_indices=swa_indices,
            swa_lengths=swa_lengths,
            swa_page_size=PAGE_SIZE,
            indexed_cache=indexed_cache,
            indexed_indices=indexed_indices,
            indexed_lengths=indexed_lengths,
            indexed_page_size=PAGE_SIZE,
            sm_scale=HEAD_DIM**-0.5,
            attn_sink=sink,
            output=reference_out,
        )
        torch.cuda.synchronize()
        flashinfer_error = (
            flashinfer_out.float() - reference_out.float()
        ).abs().max().item()
        b12x_error = (
            b12x_out.float() - reference_out.float()
        ).abs().max().item()
        flashinfer_ms = measure_ms(
            run_flashinfer, args.warmup, args.repetitions
        )
        b12x_ms = measure_ms(run_b12x, args.warmup, args.repetitions)
        print(
            f"{rows:4d}  {flashinfer_ms:13.4f}  {b12x_ms:7.4f}  "
            f"{b12x_ms / flashinfer_ms:18.3f}x  "
            f"{flashinfer_error:12.6f}  {b12x_error:14.6f}"
        )

    peak_mib = torch.cuda.max_memory_allocated() / (1024 * 1024)
    print(f"peak allocated: {peak_mib:.1f} MiB")


if __name__ == "__main__":
    main()
