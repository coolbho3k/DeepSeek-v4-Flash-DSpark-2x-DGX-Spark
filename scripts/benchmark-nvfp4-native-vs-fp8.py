#!/usr/bin/env python3
"""Small native SM120 sparse-attention comparison: 416B NVFP4 vs 584B FP8."""

from __future__ import annotations

import argparse
import statistics

import torch

from flashinfer.mla._sparse_mla_sm120 import (
    _MODEL_TYPE_DSV4,
    _MODEL_TYPE_DSV4_NVFP4,
    get_sparse_mla_sm120_module,
)
from vllm.models.deepseek_v4.nvidia.nvfp4_cache import (
    HEAD_DIM,
    RECORD_BYTES,
    insert_nvfp4_416,
)


def measure_ms(fn, warmup: int, repetitions: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def fp8_cache(slots: int, device: torch.device) -> torch.Tensor:
    pages = (slots + 63) // 64
    cache = torch.zeros((pages, 64, 584), dtype=torch.uint8, device=device)
    flat = cache.view(pages, -1)
    # DSV4's 584-byte logical record is physically 64 contiguous 576-byte
    # data records followed by 64 8-byte UE8M0 scale records per page.
    flat[:, 64 * 576 :].fill_(127)
    return cache


def nvfp4_cache(slots: int, device: torch.device) -> torch.Tensor:
    pages = (slots + 63) // 64
    cache = torch.empty((pages, 64, RECORD_BYTES), dtype=torch.uint8, device=device)
    values = torch.zeros((slots, HEAD_DIM), dtype=torch.bfloat16, device=device)
    positions = torch.arange(slots, dtype=torch.int64, device=device)
    insert_nvfp4_416(
        values,
        positions,
        positions,
        cache,
        cache_block_size=64,
        compress_ratio=1,
    )
    return cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 8, 65])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda")
    heads = 32
    swa_topk = 128
    extra_topk = 512
    max_rows = max(args.rows)
    module = get_sparse_mla_sm120_module()

    nv_swa = nvfp4_cache(swa_topk, device)
    nv_extra = nvfp4_cache(extra_topk, device)
    fp8_swa = fp8_cache(swa_topk, device)
    fp8_extra = fp8_cache(extra_topk, device)
    q_all = torch.randn(
        (max_rows, heads, HEAD_DIM), dtype=torch.bfloat16, device=device
    )
    swa_indices_all = torch.arange(
        swa_topk, dtype=torch.int32, device=device
    ).repeat(max_rows, 1)
    extra_indices_all = torch.arange(
        extra_topk, dtype=torch.int32, device=device
    ).repeat(max_rows, 1)
    swa_lengths_all = torch.full(
        (max_rows,), swa_topk, dtype=torch.int32, device=device
    )
    extra_lengths_all = torch.full(
        (max_rows,), extra_topk, dtype=torch.int32, device=device
    )
    sink = torch.zeros((heads,), dtype=torch.float32, device=device)

    print("rows  fp8_584_ms  nvfp4_416_ms  ratio_416_to_584")
    for rows in args.rows:
        q = q_all[:rows]
        swa_indices = swa_indices_all[:rows]
        extra_indices = extra_indices_all[:rows]
        swa_lengths = swa_lengths_all[:rows]
        extra_lengths = extra_lengths_all[:rows]
        output = torch.empty_like(q)
        lse = torch.empty((rows, heads), dtype=torch.float32, device=device)
        splits = (swa_topk + extra_topk) // 64
        mid_out = torch.empty(
            (rows, heads, splits, HEAD_DIM), dtype=torch.bfloat16, device=device
        )
        mid_lse = torch.empty(
            (rows, heads, splits), dtype=torch.float32, device=device
        )

        def run(model_type: int, swa_cache: torch.Tensor, extra_cache: torch.Tensor) -> None:
            module.paged_attention(
                q,
                swa_cache,
                swa_indices,
                output,
                lse,
                HEAD_DIM**-0.5,
                HEAD_DIM,
                model_type,
                swa_lengths,
                sink,
                extra_cache,
                extra_indices,
                extra_lengths,
                mid_out,
                mid_lse,
            )

        fp8_ms = measure_ms(
            lambda: run(_MODEL_TYPE_DSV4, fp8_swa, fp8_extra),
            args.warmup,
            args.repetitions,
        )
        nvfp4_ms = measure_ms(
            lambda: run(_MODEL_TYPE_DSV4_NVFP4, nv_swa, nv_extra),
            args.warmup,
            args.repetitions,
        )
        print(f"{rows:4d}  {fp8_ms:10.4f}  {nvfp4_ms:13.4f}  {nvfp4_ms / fp8_ms:17.3f}x")

    peak_mib = torch.cuda.max_memory_allocated() / (1024 * 1024)
    print(f"peak allocated: {peak_mib:.1f} MiB")


if __name__ == "__main__":
    main()
