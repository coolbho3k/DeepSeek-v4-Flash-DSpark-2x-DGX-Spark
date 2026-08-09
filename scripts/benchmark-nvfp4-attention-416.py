#!/usr/bin/env python3
"""Microbenchmark the fused 416-byte sparse-attention decode path."""

from __future__ import annotations

import argparse
import statistics

import torch

from vllm.models.deepseek_v4.nvidia.nvfp4_cache import (
    HEAD_DIM,
    RECORD_BYTES,
    insert_nvfp4_416,
    sparse_attention_nvfp4_416,
    sparse_decode_nvfp4_416_reference,
)


def measure_ms(fn, *, warmup: int, repetitions: int) -> float:
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


def make_cache(rows: int, page_size: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(416 + rows)
    values = torch.randn(
        (rows, HEAD_DIM),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    cache = torch.empty(
        ((rows + page_size - 1) // page_size, page_size, RECORD_BYTES),
        dtype=torch.uint8,
        device=device,
    )
    slots = torch.arange(rows, dtype=torch.int64, device=device)
    insert_nvfp4_416(
        values,
        slots,
        slots,
        cache,
        cache_block_size=page_size,
        compress_ratio=1,
    )
    return cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--widths", type=int, nargs="+", default=[512, 2048, 8192])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--split-tile", type=int, default=32)
    parser.add_argument("--onepass", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    device = torch.device("cuda")
    page_size = 64
    swa_width = 128
    max_width = max(args.widths)
    swa_cache = make_cache(swa_width, page_size, device)
    indexed_cache = make_cache(max_width, page_size, device)
    swa_indices = torch.arange(swa_width, dtype=torch.int32, device=device).repeat(
        args.rows, 1
    )
    swa_lengths = torch.full(
        (args.rows,), swa_width, dtype=torch.int32, device=device
    )
    generator = torch.Generator(device=device).manual_seed(418)
    q = torch.randn(
        (args.rows, args.heads, HEAD_DIM),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    sink = torch.linspace(-0.5, 0.5, args.heads, device=device)
    output = torch.empty_like(q)
    reference_output = torch.empty_like(q)

    print(
        f"rows={args.rows} heads={args.heads} swa={swa_width} "
        f"page_size={page_size} record={RECORD_BYTES}B "
        f"mode={'onepass' if args.onepass else 'split'}"
    )
    print("indexed  fused_ms  bridge_ms  speedup  max_abs_error")
    for width in args.widths:
        indexed_indices = torch.arange(
            width, dtype=torch.int32, device=device
        ).repeat(args.rows, 1)
        indexed_lengths = torch.full(
            (args.rows,), width, dtype=torch.int32, device=device
        )
        splits = (
            (swa_width + args.split_tile - 1) // args.split_tile
            + (width + args.split_tile - 1) // args.split_tile
        )
        mid_out = torch.empty(
            (args.rows, args.heads, splits, HEAD_DIM),
            dtype=torch.bfloat16,
            device=device,
        )
        mid_lse = torch.empty(
            (args.rows, args.heads, splits), dtype=torch.float32, device=device
        )
        def fused() -> None:
            sparse_attention_nvfp4_416(
                q=q,
                swa_cache=swa_cache,
                swa_indices=swa_indices,
                swa_lengths=swa_lengths,
                swa_page_size=page_size,
                indexed_cache=indexed_cache,
                indexed_indices=indexed_indices,
                indexed_lengths=indexed_lengths,
                indexed_page_size=page_size,
                sm_scale=HEAD_DIM**-0.5,
                attn_sink=sink,
                output=output,
                mid_out=None if args.onepass else mid_out,
                mid_lse=None if args.onepass else mid_lse,
                split_tile=args.split_tile,
            )

        def bridge() -> None:
            sparse_decode_nvfp4_416_reference(
                q=q,
                swa_cache=swa_cache,
                swa_indices=swa_indices,
                swa_lengths=swa_lengths,
                swa_page_size=page_size,
                indexed_cache=indexed_cache,
                indexed_indices=indexed_indices,
                indexed_lengths=indexed_lengths,
                indexed_page_size=page_size,
                sm_scale=HEAD_DIM**-0.5,
                attn_sink=sink,
                output=reference_output,
            )

        fused()
        bridge()
        torch.cuda.synchronize()
        error = (output.float() - reference_output.float()).abs().max().item()
        fused_ms = measure_ms(
            fused, warmup=args.warmup, repetitions=args.repetitions
        )
        bridge_ms = measure_ms(
            bridge, warmup=args.warmup, repetitions=args.repetitions
        )
        print(
            f"{width:7d}  {fused_ms:8.3f}  {bridge_ms:9.3f}  "
            f"{bridge_ms / fused_ms:7.2f}x  {error:.6f}"
        )


if __name__ == "__main__":
    main()
