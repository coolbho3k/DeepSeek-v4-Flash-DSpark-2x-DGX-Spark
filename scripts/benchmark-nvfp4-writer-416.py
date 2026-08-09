#!/usr/bin/env python3
"""Benchmark DSpark's full-Q and KV-only 416-byte SWA cache writers."""

from __future__ import annotations

import argparse
import statistics
from types import SimpleNamespace

import torch

from vllm.models.deepseek_v4.nvidia.nvfp4_cache import (
    HEAD_DIM,
    RECORD_BYTES,
    ROPE_DIM,
    SCALE_BYTES,
    compress_kv_nvfp4_416,
    compress_norm_rope_store_nvfp4_416,
    norm_rope_store_nvfp4_416,
    qnorm_rope_store_swa_nvfp4_416,
    rope_store_swa_nvfp4_416,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 6, 36])
    parser.add_argument("--c4-rows", type=int, nargs="+", default=[])
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=100)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(416)
    print("rows  full_q_ms  kv_only_ms  speedup")
    for rows in args.rows:
        q = torch.randn(
            (rows, args.heads, HEAD_DIM),
            generator=generator,
            dtype=torch.bfloat16,
            device=device,
        )
        kv = torch.randn(
            (rows, HEAD_DIM),
            generator=generator,
            dtype=torch.bfloat16,
            device=device,
        )
        positions = torch.arange(rows, dtype=torch.int64, device=device)
        slots = positions.clone()
        angles = torch.randn(
            (rows, ROPE_DIM // 2), generator=generator, device=device
        )
        cos_sin_cache = torch.cat((angles.cos(), angles.sin()), dim=-1)
        blocks = (rows + 63) // 64
        full_cache = torch.full(
            (blocks, 64, RECORD_BYTES), 0xA5, dtype=torch.uint8, device=device
        )
        kv_only_cache = torch.full_like(full_cache, 0xA5)

        def full_q() -> None:
            qnorm_rope_store_swa_nvfp4_416(
                q,
                kv,
                full_cache,
                slots,
                positions,
                cos_sin_cache,
                padded_heads=args.heads,
                rms_norm_eps=1e-6,
                cache_block_size=64,
            )

        def kv_only() -> None:
            rope_store_swa_nvfp4_416(
                kv,
                kv_only_cache,
                slots,
                positions,
                cos_sin_cache,
                cache_block_size=64,
            )

        full_q()
        kv_only()
        torch.cuda.synchronize()
        torch.testing.assert_close(kv_only_cache, full_cache, rtol=0, atol=0)
        full_ms = measure_ms(
            full_q, warmup=args.warmup, repetitions=args.repetitions
        )
        kv_only_ms = measure_ms(
            kv_only, warmup=args.warmup, repetitions=args.repetitions
        )
        print(
            f"{rows:4d}  {full_ms:9.4f}  {kv_only_ms:10.4f}  "
            f"{full_ms / kv_only_ms:7.2f}x"
        )

    if not args.c4_rows:
        return

    print("\nC4 rows  two_stage_ms  fused_ms  speedup")
    for rows in args.c4_rows:
        state_block_size = 4
        state_width = 2 * HEAD_DIM
        state_blocks = (rows + state_block_size - 1) // state_block_size
        state_cache = torch.randn(
            (state_blocks, state_block_size, 2 * state_width),
            generator=generator,
            dtype=torch.float32,
            device=device,
        )
        positions = torch.arange(rows, dtype=torch.int64, device=device)
        state_slots = positions.clone()
        request_indices = torch.zeros(rows, dtype=torch.int32, device=device)
        block_table = torch.arange(
            state_blocks, dtype=torch.int32, device=device
        ).view(1, -1)
        valid = (positions + 1) % 4 == 0
        kv_slots = torch.where(valid, positions // 4, -1)
        kv_rows = max(1, rows // 4)
        cache_blocks = (kv_rows + 63) // 64
        two_stage_cache = torch.full(
            (cache_blocks, 64, RECORD_BYTES),
            0xA5,
            dtype=torch.uint8,
            device=device,
        )
        fused_cache = torch.full_like(two_stage_cache, 0xA5)
        compressed = torch.empty(
            (rows, HEAD_DIM), dtype=torch.float32, device=device
        )
        norm_weight = torch.randn(
            (HEAD_DIM,), generator=generator, dtype=torch.bfloat16, device=device
        )
        angles = torch.randn(
            (rows, ROPE_DIM // 2), generator=generator, device=device
        )
        cos_sin_cache = torch.cat((angles.cos(), angles.sin()), dim=-1)
        metadata = SimpleNamespace(slot_mapping=kv_slots)

        def two_stage() -> None:
            compress_kv_nvfp4_416(
                state_cache,
                request_indices,
                positions,
                state_slots,
                block_table,
                state_block_size,
                compressed,
                state_width=state_width,
                compress_ratio=4,
                overlap=True,
            )
            norm_rope_store_nvfp4_416(
                compressed,
                positions,
                kv_slots,
                norm_weight,
                cos_sin_cache,
                two_stage_cache,
                rms_norm_eps=1e-6,
                cache_block_size=64,
                compress_ratio=4,
            )

        def fused() -> None:
            compress_norm_rope_store_nvfp4_416(
                state_cache=state_cache,
                num_actual=rows,
                token_to_req_indices=request_indices,
                positions=positions,
                slot_mapping=state_slots,
                block_table=block_table,
                block_size=state_block_size,
                state_width=state_width,
                cos_sin_cache=cos_sin_cache,
                kv_cache=fused_cache,
                k_cache_metadata=metadata,
                pdl_kwargs={},
                head_dim=HEAD_DIM,
                rope_head_dim=ROPE_DIM,
                compress_ratio=4,
                overlap=True,
                use_fp4_cache=True,
                rms_norm_weight=norm_weight,
                rms_norm_eps=1e-6,
                quant_block=16,
                token_stride=RECORD_BYTES,
                scale_dim=SCALE_BYTES,
            )

        two_stage()
        fused()
        torch.cuda.synchronize()
        torch.testing.assert_close(fused_cache, two_stage_cache, rtol=0, atol=0)
        two_stage_ms = measure_ms(
            two_stage, warmup=args.warmup, repetitions=args.repetitions
        )
        fused_ms = measure_ms(
            fused, warmup=args.warmup, repetitions=args.repetitions
        )
        print(
            f"{rows:7d}  {two_stage_ms:12.4f}  {fused_ms:8.4f}  "
            f"{two_stage_ms / fused_ms:7.2f}x"
        )


if __name__ == "__main__":
    main()
