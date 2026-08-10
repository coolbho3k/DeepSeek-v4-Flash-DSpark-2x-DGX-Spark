#!/usr/bin/env python3
"""Benchmark B12X W4A16 MoE at DeepSeek-V4 dimensions.

The serving model has 256 experts, but materializing one full synthetic layer
would consume roughly 3.4 GiB before compiler/runtime overhead.  By default the
timed kernel uses 16 experts while preserving the exact hidden, intermediate,
top-k, activation, and token dimensions.  The script separately reports the
tile plans selected for the true 256-expert routing contract.
"""

from __future__ import annotations

import argparse
import os
import statistics

import torch

try:
    from vllm.model_executor.layers.fused_moe.experts.b12x_mxfp4_moe import (
        _maybe_apply_b12x_w4a16_selector_override,
    )
except ImportError:
    from vllm.model_executor.layers.fused_moe.b12x_moe import (
        _maybe_apply_b12x_w4a16_selector_override,
    )

from b12x.moe.fused.w4a16.host import (
    make_w4a16_packed_buffers,
    select_route_block_size_m,
)
from b12x.moe.fused.w4a16.kernel import (
    _select_tile_config,
    run_w4a16_moe,
)
from b12x.moe.fused.w4a16.prepare import W4A16PackedWeights

_maybe_apply_b12x_w4a16_selector_override()


def measure_ms(fn, *, warmup: int, repetitions: int) -> float:
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


def make_prepared_weights(
    *,
    experts: int,
    hidden_size: int,
    intermediate_size: int,
    device: torch.device,
) -> W4A16PackedWeights:
    if hidden_size % 64 or intermediate_size % 64:
        raise ValueError("hidden and intermediate sizes must be multiples of 64")

    w13_rows = 2 * intermediate_size
    # These are the final B12X packed layouts.  Constructing them directly
    # avoids holding source and repacked copies simultaneously.
    w13 = torch.zeros(
        (
            experts,
            hidden_size // 16,
            (w13_rows // 64) * 128,
        ),
        dtype=torch.int32,
        device=device,
    )
    w2 = torch.zeros(
        (
            experts,
            intermediate_size // 16,
            (hidden_size // 64) * 128,
        ),
        dtype=torch.int32,
        device=device,
    )
    # E8M0 byte 127 represents a unit power-of-two scale.
    w13_scale = torch.full(
        (experts, hidden_size // 32, w13_rows),
        127,
        dtype=torch.uint8,
        device=device,
    )
    w2_scale = torch.full(
        (experts, intermediate_size // 32, hidden_size),
        127,
        dtype=torch.uint8,
        device=device,
    )
    props = torch.cuda.get_device_properties(device)
    workspace = torch.zeros(
        (int(props.multi_processor_count) * 4 + 2,),
        dtype=torch.int32,
        device=device,
    )
    unit_scale = torch.ones(experts, dtype=torch.float32, device=device)
    return W4A16PackedWeights(
        w13=w13,
        w13_scale=w13_scale,
        w13_global_scale=unit_scale,
        w2=w2,
        w2_scale=w2_scale,
        w2_global_scale=unit_scale,
        workspace=workspace,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=experts,
        is_gated=True,
        params_dtype=torch.bfloat16,
        source_format="fp4_e8m0_k32",
        w13_layout="w31",
        scale_format="e8m0_k32",
    )


def print_true_model_tile_plans(
    *,
    rows: list[int],
    hidden_size: int,
    intermediate_size: int,
    topk: int,
    model_experts: int,
    device: torch.device,
) -> None:
    props = torch.cuda.get_device_properties(device)
    sms = int(props.multi_processor_count)
    max_shared_mem = int(props.shared_memory_per_block_optin)
    print(
        "true_model_tile_plan "
        "rows route_block fc1(k,n,threads,bpsm) fc2(k,n,threads,bpsm)"
    )
    for row_count in rows:
        route_block = select_route_block_size_m(row_count, topk, model_experts)
        fc1 = _select_tile_config(
            problem_m=row_count,
            problem_n=2 * intermediate_size,
            problem_k=hidden_size,
            top_k=topk,
            moe_block_size=route_block,
            sms=sms,
            max_shared_mem=max_shared_mem,
            scale_format="e8m0_k32",
        )
        fc2 = _select_tile_config(
            problem_m=row_count,
            problem_n=hidden_size,
            problem_k=intermediate_size,
            top_k=topk,
            moe_block_size=route_block,
            sms=sms,
            max_shared_mem=max_shared_mem,
            scale_format="e8m0_k32",
        )
        print(f"{row_count:20d} {route_block:11d} {fc1!s:24s} {fc2!s}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 6, 36])
    parser.add_argument("--experts", type=int, default=16)
    parser.add_argument("--model-experts", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--intermediate-size", type=int, default=2048)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=20)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if min(args.rows) < 1 or args.experts < args.topk:
        parser.error("rows must be positive and experts must be at least topk")

    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    print_true_model_tile_plans(
        rows=args.rows,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        topk=args.topk,
        model_experts=args.model_experts,
        device=device,
    )
    prepared = make_prepared_weights(
        experts=args.experts,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        device=device,
    )
    generator = torch.Generator(device=device).manual_seed(12)

    print(
        "\ntc_decode rows direct_ms graph_ms graph_static finite "
        "synthetic_experts"
    )
    for tc_decode in (False, True):
        os.environ["B12X_W4A16_TC_DECODE"] = "1" if tc_decode else "0"
        for row_count in args.rows:
            hidden = torch.randn(
                (row_count, args.hidden_size),
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            )
            topk_ids = (
                torch.arange(
                    row_count * args.topk,
                    dtype=torch.int32,
                    device=device,
                ).view(row_count, args.topk)
                % args.experts
            )
            topk_weights = torch.full(
                (row_count, args.topk),
                1.0 / args.topk,
                dtype=torch.float32,
                device=device,
            )
            buffers = make_w4a16_packed_buffers(
                prepared,
                m=row_count,
                topk=args.topk,
                dtype=torch.bfloat16,
                device=device,
            )

            def run() -> None:
                run_w4a16_moe(
                    hidden,
                    prepared,
                    topk_weights,
                    topk_ids,
                    activation="silu",
                    intermediate_cache13=buffers.intermediate_cache13,
                    intermediate_cache2=buffers.intermediate_cache2,
                    output=buffers.output,
                    fc1_c_tmp=buffers.fc1_c_tmp,
                    fc2_c_tmp=buffers.fc2_c_tmp,
                    packed_route_indices=buffers.packed_route_indices,
                    block_expert_ids=buffers.block_expert_ids,
                    packed_route_count=buffers.packed_route_count,
                    expert_offsets=buffers.expert_offsets,
                )

            run()
            torch.cuda.synchronize()
            direct_ms = measure_ms(
                run,
                warmup=args.warmup,
                repetitions=args.repetitions,
            )
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            graph_allocated = torch.cuda.memory_allocated(device)
            graph_ms = measure_ms(
                graph.replay,
                warmup=args.warmup,
                repetitions=args.repetitions,
            )
            graph.replay()
            torch.cuda.synchronize()
            graph_static = torch.cuda.memory_allocated(device) == graph_allocated
            finite = bool(torch.isfinite(buffers.output).all().item())
            print(
                f"{int(tc_decode):9d} {row_count:4d} {direct_ms:9.4f} "
                f"{graph_ms:8.4f} {str(graph_static):12s} {str(finite):6s} "
                f"{args.experts:17d}"
            )

    peak_mib = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    print(f"peak_allocated={peak_mib:.1f} MiB")


if __name__ == "__main__":
    main()
