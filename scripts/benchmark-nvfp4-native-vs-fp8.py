#!/usr/bin/env python3
"""Small native SM120 sparse-attention comparison: 416B NVFP4 vs 584B FP8."""

from __future__ import annotations

import argparse
import os
import statistics

import torch

from flashinfer.mla._sparse_mla_sm120 import (
    _MODEL_TYPE_DSV4,
    _MODEL_TYPE_DSV4_NVFP4,
    get_sparse_mla_sm120_module,
    sparse_mla_sm120_decode_dsv4,
)
from vllm.models.deepseek_v4.nvidia.nvfp4_cache import (
    HEAD_DIM,
    RECORD_BYTES,
    insert_nvfp4_416,
    sparse_attention_nvfp4_416,
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


def random_fp8_cache(
    slots: int, device: torch.device, generator: torch.Generator
) -> torch.Tensor:
    pages = (slots + 63) // 64
    cache = torch.empty((pages, 64, 584), dtype=torch.uint8, device=device)
    flat = cache.view(pages, -1)
    data = flat[:, : 64 * 576].view(pages, 64, 576)
    data[..., :448].random_(0, 127, generator=generator)
    rope = torch.randn(
        (pages, 64, 64),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    data[..., 448:].copy_(rope.view(torch.uint8))
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


def random_nvfp4_cache(
    slots: int, device: torch.device, generator: torch.Generator
) -> torch.Tensor:
    pages = (slots + 63) // 64
    cache = torch.empty((pages, 64, RECORD_BYTES), dtype=torch.uint8, device=device)
    values = torch.randn(
        (slots, HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
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
    parser.add_argument(
        "--extra-topk",
        type=int,
        default=512,
        help="indexed/compressed entries per row (8192 models C128 at 1M)",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help=(
            "also benchmark allocation-static CUDA-graph replay while rotating "
            "the sparse index contents between iterations"
        ),
    )
    parser.add_argument("--random-cache", action="store_true")
    parser.add_argument(
        "--cache-slots",
        type=int,
        help=(
            "allocate this many records in each cache and scatter sparse "
            "indices across them; useful for exceeding the L2 working set"
        ),
    )
    parser.add_argument(
        "--index-banks",
        type=int,
        default=1,
        help=(
            "rotate through this many independently scattered index sets; "
            "with --cache-slots this prevents every timed call from reusing "
            "one L2-resident working set"
        ),
    )
    parser.add_argument("--force-prefill", action="store_true")
    parser.add_argument("--nv-first", action="store_true")
    parser.add_argument("--fp8-chunks-per-block", type=int)
    parser.add_argument("--nvfp4-chunks-per-block", type=int)
    args = parser.parse_args()

    for name in ("fp8_chunks_per_block", "nvfp4_chunks_per_block"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1")
    if args.force_prefill:
        os.environ["DSV4_NVFP4_FORCE_PREFILL"] = "1"

    device = torch.device("cuda")
    heads = 32
    swa_topk = 128
    extra_topk = args.extra_topk
    if extra_topk < 1:
        parser.error("--extra-topk must be positive")
    if args.cache_slots is not None and args.cache_slots < max(swa_topk, extra_topk):
        parser.error("--cache-slots must cover both SWA and indexed widths")
    if args.index_banks < 1:
        parser.error("--index-banks must be at least 1")
    if args.index_banks > 1 and args.cache_slots is None:
        parser.error("--index-banks greater than 1 requires --cache-slots")
    max_rows = max(args.rows)
    module = get_sparse_mla_sm120_module()

    cache_slots = args.cache_slots or extra_topk
    swa_cache_slots = cache_slots if args.cache_slots is not None else swa_topk
    extra_cache_slots = cache_slots
    if args.random_cache:
        generator = torch.Generator(device=device).manual_seed(417)
        nv_swa = random_nvfp4_cache(swa_cache_slots, device, generator)
        nv_extra = random_nvfp4_cache(extra_cache_slots, device, generator)
        fp8_swa = random_fp8_cache(swa_cache_slots, device, generator)
        fp8_extra = random_fp8_cache(extra_cache_slots, device, generator)
    else:
        nv_swa = nvfp4_cache(swa_cache_slots, device)
        nv_extra = nvfp4_cache(extra_cache_slots, device)
        fp8_swa = fp8_cache(swa_cache_slots, device)
        fp8_extra = fp8_cache(extra_cache_slots, device)
    torch.manual_seed(418)
    q_all = torch.randn(
        (max_rows, heads, HEAD_DIM), dtype=torch.bfloat16, device=device
    )
    if args.cache_slots is None:
        swa_indices_banks = torch.arange(
            swa_topk, dtype=torch.int32, device=device
        ).repeat(1, max_rows, 1)
        extra_indices_banks = torch.arange(
            extra_topk, dtype=torch.int32, device=device
        ).repeat(1, max_rows, 1)
    else:
        index_generator = torch.Generator(device=device).manual_seed(419)
        swa_indices_banks = torch.randint(
            swa_cache_slots,
            (args.index_banks, max_rows, swa_topk),
            dtype=torch.int32,
            device=device,
            generator=index_generator,
        )
        extra_indices_banks = torch.randint(
            extra_cache_slots,
            (args.index_banks, max_rows, extra_topk),
            dtype=torch.int32,
            device=device,
            generator=index_generator,
        )
    swa_lengths_all = torch.full(
        (max_rows,), swa_topk, dtype=torch.int32, device=device
    )
    extra_lengths_all = torch.full(
        (max_rows,), extra_topk, dtype=torch.int32, device=device
    )
    sink = torch.zeros((heads,), dtype=torch.float32, device=device)

    print(f"heads={heads} swa_topk={swa_topk} extra_topk={extra_topk}")
    header = "rows  fp8_584_ms  nvfp4_416_ms  ratio_416_to_584"
    if args.cuda_graph:
        header += "  fp8_graph_ms  nvfp4_graph_ms  graph_ratio  graph_static"
    print(header)
    for rows in args.rows:
        q = q_all[:rows]
        swa_indices_rows = swa_indices_banks[:, :rows]
        extra_indices_rows = extra_indices_banks[:, :rows]
        swa_indices = swa_indices_rows[0]
        extra_indices = extra_indices_rows[0]
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

        def run(
            model_type: int,
            swa_cache: torch.Tensor,
            extra_cache: torch.Tensor,
            index_bank: int = 0,
            current_swa_indices: torch.Tensor | None = None,
            current_extra_indices: torch.Tensor | None = None,
        ) -> None:
            if current_swa_indices is None:
                current_swa_indices = swa_indices_rows[index_bank]
            if current_extra_indices is None:
                current_extra_indices = extra_indices_rows[index_bank]
            chunks_per_block = (
                args.nvfp4_chunks_per_block
                if model_type == _MODEL_TYPE_DSV4_NVFP4
                else args.fp8_chunks_per_block
            )
            if chunks_per_block is not None:
                sparse_mla_sm120_decode_dsv4(
                    q,
                    swa_cache,
                    current_swa_indices,
                    mid_out,
                    mid_lse,
                    output,
                    lse,
                    HEAD_DIM**-0.5,
                    topk_length=swa_lengths,
                    attn_sink=sink,
                    extra_kv_cache=extra_cache,
                    extra_indices=current_extra_indices,
                    extra_topk_length=extra_lengths,
                    chunks_per_block=chunks_per_block,
                    model_type=model_type,
                )
                return
            module.paged_attention(
                q,
                swa_cache,
                current_swa_indices,
                output,
                lse,
                HEAD_DIM**-0.5,
                HEAD_DIM,
                model_type,
                swa_lengths,
                sink,
                extra_cache,
                current_extra_indices,
                extra_lengths,
                mid_out,
                mid_lse,
            )

        def measure_rotating(
            model_type: int,
            swa_cache: torch.Tensor,
            extra_cache: torch.Tensor,
        ) -> float:
            next_bank = 0

            def invoke() -> None:
                nonlocal next_bank
                run(model_type, swa_cache, extra_cache, next_bank)
                next_bank = (next_bank + 1) % args.index_banks

            return measure_ms(invoke, args.warmup, args.repetitions)

        measure_fp8 = lambda: measure_rotating(
            _MODEL_TYPE_DSV4, fp8_swa, fp8_extra
        )
        measure_nvfp4 = lambda: measure_rotating(
            _MODEL_TYPE_DSV4_NVFP4, nv_swa, nv_extra
        )
        if args.nv_first:
            nvfp4_ms = measure_nvfp4()
            fp8_ms = measure_fp8()
        else:
            fp8_ms = measure_fp8()
            nvfp4_ms = measure_nvfp4()

        graph_suffix = ""
        if args.cuda_graph:

            def capture_graph(
                model_type: int,
                swa_cache: torch.Tensor,
                extra_cache: torch.Tensor,
            ) -> tuple[torch.cuda.CUDAGraph, torch.Tensor, torch.Tensor]:
                graph_swa_indices = swa_indices.clone()
                graph_extra_indices = extra_indices.clone()

                def invoke() -> None:
                    run(
                        model_type,
                        swa_cache,
                        extra_cache,
                        current_swa_indices=graph_swa_indices,
                        current_extra_indices=graph_extra_indices,
                    )

                invoke()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    invoke()
                return graph, graph_swa_indices, graph_extra_indices

            def measure_graph_rotating(
                graph: torch.cuda.CUDAGraph,
                graph_swa_indices: torch.Tensor,
                graph_extra_indices: torch.Tensor,
            ) -> tuple[float, bool]:
                next_bank = 0

                def prepare() -> None:
                    nonlocal next_bank
                    graph_swa_indices.copy_(swa_indices_rows[next_bank])
                    graph_extra_indices.copy_(extra_indices_rows[next_bank])
                    next_bank = (next_bank + 1) % args.index_banks

                for _ in range(args.warmup):
                    prepare()
                    graph.replay()
                torch.cuda.synchronize()
                allocated = torch.cuda.memory_allocated(device)
                samples = []
                for _ in range(args.repetitions):
                    prepare()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    graph.replay()
                    end.record()
                    end.synchronize()
                    samples.append(start.elapsed_time(end))
                static = torch.cuda.memory_allocated(device) == allocated
                return statistics.median(samples), static

            fp8_graph = capture_graph(_MODEL_TYPE_DSV4, fp8_swa, fp8_extra)
            nvfp4_graph = capture_graph(
                _MODEL_TYPE_DSV4_NVFP4, nv_swa, nv_extra
            )
            measure_fp8_graph = lambda: measure_graph_rotating(*fp8_graph)
            measure_nvfp4_graph = lambda: measure_graph_rotating(*nvfp4_graph)
            if args.nv_first:
                nvfp4_graph_ms, nvfp4_static = measure_nvfp4_graph()
                fp8_graph_ms, fp8_static = measure_fp8_graph()
            else:
                fp8_graph_ms, fp8_static = measure_fp8_graph()
                nvfp4_graph_ms, nvfp4_static = measure_nvfp4_graph()
            graph_suffix = (
                f"  {fp8_graph_ms:12.4f}  {nvfp4_graph_ms:15.4f}  "
                f"{nvfp4_graph_ms / fp8_graph_ms:11.3f}x  "
                f"{str(fp8_static and nvfp4_static):12s}"
            )

        print(
            f"{rows:4d}  {fp8_ms:10.4f}  {nvfp4_ms:13.4f}  "
            f"{nvfp4_ms / fp8_ms:17.3f}x{graph_suffix}"
        )

        if args.check:
            if args.cache_slots is not None:
                parser.error("--check currently requires the default compact cache pool")
            generator = torch.Generator(device=device).manual_seed(416 + rows)
            check_swa = random_nvfp4_cache(swa_topk, device, generator)
            check_extra = random_nvfp4_cache(extra_topk, device, generator)
            run(_MODEL_TYPE_DSV4_NVFP4, check_swa, check_extra)
            torch.cuda.synchronize()
            native_output = output.clone()
            direct_output = torch.empty_like(output)
            os.environ["DSV4_NVFP4_ATTENTION_MODE"] = "direct"
            sparse_attention_nvfp4_416(
                q=q,
                swa_cache=check_swa,
                swa_indices=swa_indices,
                swa_lengths=swa_lengths,
                swa_page_size=64,
                indexed_cache=check_extra,
                indexed_indices=extra_indices,
                indexed_lengths=extra_lengths,
                indexed_page_size=64,
                sm_scale=HEAD_DIM**-0.5,
                attn_sink=sink,
                output=direct_output,
                mid_out=mid_out,
                mid_lse=mid_lse,
                split_tile=64,
            )
            torch.cuda.synchronize()
            error = (native_output.float() - direct_output.float()).abs()
            torch.testing.assert_close(
                native_output, direct_output, rtol=0.04, atol=0.04
            )
            print(
                f"check rows={rows}: max_error={error.max().item():.6f} "
                f"mean_error={error.mean().item():.6f}"
            )

        if args.profile:
            from torch.profiler import ProfilerActivity, profile

            for label, model_type, swa_cache, extra_cache in (
                ("fp8", _MODEL_TYPE_DSV4, fp8_swa, fp8_extra),
                ("nvfp4", _MODEL_TYPE_DSV4_NVFP4, nv_swa, nv_extra),
            ):
                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    run(model_type, swa_cache, extra_cache)
                    torch.cuda.synchronize()
                print(f"{label} rows={rows} CUDA kernels")
                print(
                    prof.key_averages().table(
                        sort_by="self_cuda_time_total", row_limit=8
                    )
                )

    peak_mib = torch.cuda.max_memory_allocated() / (1024 * 1024)
    print(f"peak allocated: {peak_mib:.1f} MiB")


if __name__ == "__main__":
    main()
