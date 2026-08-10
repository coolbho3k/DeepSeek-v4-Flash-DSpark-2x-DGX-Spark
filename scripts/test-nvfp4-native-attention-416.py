#!/usr/bin/env python3
"""Bounded smoke test for FlashInfer's native 416-byte sparse reader.

The defaults allocate fewer than 2 MiB of test tensors.  JIT compilation is
expected on the first run, so invoke the container with ``MAX_JOBS=1`` and a
host-memory limit when validating a new image.
"""

from __future__ import annotations

import os

import torch

from vllm.models.deepseek_v4.nvidia.nvfp4_cache import (
    HEAD_DIM,
    RECORD_BYTES,
    insert_nvfp4_416,
    sparse_attention_nvfp4_416,
)


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    device = torch.device("cuda")
    rows = 1
    heads = int(os.getenv("NVFP4_TEST_HEADS", "32"))
    page_size = 64
    topk = int(os.getenv("NVFP4_TEST_TOPK", "128"))
    generator = torch.Generator(device=device).manual_seed(416)

    values = torch.randn(
        (topk, HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    cache = torch.empty(
        (topk // page_size, page_size, RECORD_BYTES),
        dtype=torch.uint8,
        device=device,
    )
    slots = torch.arange(topk, dtype=torch.int64, device=device)
    insert_nvfp4_416(
        values,
        slots,
        slots,
        cache,
        cache_block_size=page_size,
        compress_ratio=1,
    )
    cache_rope = cache.view(-1, RECORD_BYTES)[:, 288:].view(torch.bfloat16)
    torch.testing.assert_close(cache_rope, values[:, 448:], rtol=0, atol=0)

    q = torch.randn(
        (rows, heads, HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    indices = torch.arange(topk, dtype=torch.int32, device=device).view(1, -1)
    valid_length = int(os.getenv("NVFP4_TEST_LENGTH", str(topk - 3)))
    lengths = torch.tensor([valid_length], dtype=torch.int32, device=device)
    sink = torch.linspace(-0.25, 0.25, heads, dtype=torch.float32, device=device)
    native_output = torch.empty_like(q)
    direct_output = torch.empty_like(q)
    # Small caller-owned buffers avoid touching the backend's 128 MiB serving
    # workspace in this compile/correctness smoke test.
    splits = topk // 32
    mid_out = torch.empty(
        (rows, heads, splits, HEAD_DIM), dtype=torch.bfloat16, device=device
    )
    mid_lse = torch.empty(
        (rows, heads, splits), dtype=torch.float32, device=device
    )
    if os.getenv("NVFP4_TEST_ZERO_SCRATCH") == "1":
        mid_out.zero_()
        mid_lse.fill_(-1.0e30)

    kwargs = dict(
        q=q,
        swa_cache=cache,
        swa_indices=indices,
        swa_lengths=lengths,
        swa_page_size=page_size,
        indexed_cache=None,
        indexed_indices=None,
        indexed_lengths=None,
        indexed_page_size=None,
        sm_scale=HEAD_DIM**-0.5,
        attn_sink=sink,
        mid_out=mid_out,
        mid_lse=mid_lse,
    )

    os.environ["DSV4_NVFP4_ATTENTION_MODE"] = "direct"
    sparse_attention_nvfp4_416(output=direct_output, **kwargs)
    if os.getenv("NVFP4_TEST_ZERO_SCRATCH") == "1":
        mid_out.zero_()
        mid_lse.fill_(-1.0e30)
    os.environ["DSV4_NVFP4_ATTENTION_MODE"] = "native"
    sparse_attention_nvfp4_416(output=native_output, **kwargs)
    torch.cuda.synchronize()

    error = (native_output.float() - direct_output.float()).abs()
    max_error = error.max().item()
    mean_error = error.mean().item()
    if os.getenv("NVFP4_TEST_DIAGNOSTICS") == "1":
        native_num_splits = 2
        partial = mid_out.flatten()[: rows * heads * native_num_splits * HEAD_DIM]
        partial = partial.view(rows, heads, native_num_splits, HEAD_DIM)
        partial_finite = torch.isfinite(partial)
        print(
            "native sections: "
            f"nope_absmax={native_output[..., :448].float().abs().max().item():.6g}, "
            f"rope_absmax={native_output[..., 448:].float().abs().max().item():.6g}, "
            f"nope_max_error={error[..., :448].max().item():.6g}, "
            f"rope_max_error={error[..., 448:].max().item():.6g}, "
            f"direct_absmax={direct_output.float().abs().max().item():.6g}, "
            f"lse={mid_lse.view(-1)[: rows * heads].tolist()}, "
            f"partial_nonfinite={(~partial_finite).sum().item()}, "
            f"partial_head0_split0_dims="
            f"{torch.nonzero(~partial_finite[0, 0, 0]).flatten().tolist()}, "
            f"partial_head0_split1_dims="
            f"{torch.nonzero(~partial_finite[0, 0, 1]).flatten().tolist()}, "
            f"partial_head0_split1_range="
            f"({partial[0, 0, 1].float().min().item()}, "
            f"{partial[0, 0, 1].float().max().item()})"
        )
    if not torch.isfinite(native_output).all():
        finite = torch.isfinite(native_output)
        finite_values = native_output.float()[finite]
        finite_range = (
            (finite_values.min().item(), finite_values.max().item())
            if finite_values.numel()
            else (float("nan"), float("nan"))
        )
        raise AssertionError(
            "native output contains non-finite values: "
            f"finite={finite.sum().item()}/{native_output.numel()}, "
            f"finite_range={finite_range}, sample={native_output.flatten()[:16].tolist()}"
            f", nonfinite_coords={torch.nonzero(~finite)[:16].tolist()}, "
            f"nonfinite_per_head={(~finite).sum(dim=-1).tolist()}, "
            f"head0_dims={torch.nonzero(~finite[0, 0]).flatten().tolist()}"
        )
    torch.testing.assert_close(
        native_output, direct_output, rtol=0.04, atol=0.04
    )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        sparse_attention_nvfp4_416(output=native_output, **kwargs)
    graph.replay()
    torch.cuda.synchronize()
    if not torch.isfinite(native_output).all():
        raise AssertionError("native graph replay contains non-finite values")

    if os.getenv("NVFP4_TEST_SKIP_DUAL") == "1":
        peak_mib = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(
            "native NVFP4-416 sparse attention: ok "
            f"(max_error={max_error:.6f}, mean_error={mean_error:.6f}, "
            f"peak_allocated={peak_mib:.1f} MiB; dual-cache skipped)"
        )
        return

    dual_native_output = torch.empty_like(q)
    dual_direct_output = torch.empty_like(q)
    dual_splits = splits * 2
    dual_mid_out = torch.empty(
        (rows, heads, dual_splits, HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    )
    dual_mid_lse = torch.empty(
        (rows, heads, dual_splits), dtype=torch.float32, device=device
    )
    dual_kwargs = kwargs | {
        "indexed_cache": cache,
        "indexed_indices": indices,
        "indexed_lengths": lengths,
        "indexed_page_size": page_size,
        "mid_out": dual_mid_out,
        "mid_lse": dual_mid_lse,
    }
    os.environ["DSV4_NVFP4_ATTENTION_MODE"] = "direct"
    sparse_attention_nvfp4_416(output=dual_direct_output, **dual_kwargs)
    os.environ["DSV4_NVFP4_ATTENTION_MODE"] = "native"
    sparse_attention_nvfp4_416(output=dual_native_output, **dual_kwargs)
    torch.cuda.synchronize()
    dual_error = (dual_native_output.float() - dual_direct_output.float()).abs()
    dual_max_error = dual_error.max().item()
    dual_mean_error = dual_error.mean().item()
    torch.testing.assert_close(
        dual_native_output, dual_direct_output, rtol=0.04, atol=0.04
    )

    dual_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(dual_graph):
        sparse_attention_nvfp4_416(output=dual_native_output, **dual_kwargs)
    dual_graph.replay()
    torch.cuda.synchronize()
    if not torch.isfinite(dual_native_output).all():
        raise AssertionError("native dual-cache graph replay contains non-finite values")

    peak_mib = torch.cuda.max_memory_allocated() / (1024 * 1024)
    print(
        "native NVFP4-416 sparse attention: ok "
        f"(max_error={max_error:.6f}, mean_error={mean_error:.6f}, "
        f"dual_max_error={dual_max_error:.6f}, "
        f"dual_mean_error={dual_mean_error:.6f}, "
        f"peak_allocated={peak_mib:.1f} MiB)"
    )


if __name__ == "__main__":
    main()
