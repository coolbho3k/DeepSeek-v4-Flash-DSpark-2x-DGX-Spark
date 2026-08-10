"""Add a native SM120 sparse-MLA reader for DeepSeek V4 NVFP4 pages.

The pinned FlashInfer runtime has a highly tuned sparse DSV4 kernel, but its
only packed cache ABI is the 584-byte FP8/UE8M0 layout.  This patch adds a
separate 416-byte model type. Its IO warps read the compact records directly
and cooperatively expand packed E2M1 to the normalized E4M3 representation
consumed by the existing fused QK, BF16 RoPE, online-softmax, PV, SWA, and
indexed-cache pipeline. No global BF16 KV or FP32 score tensor is materialized.
"""

from pathlib import Path


ROOT = Path("/usr/local/lib/python3.12/dist-packages/flashinfer")


def replace(path: str, old: str, new: str, *, count: int = 1) -> None:
    target = ROOT / path
    source = target.read_text()
    if new in source:
        return
    found = source.count(old)
    if found < count:
        raise SystemExit(
            f"missing FlashInfer NVFP4 patch anchor in {target}: "
            f"needed {count}, found {found}: {old!r}"
        )
    target.write_text(source.replace(old, new, count))


MODEL_DIR = "data/include/flashinfer/attention/sparse_mla_sm120/model"
COMMON_DIR = "data/include/flashinfer/attention/sparse_mla_sm120/common"


replace(
    f"{MODEL_DIR}/model_type.h",
    "enum class ModelType { DSV3_2, DSV4, GLM_NSA };",
    "enum class ModelType { DSV3_2, DSV4, GLM_NSA, DSV4_NVFP4 };",
)

traits_path = f"{MODEL_DIR}/kv_cache_traits.cuh"
replace(
    traits_path,
    "  static constexpr bool SCALE_IN_KV_SMEM = true;\n",
    "  static constexpr bool SCALE_IN_KV_SMEM = true;\n"
    "  static constexpr bool PAGED_LAYOUT = false;\n",
)
replace(
    traits_path,
    "  static constexpr bool SCALE_IN_KV_SMEM = false;\n",
    "  static constexpr bool SCALE_IN_KV_SMEM = false;\n"
    "  static constexpr bool PAGED_LAYOUT = true;\n",
)
replace(
    traits_path,
    "// ============================================================================\n"
    "// Shared constants across all model types\n"
    "// ============================================================================\n",
    r'''// Compact DeepSeek V4 NVFP4 cache. IO warps cooperatively convert packed
// E2M1 records to normalized E4M3 in shared memory for the sparse MMA pipeline.
template <>
struct KVCacheTraits<ModelType::DSV4_NVFP4> {
  static constexpr int D_NOPE = 448;
  static constexpr int D_ROPE = 64;
  static constexpr int D_QK = D_NOPE + D_ROPE;
  static constexpr int D_V = 512;

  static constexpr int QUANT_TILE = 64;
  static constexpr int NUM_SCALES = 7;
  static constexpr ScaleFormat SCALE_FORMAT = ScaleFormat::POW2_FP32;

  // Global record: 256B packed E2M1 + 32B E4M3 scales + 128B BF16 RoPE.
  static constexpr bool SCALE_INLINE = true;
  static constexpr bool SCALE_IN_KV_SMEM = true;
  static constexpr bool PAGED_LAYOUT = true;
  static constexpr int FP4_DATA_BYTES = 256;
  static constexpr int FP4_SCALE_BYTES = 32;
  static constexpr int FP4_SCALE_GROUP = 16;
  static constexpr int SCALE_BYTES_PER_TOKEN = NUM_SCALES * sizeof(float);
  static constexpr int KV_GMEM_STRIDE = 416;
  static constexpr int KV_SCALE_GMEM_OFFSET = FP4_DATA_BYTES;
  static constexpr int KV_ROPE_GMEM_OFFSET = FP4_DATA_BYTES + FP4_SCALE_BYTES;

  // Shared record: 448B normalized E4M3 + 7 FP32 power-of-two scales + pad.
  static constexpr int KV_SMEM_STRIDE = 480;
  static constexpr int KV_SMEM_COPY_BYTES = D_NOPE + SCALE_BYTES_PER_TOKEN;

  static constexpr int Q_NOPE_STRIDE = D_NOPE + 16;
  static constexpr int Q_NOPE_BF16_STRIDE = D_NOPE + 8;
  static constexpr bool V_HAS_ROPE = true;

  __device__ static __forceinline__ uint8_t scale_to_ue8m0(float scale) {
    return static_cast<uint8_t>((__float_as_uint(scale) >> 23) & 0xFF);
  }
};

// ============================================================================
// Shared constants across all model types
// ============================================================================
''',
)
replace(
    traits_path,
    "static_assert(KVCacheTraits<ModelType::GLM_NSA>::D_V == D_V);",
    "static_assert(KVCacheTraits<ModelType::GLM_NSA>::D_V == D_V);\n"
    "static_assert(KVCacheTraits<ModelType::DSV4_NVFP4>::D_ROPE == D_ROPE);\n"
    "static_assert(KVCacheTraits<ModelType::DSV4_NVFP4>::D_V == D_V);",
)

io_path = f"{COMMON_DIR}/kv_cache_io.cuh"
replace(
    io_path,
    '#include "../arch/barrier.cuh"',
    '#include <flashinfer/math.cuh>\n\n#include "../arch/barrier.cuh"',
)
replace(
    io_path,
    "    if constexpr (KV::SCALE_IN_KV_SMEM) {\n"
    "      src = kv_ptr + (size_t)idx * IO::IO_STRIDE;\n",
    "    if constexpr (!KV::PAGED_LAYOUT) {\n"
    "      src = kv_ptr + (size_t)idx * IO::IO_STRIDE;\n",
)
replace(
    io_path,
    "  constexpr int COPY_BYTES = KV::KV_SMEM_COPY_BYTES;\n"
    "  constexpr int SMEM_STRIDE = KV::KV_SMEM_STRIDE;\n\n"
    "  if (io_tid == 0) mbarrier_arrive_expect_tx(mbar, BI * COPY_BYTES);\n",
    r'''  constexpr int COPY_BYTES = KV::KV_SMEM_COPY_BYTES;
  constexpr int SMEM_STRIDE = KV::KV_SMEM_STRIDE;

  if constexpr (MT == ModelType::DSV4_NVFP4) {
    // Preserve the kernel's proven one-thread-per-candidate/tile schedule, but
    // decode each tile's four source scales only once and process both FP4
    // nibbles from every packed byte together.
    constexpr int FP4_TILES = KV::D_NOPE / KV::QUANT_TILE;
    constexpr int FP4_SUBGROUPS = KV::QUANT_TILE / KV::FP4_SCALE_GROUP;
    constexpr int PACKED_BYTES_PER_TILE = KV::QUANT_TILE / 2;
    static_assert(FP4_SUBGROUPS == 4);
    for (int item = io_tid; item < BI * FP4_TILES; item += IO_THREADS) {
      const int bi = item / FP4_TILES;
      const int tile = item % FP4_TILES;
      int idx = indices[bi];
      idx = (idx >= 0) ? idx : 0;
      const int block_idx = idx / PAGE_BLOCK_SIZE;
      const int local_idx = idx % PAGE_BLOCK_SIZE;
      const uint8_t* record = kv_ptr + (size_t)block_idx * stride_kv_block +
                              (size_t)local_idx * KV::KV_GMEM_STRIDE;

      const uint32_t packed_scales = *reinterpret_cast<const uint32_t*>(
          record + KV::FP4_DATA_BYTES + tile * FP4_SUBGROUPS);
      uint32_t fp16_scales01, fp16_scales23;
      asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;"
                   : "=r"(fp16_scales01)
                   : "h"(static_cast<uint16_t>(packed_scales)));
      asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;"
                   : "=r"(fp16_scales23)
                   : "h"(static_cast<uint16_t>(packed_scales >> 16)));
      const __half2 scales01 =
          *reinterpret_cast<const __half2*>(&fp16_scales01);
      const __half2 scales23 =
          *reinterpret_cast<const __half2*>(&fp16_scales23);
      const float source_scales[FP4_SUBGROUPS] = {
          __low2float(scales01), __high2float(scales01),
          __low2float(scales23), __high2float(scales23)};
      const float max_source_scale =
          fmaxf(fmaxf(source_scales[0], source_scales[1]),
                fmaxf(source_scales[2], source_scales[3]));

      float shared_scale = 1.f;
      if (max_source_scale > 0.f) {
        const float required = max_source_scale * 6.f * FP8_MAX_INV;
        uint32_t bits = __float_as_uint(required);
        if (bits & 0x007FFFFFU)
          bits = (bits + 0x00800000U) & 0x7F800000U;
        shared_scale = __uint_as_float(bits);
      }
      uint8_t* shared_record = dst + (size_t)bi * KV::KV_SMEM_STRIDE;
      reinterpret_cast<float*>(shared_record + KV::D_NOPE)[tile] = shared_scale;
      const float inv_shared_scale = 1.f / shared_scale;

#pragma unroll
      for (int j = 0; j < PACKED_BYTES_PER_TILE; j += 2) {
        const uint8_t packed0 = record[tile * PACKED_BYTES_PER_TILE + j];
        const uint8_t packed1 = record[tile * PACKED_BYTES_PER_TILE + j + 1];
        const float multiplier = source_scales[j / (KV::FP4_SCALE_GROUP / 2)] *
                                 inv_shared_scale;
        const uint16_t fp4_pair = static_cast<uint16_t>(packed0) |
                                  (static_cast<uint16_t>(packed1) << 8);
        uint32_t fp16_pair0, fp16_pair1;
        asm volatile(
            "{ .reg .b8 lo, hi;                         \n"
            "  mov.b16 {lo, hi}, %2;                    \n"
            "  cvt.rn.f16x2.e2m1x2 %0, lo;              \n"
            "  cvt.rn.f16x2.e2m1x2 %1, hi;             }\n"
            : "=r"(fp16_pair0), "=r"(fp16_pair1)
            : "h"(fp4_pair));
        const __half2 values01 =
            *reinterpret_cast<const __half2*>(&fp16_pair0);
        const __half2 values23 =
            *reinterpret_cast<const __half2*>(&fp16_pair1);
        reinterpret_cast<uint32_t*>(shared_record + tile * KV::QUANT_TILE)[j / 2] =
            flashinfer::math::fp32_vec_to_e4m3(
                __low2float(values01) * multiplier,
                __high2float(values01) * multiplier,
                __low2float(values23) * multiplier,
                __high2float(values23) * multiplier);
      }
    }
    bar_sync_t<4, IO_THREADS>();
    if (io_tid == 0) {
      __threadfence_block();
      mbarrier_arrive(mbar);
    }
    return;
  }

  if (io_tid == 0) mbarrier_arrive_expect_tx(mbar, BI * COPY_BYTES);
''',
)

xv_path = f"{COMMON_DIR}/xv_rope_mma.cuh"
replace(
    xv_path,
    "      if constexpr (KV::SCALE_IN_KV_SMEM) {\n"
    "        base = KV_cache + (size_t)idx * IO::IO_STRIDE;\n",
    "      if constexpr (!KV::PAGED_LAYOUT) {\n"
    "        base = KV_cache + (size_t)idx * IO::IO_STRIDE;\n",
    count=2,
)

prefill_path = "data/csrc/sparse_mla_sm120_prefill.cu"
replace(
    prefill_path,
    "inline bool dispatch_dsv4_single(int num_heads, int topk, const bf16* Q,",
    "template <ModelType MT>\n"
    "inline bool dispatch_dsv4_single(int num_heads, int topk, const bf16* Q,",
)
replace(
    prefill_path,
    "  launch_prefill_mg<ModelType::DSV4, ComputeMode::CM, NH, TK, 64, NHG>(",
    "  launch_prefill_mg<MT, ComputeMode::CM, NH, TK, 64, NHG>(",
)
replace(
    prefill_path,
    "inline bool dispatch_dsv4_dual(int num_heads, int topk, int topk_extra,",
    "template <ModelType MT>\n"
    "inline bool dispatch_dsv4_dual(int num_heads, int topk, int topk_extra,",
)
replace(
    prefill_path,
    "  launch_prefill_mg_dual_fulltile<ModelType::DSV4, NH, TK, 64, PBSX, NHG>(",
    "  launch_prefill_mg_dual_fulltile<MT, NH, TK, 64, PBSX, NHG>(",
)
replace(
    prefill_path,
    "  launch_prefill_mg_dual<ModelType::DSV4, ComputeMode::CM, NH, TK, 64, PBSX, NHG>(",
    "  launch_prefill_mg_dual<MT, ComputeMode::CM, NH, TK, 64, PBSX, NHG>(",
)
replace(
    prefill_path,
    "    if (mt != ModelType::DSV4) return false;\n"
    "    return dispatch_dsv4_dual(num_heads, topk, topk_extra, extra_page_block_size, Q, KV_cache,\n"
    "                              indices, extra_KV_cache, extra_indices, attn_sink, output, out_lse,\n"
    "                              sm_scale, num_tokens, stride_kv_block, stride_kv_block_extra,\n"
    "                              topk_length, extra_topk_length, stream);\n",
    "    if (mt == ModelType::DSV4) {\n"
    "      return dispatch_dsv4_dual<ModelType::DSV4>(\n"
    "          num_heads, topk, topk_extra, extra_page_block_size, Q, KV_cache, indices,\n"
    "          extra_KV_cache, extra_indices, attn_sink, output, out_lse, sm_scale, num_tokens,\n"
    "          stride_kv_block, stride_kv_block_extra, topk_length, extra_topk_length, stream);\n"
    "    }\n"
    "    if (mt == ModelType::DSV4_NVFP4) {\n"
    "      return dispatch_dsv4_dual<ModelType::DSV4_NVFP4>(\n"
    "          num_heads, topk, topk_extra, extra_page_block_size, Q, KV_cache, indices,\n"
    "          extra_KV_cache, extra_indices, attn_sink, output, out_lse, sm_scale, num_tokens,\n"
    "          stride_kv_block, stride_kv_block_extra, topk_length, extra_topk_length, stream);\n"
    "    }\n"
    "    return false;\n",
)
replace(
    prefill_path,
    "      return dispatch_dsv4_single(num_heads, topk, Q, KV_cache, indices, attn_sink, output, out_lse,\n"
    "                                  sm_scale, num_tokens, stride_kv_block, topk_length, stream);\n",
    "      return dispatch_dsv4_single<ModelType::DSV4>(\n"
    "          num_heads, topk, Q, KV_cache, indices, attn_sink, output, out_lse, sm_scale,\n"
    "          num_tokens, stride_kv_block, topk_length, stream);\n"
    "    case ModelType::DSV4_NVFP4:\n"
    "      return dispatch_dsv4_single<ModelType::DSV4_NVFP4>(\n"
    "          num_heads, topk, Q, KV_cache, indices, attn_sink, output, out_lse, sm_scale,\n"
    "          num_tokens, stride_kv_block, topk_length, stream);\n",
)

orchestrator_path = "data/csrc/sparse_mla_sm120.cu"
replace(
    orchestrator_path,
    "    TVM_FFI_ICHECK(mt == ModelType::DSV4)\n"
    "        << \"d_qk=512 supports only model_type auto or DSV4; got \" << model_type;\n",
    "    TVM_FFI_ICHECK(mt == ModelType::DSV4 || mt == ModelType::DSV4_NVFP4)\n"
    "        << \"d_qk=512 supports model_type auto, DSV4, or DSV4_NVFP4; got \"\n"
    "        << model_type;\n",
)
replace(
    orchestrator_path,
    "    case ModelType::DSV4:\n"
    "      return 584;\n",
    "    case ModelType::DSV4:\n"
    "      return 584;\n"
    "    case ModelType::DSV4_NVFP4:\n"
    "      return 416;\n",
)
replace(
    orchestrator_path,
    "  TVM_FFI_ICHECK_GT(num_tokens, 64)\n"
    "      << \"Decode (num_tokens <= 64) must go through sparse_mla_sm120_decode_dsv3_2 \"\n"
    "         \"or sparse_mla_sm120_decode_dsv4; got num_tokens=\"\n"
    "      << num_tokens;\n",
    "  TVM_FFI_ICHECK(mt == ModelType::DSV4_NVFP4 || num_tokens > 64)\n"
    "      << \"Decode (num_tokens <= 64) must go through a standalone sparse-MLA kernel; \"\n"
    "         \"got num_tokens=\"\n"
    "      << num_tokens;\n",
)
replace(
    orchestrator_path,
    "                                                 : (mt == ModelType::GLM_NSA ? \"GLM_NSA\" : \"DSV4\"))\n",
    "                                                 : (mt == ModelType::GLM_NSA\n"
    "                                                        ? \"GLM_NSA\"\n"
    "                                                        : (mt == ModelType::DSV4_NVFP4\n"
    "                                                               ? \"DSV4_NVFP4\"\n"
    "                                                               : \"DSV4\")))\n",
)

python_path = "mla/_sparse_mla_sm120.py"
replace(
    python_path,
    "_MODEL_TYPE_GLM_NSA = 2\n",
    "_MODEL_TYPE_GLM_NSA = 2\n_MODEL_TYPE_DSV4_NVFP4 = 3\n",
)
replace(
    python_path,
    "_BPT_DSV4 = 584\n",
    "_BPT_DSV4 = 584\n_BPT_DSV4_NVFP4 = 416\n",
)
replace(
    python_path,
    "    if model_type == _MODEL_TYPE_DSV4:\n"
    "        return _BPT_DSV4\n",
    "    if model_type == _MODEL_TYPE_DSV4:\n"
    "        return _BPT_DSV4\n"
    "    if model_type == _MODEL_TYPE_DSV4_NVFP4:\n"
    "        return _BPT_DSV4_NVFP4\n",
)

# Give the patched sources their own JIT cache key, even if the base FP8 module
# was compiled on the host before this image was built.
replace(
    "jit/mla.py",
    '        "sparse_mla_sm120",\n',
    '        "sparse_mla_sm120_nvfp4_416",\n',
)

print("FlashInfer native SM120 sparse NVFP4-416 patch applied")
