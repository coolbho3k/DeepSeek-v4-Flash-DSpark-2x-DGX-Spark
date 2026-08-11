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


def replace(
    path: str,
    old: str,
    new: str,
    *,
    count: int = 1,
    present: str | None = None,
) -> None:
    target = ROOT / path
    source = target.read_text()
    if new in source or (present is not None and present in source):
        return
    found = source.count(old)
    if found < count:
        raise SystemExit(
            f"missing FlashInfer NVFP4 patch anchor in {target}: "
            f"needed {count}, found {found}: {old!r}"
        )
    target.write_text(source.replace(old, new, count))


def replace_in_region(
    path: str,
    start: str,
    end: str,
    old: str,
    new: str,
    *,
    count: int = 1,
    present: str | None = None,
) -> None:
    """Replace an anchor only inside one generated-source region."""
    target = ROOT / path
    source = target.read_text()
    start_at = source.find(start)
    if start_at < 0:
        raise SystemExit(f"missing FlashInfer region start in {target}: {start!r}")
    end_at = source.find(end, start_at + len(start))
    if end_at < 0:
        raise SystemExit(f"missing FlashInfer region end in {target}: {end!r}")
    region = source[start_at:end_at]
    if new in region or (present is not None and present in region):
        return
    found = region.count(old)
    if found < count:
        raise SystemExit(
            f"missing FlashInfer NVFP4 regional anchor in {target}: "
            f"needed {count}, found {found}: {old!r}"
        )
    region = region.replace(old, new, count)
    target.write_text(source[:start_at] + region + source[end_at:])


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
    "template <ModelType MT, int PAGE_BLOCK_SIZE, bool USE_L2_HINT = false>\n"
    "__device__ __forceinline__ void io_bulk_gather_tile",
    "template <ModelType MT, int PAGE_BLOCK_SIZE, bool USE_L2_HINT = false,\n"
    "          bool RAW_COMPACT = false>\n"
    "__device__ __forceinline__ void io_bulk_gather_tile",
)
replace(
    io_path,
    "  constexpr int COPY_BYTES = KV::KV_SMEM_COPY_BYTES;\n"
    "  constexpr int SMEM_STRIDE = KV::KV_SMEM_STRIDE;\n\n"
    "  if (io_tid == 0) mbarrier_arrive_expect_tx(mbar, BI * COPY_BYTES);\n",
    r'''  constexpr int COPY_BYTES = KV::KV_SMEM_COPY_BYTES;
  constexpr int SMEM_STRIDE = KV::KV_SMEM_STRIDE;

  if constexpr (MT == ModelType::DSV4_NVFP4 && RAW_COMPACT) {
    // Keep the 416-byte cache record packed through global and shared memory.
    // Only the 256-byte FP4 payload and its 32 E4M3 scales are needed by the
    // NoPE QK/PV pipeline; RoPE is fetched directly from the original record.
    constexpr int RAW_BYTES = KV::FP4_DATA_BYTES + KV::FP4_SCALE_BYTES;
    constexpr int VECTOR_BYTES = 16;
    constexpr int VECTORS_PER_RECORD = RAW_BYTES / VECTOR_BYTES;
    static_assert(RAW_BYTES == 288 && RAW_BYTES % VECTOR_BYTES == 0);
    for (int item = io_tid; item < BI * VECTORS_PER_RECORD;
         item += IO_THREADS) {
      const int bi = item / VECTORS_PER_RECORD;
      const int vec = item % VECTORS_PER_RECORD;
      int idx = indices[bi];
      idx = (idx >= 0) ? idx : 0;
      const int block_idx = idx / PAGE_BLOCK_SIZE;
      const int local_idx = idx % PAGE_BLOCK_SIZE;
      const uint8_t* record =
          kv_ptr + (size_t)block_idx * stride_kv_block +
          (size_t)local_idx * KV::KV_GMEM_STRIDE;
      *reinterpret_cast<uint4*>(dst + (size_t)bi * KV::KV_SMEM_STRIDE +
                                vec * VECTOR_BYTES) =
          *reinterpret_cast<const uint4*>(record + vec * VECTOR_BYTES);
    }
    bar_sync_t<4, IO_THREADS>();
    if (io_tid == 0) {
      __threadfence_block();
      mbarrier_arrive(mbar);
    }
    return;
  }

  if constexpr (MT == ModelType::DSV4_NVFP4 && !RAW_COMPACT) {
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
      const uint8_t* record =
          kv_ptr + (size_t)block_idx * stride_kv_block +
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
    present="Preserve the kernel's proven one-thread-per-candidate/tile schedule",
)

# E2M1 has only eight magnitudes.  Build their scaled E4M3 encodings once per
# 16-value source-scale group, then expand four nibbles with one PRMT instead
# of running an E2M1->F16->E4M3 conversion chain for every value.
prefill_half2_loop = r'''#pragma unroll
      for (int subgroup = 0; subgroup < FP4_SUBGROUPS; ++subgroup) {
        const __half2 multiplier2 = __float2half2_rn(
            source_scales[subgroup] * inv_shared_scale);
        const uint64_t packed = *reinterpret_cast<const uint64_t*>(
            record + tile * PACKED_BYTES_PER_TILE +
            subgroup * (KV::FP4_SCALE_GROUP / 2));
#pragma unroll
        for (int pair = 0; pair < KV::FP4_SCALE_GROUP / 4; ++pair) {
          const uint16_t fp4_pair =
              static_cast<uint16_t>(packed >> (pair * 16));
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
          const __half2 scaled01 = __hmul2(values01, multiplier2);
          const __half2 scaled23 = __hmul2(values23, multiplier2);
          uint16_t fp8_pair01, fp8_pair23;
          asm volatile("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;"
                       : "=h"(fp8_pair01)
                       : "r"(*reinterpret_cast<const uint32_t*>(&scaled01)));
          asm volatile("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;"
                       : "=h"(fp8_pair23)
                       : "r"(*reinterpret_cast<const uint32_t*>(&scaled23)));
          reinterpret_cast<uint32_t*>(
              shared_record + tile * KV::QUANT_TILE +
              subgroup * KV::FP4_SCALE_GROUP)[pair] =
              static_cast<uint32_t>(fp8_pair01) |
              (static_cast<uint32_t>(fp8_pair23) << 16);
        }
      }
'''
prefill_lookup_loop = r'''#pragma unroll
      for (int subgroup = 0; subgroup < FP4_SUBGROUPS; ++subgroup) {
        const __half2 multiplier2 = __float2half2_rn(
            source_scales[subgroup] * inv_shared_scale);
        const __half2 scaled01 = __hmul2(
            multiplier2, __floats2half2_rn(0.f, 0.5f));
        const __half2 scaled23 = __hmul2(
            multiplier2, __floats2half2_rn(1.f, 1.5f));
        const __half2 scaled45 = __hmul2(
            multiplier2, __floats2half2_rn(2.f, 3.f));
        const __half2 scaled67 = __hmul2(
            multiplier2, __floats2half2_rn(4.f, 6.f));
        uint16_t fp8_pair01, fp8_pair23, fp8_pair45, fp8_pair67;
        asm volatile("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;"
                     : "=h"(fp8_pair01)
                     : "r"(*reinterpret_cast<const uint32_t*>(&scaled01)));
        asm volatile("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;"
                     : "=h"(fp8_pair23)
                     : "r"(*reinterpret_cast<const uint32_t*>(&scaled23)));
        asm volatile("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;"
                     : "=h"(fp8_pair45)
                     : "r"(*reinterpret_cast<const uint32_t*>(&scaled45)));
        asm volatile("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;"
                     : "=h"(fp8_pair67)
                     : "r"(*reinterpret_cast<const uint32_t*>(&scaled67)));
        const uint32_t lut_lo = static_cast<uint32_t>(fp8_pair01) |
                                (static_cast<uint32_t>(fp8_pair23) << 16);
        const uint32_t lut_hi = static_cast<uint32_t>(fp8_pair45) |
                                (static_cast<uint32_t>(fp8_pair67) << 16);
        const uint64_t packed = *reinterpret_cast<const uint64_t*>(
            record + tile * PACKED_BYTES_PER_TILE +
            subgroup * (KV::FP4_SCALE_GROUP / 2));
#pragma unroll
        for (int pair = 0; pair < KV::FP4_SCALE_GROUP / 4; ++pair) {
          const uint32_t fp4_pair =
              static_cast<uint16_t>(packed >> (pair * 16));
          const uint32_t selector = fp4_pair & 0x7777U;
          const uint32_t signs =
              ((fp4_pair & 0x0008U) << 4) |
              ((fp4_pair & 0x0080U) << 8) |
              ((fp4_pair & 0x0800U) << 12) |
              ((fp4_pair & 0x8000U) << 16);
          reinterpret_cast<uint32_t*>(
              shared_record + tile * KV::QUANT_TILE +
              subgroup * KV::FP4_SCALE_GROUP)[pair] =
              __byte_perm(lut_lo, lut_hi, selector) ^ signs;
        }
      }
'''

# The initial NVFP4 producer above is also the migration anchor for older
# experimental images.  Convert four values at a time through half2 registers
# so prefill does not bounce every E2M1 value through scalar FP32 merely to
# write E4M3 shared memory.
replace(
    io_path,
    r'''#pragma unroll
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
''',
    r'''#pragma unroll
      for (int subgroup = 0; subgroup < FP4_SUBGROUPS; ++subgroup) {
        const __half2 multiplier2 = __float2half2_rn(
            source_scales[subgroup] * inv_shared_scale);
        const uint64_t packed = *reinterpret_cast<const uint64_t*>(
            record + tile * PACKED_BYTES_PER_TILE +
            subgroup * (KV::FP4_SCALE_GROUP / 2));
#pragma unroll
        for (int pair = 0; pair < KV::FP4_SCALE_GROUP / 4; ++pair) {
          const uint16_t fp4_pair =
              static_cast<uint16_t>(packed >> (pair * 16));
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
          const __half2 scaled01 = __hmul2(values01, multiplier2);
          const __half2 scaled23 = __hmul2(values23, multiplier2);
          uint16_t fp8_pair01, fp8_pair23;
          asm volatile("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;"
                       : "=h"(fp8_pair01)
                       : "r"(*reinterpret_cast<const uint32_t*>(&scaled01)));
          asm volatile("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;"
                       : "=h"(fp8_pair23)
                       : "r"(*reinterpret_cast<const uint32_t*>(&scaled23)));
          reinterpret_cast<uint32_t*>(
              shared_record + tile * KV::QUANT_TILE +
              subgroup * KV::FP4_SCALE_GROUP)[pair] =
              static_cast<uint32_t>(fp8_pair01) |
              (static_cast<uint32_t>(fp8_pair23) << 16);
        }
      }
''',
    present="const uint32_t selector = fp4_pair & 0x7777U;",
)

replace(
    io_path,
    prefill_half2_loop,
    prefill_lookup_loop,
    present="const uint32_t selector = fp4_pair & 0x7777U;",
)

replace(
    io_path,
    r'''          const uint32_t signs =
              ((fp4_pair & 0x0008U) << 4) |
              ((fp4_pair & 0x0080U) << 8) |
              ((fp4_pair & 0x0800U) << 12) |
              ((fp4_pair & 0x8000U) << 16);
''',
    r'''          const uint32_t sign_selector =
              (fp4_pair & 0x8888U) >> 1;
          const uint32_t signs =
              __byte_perm(0U, 0x80808080U, sign_selector);
''',
)

# Migration for images that already contain the expanded compact producer.
# Fresh images receive this block as part of the initial insertion above.
replace(
    io_path,
    "  if constexpr (MT == ModelType::DSV4_NVFP4) {\n"
    "    // Preserve the kernel's proven one-thread-per-candidate/tile schedule, but\n",
    r'''  if constexpr (MT == ModelType::DSV4_NVFP4 && RAW_COMPACT) {
    // Keep the 416-byte cache record packed through global and shared memory.
    // Only the 256-byte FP4 payload and its 32 E4M3 scales are needed by the
    // NoPE QK/PV pipeline; RoPE is fetched directly from the original record.
    constexpr int RAW_BYTES = KV::FP4_DATA_BYTES + KV::FP4_SCALE_BYTES;
    constexpr int VECTOR_BYTES = 16;
    constexpr int VECTORS_PER_RECORD = RAW_BYTES / VECTOR_BYTES;
    static_assert(RAW_BYTES == 288 && RAW_BYTES % VECTOR_BYTES == 0);
    for (int item = io_tid; item < BI * VECTORS_PER_RECORD;
         item += IO_THREADS) {
      const int bi = item / VECTORS_PER_RECORD;
      const int vec = item % VECTORS_PER_RECORD;
      int idx = indices[bi];
      idx = (idx >= 0) ? idx : 0;
      const int block_idx = idx / PAGE_BLOCK_SIZE;
      const int local_idx = idx % PAGE_BLOCK_SIZE;
      const uint8_t* record =
          kv_ptr + (size_t)block_idx * stride_kv_block +
          (size_t)local_idx * KV::KV_GMEM_STRIDE;
      *reinterpret_cast<uint4*>(dst + (size_t)bi * KV::KV_SMEM_STRIDE +
                                vec * VECTOR_BYTES) =
          *reinterpret_cast<const uint4*>(record + vec * VECTOR_BYTES);
    }
    bar_sync_t<4, IO_THREADS>();
    if (io_tid == 0) {
      __threadfence_block();
      mbarrier_arrive(mbar);
    }
    return;
  }

  if constexpr (MT == ModelType::DSV4_NVFP4 && !RAW_COMPACT) {
    // Preserve the kernel's proven one-thread-per-candidate/tile schedule, but
''',
    present="Keep the 416-byte cache record packed through global and shared memory",
)
replace(
    io_path,
    r'''    constexpr int VECTOR_BYTES = 16;
    constexpr int VECTORS_PER_RECORD = RAW_BYTES / VECTOR_BYTES;
    static_assert(RAW_BYTES == 288 && RAW_BYTES % VECTOR_BYTES == 0);
    for (int item = io_tid; item < BI * VECTORS_PER_RECORD;
         item += IO_THREADS) {
      const int bi = item / VECTORS_PER_RECORD;
      const int vec = item % VECTORS_PER_RECORD;
      int idx = indices[bi];
      idx = (idx >= 0) ? idx : 0;
      const int block_idx = idx / PAGE_BLOCK_SIZE;
      const int local_idx = idx % PAGE_BLOCK_SIZE;
      const uint8_t* record =
          kv_ptr + (size_t)block_idx * stride_kv_block +
          (size_t)local_idx * KV::KV_GMEM_STRIDE;
      *reinterpret_cast<uint4*>(dst + (size_t)bi * KV::KV_SMEM_STRIDE +
                                vec * VECTOR_BYTES) =
          *reinterpret_cast<const uint4*>(record + vec * VECTOR_BYTES);
    }
    bar_sync_t<4, IO_THREADS>();
    if (io_tid == 0) {
      __threadfence_block();
      mbarrier_arrive(mbar);
    }
    return;
''',
    r'''    static_assert(RAW_BYTES == 288 && RAW_BYTES % 16 == 0);
    if (io_tid == 0)
      mbarrier_arrive_expect_tx(mbar, BI * RAW_BYTES);
#pragma unroll 1
    for (int bi = io_tid; bi < BI; bi += IO_THREADS) {
      int idx = indices[bi];
      idx = (idx >= 0) ? idx : 0;
      const int block_idx = idx / PAGE_BLOCK_SIZE;
      const int local_idx = idx % PAGE_BLOCK_SIZE;
      const uint8_t* record =
          kv_ptr + (size_t)block_idx * stride_kv_block +
          (size_t)local_idx * KV::KV_GMEM_STRIDE;
      if constexpr (USE_L2_HINT) {
        cp_async_bulk_g2s_l2hint(
            dst + (size_t)bi * KV::KV_SMEM_STRIDE, record,
            RAW_BYTES, mbar, cache_policy);
      } else {
        cp_async_bulk_g2s(
            dst + (size_t)bi * KV::KV_SMEM_STRIDE, record,
            RAW_BYTES, mbar);
      }
    }
    return;
''',
    present="mbarrier_arrive_expect_tx(mbar, BI * RAW_BYTES)",
)

decode_path = (
    "data/include/flashinfer/attention/sparse_mla_sm120/"
    "decode_dsv4_kernel.cuh"
)
replace(
    decode_path,
    '#include "arch/barrier.cuh"',
    '#include <flashinfer/math.cuh>\n\n#include "arch/barrier.cuh"',
)
replace(
    decode_path,
    "constexpr int DSV4_QK_N_TILES = DSV4_ENTRIES_PER_WARP / 8;     // 1\n",
    r'''constexpr int DSV4_QK_N_TILES = DSV4_ENTRIES_PER_WARP / 8;     // 1

template <ModelType MT>
inline constexpr int DSV4_IO_THREADS_FOR =
    (MT == ModelType::DSV4_NVFP4 ? 1 : DSV4_IO_WARPS) * 32;
template <ModelType MT>
inline constexpr int DSV4_BLOCK_THREADS_FOR =
    DSV4_MATH_THREADS + DSV4_IO_THREADS_FOR<MT>;
template <ModelType MT>
inline constexpr int DSV4_KV_BUF_COUNT_FOR =
    MT == ModelType::DSV4_NVFP4 ? 2 : DSV4_KV_BUF_COUNT;
template <ModelType MT>
inline constexpr int DSV4_KV_SMEM_STRIDE_FOR =
    MT == ModelType::DSV4_NVFP4
        ? KVCacheTraits<MT>::KV_GMEM_STRIDE
        : KVCacheTraits<MT>::KV_SMEM_STRIDE;

// The native DSV4 cache keeps one UE8M0 byte per tile in a separate shared
// buffer. The compact NVFP4 producer stores one exact power-of-two FP32 scale
// per tile there instead. Keep the fused MMA/PV stages layout-agnostic.
template <ModelType MT>
__device__ __forceinline__ uint8_t decode_dsv4_scale_to_ue8m0(
    const uint8_t* scale_base, int entry, int tile) {
  using KV = KVCacheTraits<MT>;
  const uint8_t* row = scale_base + (size_t)entry * KV::SCALE_BYTES_PER_TOKEN;
  if constexpr (MT == ModelType::DSV4_NVFP4) {
    return KV::scale_to_ue8m0(reinterpret_cast<const float*>(row)[tile]);
  } else {
    return row[tile];
  }
}

template <ModelType MT>
__device__ __forceinline__ float decode_dsv4_scale_to_fp32(
    const uint8_t* scale_base, int entry, int tile) {
  using KV = KVCacheTraits<MT>;
  const uint8_t* row = scale_base + (size_t)entry * KV::SCALE_BYTES_PER_TOKEN;
  if constexpr (MT == ModelType::DSV4_NVFP4) {
    return reinterpret_cast<const float*>(row)[tile];
  } else {
    return ue8m0_to_fp32(row[tile]);
  }
}

// Expand one staged 64-value E2M1 tile into the normalized E4M3 layout used
// by the existing fused QK/PV math. Packed NoPE is staged at byte 224 and its
// 32 E4M3 scale bytes at byte 448, both inside the otherwise-unused tail of
// the 480-byte shared record.
__device__ __forceinline__ void decode_dsv4_expand_staged_nvfp4_tile(
    uint8_t* shared_record, uint8_t* shared_scales, int tile) {
  using KV = KVCacheTraits<ModelType::DSV4_NVFP4>;
  constexpr int STAGED_DATA_OFFSET = 224;
  constexpr int STAGED_SCALE_OFFSET = 448;
  constexpr int FP4_SUBGROUPS = KV::QUANT_TILE / KV::FP4_SCALE_GROUP;
  constexpr int PACKED_BYTES_PER_TILE = KV::QUANT_TILE / 2;
  static_assert(FP4_SUBGROUPS == 4);

  const uint32_t packed_scales = *reinterpret_cast<const uint32_t*>(
      shared_record + STAGED_SCALE_OFFSET + tile * FP4_SUBGROUPS);
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
    if (bits & 0x007fffffU) {
      bits = (bits + 0x00800000U) & 0x7f800000U;
    }
    shared_scale = __uint_as_float(bits);
  }
  reinterpret_cast<float*>(shared_scales)[tile] = shared_scale;
  const float inv_shared_scale = 1.f / shared_scale;

#pragma unroll
  for (int subgroup = 0; subgroup < FP4_SUBGROUPS; ++subgroup) {
    const __half2 multiplier2 =
        __float2half2_rn(source_scales[subgroup] * inv_shared_scale);
    const uint64_t packed = *reinterpret_cast<const uint64_t*>(
        shared_record + STAGED_DATA_OFFSET + tile * PACKED_BYTES_PER_TILE +
        subgroup * (KV::FP4_SCALE_GROUP / 2));
#pragma unroll
    for (int pair = 0; pair < KV::FP4_SCALE_GROUP / 4; ++pair) {
      const uint16_t fp4_pair =
          static_cast<uint16_t>(packed >> (pair * 16));
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
      const __half2 scaled01 = __hmul2(values01, multiplier2);
      const __half2 scaled23 = __hmul2(values23, multiplier2);
      uint16_t fp8_pair01, fp8_pair23;
      asm volatile("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;"
                   : "=h"(fp8_pair01)
                   : "r"(*reinterpret_cast<const uint32_t*>(&scaled01)));
      asm volatile("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;"
                   : "=h"(fp8_pair23)
                   : "r"(*reinterpret_cast<const uint32_t*>(&scaled23)));
      reinterpret_cast<uint32_t*>(
          shared_record + tile * KV::QUANT_TILE +
          subgroup * KV::FP4_SCALE_GROUP)[pair] =
          static_cast<uint32_t>(fp8_pair01) |
          (static_cast<uint32_t>(fp8_pair23) << 16);
    }
  }
}

struct MmaNvfp4Result {
  float d0, d1, d2, d3;
};

__device__ __forceinline__ uint8_t decode_dsv4_fp32x2_to_e2m1(
    float lo, float hi) {
  uint32_t packed;
  asm volatile(
      "{ .reg .b8 tmp;                           \n"
      "  cvt.rn.satfinite.e2m1x2.f32 tmp, %1, %2;\n"
      "  cvt.u32.u8 %0, tmp;                    }\n"
      : "=r"(packed)
      : "f"(hi), "f"(lo));
  return static_cast<uint8_t>(packed);
}

__device__ __forceinline__ float2 decode_dsv4_e2m1x2_to_float(
    uint8_t packed) {
  uint32_t fp16_pair;
  asm volatile(
      "{ .reg .b8 lo, hi;              \n"
      "  mov.b16 {lo, hi}, %1;         \n"
      "  cvt.rn.f16x2.e2m1x2 %0, lo;  }\n"
      : "=r"(fp16_pair)
      : "h"(static_cast<uint16_t>(packed)));
  const __half2 values = *reinterpret_cast<const __half2*>(&fp16_pair);
  return make_float2(__low2float(values), __high2float(values));
}

__device__ __forceinline__ float2 decode_dsv4_e4m3x2_to_float(
    uint8_t lo, uint8_t hi) {
  const uint16_t packed = static_cast<uint16_t>(lo) |
                          (static_cast<uint16_t>(hi) << 8);
  uint32_t fp16_pair;
  asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;"
               : "=r"(fp16_pair)
               : "h"(packed));
  const __half2 values = *reinterpret_cast<const __half2*>(&fp16_pair);
  return make_float2(__low2float(values), __high2float(values));
}

__device__ __forceinline__ uint16_t
decode_scaled_dsv4_e2m1x2_to_e4m3x2(
    uint8_t fp4_pair, uint8_t scale_lo, uint8_t scale_hi) {
  const uint16_t scale_pair = static_cast<uint16_t>(scale_lo) |
                              (static_cast<uint16_t>(scale_hi) << 8);
  uint32_t value_half2;
  uint32_t scale_half2;
  asm volatile(
      "{ .reg .b8 lo, hi;              \n"
      "  mov.b16 {lo, hi}, %2;         \n"
      "  cvt.rn.f16x2.e2m1x2 %0, lo;  \n"
      "  cvt.rn.f16x2.e4m3x2 %1, %3;  }\n"
      : "=r"(value_half2), "=r"(scale_half2)
      : "h"(static_cast<uint16_t>(fp4_pair)), "h"(scale_pair));
  const __half2 scaled = __hmul2(
      *reinterpret_cast<const __half2*>(&value_half2),
      *reinterpret_cast<const __half2*>(&scale_half2));
  uint16_t fp8_pair;
  asm volatile("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;"
               : "=h"(fp8_pair)
               : "r"(*reinterpret_cast<const uint32_t*>(&scaled)));
  return fp8_pair;
}

__device__ __forceinline__ float decode_dsv4_e2m1_to_float(uint8_t code) {
  const uint8_t magnitude_code = code & 7U;
  float magnitude;
  switch (magnitude_code) {
    case 0: magnitude = 0.f; break;
    case 1: magnitude = 0.5f; break;
    case 2: magnitude = 1.f; break;
    case 3: magnitude = 1.5f; break;
    case 4: magnitude = 2.f; break;
    case 5: magnitude = 3.f; break;
    case 6: magnitude = 4.f; break;
    default: magnitude = 6.f; break;
  }
  return (code & 8U) != 0 ? -magnitude : magnitude;
}

__device__ __forceinline__ MmaNvfp4Result mma_nvfp4_m16n8k64(
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1, float c0, float c1, float c2, float c3,
    uint32_t scale_a, uint32_t scale_b) {
  MmaNvfp4Result r;
  asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X."
      "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},"
      "{%14},{%15,%16},{%17},{%18,%19};\n"
      : "=f"(r.d0), "=f"(r.d1), "=f"(r.d2), "=f"(r.d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
        "f"(c0), "f"(c1), "f"(c2), "f"(c3), "r"(scale_a),
        "n"(static_cast<uint16_t>(0)), "n"(static_cast<uint16_t>(0)),
        "r"(scale_b), "n"(static_cast<uint16_t>(0)),
        "n"(static_cast<uint16_t>(0)));
  return r;
}

// SM120's mixed MMA consumes one E2M1 value in each 8-bit B container.
// The four-bit FP4 payload must occupy bits 2..5 (CUTLASS fp4_shift_B), not
// the low nibble used by packed cache storage. Scale B is one because the
// cache's E4M3 scale is either applied to the QK result or folded into P.
__device__ __forceinline__ MmaNvfp4Result
mma_mixed_fp8_fp4_m16n8k32(
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1, float c0, float c1, float c2, float c3,
    uint8_t scale_a) {
  MmaNvfp4Result r;
  constexpr uint8_t UNIT_UE8M0 = 0x7f;
  asm volatile(
      "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X."
      "m16n8k32.row.col.f32.e4m3.e2m1.f32.ue8m0 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},"
      "{%14},{%15,%16},{%17},{%18,%19};\n"
      : "=f"(r.d0), "=f"(r.d1), "=f"(r.d2), "=f"(r.d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
        "f"(c0), "f"(c1), "f"(c2), "f"(c3),
        "r"(static_cast<uint32_t>(scale_a)),
        "n"(static_cast<uint16_t>(0)), "n"(static_cast<uint16_t>(0)),
        "r"(static_cast<uint32_t>(UNIT_UE8M0)),
        "n"(static_cast<uint16_t>(0)), "n"(static_cast<uint16_t>(0)));
  return r;
}

__device__ __forceinline__ void decode_dsv4_pair_barrier(int barrier_id) {
  asm volatile("barrier.cta.sync %0, 64;\n" : : "r"(barrier_id) : "memory");
}

__device__ __forceinline__ void ldmatrix_load_A_nvfp4(
    uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3,
    const uint8_t* smem_base, int stride, int lane) {
  const int row = (lane & 7) + ((lane >> 3) & 1) * 8;
  const int col = (lane >> 4) * 16;
  ldmatrix_x4(a0, a1, a2, a3, smem_base + row * stride + col);
}

__device__ __forceinline__ void ldmatrix_load_B_nvfp4(
    uint32_t& b0, uint32_t& b1, const uint8_t* smem_base, int stride,
    int lane) {
  const int row = lane & 7;
  const int col = ((lane >> 3) & 1) * 16;
  ldmatrix_x2(b0, b1, smem_base + row * stride + col);
}

template <int KV_STRIDE>
__device__ __forceinline__ void d2_load_b_nvfp4(
    uint32_t& b0, uint32_t& b1, const uint8_t* kv_smem, int dim,
    int lane) {
  const int gid = lane >> 2;
  const int tid = lane & 3;
  const int d = dim + gid;
  const int byte_col = d >> 1;
  const int shift = (d & 1) * 4;
  b0 = 0;
  b1 = 0;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int k0 = tid * 8 + j;
    const int k1 = 32 + tid * 8 + j;
    const uint32_t v0 =
        (kv_smem[(size_t)k0 * KV_STRIDE + byte_col] >> shift) & 0xFU;
    const uint32_t v1 =
        (kv_smem[(size_t)k1 * KV_STRIDE + byte_col] >> shift) & 0xFU;
    b0 |= v0 << (j * 4);
    b1 |= v1 << (j * 4);
  }
}

// XQA's SM12x fast path keeps the cache packed and converts only the two
// values consumed by an MMA register.  CUDA 13.0 has native E2M1x2->F16x2,
// so apply the two E4M3 scales with one packed half2 multiply.
__device__ __forceinline__ uint32_t
decode_scaled_dsv4_e2m1x2_to_f16x2(
    uint8_t fp4_pair, uint8_t scale_lo, uint8_t scale_hi) {
  const uint16_t scale_pair = static_cast<uint16_t>(scale_lo) |
                              (static_cast<uint16_t>(scale_hi) << 8);
  uint32_t value_f16x2;
  uint32_t scale_f16x2;
  uint32_t scaled_f16x2;
  asm volatile(
      "{ .reg .b8 fp4;                         \n"
      "  mov.b16 {fp4, _}, %3;                 \n"
      "  cvt.rn.f16x2.e2m1x2 %0, fp4;          \n"
      "  cvt.rn.f16x2.e4m3x2 %1, %4;           \n"
      "  mul.rn.f16x2 %2, %0, %1;             }\n"
      : "=r"(value_f16x2), "=r"(scale_f16x2), "=r"(scaled_f16x2)
      : "h"(static_cast<uint16_t>(fp4_pair)), "h"(scale_pair));
  return scaled_f16x2;
}

__device__ __forceinline__ MmaBf16Result mma_f16_m16n8k16(
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1, float c0, float c1, float c2, float c3) {
  MmaBf16Result r;
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
      : "=f"(r.d0), "=f"(r.d1), "=f"(r.d2), "=f"(r.d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
        "f"(c0), "f"(c1), "f"(c2), "f"(c3));
  return r;
}

// Build the ordinary m16n8k16 column-major B fragment directly from four
// compact cache records.  Only F16 register fragments are materialized.
template <int KV_STRIDE>
__device__ __forceinline__ void d2_load_b_dequant_f16(
    uint32_t& b0, uint32_t& b1, const uint8_t* kv_smem, int cand_base,
    int dim, int lane) {
  using CompactKV = KVCacheTraits<ModelType::DSV4_NVFP4>;
  const int gid = lane >> 2;
  const int tid = lane & 3;
  const int d = dim + gid;
  const int byte_col = d >> 1;
  const int shift = (d & 1) * 4;
  const int scale_group = d / CompactKV::FP4_SCALE_GROUP;
  const int cand0 = cand_base + tid * 2;
  const int cand1 = cand0 + 1;
  const int cand8 = cand0 + 8;
  const int cand9 = cand8 + 1;
  const uint8_t* row0 = kv_smem + (size_t)cand0 * KV_STRIDE;
  const uint8_t* row1 = kv_smem + (size_t)cand1 * KV_STRIDE;
  const uint8_t* row8 = kv_smem + (size_t)cand8 * KV_STRIDE;
  const uint8_t* row9 = kv_smem + (size_t)cand9 * KV_STRIDE;
  const uint8_t code0 = (row0[byte_col] >> shift) & 0xfU;
  const uint8_t code1 = (row1[byte_col] >> shift) & 0xfU;
  const uint8_t code8 = (row8[byte_col] >> shift) & 0xfU;
  const uint8_t code9 = (row9[byte_col] >> shift) & 0xfU;
  b0 = decode_scaled_dsv4_e2m1x2_to_f16x2(
      code0 | (code1 << 4),
      row0[CompactKV::FP4_DATA_BYTES + scale_group],
      row1[CompactKV::FP4_DATA_BYTES + scale_group]);
  b1 = decode_scaled_dsv4_e2m1x2_to_f16x2(
      code8 | (code9 << 4),
      row8[CompactKV::FP4_DATA_BYTES + scale_group],
      row9[CompactKV::FP4_DATA_BYTES + scale_group]);
}

// Load a K32 x N8 QK B tile directly from N-major packed cache records.
// Each invocation exposes only one 16-wide scale group and zeroes the other
// register half. The two results can therefore be scaled independently and
// accumulated without normalizing or expanding the cache values.
template <int KV_STRIDE>
__device__ __forceinline__ void qk_load_b_mixed_fp4(
    uint32_t& b0, uint32_t& b1, const uint8_t* kv_smem,
    int cand_row_base, int dim_base, int scale_half, int lane) {
  const int gid = lane >> 2;
  const int tid = lane & 3;
  const int entry = cand_row_base + gid;
  const int first_dim = dim_base + scale_half * 16 + tid * 4;
  uint32_t packed = 0;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int d = first_dim + j;
    const uint8_t byte =
        kv_smem[(size_t)entry * KV_STRIDE + (d >> 1)];
    const uint32_t fp4 = (byte >> ((d & 1) * 4)) & 0xfU;
    packed |= (fp4 << 2) << (j * 8);
  }
  b0 = scale_half == 0 ? packed : 0;
  b1 = scale_half == 0 ? 0 : packed;
}

// Load a K32 x N8 PV B tile where K indexes candidates and N indexes output
// dimensions. Cache nibbles are shifted into the mixed-MMA byte container.
template <int KV_STRIDE>
__device__ __forceinline__ void d2_load_b_mixed_fp4(
    uint32_t& b0, uint32_t& b1, const uint8_t* kv_smem, int cand_base,
    int dim, int lane) {
  const int gid = lane >> 2;
  const int tid = lane & 3;
  const int d = dim + gid;
  const int byte_col = d >> 1;
  const int shift = (d & 1) * 4;
  b0 = 0;
  b1 = 0;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int k0 = cand_base + tid * 4 + j;
    const int k1 = cand_base + 16 + tid * 4 + j;
    const uint32_t v0 =
        (kv_smem[(size_t)k0 * KV_STRIDE + byte_col] >> shift) & 0xfU;
    const uint32_t v1 =
        (kv_smem[(size_t)k1 * KV_STRIDE + byte_col] >> shift) & 0xfU;
    b0 |= (v0 << 2) << (j * 8);
    b1 |= (v1 << 2) << (j * 8);
  }
}

// Form the ordinary E4M3 PV B registers directly from compact cache records.
// Values live only in registers: no BF16/FP32 tensor and no expanded shared
// cache are materialized. This lets all output-dimension groups reuse one
// quantized P matrix instead of folding four different V scales into P.
template <int KV_STRIDE>
__device__ __forceinline__ void d2_load_b_dequant_fp8(
    uint32_t& b0, uint32_t& b1, const uint8_t* kv_smem, int cand_base,
    int dim, int lane) {
  using CompactKV = KVCacheTraits<ModelType::DSV4_NVFP4>;
  const int gid = lane >> 2;
  const int tid = lane & 3;
  const int d = dim + gid;
  const int byte_col = d >> 1;
  const int shift = (d & 1) * 4;
  const int scale_group = d / CompactKV::FP4_SCALE_GROUP;
  b0 = 0;
  b1 = 0;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int k0 = cand_base + tid * 4 + j;
    const int k1 = cand_base + 16 + tid * 4 + j;
    const uint8_t* row0 = kv_smem + (size_t)k0 * KV_STRIDE;
    const uint8_t* row1 = kv_smem + (size_t)k1 * KV_STRIDE;
    const uint32_t fp4_0 = (row0[byte_col] >> shift) & 0xfU;
    const uint32_t fp4_1 = (row1[byte_col] >> shift) & 0xfU;
    const uint16_t fp8_pair = decode_scaled_dsv4_e2m1x2_to_e4m3x2(
        static_cast<uint8_t>(fp4_0 | (fp4_1 << 4)),
        row0[CompactKV::FP4_DATA_BYTES + scale_group],
        row1[CompactKV::FP4_DATA_BYTES + scale_group]);
    b0 |= static_cast<uint32_t>(fp8_pair & 0xffU) << (j * 8);
    b1 |= static_cast<uint32_t>(fp8_pair >> 8) << (j * 8);
  }
}

template <int MATH_THREADS>
__device__ __forceinline__ void quantize_q_nvfp4_to_smem(
    uint8_t* q_records, bf16* q_rope, const bf16* q_base, int valid_hpb) {
  constexpr int Q_RECORD_STRIDE = 288;
  constexpr int Q_TERM_BYTES = HPB * Q_RECORD_STRIDE;
  constexpr int Q_GROUPS = 28;
  constexpr int Q_GROUP_SIZE = 16;
  constexpr int Q_PACKED_BYTES_PER_GROUP = Q_GROUP_SIZE / 2;
  constexpr int Q_DIM = 512;

  for (int i = threadIdx.x; i < HPB * D_ROPE; i += MATH_THREADS) {
    const int h = i / D_ROPE;
    const int d = i % D_ROPE;
    q_rope[i] = h < valid_hpb ? q_base[(size_t)h * Q_DIM + 448 + d]
                              : __float2bfloat16(0.f);
  }
  for (int item = threadIdx.x; item < HPB * Q_GROUPS;
       item += MATH_THREADS) {
    const int h = item / Q_GROUPS;
    const int group = item % Q_GROUPS;
    float amax = 0.f;
#pragma unroll
    for (int j = 0; j < Q_GROUP_SIZE; ++j) {
      const float value =
          h < valid_hpb
              ? __bfloat162float(
                    q_base[(size_t)h * Q_DIM + group * Q_GROUP_SIZE + j])
              : 0.f;
      amax = fmaxf(amax, fabsf(value));
    }
    __nv_fp8_e4m3 scale0_fp8(amax * (1.f / 6.f));
    const float scale0 = static_cast<float>(scale0_fp8);
    const float inv_scale0 =
        (amax != 0.f && scale0 != 0.f) ? 1.f / scale0 : 0.f;
    uint8_t* q_record0 = q_records + (size_t)h * Q_RECORD_STRIDE;
    uint8_t* q_record1 = q_record0 + Q_TERM_BYTES;
    q_record0[256 + group] = scale0_fp8.__x;
    // With the primary value normalized into [-6, 6], E2M1's largest
    // adjacent-level gap is two, so its residual magnitude is at most one.
    // A scale of primary/6 therefore covers the complete residual range and
    // avoids a second amax pass.
    __nv_fp8_e4m3 scale1_fp8(scale0 * (1.f / 6.f));
    const float scale1 = static_cast<float>(scale1_fp8);
    const float inv_scale1 = scale1 != 0.f ? 1.f / scale1 : 0.f;
    q_record1[256 + group] = scale1_fp8.__x;
#pragma unroll
    for (int j = 0; j < Q_PACKED_BYTES_PER_GROUP; ++j) {
      const float value_lo =
          h < valid_hpb
              ? __bfloat162float(q_base[(size_t)h * Q_DIM +
                                        group * Q_GROUP_SIZE + j * 2])
              : 0.f;
      const float value_hi =
          h < valid_hpb
              ? __bfloat162float(q_base[(size_t)h * Q_DIM +
                                        group * Q_GROUP_SIZE + j * 2 + 1])
              : 0.f;
      const uint8_t packed0 = decode_dsv4_fp32x2_to_e2m1(
          value_lo * inv_scale0, value_hi * inv_scale0);
      q_record0[group * Q_PACKED_BYTES_PER_GROUP + j] = packed0;
      const float2 primary = decode_dsv4_e2m1x2_to_float(packed0);
      const float residual_lo =
          value_lo - primary.x * scale0;
      const float residual_hi =
          value_hi - primary.y * scale0;
      q_record1[group * Q_PACKED_BYTES_PER_GROUP + j] =
          decode_dsv4_fp32x2_to_e2m1(
              residual_lo * inv_scale1, residual_hi * inv_scale1);
    }
  }
  bar_sync_t<2, MATH_THREADS>();
}
''',
)
replace(
    decode_path,
    "  static_assert(MT == ModelType::DSV4);\n\n"
    "  static constexpr int N_V_CHUNKS = KV::D_NOPE / KV::QUANT_TILE;\n",
    "  static_assert(MT == ModelType::DSV4 || "
    "MT == ModelType::DSV4_NVFP4);\n\n"
    "  static constexpr bool COMPACT = "
    "MT == ModelType::DSV4_NVFP4;\n"
    "  static constexpr int KV_BUF_COUNT = "
    "DSV4_KV_BUF_COUNT_FOR<MT>;\n"
    "  static constexpr int N_V_CHUNKS = "
    "KV::D_NOPE / KV::QUANT_TILE;\n",
)
replace(
    decode_path,
    "  static constexpr size_t SMEM_Q_FP8 = HPB * KV::Q_NOPE_STRIDE;\n",
    "  static constexpr size_t SMEM_Q_FP8 =\n"
    "      COMPACT ? 2 * HPB * 288 : HPB * KV::Q_NOPE_STRIDE;\n",
)
replace(
    decode_path,
    "  static constexpr size_t SMEM_KV_FP8_BUF = "
    "DSV4_BI * KV::KV_SMEM_STRIDE;\n"
    "  static constexpr size_t SMEM_KV_SC_BUF = "
    "DSV4_BI * KV::SCALE_BYTES_PER_TOKEN;\n"
    "  static constexpr size_t SMEM_KV_ROPE_BUF = "
    "DSV4_BI * KV::D_ROPE * sizeof(bf16);\n"
    "  static constexpr size_t SMEM_MBAR_PAIR = "
    "2 * sizeof(uint64_t);\n",
    "  static constexpr size_t SMEM_KV_FP8_BUF =\n"
    "      DSV4_BI * DSV4_KV_SMEM_STRIDE_FOR<MT>;\n"
    "  static constexpr size_t SMEM_KV_SC_BUF =\n"
    "      COMPACT ? 0 : DSV4_BI * KV::SCALE_BYTES_PER_TOKEN;\n"
    "  static constexpr size_t SMEM_KV_ROPE_BUF =\n"
    "      COMPACT ? 0 : DSV4_BI * KV::D_ROPE * sizeof(bf16);\n"
    "  static constexpr size_t SMEM_MBAR_ARRAY =\n"
    "      KV_BUF_COUNT * sizeof(uint64_t);\n",
)
replace(
    decode_path,
    "  static constexpr size_t OFF_KV_SC = "
    "OFF_KV_FP8 + DSV4_KV_BUF_COUNT * SMEM_KV_FP8_BUF;\n"
    "  static constexpr size_t OFF_KV_ROPE = "
    "OFF_KV_SC + DSV4_KV_BUF_COUNT * SMEM_KV_SC_BUF;\n"
    "  static constexpr size_t OFF_MBAR_FULL_UNALIGNED =\n"
    "      OFF_KV_ROPE + DSV4_KV_BUF_COUNT * SMEM_KV_ROPE_BUF;\n"
    "  static constexpr size_t OFF_MBAR_FULL = "
    "(OFF_MBAR_FULL_UNALIGNED + 15) / 16 * 16;\n"
    "  static constexpr size_t OFF_MBAR_EMPTY = "
    "OFF_MBAR_FULL + SMEM_MBAR_PAIR;\n"
    "  static constexpr size_t OFF_REDUCE = "
    "OFF_MBAR_EMPTY + SMEM_MBAR_PAIR;\n",
    "  static constexpr size_t OFF_KV_SC =\n"
    "      OFF_KV_FP8 + KV_BUF_COUNT * SMEM_KV_FP8_BUF;\n"
    "  static constexpr size_t OFF_KV_ROPE =\n"
    "      OFF_KV_SC + KV_BUF_COUNT * SMEM_KV_SC_BUF;\n"
    "  static constexpr size_t OFF_MBAR_FULL_UNALIGNED =\n"
    "      OFF_KV_ROPE + KV_BUF_COUNT * SMEM_KV_ROPE_BUF;\n"
    "  static constexpr size_t OFF_MBAR_FULL = "
    "(OFF_MBAR_FULL_UNALIGNED + 15) / 16 * 16;\n"
    "  static constexpr size_t OFF_MBAR_EMPTY =\n"
    "      OFF_MBAR_FULL + SMEM_MBAR_ARRAY;\n"
    "  static constexpr size_t OFF_REDUCE =\n"
    "      OFF_MBAR_EMPTY + SMEM_MBAR_ARRAY;\n",
)
replace(
    decode_path,
    "__global__ void __launch_bounds__(DSV4_BLOCK_THREADS) sparse_mla_decode_dsv4_kernel(",
    "__global__ void __launch_bounds__(DSV4_BLOCK_THREADS_FOR<MT>) "
    "sparse_mla_decode_dsv4_kernel(",
)
replace(
    decode_path,
    "  static_assert(MT == ModelType::DSV4);",
    "  static_assert(MT == ModelType::DSV4 || MT == ModelType::DSV4_NVFP4);",
)
replace(
    decode_path,
    '  static_assert(MT == ModelType::DSV4, "decode-dsv4 currently DSV4-only");',
    '  static_assert(MT == ModelType::DSV4 || MT == ModelType::DSV4_NVFP4,\n'
    '                "decode-dsv4 supports native FP8 and compact NVFP4");',
)
replace(
    decode_path,
    "  constexpr int KV_SMEM_STRIDE = KV::KV_SMEM_STRIDE;                // 464\n",
    "  constexpr int KV_SMEM_STRIDE = DSV4_KV_SMEM_STRIDE_FOR<MT>;\n",
)
replace(
    decode_path,
    "  constexpr int IO_STRIDE = D_NOPE + D_ROPE_C * 2;                  // 576\n",
    "  constexpr int IO_STRIDE =\n"
    "      MT == ModelType::DSV4_NVFP4 ? KV::KV_GMEM_STRIDE : D_NOPE + D_ROPE_C * 2;\n",
)
replace(
    decode_path,
    "  constexpr int pbs = PAGE_BLOCK_SIZE;\n",
    "  constexpr int pbs = PAGE_BLOCK_SIZE;\n"
    "  constexpr int IO_THREADS = DSV4_IO_THREADS_FOR<MT>;\n"
    "  constexpr int KV_BUF_COUNT = DSV4_KV_BUF_COUNT_FOR<MT>;\n",
)
replace(
    decode_path,
    "    for (int s = 0; s < DSV4_KV_BUF_COUNT; ++s) {\n",
    "    for (int s = 0; s < KV_BUF_COUNT; ++s) {\n",
)
replace(
    decode_path,
    "      const int buf = (chunk_idx - chunk_lo) & 1;\n",
    "      const int buf = (chunk_idx - chunk_lo) % KV_BUF_COUNT;\n",
)
replace(
    decode_path,
    "    const int buf = (chunk_idx - chunk_lo) & 1;\n"
    "    // Dispatch chunk to main vs extra section.",
    "    const int buf = (chunk_idx - chunk_lo) % KV_BUF_COUNT;\n"
    "    // Dispatch chunk to main vs extra section.",
)
replace(
    decode_path,
    "      if (prod_idx == DSV4_KV_BUF_COUNT) {\n",
    "      if (prod_idx == KV_BUF_COUNT) {\n",
)
replace(
    decode_path,
    "    if (cons_idx == DSV4_KV_BUF_COUNT) {\n",
    "    if (cons_idx == KV_BUF_COUNT) {\n",
)
replace(
    decode_path,
    "    uint8_t* kv_sc_dst = sm.kv_sc(buf);\n\n#pragma unroll\n",
    r'''    uint8_t* kv_sc_dst = sm.kv_sc(buf);

    if constexpr (MT == ModelType::DSV4_NVFP4) {
      // Keep packed E2M1 and inline E4M3 scales in the per-candidate shared
      // record. A single 416-byte bulk also preserves inline BF16 RoPE and
      // avoids the separate legacy RoPE/scales allocations.
      static_assert(KV::KV_ROPE_GMEM_OFFSET +
                        D_ROPE_C * static_cast<int>(sizeof(bf16)) ==
                    KV::KV_GMEM_STRIDE);
      static_assert(DSV4_KV_SMEM_STRIDE_FOR<MT> ==
                    KV::KV_GMEM_STRIDE);
      if (io_tid == 0) {
        mbarrier_arrive_expect_tx(
            sm.mbar_full(buf),
            static_cast<uint32_t>(DSV4_BI * KV::KV_GMEM_STRIDE));
      }
#pragma unroll
      for (int eo = 0; eo < DSV4_BI; eo += IO_THREADS) {
        const int entry_idx = eo + io_tid;
        if (entry_idx >= DSV4_BI) break;
        const int cand_pos = g_start + entry_idx;
        const int idx_raw =
            (cand_pos < g_end) ? section_idx_base[cand_pos] : -1;
        const int idx = (idx_raw >= 0) ? idx_raw : 0;
        const int block_idx_g = idx / section_pbs;
        const int local_idx_g = idx - block_idx_g * section_pbs;
        const uint8_t* record =
            section_kv + (size_t)block_idx_g * section_stride +
            (size_t)local_idx_g * KV::KV_GMEM_STRIDE;
        cp_async_bulk_g2s(
            kv_fp8_dst + (size_t)entry_idx * KV_SMEM_STRIDE,
            record, static_cast<uint32_t>(KV::KV_GMEM_STRIDE),
            sm.mbar_full(buf));
      }
      return;

#if 0
      constexpr int STAGED_DATA_OFFSET = 224;
      constexpr int STAGED_SCALE_OFFSET = 448;
      constexpr int COPY_LANES = 24;
      const int copy_warp = io_tid >> 5;
      const int copy_lane = io_tid & 31;

      // Four producer warps each stream 16 records. Lanes 0..13 copy the
      // packed 448-dimensional NoPE payload, lanes 14..15 copy its 32 scales,
      // and lanes 16..23 copy BF16 RoPE. Every transfer is 16-byte aligned.
      for (int entry_idx = copy_warp; entry_idx < DSV4_BI;
           entry_idx += IO_THREADS / 32) {
        const int cand_pos = g_start + entry_idx;
        const int idx_raw =
            (cand_pos < g_end) ? section_idx_base[cand_pos] : -1;
        const int idx = (idx_raw >= 0) ? idx_raw : 0;
        const int block_idx_g = idx / section_pbs;
        const int local_idx_g = idx - block_idx_g * section_pbs;
        const uint8_t* record =
            section_kv + (size_t)block_idx_g * section_stride +
            (size_t)local_idx_g * KV::KV_GMEM_STRIDE;
        uint8_t* shared_record =
            kv_fp8_dst + (size_t)entry_idx * KV::KV_SMEM_STRIDE;

        if (copy_lane < 14) {
          cp_async_16B_l2(shared_record + STAGED_DATA_OFFSET + copy_lane * 16,
                          record + copy_lane * 16);
        } else if (copy_lane < 16) {
          cp_async_16B_l2(
              shared_record + STAGED_SCALE_OFFSET + (copy_lane - 14) * 16,
              record + KV::FP4_DATA_BYTES + (copy_lane - 14) * 16);
        } else if (copy_lane < COPY_LANES) {
          cp_async_16B_l2(
              reinterpret_cast<uint8_t*>(
                  kv_rope_dst + (size_t)entry_idx * D_ROPE_C) +
                  (copy_lane - 16) * 16,
              record + KV::KV_ROPE_GMEM_OFFSET + (copy_lane - 16) * 16);
        }
      }
      cp_async_commit();
      cp_async_wait_all();
      bar_sync_t<4, IO_THREADS>();

      // Destination bytes for tiles 0..2 are below the staging tail. Tiles
      // 3+ overwrite only source tiles completed by an earlier phase.
      for (int item = io_tid; item < DSV4_BI * 3; item += IO_THREADS) {
        const int tile = item / DSV4_BI;
        const int entry_idx = item % DSV4_BI;
        decode_dsv4_expand_staged_nvfp4_tile(
            kv_fp8_dst + (size_t)entry_idx * KV::KV_SMEM_STRIDE,
            kv_sc_dst + (size_t)entry_idx * SCALE_BYTES_PER_TOKEN, tile);
      }
      bar_sync_t<4, IO_THREADS>();
      for (int item = io_tid; item < DSV4_BI * 2; item += IO_THREADS) {
        const int tile = 3 + item / DSV4_BI;
        const int entry_idx = item % DSV4_BI;
        decode_dsv4_expand_staged_nvfp4_tile(
            kv_fp8_dst + (size_t)entry_idx * KV::KV_SMEM_STRIDE,
            kv_sc_dst + (size_t)entry_idx * SCALE_BYTES_PER_TOKEN, tile);
      }
      bar_sync_t<4, IO_THREADS>();
      for (int entry_idx = io_tid; entry_idx < DSV4_BI;
           entry_idx += IO_THREADS) {
        decode_dsv4_expand_staged_nvfp4_tile(
            kv_fp8_dst + (size_t)entry_idx * KV::KV_SMEM_STRIDE,
            kv_sc_dst + (size_t)entry_idx * SCALE_BYTES_PER_TOKEN, 5);
      }
      bar_sync_t<4, IO_THREADS>();
      for (int entry_idx = io_tid; entry_idx < DSV4_BI;
           entry_idx += IO_THREADS) {
        decode_dsv4_expand_staged_nvfp4_tile(
            kv_fp8_dst + (size_t)entry_idx * KV::KV_SMEM_STRIDE,
            kv_sc_dst + (size_t)entry_idx * SCALE_BYTES_PER_TOKEN, 6);
      }
      bar_sync_t<4, IO_THREADS>();
      if (io_tid == 0) {
        __threadfence_block();
        mbarrier_arrive(sm.mbar_full(buf));
      }
      return;

#else
      // Producer warps expand compact records directly into the normalized
      // E4M3 + power-of-two scale representation consumed by the downstream
      // fused QK/online-softmax/PV pipeline. RoPE stays BF16.
      constexpr int FP4_TILES = D_NOPE / QUANT_TILE;
      constexpr int FP4_SUBGROUPS = QUANT_TILE / KV::FP4_SCALE_GROUP;
      constexpr int PACKED_BYTES_PER_TILE = QUANT_TILE / 2;
      static_assert(FP4_SUBGROUPS == 4);

      for (int item = io_tid; item < DSV4_BI * FP4_TILES;
           item += IO_THREADS) {
        const int entry_idx = item / FP4_TILES;
        const int tile = item % FP4_TILES;
        const int cand_pos = g_start + entry_idx;
        const int idx_raw = (cand_pos < g_end) ? section_idx_base[cand_pos] : -1;
        const int idx = (idx_raw >= 0) ? idx_raw : 0;
        const int block_idx_g = idx / section_pbs;
        const int local_idx_g = idx - block_idx_g * section_pbs;
        const uint8_t* record =
            section_kv + (size_t)block_idx_g * section_stride +
            (size_t)local_idx_g * KV::KV_GMEM_STRIDE;

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
        reinterpret_cast<float*>(
            kv_sc_dst + (size_t)entry_idx * SCALE_BYTES_PER_TOKEN)[tile] =
            shared_scale;
        uint8_t* shared_record =
            kv_fp8_dst + (size_t)entry_idx * KV_SMEM_STRIDE;
        const float inv_shared_scale = 1.f / shared_scale;

#pragma unroll
        for (int subgroup = 0; subgroup < FP4_SUBGROUPS; ++subgroup) {
          const __half2 multiplier2 = __float2half2_rn(
              source_scales[subgroup] * inv_shared_scale);
          const uint64_t packed = *reinterpret_cast<const uint64_t*>(
              record + tile * PACKED_BYTES_PER_TILE +
              subgroup * (KV::FP4_SCALE_GROUP / 2));
#pragma unroll
          for (int pair = 0; pair < KV::FP4_SCALE_GROUP / 4; ++pair) {
            const uint16_t fp4_pair =
                static_cast<uint16_t>(packed >> (pair * 16));
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
            const __half2 scaled01 = __hmul2(values01, multiplier2);
            const __half2 scaled23 = __hmul2(values23, multiplier2);
            uint16_t fp8_pair01, fp8_pair23;
            asm volatile("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;"
                         : "=h"(fp8_pair01)
                         : "r"(*reinterpret_cast<const uint32_t*>(&scaled01)));
            asm volatile("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;"
                         : "=h"(fp8_pair23)
                         : "r"(*reinterpret_cast<const uint32_t*>(&scaled23)));
            reinterpret_cast<uint32_t*>(
                shared_record + tile * QUANT_TILE +
                subgroup * KV::FP4_SCALE_GROUP)[pair] =
                static_cast<uint32_t>(fp8_pair01) |
                (static_cast<uint32_t>(fp8_pair23) << 16);
          }
        }
      }

      // Named barrier 4 publishes every producer lane's conversion stores;
      // the mbar transaction then couples that data with completion of all
      // 128-byte RoPE copies.
      bar_sync_t<4, IO_THREADS>();
      if (io_tid == 0) {
        __threadfence_block();
        mbarrier_arrive_expect_tx(
            sm.mbar_full(buf),
            (uint32_t)DSV4_BI * D_ROPE_C * sizeof(bf16));
      }
#pragma unroll
      for (int eo = 0; eo < DSV4_BI; eo += IO_THREADS) {
        const int entry_idx = eo + io_tid;
        if (entry_idx >= DSV4_BI) break;
        const int cand_pos = g_start + entry_idx;
        const int idx_raw = (cand_pos < g_end) ? section_idx_base[cand_pos] : -1;
        const int idx = (idx_raw >= 0) ? idx_raw : 0;
        const int block_idx_g = idx / section_pbs;
        const int local_idx_g = idx - block_idx_g * section_pbs;
        const uint8_t* record =
            section_kv + (size_t)block_idx_g * section_stride +
            (size_t)local_idx_g * KV::KV_GMEM_STRIDE;
        cp_async_bulk_g2s(
            kv_rope_dst + (size_t)entry_idx * D_ROPE_C,
            record + KV::KV_ROPE_GMEM_OFFSET,
            (uint32_t)D_ROPE_C * sizeof(bf16), sm.mbar_full(buf));
      }
      return;
#endif
    }

#pragma unroll
''',
)
replace(
    decode_path,
    "  const bool is_io = (warp_id >= DSV4_N_WARPS);\n",
    "  const bool is_io = (warp_id >= DSV4_N_WARPS);\n"
    "  const int io_tid = threadIdx.x - DSV4_MATH_THREADS;\n",
)
replace(
    decode_path,
    "    for (int eo = 0; eo < DSV4_BI; eo += DSV4_IO_THREADS) {\n"
    "      const int entry_idx = eo + lane;",
    "    for (int eo = 0; eo < DSV4_BI; eo += IO_THREADS) {\n"
    "      const int entry_idx = eo + io_tid;",
    count=2,
)
replace(
    decode_path,
    "    __threadfence_block();\n\n"
    "    if (lane == 0) {\n",
    "    bar_sync_t<4, IO_THREADS>();\n"
    "    __threadfence_block();\n\n"
    "    if (io_tid == 0) {\n",
)
replace(
    decode_path,
    "  quantize_q_to_smem<MT, DSV4_MATH_THREADS>(sm.q_fp8(), sm.q_sc(), "
    "sm.q_rope(), q_base, sm.reduce(),\n"
    "                                            VALID_HPB);\n",
    "  if constexpr (MT == ModelType::DSV4_NVFP4) {\n"
    "    quantize_q_nvfp4_to_smem<DSV4_MATH_THREADS>(\n"
    "        sm.q_fp8(), sm.q_rope(), q_base, VALID_HPB);\n"
    "  } else {\n"
    "    quantize_q_to_smem<MT, DSV4_MATH_THREADS>(\n"
    "        sm.q_fp8(), sm.q_sc(), sm.q_rope(), q_base, sm.reduce(),\n"
    "        VALID_HPB);\n"
    "  }\n",
)
replace(
    decode_path,
    "    float qk[DSV4_QK_N_TILES][4] = {0};\n"
    "    {\n",
    r'''    float qk[DSV4_QK_N_TILES][4] = {0};
    if constexpr (MT == ModelType::DSV4_NVFP4 && false) {
      // Q remains in the baseline E4M3/power-of-two representation. Each
      // packed K32 tile has two independent E4M3 cache scales, so issue one
      // mixed MMA per K16 half with the other half zeroed, then apply that
      // candidate's exact source scale to the unaccumulated result.
      const int warp_first_cand = warp_id * DSV4_ENTRIES_PER_WARP;
#pragma unroll
      for (int blk = 0; blk < NUM_SCALES; ++blk) {
        const uint8_t sfa = fp32_to_ue8m0(
            sm.q_sc()[(gid + (lane & 1) * 8) * NUM_SCALES + blk]);
#pragma unroll
        for (int ks = 0; ks < QUANT_TILE / 32; ++ks) {
          const int ko = blk * QUANT_TILE + ks * 32;
          uint32_t a0, a1, a2, a3;
          ldmatrix_load_A_fp8(
              a0, a1, a2, a3, sm.q_fp8() + ko, Q_NOPE_STRIDE, lane);
          for (int scale_half = 0; scale_half < 2; ++scale_half) {
            const int source_scale_group =
                blk * (QUANT_TILE / KV::FP4_SCALE_GROUP) +
                ks * (32 / KV::FP4_SCALE_GROUP) + scale_half;
#pragma unroll
            for (int nt = 0; nt < DSV4_QK_N_TILES; ++nt) {
              const int cand_row_base = warp_first_cand + nt * 8;
              uint32_t b0, b1;
              qk_load_b_mixed_fp4<KV_SMEM_STRIDE>(
                  b0, b1, sm_kv_fp8, cand_row_base, ko, scale_half,
                  lane);
              const MmaNvfp4Result r =
                  mma_mixed_fp8_fp4_m16n8k32(
                      a0, a1, a2, a3, b0, b1,
                      0.f, 0.f, 0.f, 0.f, sfa);
              const int cand0 = cand_row_base + tid * 2;
              const int cand1 = cand0 + 1;
              const float2 source_scales =
                  decode_dsv4_e4m3x2_to_float(
                      sm_kv_fp8[(size_t)cand0 * KV_SMEM_STRIDE +
                                 KV::FP4_DATA_BYTES +
                                 source_scale_group],
                      sm_kv_fp8[(size_t)cand1 * KV_SMEM_STRIDE +
                                 KV::FP4_DATA_BYTES +
                                 source_scale_group]);
              qk[nt][0] += r.d0 * source_scales.x;
              qk[nt][1] += r.d1 * source_scales.y;
              qk[nt][2] += r.d2 * source_scales.x;
              qk[nt][3] += r.d3 * source_scales.y;
            }
          }
        }
      }
    } else if constexpr (MT == ModelType::DSV4_NVFP4) {
      constexpr int Q_NVFP4_STRIDE = 288;
      constexpr int Q_TERM_BYTES = HPB * Q_NVFP4_STRIDE;
      constexpr int FP4_PACKED_BYTES_PER_TILE = QUANT_TILE / 2;
      const int warp_first_cand = warp_id * DSV4_ENTRIES_PER_WARP;
#pragma unroll
      for (int blk = 0; blk < NUM_SCALES; ++blk) {
#pragma unroll
        for (int term = 0; term < 2; ++term) {
          const uint8_t* q_term = sm.q_fp8() + term * Q_TERM_BYTES;
          const int q_head = gid + (lane & 1) * 8;
          const uint8_t* q_record =
              q_term + (size_t)q_head * Q_NVFP4_STRIDE;
          const uint32_t sfa = *reinterpret_cast<const uint32_t*>(
              q_record + KV::FP4_DATA_BYTES + blk * 4);
          uint32_t a0, a1, a2, a3;
          ldmatrix_load_A_nvfp4(
              a0, a1, a2, a3,
              q_term + blk * FP4_PACKED_BYTES_PER_TILE,
              Q_NVFP4_STRIDE, lane);
#pragma unroll
          for (int nt = 0; nt < DSV4_QK_N_TILES; ++nt) {
            const int cand_row_base = warp_first_cand + nt * 8;
            const uint8_t* kv_record =
                sm_kv_fp8 + (size_t)(cand_row_base + gid) * KV_SMEM_STRIDE;
            const uint32_t sfb = *reinterpret_cast<const uint32_t*>(
                kv_record + KV::FP4_DATA_BYTES + blk * 4);
            uint32_t b0, b1;
            ldmatrix_load_B_nvfp4(
                b0, b1,
                sm_kv_fp8 + (size_t)cand_row_base * KV_SMEM_STRIDE +
                    blk * FP4_PACKED_BYTES_PER_TILE,
                KV_SMEM_STRIDE, lane);
            const MmaNvfp4Result r = mma_nvfp4_m16n8k64(
                a0, a1, a2, a3, b0, b1, qk[nt][0], qk[nt][1],
                qk[nt][2], qk[nt][3], sfa, sfb);
            qk[nt][0] = r.d0;
            qk[nt][1] = r.d1;
            qk[nt][2] = r.d2;
            qk[nt][3] = r.d3;
          }
        }
      }
    } else {
''',
)
replace(
    decode_path,
    "            uint8_t sfb = sm_kv_sc[(cand_row_base + gid) * SCALE_BYTES_PER_TOKEN + blk];",
    "            uint8_t sfb =\n"
    "                decode_dsv4_scale_to_ue8m0<MT>(sm_kv_sc, cand_row_base + gid, blk);",
)
replace(
    decode_path,
    "          const bf16* kv_rope_row = sm_kv_rope + "
    "(size_t)entry * D_ROPE_C + ks * 16;\n",
    "          const bf16* kv_rope_row;\n"
    "          if constexpr (MT == ModelType::DSV4_NVFP4) {\n"
    "            kv_rope_row = reinterpret_cast<const bf16*>(\n"
    "                sm_kv_fp8 + (size_t)entry * KV_SMEM_STRIDE +\n"
    "                KV::KV_ROPE_GMEM_OFFSET) + ks * 16;\n"
    "          } else {\n"
    "            kv_rope_row = sm_kv_rope + "
    "(size_t)entry * D_ROPE_C + ks * 16;\n"
    "          }\n",
)
replace(
    decode_path,
    "          uint16_t v0 =\n"
    "              *reinterpret_cast<const uint16_t*>(sm_kv_rope + "
    "(size_t)ent0 * D_ROPE_C + col);\n"
    "          uint16_t v1 =\n"
    "              *reinterpret_cast<const uint16_t*>(sm_kv_rope + "
    "(size_t)ent1 * D_ROPE_C + col);\n"
    "          uint16_t v8 =\n"
    "              *reinterpret_cast<const uint16_t*>(sm_kv_rope + "
    "(size_t)ent8 * D_ROPE_C + col);\n"
    "          uint16_t v9 =\n"
    "              *reinterpret_cast<const uint16_t*>(sm_kv_rope + "
    "(size_t)ent9 * D_ROPE_C + col);\n",
    "          const bf16* rope_row0;\n"
    "          const bf16* rope_row1;\n"
    "          const bf16* rope_row8;\n"
    "          const bf16* rope_row9;\n"
    "          if constexpr (MT == ModelType::DSV4_NVFP4) {\n"
    "            rope_row0 = reinterpret_cast<const bf16*>(\n"
    "                sm_kv_fp8 + (size_t)ent0 * KV_SMEM_STRIDE +\n"
    "                KV::KV_ROPE_GMEM_OFFSET);\n"
    "            rope_row1 = reinterpret_cast<const bf16*>(\n"
    "                sm_kv_fp8 + (size_t)ent1 * KV_SMEM_STRIDE +\n"
    "                KV::KV_ROPE_GMEM_OFFSET);\n"
    "            rope_row8 = reinterpret_cast<const bf16*>(\n"
    "                sm_kv_fp8 + (size_t)ent8 * KV_SMEM_STRIDE +\n"
    "                KV::KV_ROPE_GMEM_OFFSET);\n"
    "            rope_row9 = reinterpret_cast<const bf16*>(\n"
    "                sm_kv_fp8 + (size_t)ent9 * KV_SMEM_STRIDE +\n"
    "                KV::KV_ROPE_GMEM_OFFSET);\n"
    "          } else {\n"
    "            rope_row0 = sm_kv_rope + "
    "(size_t)ent0 * D_ROPE_C;\n"
    "            rope_row1 = sm_kv_rope + "
    "(size_t)ent1 * D_ROPE_C;\n"
    "            rope_row8 = sm_kv_rope + "
    "(size_t)ent8 * D_ROPE_C;\n"
    "            rope_row9 = sm_kv_rope + "
    "(size_t)ent9 * D_ROPE_C;\n"
    "          }\n"
    "          const uint16_t v0 = "
    "*reinterpret_cast<const uint16_t*>(rope_row0 + col);\n"
    "          const uint16_t v1 = "
    "*reinterpret_cast<const uint16_t*>(rope_row1 + col);\n"
    "          const uint16_t v8 = "
    "*reinterpret_cast<const uint16_t*>(rope_row8 + col);\n"
    "          const uint16_t v9 = "
    "*reinterpret_cast<const uint16_t*>(rope_row9 + col);\n",
)
replace(
    decode_path,
    "          const float vsc0 = ue8m0_to_fp32(sm_kv_sc[(size_t)cand_e0 * SCALE_BYTES_PER_TOKEN + vc]);\n"
    "          const float vsc1 = ue8m0_to_fp32(sm_kv_sc[(size_t)cand_e1 * SCALE_BYTES_PER_TOKEN + vc]);",
    "          const float vsc0 = decode_dsv4_scale_to_fp32<MT>(sm_kv_sc, cand_e0, vc);\n"
    "          const float vsc1 = decode_dsv4_scale_to_fp32<MT>(sm_kv_sc, cand_e1, vc);",
    count=2,
)

# Disposable development images may already contain the rejected F16 P
# staging experiment. Remove it before applying the current decode branch;
# fresh upstream sources simply do not contain this exact block.
legacy_f16_p_staging = r'''      sm_p_full[gid][cand_col_base + c0] = __float2bfloat16(w_pre[nt][0]);
      sm_p_full[gid][cand_col_base + c1] = __float2bfloat16(w_pre[nt][1]);
      sm_p_full[gid + 8][cand_col_base + c0] = __float2bfloat16(w_pre[nt][2]);
      sm_p_full[gid + 8][cand_col_base + c1] = __float2bfloat16(w_pre[nt][3]);
      if constexpr (MT == ModelType::DSV4_NVFP4) {
        // The two existing FP8 weight buffers are exactly 2 KiB together.
        // Reuse them for an F16 copy of P so register-fragment PV avoids the
        // atomic max and FP8 requantization prologue.
        __half* sm_p_f16 = reinterpret_cast<__half*>(sm.w_fp8(0));
        sm_p_f16[(size_t)gid * DSV4_BI + cand_col_base + c0] =
            __float2half_rn(w_pre[nt][0]);
        sm_p_f16[(size_t)gid * DSV4_BI + cand_col_base + c1] =
            __float2half_rn(w_pre[nt][1]);
        sm_p_f16[(size_t)(gid + 8) * DSV4_BI + cand_col_base + c0] =
            __float2half_rn(w_pre[nt][2]);
        sm_p_f16[(size_t)(gid + 8) * DSV4_BI + cand_col_base + c1] =
            __float2half_rn(w_pre[nt][3]);
      }
'''
plain_p_staging = r'''      sm_p_full[gid][cand_col_base + c0] = __float2bfloat16(w_pre[nt][0]);
      sm_p_full[gid][cand_col_base + c1] = __float2bfloat16(w_pre[nt][1]);
      sm_p_full[gid + 8][cand_col_base + c0] = __float2bfloat16(w_pre[nt][2]);
      sm_p_full[gid + 8][cand_col_base + c1] = __float2bfloat16(w_pre[nt][3]);
'''
decode_target = ROOT / decode_path
decode_source = decode_target.read_text()
if legacy_f16_p_staging in decode_source:
    decode_target.write_text(
        decode_source.replace(legacy_f16_p_staging, plain_p_staging, 1)
    )

replace(
    decode_path,
    r'''    // Zero-init sm_w_head_sc here (different smem buffer than sm_p_full above),
    // so the single bar_sync below covers both write groups.
    for (int i = threadIdx.x; i < N_V_CHUNKS * HPB; i += DSV4_MATH_THREADS) {
      sm.w_head_sc()[i] = 0.f;
    }
    bar_sync_t<3, DSV4_MATH_THREADS>();
''',
    r'''    if constexpr (MT != ModelType::DSV4_NVFP4) {
      // Only the baseline FP8 PV path needs the per-output-chunk reductions.
      for (int i = threadIdx.x; i < N_V_CHUNKS * HPB;
           i += DSV4_MATH_THREADS) {
        sm.w_head_sc()[i] = 0.f;
      }
      bar_sync_t<3, DSV4_MATH_THREADS>();
    }
''',
    present="Only the baseline FP8 PV path needs",
)
replace(
    decode_path,
    "    // ── Stage 3 NoPE FP8 ──────────────────────────────────────\n"
    "    {\n",
    r'''    if constexpr (MT == ModelType::DSV4_NVFP4 && false) {
      // QK consumes the packed record directly. Reformat NoPE in-place only
      // after QK, then reuse the proven normalized-E4M3 FP8 PV path.
      constexpr int FP4_TILES = D_NOPE / QUANT_TILE;
      constexpr int FP4_SUBGROUPS = QUANT_TILE / KV::FP4_SCALE_GROUP;
      static_assert(FP4_SUBGROUPS == 4);
      for (int item = threadIdx.x; item < DSV4_BI * FP4_TILES;
           item += DSV4_MATH_THREADS) {
        const int entry = item / FP4_TILES;
        const int tile = item % FP4_TILES;
        *reinterpret_cast<uint32_t*>(
            sm_kv_sc + (size_t)entry * SCALE_BYTES_PER_TOKEN + tile * 4) =
            *reinterpret_cast<const uint32_t*>(
                sm_kv_fp8 + (size_t)entry * KV_SMEM_STRIDE +
                KV::FP4_DATA_BYTES + tile * 4);
      }
      bar_sync_t<3, DSV4_MATH_THREADS>();

      const int expand_entry = threadIdx.x >> 2;
      const int expand_subgroup = threadIdx.x & 3;
#pragma unroll
      for (int tile = FP4_TILES - 1; tile >= 0; --tile) {
        uint8_t* record =
            sm_kv_fp8 + (size_t)expand_entry * KV_SMEM_STRIDE;
        const uint64_t packed = *reinterpret_cast<const uint64_t*>(
            record + tile * (QUANT_TILE / 2) + expand_subgroup * 8);
        const uint8_t source_scale_bits =
            sm_kv_sc[(size_t)expand_entry * SCALE_BYTES_PER_TOKEN +
                     tile * 4 + expand_subgroup];
        const float source_scale = decode_dsv4_e4m3x2_to_float(
                                       source_scale_bits, source_scale_bits)
                                       .x;
        float max_source_scale = source_scale;
        max_source_scale = fmaxf(
            max_source_scale,
            __shfl_xor_sync(0xffffffffU, max_source_scale, 1));
        max_source_scale = fmaxf(
            max_source_scale,
            __shfl_xor_sync(0xffffffffU, max_source_scale, 2));

        float shared_scale = 1.f;
        if (max_source_scale > 0.f) {
          const float required = max_source_scale * 6.f * FP8_MAX_INV;
          uint32_t bits = __float_as_uint(required);
          if (bits & 0x007fffffU) {
            bits = (bits + 0x00800000U) & 0x7f800000U;
          }
          shared_scale = __uint_as_float(bits);
        }
        const float multiplier = source_scale / shared_scale;
        // All four lanes in every record group retain their source fragment
        // and scale across this barrier before tile 0 expands in-place.
        bar_sync_t<3, DSV4_MATH_THREADS>();
        if (expand_subgroup == 0) {
          reinterpret_cast<float*>(
              sm_kv_sc +
              (size_t)expand_entry * SCALE_BYTES_PER_TOKEN)[tile] =
              shared_scale;
        }
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
          const uint8_t packed0 = (packed >> (pair * 16)) & 0xffU;
          const uint8_t packed1 = (packed >> (pair * 16 + 8)) & 0xffU;
          reinterpret_cast<uint32_t*>(
              record + tile * QUANT_TILE + expand_subgroup * 16)[pair] =
              flashinfer::math::fp32_vec_to_e4m3(
                  decode_dsv4_e2m1_to_float(packed0 & 0xfU) * multiplier,
                  decode_dsv4_e2m1_to_float(packed0 >> 4) * multiplier,
                  decode_dsv4_e2m1_to_float(packed1 & 0xfU) * multiplier,
                  decode_dsv4_e2m1_to_float(packed1 >> 4) * multiplier);
        }
        bar_sync_t<3, DSV4_MATH_THREADS>();
      }
    }

    // ── Stage 3 NoPE ──────────────────────────────────────────
    if constexpr (MT == ModelType::DSV4_NVFP4) {
#if 0
      // A fixed power-of-two scale keeps P*V_scale representable as E4M3
      // without a per-head reduction. Each adjacent warp pair shares one
      // 16x64 effective-weight tile and consumes raw E2M1 V directly.
      constexpr float FIXED_WEIGHT_MULTIPLIER = 64.f;
      constexpr uint8_t FIXED_WEIGHT_UE8M0 = 0x79;  // 2^-6
      constexpr int W_MIXED_STRIDE = DSV4_BI;
      constexpr int W_MIXED_GROUP_BYTES = HPB * W_MIXED_STRIDE;
      const int output_group = warp_id >> 1;
      const int group_tid = (warp_id & 1) * 32 + lane;
      const int pair_barrier = 5 + output_group;
      uint8_t* weight_records =
          output_group < 3
              ? sm.kv_sc(0) + output_group * W_MIXED_GROUP_BYTES
              : sm.w_fp8(0);

#pragma unroll
      for (int vc = 0; vc < N_V_CHUNKS; ++vc) {
        const int source_scale_group = vc * 4 + output_group;
        for (int item = group_tid; item < W_MIXED_GROUP_BYTES;
             item += 64) {
          const int h = item / W_MIXED_STRIDE;
          const int cand = item % W_MIXED_STRIDE;
          const uint8_t source_scale_bits =
              sm_kv_fp8[(size_t)cand * KV_SMEM_STRIDE +
                         KV::FP4_DATA_BYTES + source_scale_group];
          const float source_scale =
              decode_dsv4_e4m3x2_to_float(
                  source_scale_bits, source_scale_bits).x;
          const float p = __bfloat162float(sm_p_full[h][cand]);
          const __nv_fp8_e4m3 quantized(
              p * source_scale * FIXED_WEIGHT_MULTIPLIER);
          weight_records[item] = quantized.__x;
        }
        decode_dsv4_pair_barrier(pair_barrier);

        const int dim = vc * V_CHUNK + warp_id * 8;
        float xv[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int kstep = 0; kstep < XV_KSTEPS; ++kstep) {
          uint32_t a0, a1, a2, a3, b0, b1;
          ldmatrix_load_A_fp8(
              a0, a1, a2, a3, weight_records + kstep * 32,
              W_MIXED_STRIDE, lane);
          d2_load_b_mixed_fp4<KV_SMEM_STRIDE>(
              b0, b1, sm_kv_fp8, kstep * 32, dim, lane);
          const MmaNvfp4Result r =
              mma_mixed_fp8_fp4_m16n8k32(
                  a0, a1, a2, a3, b0, b1,
                  xv[0], xv[1], xv[2], xv[3],
                  FIXED_WEIGHT_UE8M0);
          xv[0] = r.d0;
          xv[1] = r.d1;
          xv[2] = r.d2;
          xv[3] = r.d3;
        }
        acc_nope[vc][0][0] += xv[0];
        acc_nope[vc][0][1] += xv[1];
        acc_nope[vc][0][2] += xv[2];
        acc_nope[vc][0][3] += xv[3];
        decode_dsv4_pair_barrier(pair_barrier);
      }

#endif
      // Quantize P once with a fixed safe E4M3 scale, then reconstruct only
      // the compact V fragments consumed by each FP8 MMA. A fixed scale
      // removes the per-head atomic-max reduction and its two block barriers.
      constexpr float P_MULTIPLIER = 256.f;
      constexpr float P_SCALE = 1.f / P_MULTIPLIER;
      const int warp_first_cand_xv =
          warp_id * DSV4_ENTRIES_PER_WARP;
      uint8_t* sm_p_fp8 = sm.w_fp8(0);
#pragma unroll
      for (int nt = 0; nt < DSV4_QK_N_TILES; ++nt) {
        const int cand0 =
            warp_first_cand_xv + nt * 8 + tid * 2;
        const int cand1 = cand0 + 1;
        const __nv_fp8_e4m3 p00(
            fmaxf(FP8_MIN, fminf(FP8_MAX,
                                 w_pre[nt][0] * P_MULTIPLIER)));
        const __nv_fp8_e4m3 p01(
            fmaxf(FP8_MIN, fminf(FP8_MAX,
                                 w_pre[nt][1] * P_MULTIPLIER)));
        const __nv_fp8_e4m3 p10(
            fmaxf(FP8_MIN, fminf(FP8_MAX,
                                 w_pre[nt][2] * P_MULTIPLIER)));
        const __nv_fp8_e4m3 p11(
            fmaxf(FP8_MIN, fminf(FP8_MAX,
                                 w_pre[nt][3] * P_MULTIPLIER)));
        sm_p_fp8[(size_t)gid * W_FP8_STRIDE + cand0] = p00.__x;
        sm_p_fp8[(size_t)gid * W_FP8_STRIDE + cand1] = p01.__x;
        sm_p_fp8[(size_t)(gid + 8) * W_FP8_STRIDE + cand0] = p10.__x;
        sm_p_fp8[(size_t)(gid + 8) * W_FP8_STRIDE + cand1] = p11.__x;
      }
      bar_sync_t<3, DSV4_MATH_THREADS>();

#pragma unroll
      for (int vc = 0; vc < N_V_CHUNKS; ++vc) {
        const int dim = vc * V_CHUNK + warp_id * 8;
        float xv[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int kstep = 0; kstep < XV_KSTEPS; ++kstep) {
          uint32_t a0, a1, a2, a3, b0, b1;
          ldmatrix_load_A_fp8(
              a0, a1, a2, a3, sm_p_fp8 + kstep * 32,
              W_FP8_STRIDE, lane);
          d2_load_b_dequant_fp8<KV_SMEM_STRIDE>(
              b0, b1, sm_kv_fp8, kstep * 32, dim, lane);
          const MmaFp8Result r = mma_fp8_m16n8k32(
              a0, a1, a2, a3, b0, b1,
              xv[0], xv[1], xv[2], xv[3]);
          xv[0] = r.d0;
          xv[1] = r.d1;
          xv[2] = r.d2;
          xv[3] = r.d3;
        }
        acc_nope[vc][0][0] += xv[0] * P_SCALE;
        acc_nope[vc][0][1] += xv[1] * P_SCALE;
        acc_nope[vc][0][2] += xv[2] * P_SCALE;
        acc_nope[vc][0][3] += xv[3] * P_SCALE;
      }

#if 0
      // QK has already consumed the original per-candidate scales. Reblock V
      // in place into the scale_vec::4X orientation required by PV, then use
      // one packed K64 tensor-core operation per eight output dimensions.
      decode_dsv4_reblock_v_for_nvfp4_pv<KV_SMEM_STRIDE>(
          sm_kv_fp8, warp_id, lane);
      bar_sync_t<3, DSV4_MATH_THREADS>();

      constexpr int P_NVFP4_STRIDE = 48;
      constexpr int P_DATA_BYTES = DSV4_BI / 2;
      constexpr int P_SCALE_OFFSET = P_DATA_BYTES;
      uint8_t* sm_p_nvfp4 = sm.w_fp8(0);
      for (int item = threadIdx.x; item < HPB * 4;
           item += DSV4_MATH_THREADS) {
        const int head = item >> 2;
        const int candidate_block = item & 3;
        float amax = 0.f;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
          amax = fmaxf(
              amax,
              fabsf(__bfloat162float(
                  sm_p_full[head][candidate_block * 16 + j])));
        }
        const __nv_fp8_e4m3 p_scale(
            fmaxf(amax * (1.f / 6.f), 0.001953125f));
        const float rounded_scale = static_cast<float>(p_scale);
        const float inv_scale =
            rounded_scale != 0.f ? 1.f / rounded_scale : 0.f;
        uint8_t* p_record =
            sm_p_nvfp4 + (size_t)head * P_NVFP4_STRIDE;
        p_record[P_SCALE_OFFSET + candidate_block] = p_scale.__x;
#pragma unroll
        for (int j = 0; j < 16; j += 2) {
          p_record[candidate_block * 8 + j / 2] =
              decode_dsv4_fp32x2_to_e2m1(
                  __bfloat162float(
                      sm_p_full[head][candidate_block * 16 + j]) *
                      inv_scale,
                  __bfloat162float(
                      sm_p_full[head][candidate_block * 16 + j + 1]) *
                      inv_scale);
        }
      }
      bar_sync_t<3, DSV4_MATH_THREADS>();

      const int p_head = gid + (lane & 1) * 8;
      const uint32_t sfa = *reinterpret_cast<const uint32_t*>(
          sm_p_nvfp4 + (size_t)p_head * P_NVFP4_STRIDE +
          P_SCALE_OFFSET);
      uint32_t a0, a1, a2, a3;
      ldmatrix_load_A_nvfp4(a0, a1, a2, a3, sm_p_nvfp4,
                            P_NVFP4_STRIDE, lane);
#pragma unroll
      for (int vc = 0; vc < N_V_CHUNKS; ++vc) {
        const int dim = vc * V_CHUNK + warp_id * 8;
        uint32_t b0, b1;
        d2_load_b_nvfp4<KV_SMEM_STRIDE>(
            b0, b1, sm_kv_fp8, dim, lane);
        const uint32_t sfb =
            decode_dsv4_load_pv_scales_nvfp4<KV_SMEM_STRIDE>(
                sm_kv_fp8, dim + gid);
        const MmaNvfp4Result r = mma_nvfp4_m16n8k64(
            a0, a1, a2, a3, b0, b1,
            0.f, 0.f, 0.f, 0.f, sfa, sfb);
        acc_nope[vc][0][0] += r.d0;
        acc_nope[vc][0][1] += r.d1;
        acc_nope[vc][0][2] += r.d2;
        acc_nope[vc][0][3] += r.d3;
      }
#endif

#if 0
      // Fold each candidate's 16-value cache scale into the attention
      // weights, retain those weights at E4M3 precision, and consume raw E2M1
      // V values with mixed mxf8f6f4 MMA. Each adjacent warp pair owns one
      // 16-output-dimension source-scale group and one named barrier.
      constexpr int W_MIXED_STRIDE = DSV4_BI;
      constexpr int W_MIXED_GROUP_BYTES = HPB * W_MIXED_STRIDE;
      const int output_group = warp_id >> 1;
      const int group_tid = (warp_id & 1) * 32 + lane;
      const int pair_barrier = 5 + output_group;
      uint8_t* weight_records =
          output_group < 3
              ? sm.kv_sc(0) + output_group * W_MIXED_GROUP_BYTES
              : sm.w_fp8(0);
      uint8_t* weight_scales =
          reinterpret_cast<uint8_t*>(sm.w_head_sc()) + output_group * HPB * 2;

#pragma unroll
      for (int vc = 0; vc < N_V_CHUNKS; ++vc) {
        const int source_scale_group = vc * 4 + output_group;

        // One lane computes each (head, K32) power-of-two weight scale.
        if (group_tid < HPB * 2) {
          const int scale_head = group_tid >> 1;
          const int kstep = group_tid & 1;
          float amax = 0.f;
#pragma unroll
          for (int j = 0; j < 32; ++j) {
            const int cand = kstep * 32 + j;
            const uint8_t source_scale_bits =
                sm_kv_fp8[(size_t)cand * KV_SMEM_STRIDE +
                           KV::FP4_DATA_BYTES + source_scale_group];
            const float source_scale =
                decode_dsv4_e4m3x2_to_float(source_scale_bits,
                                            source_scale_bits).x;
            const float p = __bfloat162float(sm_p_full[scale_head][cand]);
            amax = fmaxf(amax, fabsf(p * source_scale));
          }
          float weight_scale = 1.f;
          if (amax > 0.f) {
            uint32_t bits = __float_as_uint(amax * FP8_MAX_INV);
            if (bits & 0x007fffffU) {
              bits = (bits + 0x00800000U) & 0x7f800000U;
            }
            weight_scale = __uint_as_float(bits);
          }
          weight_scales[scale_head * 2 + kstep] =
              fp32_to_ue8m0(weight_scale);
        }
        decode_dsv4_pair_barrier(pair_barrier);

        // Quantize P*V_scale to E4M3. The cache's raw E2M1 V operand then has
        // a unit UE8M0 scale in the MMA.
        for (int item = group_tid; item < W_MIXED_GROUP_BYTES; item += 64) {
          const int h = item / W_MIXED_STRIDE;
          const int cand = item % W_MIXED_STRIDE;
          const uint8_t source_scale_bits =
              sm_kv_fp8[(size_t)cand * KV_SMEM_STRIDE +
                         KV::FP4_DATA_BYTES + source_scale_group];
          const float source_scale =
              decode_dsv4_e4m3x2_to_float(source_scale_bits,
                                          source_scale_bits).x;
          const float p = __bfloat162float(sm_p_full[h][cand]);
          const float inv_scale =
              1.f / ue8m0_to_fp32(weight_scales[h * 2 + (cand >> 5)]);
          __nv_fp8_e4m3 quantized(p * source_scale * inv_scale);
          weight_records[item] = quantized.__x;
        }
        decode_dsv4_pair_barrier(pair_barrier);

        const int dim = vc * V_CHUNK + warp_id * 8;
        const int weight_head = gid + (lane & 1) * 8;
        float xv[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int kstep = 0; kstep < XV_KSTEPS; ++kstep) {
          uint32_t a0, a1, a2, a3, b0, b1;
          ldmatrix_load_A_fp8(a0, a1, a2, a3,
                              weight_records + kstep * 32,
                              W_MIXED_STRIDE, lane);
          d2_load_b_mixed_fp4<KV_SMEM_STRIDE>(
              b0, b1, sm_kv_fp8, kstep * 32, dim, lane);
          const uint8_t sfa =
              weight_scales[weight_head * 2 + kstep];
          const MmaNvfp4Result r = mma_mixed_fp8_fp4_m16n8k32(
              a0, a1, a2, a3, b0, b1, xv[0], xv[1], xv[2], xv[3], sfa);
          xv[0] = r.d0;
          xv[1] = r.d1;
          xv[2] = r.d2;
          xv[3] = r.d3;
        }
        acc_nope[vc][0][0] += xv[0];
        acc_nope[vc][0][1] += xv[1];
        acc_nope[vc][0][2] += xv[2];
        acc_nope[vc][0][3] += xv[3];
        decode_dsv4_pair_barrier(pair_barrier);
      }
#endif
    }
#if 0
    if constexpr (MT == ModelType::DSV4_NVFP4 && false) {
      // Four output-dimension scale groups run concurrently. Each two-warp
      // group quantizes the full 16x64 effective-weight tile
      //   P[h,k] * V_scale[k,dim_group]
      // to NVFP4, then its two warps consume the tile for 16 output dims.
      // V's source scale is folded into P, so the raw E2M1 V operand uses an
      // all-ones UE4M3 scale vector.
      constexpr int W_NVFP4_STRIDE = 48;
      constexpr int W_NVFP4_DATA_BYTES = DSV4_BI / 2;
      constexpr int W_NVFP4_SCALE_OFFSET = W_NVFP4_DATA_BYTES;
      constexpr uint32_t NVFP4_ONE_SCALES = 0x38383838U;
      const int output_group = warp_id >> 1;
      const int group_tid = (warp_id & 1) * 32 + lane;
      uint8_t* all_weight_records = sm.kv_sc(0);
      uint8_t* weight_records =
          all_weight_records + output_group * HPB * W_NVFP4_STRIDE;

#pragma unroll
      for (int vc = 0; vc < N_V_CHUNKS; ++vc) {
        const int source_scale_group = vc * 4 + output_group;

        // One thread computes each (head, candidate-block-of-16) scale.
        const int scale_item = group_tid;
        const int scale_head = scale_item >> 2;
        const int candidate_block = scale_item & 3;
        float amax = 0.f;
#pragma unroll
        for (int j = 0; j < 16; j += 2) {
          const int cand0 = candidate_block * 16 + j;
          const int cand1 = cand0 + 1;
          const uint8_t* kv0 =
              sm_kv_fp8 + (size_t)cand0 * KV_SMEM_STRIDE;
          const uint8_t* kv1 =
              sm_kv_fp8 + (size_t)cand1 * KV_SMEM_STRIDE;
          const float2 source_scales = decode_dsv4_e4m3x2_to_float(
              kv0[KV::FP4_DATA_BYTES + source_scale_group],
              kv1[KV::FP4_DATA_BYTES + source_scale_group]);
          const float p0 = __bfloat162float(sm_p_full[scale_head][cand0]);
          const float p1 = __bfloat162float(sm_p_full[scale_head][cand1]);
          amax = fmaxf(amax, fabsf(p0 * source_scales.x));
          amax = fmaxf(amax, fabsf(p1 * source_scales.y));
        }
        __nv_fp8_e4m3 weight_scale(
            fmaxf(amax * (1.f / 6.f), 0.001953125f));
        weight_records[(size_t)scale_head * W_NVFP4_STRIDE +
                       W_NVFP4_SCALE_OFFSET + candidate_block] =
            weight_scale.__x;
        bar_sync_t<3, DSV4_MATH_THREADS>();

        // Pack two adjacent candidates into each E2M1 byte.
        for (int byte_item = group_tid;
             byte_item < HPB * W_NVFP4_DATA_BYTES; byte_item += 64) {
          const int h = byte_item / W_NVFP4_DATA_BYTES;
          const int byte_col = byte_item % W_NVFP4_DATA_BYTES;
          const int cand0 = byte_col * 2;
          const int cand1 = cand0 + 1;
          const int cb = cand0 >> 4;
          uint8_t* weight_row =
              weight_records + (size_t)h * W_NVFP4_STRIDE;
          const uint8_t scale_bits =
              weight_row[W_NVFP4_SCALE_OFFSET + cb];
          const float rounded_scale =
              decode_dsv4_e4m3x2_to_float(scale_bits, scale_bits).x;
          const float inv_scale =
              rounded_scale != 0.f ? 1.f / rounded_scale : 0.f;
          const uint8_t* kv0 =
              sm_kv_fp8 + (size_t)cand0 * KV_SMEM_STRIDE;
          const uint8_t* kv1 =
              sm_kv_fp8 + (size_t)cand1 * KV_SMEM_STRIDE;
          const float2 source_scales = decode_dsv4_e4m3x2_to_float(
              kv0[KV::FP4_DATA_BYTES + source_scale_group],
              kv1[KV::FP4_DATA_BYTES + source_scale_group]);
          const float p0 = __bfloat162float(sm_p_full[h][cand0]);
          const float p1 = __bfloat162float(sm_p_full[h][cand1]);
          weight_row[byte_col] = decode_dsv4_fp32x2_to_e2m1(
              p0 * source_scales.x * inv_scale,
              p1 * source_scales.y * inv_scale);
        }
        bar_sync_t<3, DSV4_MATH_THREADS>();

        const int dim = vc * V_CHUNK + warp_id * 8;
        const int weight_head = gid + (lane & 1) * 8;
        const uint8_t* weight_row =
            weight_records + (size_t)weight_head * W_NVFP4_STRIDE;
        const uint32_t sfa = *reinterpret_cast<const uint32_t*>(
            weight_row + W_NVFP4_SCALE_OFFSET);
        uint32_t a0, a1, a2, a3, b0, b1;
        ldmatrix_load_A_nvfp4(a0, a1, a2, a3, weight_records,
                              W_NVFP4_STRIDE, lane);
        uint8_t* v_tile = sm.w_fp8(0) + warp_id * 256;
        for (int item = lane; item < 8 * 32; item += 32) {
          const int out_dim = item / 32;
          const int k_byte = item % 32;
          const int cand0 = k_byte * 2;
          const int cand1 = cand0 + 1;
          const int d = dim + out_dim;
          const int source_byte = d >> 1;
          const int source_shift = (d & 1) * 4;
          const uint8_t lo =
              (sm_kv_fp8[(size_t)cand0 * KV_SMEM_STRIDE + source_byte] >>
               source_shift) &
              0xFU;
          const uint8_t hi =
              (sm_kv_fp8[(size_t)cand1 * KV_SMEM_STRIDE + source_byte] >>
               source_shift) &
              0xFU;
          v_tile[out_dim * 32 + k_byte] = lo | (hi << 4);
        }
        bar_sync_t<3, DSV4_MATH_THREADS>();
        ldmatrix_load_B_nvfp4(b0, b1, v_tile, 32, lane);
        const MmaNvfp4Result r = mma_nvfp4_m16n8k64(
            a0, a1, a2, a3, b0, b1, acc_nvfp4[vc].d0,
            acc_nvfp4[vc].d1, acc_nvfp4[vc].d2, acc_nvfp4[vc].d3, sfa,
            NVFP4_ONE_SCALES);
        acc_nvfp4[vc] = r;
      }
    } else {
#endif
    if constexpr (MT != ModelType::DSV4_NVFP4) {
''',
)

replace(
    decode_path,
    "    }\n"
    "    bar_sync_t<3, DSV4_MATH_THREADS>();\n"
    "    for (int i = threadIdx.x; i < N_V_CHUNKS * HPB; "
    "i += DSV4_MATH_THREADS) {\n"
    "      sm.w_head_sc()[i] = fmaxf(sm.w_head_sc()[i], 1e-10f) / FP8_MAX;\n",
    "    }\n"
    "    if constexpr (MT != ModelType::DSV4_NVFP4) {\n"
    "    bar_sync_t<3, DSV4_MATH_THREADS>();\n"
    "    for (int i = threadIdx.x; i < N_V_CHUNKS * HPB; "
    "i += DSV4_MATH_THREADS) {\n"
    "      sm.w_head_sc()[i] = fmaxf(sm.w_head_sc()[i], 1e-10f) / FP8_MAX;\n",
)

replace(
    decode_path,
    "        acc_nope[vc][nt][3] += xv[3] * sc1;\n"
    "      }\n"
    "    }\n\n"
    "    // ── Stage 3 RoPE bf16 ─────────────────────────────────────\n",
    "        acc_nope[vc][nt][3] += xv[3] * sc1;\n"
    "      }\n"
    "    }\n"
    "    }\n\n"
    "    // ── Stage 3 RoPE bf16 ─────────────────────────────────────\n",
)

replace(
    decode_path,
    "  float acc_nope[N_V_CHUNKS][NT_PER_WARP_XV][4] = {0};\n"
    "  float acc_rope[ROPE_N_TILES][4] = {0};",
    "  float acc_nope[N_V_CHUNKS][NT_PER_WARP_XV][4] = {0};\n"
    "  MmaNvfp4Result acc_nvfp4[N_V_CHUNKS] = {};\n"
    "  float acc_rope[ROPE_N_TILES][4] = {0};",
)
decode_launcher_path = "data/csrc/sparse_mla_sm120_decode_dsv4.cu"
replace(
    decode_launcher_path,
    "  constexpr int N_V_CHUNKS_LAUNCH = "
    "KV::D_NOPE / KV::QUANT_TILE;  // 7\n"
    "  constexpr int DYN_SMEM_BYTES =\n",
    "  constexpr int N_V_CHUNKS_LAUNCH = "
    "KV::D_NOPE / KV::QUANT_TILE;  // 7\n"
    "  constexpr int KV_BUF_COUNT_LAUNCH = "
    "DSV4_KV_BUF_COUNT_FOR<MT>;\n"
    "  constexpr int DYN_SMEM_BYTES =\n",
)
replace(
    decode_launcher_path,
    "      + HPB * KV::Q_NOPE_STRIDE                                       "
    "// sm_q_fp8\n",
    "      + (MT == ModelType::DSV4_NVFP4 ? 2 * HPB * 288                  "
    "// two compact Q terms\n"
    "                                         : HPB * KV::Q_NOPE_STRIDE)   "
    "// native FP8 Q\n",
)
replace(
    decode_launcher_path,
    "      + DSV4_KV_BUF_COUNT * DSV4_BI * "
    "KV::KV_SMEM_STRIDE              // sm_kv_fp8 ×2\n",
    "      + KV_BUF_COUNT_LAUNCH * DSV4_BI * "
    "DSV4_KV_SMEM_STRIDE_FOR<MT>      // compact: one inline record buf\n",
)
replace(
    decode_launcher_path,
    "      + DSV4_KV_BUF_COUNT * DSV4_BI * "
    "KV::SCALE_BYTES_PER_TOKEN       // sm_kv_sc ×2\n",
    "      + (MT == ModelType::DSV4_NVFP4\n"
    "             ? 0\n"
    "             : KV_BUF_COUNT_LAUNCH * DSV4_BI * "
    "KV::SCALE_BYTES_PER_TOKEN)       // compact: scales inline\n",
)
replace(
    decode_launcher_path,
    "      + DSV4_KV_BUF_COUNT * DSV4_BI * KV::D_ROPE * "
    "(int)sizeof(bf16)  // sm_kv_rope ×2\n",
    "      + (MT == ModelType::DSV4_NVFP4\n"
    "             ? 0\n"
    "             : KV_BUF_COUNT_LAUNCH * DSV4_BI * KV::D_ROPE * "
    "(int)sizeof(bf16))  // compact: RoPE inline\n",
)
replace(
    decode_launcher_path,
    "      + 4 * (int)sizeof(uint64_t)                                     "
    "// mbar_full+empty\n",
    "      + 2 * KV_BUF_COUNT_LAUNCH * (int)sizeof(uint64_t)               "
    "// mbar_full+empty\n",
)
replace(
    decode_launcher_path,
    "      + 2 * HPB * (DSV4_BI + 16);                                     "
    "// sm_w_fp8 ×2 (vc parity)\n",
    "      + 2 * HPB * (DSV4_BI + 16);                              "
    "// both P scratch buffers\n",
)
replace(
    decode_launcher_path,
    "  if (mt != ModelType::DSV4 || page_block_size != 64) return false;",
    "  if ((mt != ModelType::DSV4 && mt != ModelType::DSV4_NVFP4) ||\n"
    "      page_block_size != 64)\n"
    "    return false;",
)
replace(
    decode_launcher_path,
    "  dim3 block1(DSV4_BLOCK_THREADS);\n",
    "  dim3 block1(DSV4_BLOCK_THREADS_FOR<MT>);\n",
)
replace(
    decode_launcher_path,
    r'''#define DSV4_DISPATCH(H, K)                                                                 \
  if (num_heads == (H) && topk == (K)) {                                                    \
    return launch_decode_dsv4_impl<ModelType::DSV4, (H), (K), 64>(                          \
        Q, KV_cache, indices, mid_out, mid_lse, topk_length, output, out_lse, attn_sink,    \
        extra_KV_cache, extra_indices, extra_topk_length, extra_topk, pbs_extra,            \
        stride_extra_kv_block, num_tokens, num_splits, chunks_per_block_override, sm_scale, \
        stride_kv_block, stream);                                                           \
  }
''',
    r'''#define DSV4_DISPATCH(H, K)                                                                 \
  if (num_heads == (H) && topk == (K)) {                                                    \
    if (mt == ModelType::DSV4) {                                                            \
      return launch_decode_dsv4_impl<ModelType::DSV4, (H), (K), 64>(                        \
          Q, KV_cache, indices, mid_out, mid_lse, topk_length, output, out_lse, attn_sink,  \
          extra_KV_cache, extra_indices, extra_topk_length, extra_topk, pbs_extra,          \
          stride_extra_kv_block, num_tokens, num_splits, chunks_per_block_override,         \
          sm_scale, stride_kv_block, stream);                                               \
    }                                                                                       \
    return launch_decode_dsv4_impl<ModelType::DSV4_NVFP4, (H), (K), 64>(                   \
        Q, KV_cache, indices, mid_out, mid_lse, topk_length, output, out_lse, attn_sink,    \
        extra_KV_cache, extra_indices, extra_topk_length, extra_topk, pbs_extra,            \
        stride_extra_kv_block, num_tokens, num_splits, chunks_per_block_override, sm_scale, \
        stride_kv_block, stream);                                                           \
  }
''',
)

decode_binding_path = "data/csrc/sparse_mla_sm120_jit_binding.cu"
replace(
    decode_binding_path,
    "                              TensorView out_lse, int64_t num_splits, double sm_scale,\n"
    "                              Optional<TensorView> topk_length, Optional<TensorView> attn_sink,",
    "                              TensorView out_lse, int64_t num_splits, double sm_scale,\n"
    "                              int64_t model_type, Optional<TensorView> topk_length,\n"
    "                              Optional<TensorView> attn_sink,",
)
replace(
    decode_binding_path,
    r'''  ModelType mt = (d_qk == 512) ? ModelType::DSV4 : ModelType::DSV3_2;
  // Currently the kernel only supports DSV4.
  TVM_FFI_ICHECK_EQ(static_cast<int>(mt), static_cast<int>(ModelType::DSV4))
      << "decode-dsv4 currently DSV4-only";

  constexpr int BPT_DSV4 = 584;
  const PagedKVLayout kv_layout = parse_paged_kv_layout(kv_cache, BPT_DSV4, "kv_cache");
''',
    r'''  TVM_FFI_ICHECK_EQ(d_qk, 512) << "decode-dsv4 expects d_qk=512; got " << d_qk;
  const auto mt = static_cast<ModelType>(model_type);
  TVM_FFI_ICHECK(mt == ModelType::DSV4 || mt == ModelType::DSV4_NVFP4)
      << "decode-dsv4 expects model_type DSV4 or DSV4_NVFP4; got " << model_type;

  const int bytes_per_token = mt == ModelType::DSV4_NVFP4 ? 416 : 584;
  const PagedKVLayout kv_layout =
      parse_paged_kv_layout(kv_cache, bytes_per_token, "kv_cache");
''',
)
replace(
    decode_binding_path,
    '    const PagedKVLayout extra_layout = parse_paged_kv_layout(ekv, BPT_DSV4, "extra_kv_cache");',
    '    const PagedKVLayout extra_layout =\n'
    '        parse_paged_kv_layout(ekv, bytes_per_token, "extra_kv_cache");',
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

prefill_kernel_path = (
    "data/include/flashinfer/attention/sparse_mla_sm120/"
    "prefill_kernel.cuh"
)
replace(
    prefill_kernel_path,
    '#include "model/scale_convert.cuh"\n\n',
    r'''#include "model/scale_convert.cuh"

// Convert one adjacent E2M1 pair and its E4M3 block scale directly to the
// BF16 register fragment consumed by m16n8k16 QK. No expanded KV tile is
// materialized in shared memory.
__device__ __forceinline__ uint32_t
prefill_nvfp4_scaled_pair_to_bf16x2(uint8_t fp4_pair,
                                    uint8_t scale_bits) {
  const uint16_t scale_pair =
      static_cast<uint16_t>(scale_bits) |
      (static_cast<uint16_t>(scale_bits) << 8);
  uint32_t value_f16x2;
  uint32_t scale_f16x2;
  asm volatile(
      "{ .reg .b8 lo, hi;              \n"
      "  mov.b16 {lo, hi}, %2;         \n"
      "  cvt.rn.f16x2.e2m1x2 %0, lo;  \n"
      "  cvt.rn.f16x2.e4m3x2 %1, %3;  }\n"
      : "=r"(value_f16x2), "=r"(scale_f16x2)
      : "h"(static_cast<uint16_t>(fp4_pair)), "h"(scale_pair));
  const __half2 scaled = __hmul2(
      *reinterpret_cast<const __half2*>(&value_f16x2),
      *reinterpret_cast<const __half2*>(&scale_f16x2));
  const float lo = __low2float(scaled);
  const float hi = __high2float(scaled);
  uint32_t result;
  asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;"
               : "=r"(result)
               : "f"(hi), "f"(lo));
  return result;
}

// Build an ordinary E4M3 PV B fragment directly from packed candidate rows.
// The E2M1 values and per-token/per-D16 scales live only in registers.
template <int KV_STRIDE>
__device__ __forceinline__ void prefill_nvfp4_d2_load_b_fp8(
    uint32_t& b0, uint32_t& b1, const uint8_t* kv_smem, int cand_base,
    int dim, int lane) {
  using CompactKV = KVCacheTraits<ModelType::DSV4_NVFP4>;
  const int gid = lane >> 2;
  const int tid = lane & 3;
  const int d = dim + gid;
  const int byte_col = d >> 1;
  const int shift = (d & 1) * 4;
  const int scale_group = d / CompactKV::FP4_SCALE_GROUP;
  b0 = 0;
  b1 = 0;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int k0 = cand_base + tid * 4 + j;
    const int k1 = cand_base + 16 + tid * 4 + j;
    const uint8_t* row0 = kv_smem + (size_t)k0 * KV_STRIDE;
    const uint8_t* row1 = kv_smem + (size_t)k1 * KV_STRIDE;
    const uint8_t fp4_0 = (row0[byte_col] >> shift) & 0xfU;
    const uint8_t fp4_1 = (row1[byte_col] >> shift) & 0xfU;
    const uint16_t scale_pair =
        static_cast<uint16_t>(
            row0[CompactKV::FP4_DATA_BYTES + scale_group]) |
        (static_cast<uint16_t>(
             row1[CompactKV::FP4_DATA_BYTES + scale_group])
         << 8);
    uint32_t value_f16x2;
    uint32_t scale_f16x2;
    asm volatile(
        "{ .reg .b8 lo, hi;              \n"
        "  mov.b16 {lo, hi}, %2;         \n"
        "  cvt.rn.f16x2.e2m1x2 %0, lo;  \n"
        "  cvt.rn.f16x2.e4m3x2 %1, %3;  }\n"
        : "=r"(value_f16x2), "=r"(scale_f16x2)
        : "h"(static_cast<uint16_t>(fp4_0 | (fp4_1 << 4))),
          "h"(scale_pair));
    const __half2 scaled = __hmul2(
        *reinterpret_cast<const __half2*>(&value_f16x2),
        *reinterpret_cast<const __half2*>(&scale_f16x2));
    uint16_t fp8_pair;
    asm volatile("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;"
                 : "=h"(fp8_pair)
                 : "r"(*reinterpret_cast<const uint32_t*>(&scaled)));
    b0 |= static_cast<uint32_t>(fp8_pair & 0xffU) << (j * 8);
    b1 |= static_cast<uint32_t>(fp8_pair >> 8) << (j * 8);
  }
}

''',
    present="prefill_nvfp4_scaled_pair_to_bf16x2",
)
replace(
    prefill_kernel_path,
    "// ============================================================================\n"
    "// Sparse MLA Prefill Kernel — single-pass (no split-KV, no combine)\n",
    r'''__device__ __forceinline__ uint8_t
prefill_nvfp4_fp32x2_to_e2m1(float lo, float hi) {
  uint32_t packed;
  asm volatile(
      "{ .reg .b8 tmp;                           \n"
      "  cvt.rn.satfinite.e2m1x2.f32 tmp, %1, %2;\n"
      "  cvt.u32.u8 %0, tmp;                    }\n"
      : "=r"(packed)
      : "f"(hi), "f"(lo));
  return static_cast<uint8_t>(packed);
}

__device__ __forceinline__ float2
prefill_nvfp4_e2m1x2_to_float(uint8_t packed) {
  uint32_t fp16_pair;
  asm volatile(
      "{ .reg .b8 lo, hi;              \n"
      "  mov.b16 {lo, hi}, %1;         \n"
      "  cvt.rn.f16x2.e2m1x2 %0, lo;  }\n"
      : "=r"(fp16_pair)
      : "h"(static_cast<uint16_t>(packed)));
  const __half2 values =
      *reinterpret_cast<const __half2*>(&fp16_pair);
  return make_float2(__low2float(values), __high2float(values));
}

template <int MATH_THREADS>
__device__ __forceinline__ void prefill_nvfp4_quantize_q(
    uint8_t* q_records, bf16* q_rope, const bf16* q_base) {
  constexpr int Q_RECORD_STRIDE = 288;
  constexpr int Q_TERM_BYTES = HPB * Q_RECORD_STRIDE;
  constexpr int Q_GROUPS = 28;
  constexpr int Q_GROUP_SIZE = 16;
  constexpr int Q_PACKED_BYTES_PER_GROUP = Q_GROUP_SIZE / 2;
  constexpr int Q_DIM = 512;

  for (int i = threadIdx.x; i < HPB * D_ROPE;
       i += MATH_THREADS) {
    const int h = i / D_ROPE;
    const int d = i % D_ROPE;
    q_rope[i] = q_base[(size_t)h * Q_DIM + 448 + d];
  }
  for (int item = threadIdx.x; item < HPB * Q_GROUPS;
       item += MATH_THREADS) {
    const int h = item / Q_GROUPS;
    const int group = item % Q_GROUPS;
    float amax = 0.f;
#pragma unroll
    for (int j = 0; j < Q_GROUP_SIZE; ++j) {
      const float value = __bfloat162float(
          q_base[(size_t)h * Q_DIM + group * Q_GROUP_SIZE + j]);
      amax = fmaxf(amax, fabsf(value));
    }
    __nv_fp8_e4m3 scale0_fp8(amax * (1.f / 6.f));
    const float scale0 = static_cast<float>(scale0_fp8);
    const float inv_scale0 =
        (amax != 0.f && scale0 != 0.f) ? 1.f / scale0 : 0.f;
    uint8_t* q_record0 =
        q_records + (size_t)h * Q_RECORD_STRIDE;
    uint8_t* q_record1 = q_record0 + Q_TERM_BYTES;
    q_record0[256 + group] = scale0_fp8.__x;
    __nv_fp8_e4m3 scale1_fp8(scale0 * (1.f / 6.f));
    const float scale1 = static_cast<float>(scale1_fp8);
    const float inv_scale1 =
        scale1 != 0.f ? 1.f / scale1 : 0.f;
    q_record1[256 + group] = scale1_fp8.__x;
#pragma unroll
    for (int j = 0; j < Q_PACKED_BYTES_PER_GROUP; ++j) {
      const float value_lo = __bfloat162float(
          q_base[(size_t)h * Q_DIM +
                 group * Q_GROUP_SIZE + j * 2]);
      const float value_hi = __bfloat162float(
          q_base[(size_t)h * Q_DIM +
                 group * Q_GROUP_SIZE + j * 2 + 1]);
      const uint8_t packed0 = prefill_nvfp4_fp32x2_to_e2m1(
          value_lo * inv_scale0, value_hi * inv_scale0);
      q_record0[group * Q_PACKED_BYTES_PER_GROUP + j] = packed0;
      const float2 primary =
          prefill_nvfp4_e2m1x2_to_float(packed0);
      q_record1[group * Q_PACKED_BYTES_PER_GROUP + j] =
          prefill_nvfp4_fp32x2_to_e2m1(
              (value_lo - primary.x * scale0) * inv_scale1,
              (value_hi - primary.y * scale0) * inv_scale1);
    }
  }
  bar_sync_t<2, MATH_THREADS>();
}

__device__ __forceinline__ void prefill_nvfp4_ldmatrix_A(
    uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3,
    const uint8_t* smem_base, int stride, int lane) {
  const int row = (lane & 7) + ((lane >> 3) & 1) * 8;
  const int col = (lane >> 4) * 16;
  ldmatrix_x4(a0, a1, a2, a3,
              smem_base + row * stride + col);
}

__device__ __forceinline__ void prefill_nvfp4_ldmatrix_B(
    uint32_t& b0, uint32_t& b1, const uint8_t* smem_base,
    int stride, int lane) {
  const int row = lane & 7;
  const int col = ((lane >> 3) & 1) * 16;
  ldmatrix_x2(b0, b1, smem_base + row * stride + col);
}

__device__ __forceinline__ MmaFp8Result prefill_nvfp4_mma_m16n8k64(
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1, float c0, float c1, float c2,
    float c3, uint32_t scale_a, uint32_t scale_b) {
  MmaFp8Result r;
  asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X."
      "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},"
      "{%10,%11,%12,%13},{%14},{%15,%16},{%17},{%18,%19};\n"
      : "=f"(r.d0), "=f"(r.d1), "=f"(r.d2), "=f"(r.d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0),
        "r"(b1), "f"(c0), "f"(c1), "f"(c2), "f"(c3),
        "r"(scale_a), "n"(static_cast<uint16_t>(0)),
        "n"(static_cast<uint16_t>(0)), "r"(scale_b),
        "n"(static_cast<uint16_t>(0)),
        "n"(static_cast<uint16_t>(0)));
  return r;
}

// ============================================================================
// Sparse MLA Prefill Kernel — single-pass (no split-KV, no combine)
''',
    present="prefill_nvfp4_quantize_q",
)

mg_region_start = "__device__ __forceinline__ void prefill_mg_impl("
mg_region_end = "// Single-cache __global__ wrapper."
raw_compact_predicate = (
    "(MT == ModelType::DSV4_NVFP4 && CM == ComputeMode::BF16)"
)
replace_in_region(
    prefill_kernel_path,
    mg_region_start,
    mg_region_end,
    "io_bulk_gather_tile<MT, PAGE_BLOCK_SIZE, true>",
    "io_bulk_gather_tile<MT, PAGE_BLOCK_SIZE, true,\n"
    f"                                  {raw_compact_predicate}>",
    count=5,
    present="Raw compact MG/BF16 gather",
)
replace_in_region(
    prefill_kernel_path,
    mg_region_start,
    mg_region_end,
    "io_bulk_gather_tile<MT, PAGE_BLOCK_SIZE_EXTRA, true>",
    "io_bulk_gather_tile<MT, PAGE_BLOCK_SIZE_EXTRA, true,\n"
    f"                                  {raw_compact_predicate}>",
    count=2,
)
replace_in_region(
    prefill_kernel_path,
    mg_region_start,
    mg_region_end,
    "      // ── QK + softmax for both groups ────────────────────────\n",
    "      // Raw compact MG/BF16 gather: QK and PV consume the first\n"
    "      // 288 bytes of each 416-byte record without shared expansion.\n"
    "      // ── QK + softmax for both groups ────────────────────────\n",
)
replace_in_region(
    prefill_kernel_path,
    mg_region_start,
    mg_region_end,
    r'''      // Init per-group w_head_sc_all
      for (int i = threadIdx.x; i < MG_N_HG * CT::N_V_CHUNKS * HPB; i += MATH_THREADS)
        sm.w_head_sc_all()[i] = 0.f;
''',
    f'''      if constexpr (!{raw_compact_predicate}) {{
        // Init per-group w_head_sc_all
        for (int i = threadIdx.x;
             i < MG_N_HG * CT::N_V_CHUNKS * HPB;
             i += MATH_THREADS) {{
          sm.w_head_sc_all()[i] = 0.f;
        }}
      }}
''',
)
replace_in_region(
    prefill_kernel_path,
    mg_region_start,
    mg_region_end,
    r'''      if constexpr (CM == ComputeMode::BF16) {
        load_q_bf16_to_smem<MT, MATH_THREADS>(sm.q_nope_bf16(g), sm.q_rope() + g * HPB * D_ROPE,
                                              q_base_g);
      } else {
''',
    f'''      if constexpr ({raw_compact_predicate}) {{
        prefill_nvfp4_quantize_q<MATH_THREADS>(
            reinterpret_cast<uint8_t*>(sm.q_nope_bf16(g)),
            sm.q_rope() + g * HPB * D_ROPE, q_base_g);
      }} else if constexpr (CM == ComputeMode::BF16) {{
        load_q_bf16_to_smem<MT, MATH_THREADS>(
            sm.q_nope_bf16(g),
            sm.q_rope() + g * HPB * D_ROPE, q_base_g);
      }} else {{
''',
)

mg_fused_scale_load = r'''          float scale_f;
          if constexpr (KV::SCALE_IN_KV_SMEM) {
            scale_f = reinterpret_cast<const float*>(kv_gid_base + KV::D_NOPE)[blk];
          } else {
            scale_f = ue8m0_to_fp32(
                sm.kv_scale_buf(ti & 1)[(qk_nb + gid) * KV::SCALE_BYTES_PER_TOKEN + blk]);
          }
'''
mg_fused_scale_load_direct = r'''          float scale_f = 1.f;
          if constexpr (MT != ModelType::DSV4_NVFP4) {
            if constexpr (KV::SCALE_IN_KV_SMEM) {
              scale_f = reinterpret_cast<const float*>(
                  kv_gid_base + KV::D_NOPE)[blk];
            } else {
              scale_f = ue8m0_to_fp32(
                  sm.kv_scale_buf(ti & 1)[
                      (qk_nb + gid) * KV::SCALE_BYTES_PER_TOKEN + blk]);
            }
          }
'''
replace_in_region(
    prefill_kernel_path,
    mg_region_start,
    mg_region_end,
    mg_fused_scale_load,
    mg_fused_scale_load_direct,
)

mg_fallback_scale_load = r'''              float scale_f;
              if constexpr (KV::SCALE_IN_KV_SMEM) {
                scale_f = reinterpret_cast<const float*>(kv_gid_base + KV::D_NOPE)[blk];
              } else {
                scale_f = ue8m0_to_fp32(
                    sm.kv_scale_buf(ti & 1)[(qk_nb + gid) * KV::SCALE_BYTES_PER_TOKEN + blk]);
              }
'''
mg_fallback_scale_load_direct = r'''              float scale_f = 1.f;
              if constexpr (MT != ModelType::DSV4_NVFP4) {
                if constexpr (KV::SCALE_IN_KV_SMEM) {
                  scale_f = reinterpret_cast<const float*>(
                      kv_gid_base + KV::D_NOPE)[blk];
                } else {
                  scale_f = ue8m0_to_fp32(
                      sm.kv_scale_buf(ti & 1)[
                          (qk_nb + gid) * KV::SCALE_BYTES_PER_TOKEN + blk]);
                }
              }
'''
replace_in_region(
    prefill_kernel_path,
    mg_region_start,
    mg_region_end,
    mg_fallback_scale_load,
    mg_fallback_scale_load_direct,
)

mg_fused_k_fragment = r'''            uint16_t p0 = *reinterpret_cast<const uint16_t*>(kv_gid_base + ko + 2 * tid);
            uint16_t p1 = *reinterpret_cast<const uint16_t*>(kv_gid_base + ko + 2 * tid + 8);
            uint32_t f16x2_0, f16x2_1;
            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(f16x2_0) : "h"(p0));
            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(f16x2_1) : "h"(p1));
            __half2 h2_0 = *reinterpret_cast<__half2*>(&f16x2_0);
            __half2 h2_1 = *reinterpret_cast<__half2*>(&f16x2_1);
            float fk0 = __low2float(h2_0) * scale_f, fk1 = __high2float(h2_0) * scale_f;
            float fk2 = __low2float(h2_1) * scale_f, fk3 = __high2float(h2_1) * scale_f;
            uint32_t b0, b1;
            asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(b0) : "f"(fk1), "f"(fk0));
            asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(b1) : "f"(fk3), "f"(fk2));
'''
mg_fused_k_fragment_direct = r'''            uint32_t b0, b1;
            if constexpr (MT == ModelType::DSV4_NVFP4) {
              const uint8_t source_scale =
                  kv_gid_base[KV::FP4_DATA_BYTES + blk * 4 + ks];
              const int d0 = ko + 2 * tid;
              const int d1 = d0 + 8;
              b0 = prefill_nvfp4_scaled_pair_to_bf16x2(
                  kv_gid_base[d0 >> 1], source_scale);
              b1 = prefill_nvfp4_scaled_pair_to_bf16x2(
                  kv_gid_base[d1 >> 1], source_scale);
            } else {
              uint16_t p0 =
                  *reinterpret_cast<const uint16_t*>(
                      kv_gid_base + ko + 2 * tid);
              uint16_t p1 =
                  *reinterpret_cast<const uint16_t*>(
                      kv_gid_base + ko + 2 * tid + 8);
              uint32_t f16x2_0, f16x2_1;
              asm("cvt.rn.f16x2.e4m3x2 %0, %1;"
                  : "=r"(f16x2_0)
                  : "h"(p0));
              asm("cvt.rn.f16x2.e4m3x2 %0, %1;"
                  : "=r"(f16x2_1)
                  : "h"(p1));
              __half2 h2_0 = *reinterpret_cast<__half2*>(&f16x2_0);
              __half2 h2_1 = *reinterpret_cast<__half2*>(&f16x2_1);
              float fk0 = __low2float(h2_0) * scale_f;
              float fk1 = __high2float(h2_0) * scale_f;
              float fk2 = __low2float(h2_1) * scale_f;
              float fk3 = __high2float(h2_1) * scale_f;
              asm("cvt.rn.bf16x2.f32 %0, %1, %2;"
                  : "=r"(b0)
                  : "f"(fk1), "f"(fk0));
              asm("cvt.rn.bf16x2.f32 %0, %1, %2;"
                  : "=r"(b1)
                  : "f"(fk3), "f"(fk2));
            }
'''
replace_in_region(
    prefill_kernel_path,
    mg_region_start,
    mg_region_end,
    mg_fused_k_fragment,
    mg_fused_k_fragment_direct,
)

mg_fallback_k_fragment = mg_fused_k_fragment.replace(
    "\n            ", "\n                "
).replace("            uint16_t", "                uint16_t", 1)
mg_fallback_k_fragment_direct = mg_fused_k_fragment_direct.replace(
    "\n            ", "\n                "
).replace("            uint32_t", "                uint32_t", 1)
replace_in_region(
    prefill_kernel_path,
    mg_region_start,
    mg_region_end,
    mg_fallback_k_fragment,
    mg_fallback_k_fragment_direct,
)

mg_fused_qk_begin = r'''        float qk_grp[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};

#pragma unroll
        for (int blk = 0; blk < KV::NUM_SCALES; blk++) {
'''
mg_fused_qk_native = (
    r'''        float qk_grp[2][4] = {
            {0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};

'''
    f'''        if constexpr ({raw_compact_predicate}) {{
'''
    + r'''          constexpr int Q_NVFP4_STRIDE = 288;
          constexpr int Q_TERM_BYTES =
              HPB * Q_NVFP4_STRIDE;
#pragma unroll
          for (int blk = 0; blk < KV::NUM_SCALES; ++blk) {
            const uint8_t* kv_record =
                kv_warp_base +
                (size_t)gid * KV::KV_SMEM_STRIDE;
            const uint32_t sfb =
                *reinterpret_cast<const uint32_t*>(
                    kv_record + KV::FP4_DATA_BYTES + blk * 4);
            uint32_t b0, b1;
            prefill_nvfp4_ldmatrix_B(
                b0, b1, kv_warp_base + blk * 32,
                KV::KV_SMEM_STRIDE, lane);
#pragma unroll
            for (int term = 0; term < 2; ++term) {
#pragma unroll
              for (int g = 0; g < 2; ++g) {
                const uint8_t* q_group =
                    reinterpret_cast<const uint8_t*>(
                        sm.q_nope_bf16(g));
                const uint8_t* q_term =
                    q_group + term * Q_TERM_BYTES;
                const int q_head = gid + (lane & 1) * 8;
                const uint32_t sfa =
                    *reinterpret_cast<const uint32_t*>(
                        q_term +
                        (size_t)q_head * Q_NVFP4_STRIDE +
                        KV::FP4_DATA_BYTES + blk * 4);
                uint32_t a0, a1, a2, a3;
                prefill_nvfp4_ldmatrix_A(
                    a0, a1, a2, a3, q_term + blk * 32,
                    Q_NVFP4_STRIDE, lane);
                MmaFp8Result r = prefill_nvfp4_mma_m16n8k64(
                    a0, a1, a2, a3, b0, b1,
                    qk_grp[g][0], qk_grp[g][1],
                    qk_grp[g][2], qk_grp[g][3], sfa, sfb);
                qk_grp[g][0] = r.d0;
                qk_grp[g][1] = r.d1;
                qk_grp[g][2] = r.d2;
                qk_grp[g][3] = r.d3;
              }
            }
          }
        } else {
#pragma unroll
        for (int blk = 0; blk < KV::NUM_SCALES; blk++) {
'''
)
replace_in_region(
    prefill_kernel_path,
    mg_region_start,
    mg_region_end,
    mg_fused_qk_begin,
    mg_fused_qk_native,
)
replace_in_region(
    prefill_kernel_path,
    mg_region_start,
    mg_region_end,
    r'''          }
        }

#pragma unroll
        for (int g = 0; g < 2; g++) {
''',
    r'''          }
        }
        }

#pragma unroll
        for (int g = 0; g < 2; g++) {
''',
)

mg_vscale_cache = r'''#pragma unroll
      for (int vc = 0; vc < CT::N_V_CHUNKS; vc++) {
        if constexpr (KV::SCALE_IN_KV_SMEM) {
          vsc_cache[vc][0] = reinterpret_cast<const float*>(e0_base + KV::D_NOPE)[vc];
          vsc_cache[vc][1] = reinterpret_cast<const float*>(e1_base + KV::D_NOPE)[vc];
        } else {
          vsc_cache[vc][0] =
              ue8m0_to_fp32(sm.kv_scale_buf(ti & 1)[e0i * KV::SCALE_BYTES_PER_TOKEN + vc]);
          vsc_cache[vc][1] =
              ue8m0_to_fp32(sm.kv_scale_buf(ti & 1)[e1i * KV::SCALE_BYTES_PER_TOKEN + vc]);
        }
      }
'''
mg_vscale_cache_direct = (
    f'''      if constexpr (!{raw_compact_predicate}) {{\n'''
    + mg_vscale_cache
    + "      }\n"
)
replace_in_region(
    prefill_kernel_path,
    mg_region_start,
    mg_region_end,
    mg_vscale_cache,
    mg_vscale_cache_direct,
)

mg_vscale_max = r'''        // V-scale max for W quantization.
#pragma unroll
        for (int vc = 0; vc < CT::N_V_CHUNKS; vc++) {
          float vsc0 = vsc_cache[vc][0], vsc1 = vsc_cache[vc][1];
          float ws00 = w0 * vsc0, ws01 = w1 * vsc1;
          float ws10 = w2 * vsc0, ws11 = w3 * vsc1;
          atomicMax(
              reinterpret_cast<int*>(&sm.w_head_sc_all()[g * SMG::WSC_GRP_STRIDE + vc * HPB + gid]),
              __float_as_int(fmaxf(ws00, ws01)));
          atomicMax(reinterpret_cast<int*>(
                        &sm.w_head_sc_all()[g * SMG::WSC_GRP_STRIDE + vc * HPB + gid + 8]),
                    __float_as_int(fmaxf(ws10, ws11)));
        }
'''
mg_vscale_max_direct = (
    f'''        if constexpr ({raw_compact_predicate}) {{\n'''
    + r'''          // Compact P uses a fixed power-of-two scale because every
          // online-softmax weight is in [0, 1]. No reduction is required.
        } else {
'''
    + mg_vscale_max.replace("        ", "          ")
    + "        }\n"
)
replace_in_region(
    prefill_kernel_path,
    mg_region_start,
    mg_region_end,
    mg_vscale_max,
    mg_vscale_max_direct,
)

mg_normalize_wscale = r'''      bar_sync_t<2, MATH_THREADS>();

      // Normalize w_head_sc_all (both groups)
      for (int i = threadIdx.x; i < MG_N_HG * CT::N_V_CHUNKS * HPB; i += MATH_THREADS)
        sm.w_head_sc_all()[i] = fmaxf(sm.w_head_sc_all()[i], 1e-10f) / FP8_MAX;
      bar_sync_t<2, MATH_THREADS>();
'''
mg_normalize_wscale_direct = (
    f'''      if constexpr (!{raw_compact_predicate}) {{\n'''
    + r'''        bar_sync_t<2, MATH_THREADS>();
        // Normalize w_head_sc_all (both groups)
        for (int i = threadIdx.x;
             i < MG_N_HG * CT::N_V_CHUNKS * HPB;
             i += MATH_THREADS) {
          sm.w_head_sc_all()[i] =
              fmaxf(sm.w_head_sc_all()[i], 1e-10f) / FP8_MAX;
        }
        bar_sync_t<2, MATH_THREADS>();
      }
'''
)
replace_in_region(
    prefill_kernel_path,
    mg_region_start,
    mg_region_end,
    mg_normalize_wscale,
    mg_normalize_wscale_direct,
)

mg_xv_open = r'''      {
        if constexpr (KV::SCALE_FORMAT == ScaleFormat::ARBITRARY_FP32) {
'''
mg_xv_direct = (
    "      {\n"
    f"        if constexpr ({raw_compact_predicate}) {{\n"
    + r'''          // Map P into E4M3 with a fixed power-of-two multiplier.
          // This preserves weights down to roughly 7.6e-6 while avoiding
          // per-head reductions; V's exact scale stays in the B fragment.
          constexpr float P_MULTIPLIER = 256.f;
          constexpr float P_SCALE = 1.f / P_MULTIPLIER;
#pragma unroll
          for (int g = 0; g < MG_N_HG; ++g) {
            uint8_t* cur_p =
                sm.w_fp8() + g * SMG::WFP8_GRP_SIZE;
            const float w0 = w_grp[g][0];
            const float w1 = w_grp[g][1];
            const float w2 = w_grp[g][2];
            const float w3 = w_grp[g][3];
            const __nv_fp8_e4m3 p00(
                fmaxf(FP8_MIN, fminf(FP8_MAX, w0 * P_MULTIPLIER)));
            const __nv_fp8_e4m3 p01(
                fmaxf(FP8_MIN, fminf(FP8_MAX, w1 * P_MULTIPLIER)));
            const __nv_fp8_e4m3 p10(
                fmaxf(FP8_MIN, fminf(FP8_MAX, w2 * P_MULTIPLIER)));
            const __nv_fp8_e4m3 p11(
                fmaxf(FP8_MIN, fminf(FP8_MAX, w3 * P_MULTIPLIER)));
            int row0 = gid;
            int row1 = gid + 8;
            if constexpr (USE_WFP8_ROW_XOR) {
              row0 = wfp8_row_xor(row0);
              row1 = wfp8_row_xor(row1);
            }
            cur_p[row0 * (BI + 16) + e0i] = p00.__x;
            cur_p[row0 * (BI + 16) + e1i] = p01.__x;
            cur_p[row1 * (BI + 16) + e0i] = p10.__x;
            cur_p[row1 * (BI + 16) + e1i] = p11.__x;
          }
          bar_sync_t<2, MATH_THREADS>();

#pragma unroll
          for (int vc = 0; vc < CT::N_V_CHUNKS; ++vc) {
            if constexpr (MG_N_HG == 2) {
              uint8_t* cur_p0 = sm.w_fp8();
              uint8_t* cur_p1 =
                  sm.w_fp8() + SMG::WFP8_GRP_SIZE;
#pragma unroll
              for (int nt = 0; nt < CT::NT_PER_WARP_XV; ++nt) {
                const int ti_acc = vc * CT::NT_PER_WARP_XV + nt;
                const int dim =
                    vc * CT::V_CHUNK +
                    mwarp * (CT::NT_PER_WARP_XV * 8) + nt * 8;
                float xv0[4] = {0.f, 0.f, 0.f, 0.f};
                float xv1[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
                for (int kstep = 0; kstep < CT::XV_KSTEPS; ++kstep) {
                  const int ko = kstep * 32;
                  uint32_t b0, b1;
                  prefill_nvfp4_d2_load_b_fp8<KV::KV_SMEM_STRIDE>(
                      b0, b1, kv_smem, ko, dim, lane);
                  uint32_t a00, a01, a02, a03;
                  ldmatrix_load_A_fp8_layout<USE_WFP8_ROW_XOR>(
                      a00, a01, a02, a03, cur_p0 + ko,
                      BI + 16, lane);
                  uint32_t a10, a11, a12, a13;
                  ldmatrix_load_A_fp8_layout<USE_WFP8_ROW_XOR>(
                      a10, a11, a12, a13, cur_p1 + ko,
                      BI + 16, lane);
                  MmaFp8Result r0 = mma_fp8_m16n8k32(
                      a00, a01, a02, a03, b0, b1,
                      xv0[0], xv0[1], xv0[2], xv0[3]);
                  xv0[0] = r0.d0;
                  xv0[1] = r0.d1;
                  xv0[2] = r0.d2;
                  xv0[3] = r0.d3;
                  MmaFp8Result r1 = mma_fp8_m16n8k32(
                      a10, a11, a12, a13, b0, b1,
                      xv1[0], xv1[1], xv1[2], xv1[3]);
                  xv1[0] = r1.d0;
                  xv1[1] = r1.d1;
                  xv1[2] = r1.d2;
                  xv1[3] = r1.d3;
                }
                acc_o[0][ti_acc][0] += xv0[0] * P_SCALE;
                acc_o[0][ti_acc][1] += xv0[1] * P_SCALE;
                acc_o[0][ti_acc][2] += xv0[2] * P_SCALE;
                acc_o[0][ti_acc][3] += xv0[3] * P_SCALE;
                acc_o[1][ti_acc][0] += xv1[0] * P_SCALE;
                acc_o[1][ti_acc][1] += xv1[1] * P_SCALE;
                acc_o[1][ti_acc][2] += xv1[2] * P_SCALE;
                acc_o[1][ti_acc][3] += xv1[3] * P_SCALE;
              }
            } else {
#pragma unroll
              for (int g = 0; g < MG_N_HG; ++g) {
                uint8_t* cur_p =
                    sm.w_fp8() + g * SMG::WFP8_GRP_SIZE;
#pragma unroll
                for (int nt = 0; nt < CT::NT_PER_WARP_XV; ++nt) {
                  const int ti_acc = vc * CT::NT_PER_WARP_XV + nt;
                  const int dim =
                      vc * CT::V_CHUNK +
                      mwarp * (CT::NT_PER_WARP_XV * 8) + nt * 8;
                  float xv[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
                  for (int kstep = 0;
                       kstep < CT::XV_KSTEPS; ++kstep) {
                    const int ko = kstep * 32;
                    uint32_t a0, a1, a2, a3, b0, b1;
                    ldmatrix_load_A_fp8_layout<USE_WFP8_ROW_XOR>(
                        a0, a1, a2, a3, cur_p + ko,
                        BI + 16, lane);
                    prefill_nvfp4_d2_load_b_fp8<KV::KV_SMEM_STRIDE>(
                        b0, b1, kv_smem, ko, dim, lane);
                    MmaFp8Result r = mma_fp8_m16n8k32(
                        a0, a1, a2, a3, b0, b1,
                        xv[0], xv[1], xv[2], xv[3]);
                    xv[0] = r.d0;
                    xv[1] = r.d1;
                    xv[2] = r.d2;
                    xv[3] = r.d3;
                  }
                  acc_o[g][ti_acc][0] += xv[0] * P_SCALE;
                  acc_o[g][ti_acc][1] += xv[1] * P_SCALE;
                  acc_o[g][ti_acc][2] += xv[2] * P_SCALE;
                  acc_o[g][ti_acc][3] += xv[3] * P_SCALE;
                }
              }
            }
          }
        } else if constexpr (KV::SCALE_FORMAT ==
                             ScaleFormat::ARBITRARY_FP32) {
'''
)
replace_in_region(
    prefill_kernel_path,
    mg_region_start,
    mg_region_end,
    mg_xv_open,
    mg_xv_direct,
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
    "  TVM_FFI_ICHECK_GT(num_tokens, 64)\n"
    "      << \"Decode (num_tokens <= 64) must go through sparse_mla_sm120_decode_dsv3_2 \"\n",
    "  TVM_FFI_ICHECK(num_tokens > 64 || mt == ModelType::DSV4_NVFP4)\n"
    "      << \"Decode (num_tokens <= 64) must go through sparse_mla_sm120_decode_dsv3_2 \"\n",
)
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
replace(
    python_path,
    "            model_type == _MODEL_TYPE_DSV4\n"
    "            and kv_pbs == _DECODE_DSV4_PAGE_BLOCK_SIZE\n",
    "            model_type in (_MODEL_TYPE_DSV4, _MODEL_TYPE_DSV4_NVFP4)\n"
    "            and kv_pbs == _DECODE_DSV4_PAGE_BLOCK_SIZE\n",
)
replace(
    python_path,
    "                extra_topk_length=extra_topk_length,\n"
    "            )\n"
    "            return\n",
    "                extra_topk_length=extra_topk_length,\n"
    "                model_type=model_type,\n"
    "            )\n"
    "            return\n",
)
replace(
    python_path,
    "def _get_sparse_mla_decode_dsv4_module():\n",
    "def _get_sparse_mla_decode_dsv4_module(model_type: int):\n",
)
replace(
    python_path,
    "            return (\n"
    "                topk_length is not None,\n",
    "            return (\n"
    "                int(model_type),\n"
    "                topk_length is not None,\n",
)
replace(
    python_path,
    "                num_splits,\n"
    "                sm_scale,\n"
    "                topk_length,\n"
    "                attn_sink,\n"
    "                extra_kv_cache,\n",
    "                num_splits,\n"
    "                sm_scale,\n"
    "                int(model_type),\n"
    "                topk_length,\n"
    "                attn_sink,\n"
    "                extra_kv_cache,\n",
)
replace(
    python_path,
    "        if (\n"
    "            model_type in (_MODEL_TYPE_DSV4, _MODEL_TYPE_DSV4_NVFP4)\n",
    "        force_nvfp4_prefill = (\n"
    "            model_type == _MODEL_TYPE_DSV4_NVFP4\n"
    "            and os.getenv(\"DSV4_NVFP4_FORCE_PREFILL\") == \"1\"\n"
    "        )\n"
    "        if (\n"
    "            not force_nvfp4_prefill\n"
    "            and model_type in (_MODEL_TYPE_DSV4, _MODEL_TYPE_DSV4_NVFP4)\n",
    present="auto_nvfp4_prefill =",
)
replace(
    python_path,
    "        force_nvfp4_prefill = (\n"
    "            model_type == _MODEL_TYPE_DSV4_NVFP4\n"
    "            and os.getenv(\"DSV4_NVFP4_FORCE_PREFILL\") == \"1\"\n"
    "        )\n"
    "        if (\n"
    "            not force_nvfp4_prefill\n"
    "            and model_type in (_MODEL_TYPE_DSV4, _MODEL_TYPE_DSV4_NVFP4)\n",
    "        force_nvfp4_prefill = (\n"
    "            model_type == _MODEL_TYPE_DSV4_NVFP4\n"
    "            and os.getenv(\"DSV4_NVFP4_FORCE_PREFILL\") == \"1\"\n"
    "        )\n"
    "        # The compact prefill kernel amortizes its fixed launch cost by ten\n"
    "        # rows and then beats both compact decode and the 584-byte FP8 path.\n"
    "        auto_nvfp4_prefill = (\n"
    "            model_type == _MODEL_TYPE_DSV4_NVFP4 and num_tokens >= 10\n"
    "        )\n"
    "        if (\n"
    "            not (force_nvfp4_prefill or auto_nvfp4_prefill)\n"
    "            and model_type in (_MODEL_TYPE_DSV4, _MODEL_TYPE_DSV4_NVFP4)\n",
)
replace(
    python_path,
    "def _decode_dsv4_runner_singleton():\n"
    "    return _get_sparse_mla_decode_dsv4_module().runner_cls()\n",
    "def _decode_dsv4_runner_singleton(model_type: int):\n"
    "    return _get_sparse_mla_decode_dsv4_module(int(model_type)).runner_cls()\n",
)
replace(
    python_path,
    "    extra_topk_length: Optional[torch.Tensor] = None,\n"
    "    chunks_per_block: Optional[int] = None,\n",
    "    extra_topk_length: Optional[torch.Tensor] = None,\n"
    "    chunks_per_block: Optional[int] = None,\n"
    "    model_type: int = _MODEL_TYPE_DSV4,\n",
)
replace(
    python_path,
    "    runner = _decode_dsv4_runner_singleton()\n",
    "    runner = _decode_dsv4_runner_singleton(model_type)\n",
)

# Give the patched sources their own JIT cache key, even if the base FP8 module
# was compiled on the host before this image was built.
jit_path = ROOT / "jit/mla.py"
jit_source = jit_path.read_text()
jit_key = '        "sparse_mla_sm120_nvfp4_416_decodefast80",\n'
if jit_key not in jit_source:
    for previous_key in (
        '        "sparse_mla_sm120_nvfp4_416_pvf16_73",\n',
        '        "sparse_mla_sm120_nvfp4_416_pvblock72",\n',
        '        "sparse_mla_sm120_nvfp4_416_prefillbulk67",\n',
        '        "sparse_mla_sm120_nvfp4_416_rawmixed66",\n',
        '        "sparse_mla_sm120_nvfp4_416_rawmixed65",\n',
        '        "sparse_mla_sm120_nvfp4_416_rawpv64",\n',
        '        "sparse_mla_sm120_nvfp4_416_qpv59",\n',
        '        "sparse_mla_sm120_nvfp4_416_prefillshift63",\n',
        '        "sparse_mla_sm120_nvfp4_416_prefillio8_62",\n',
        '        "sparse_mla_sm120_nvfp4_416_prefilllut61",\n',
        '        "sparse_mla_sm120_nvfp4_416_prefillfp8_60",\n',
        '        "sparse_mla_sm120_nvfp4_416_k32pair58",\n',
        '        "sparse_mla_sm120_nvfp4_416_k32pair57",\n',
        '        "sparse_mla_sm120_nvfp4_416_qfix56",\n',
        '        "sparse_mla_sm120_nvfp4_416_qres55",\n',
        '        "sparse_mla_sm120_nvfp4_416_qres54",\n',
        '        "sparse_mla_sm120_nvfp4_416_qfp4regpv53",\n',
        '        "sparse_mla_sm120_nvfp4_416_io2_52",\n',
        '        "sparse_mla_sm120_nvfp4_416_dbuf51",\n',
        '        "sparse_mla_sm120_nvfp4_416_prefill49",\n',
        '        "sparse_mla_sm120_nvfp4_416_prefill48",\n',
        '        "sparse_mla_sm120_nvfp4_416_prefill47",\n',
        '        "sparse_mla_sm120_nvfp4_416_prefill46",\n',
        '        "sparse_mla_sm120_nvfp4_416_compact45",\n',
        '        "sparse_mla_sm120_nvfp4_416_compact44",\n',
        '        "sparse_mla_sm120",\n',
    ):
        if previous_key in jit_source:
            jit_path.write_text(jit_source.replace(previous_key, jit_key, 1))
            break
    else:
        raise SystemExit(f"missing FlashInfer JIT key anchor in {jit_path}")

print("FlashInfer native SM120 sparse NVFP4-416 patch applied")
