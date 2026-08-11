#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

struct MmaResult {
  float d0, d1, d2, d3;
};

__device__ __forceinline__ MmaResult mma_immediate(
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1, float c0, float c1, float c2, float c3,
    uint32_t scale_a, uint32_t scale_b) {
  MmaResult r;
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

__device__ __forceinline__ MmaResult mma_register(
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1, float c0, float c1, float c2, float c3,
    uint32_t scale_a, uint32_t scale_b) {
  MmaResult r;
  const uint16_t zero = 0;
  asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X."
      "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},"
      "{%14},{%15,%16},{%17},{%18,%19};\n"
      : "=f"(r.d0), "=f"(r.d1), "=f"(r.d2), "=f"(r.d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
        "f"(c0), "f"(c1), "f"(c2), "f"(c3), "r"(scale_a),
        "h"(zero), "h"(zero), "r"(scale_b), "h"(zero), "h"(zero));
  return r;
}

__device__ __forceinline__ MmaResult mma_mixed_fp8_fp4(
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1, float c0, float c1, float c2, float c3,
    uint8_t scale_a, uint8_t scale_b) {
  MmaResult r;
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
        "r"(static_cast<uint32_t>(scale_b)),
        "n"(static_cast<uint16_t>(0)), "n"(static_cast<uint16_t>(0)));
  return r;
}

__global__ void probe(float* output, int mode, int use_register_controls) {
  const int lane = threadIdx.x & 31;
  uint32_t a = 0;
  uint32_t b = 0;
  if (mode == 1) {
    a = 0x11111111U;
    b = 0x11111111U;
  } else if (mode == 2) {
    a = 0x77777777U;
    b = 0x77777777U;
  } else if (mode == 3) {
    a = 0x9e3779b9U * static_cast<uint32_t>(lane + 1);
    b = 0x85ebca6bU * static_cast<uint32_t>(lane + 3);
  }
  const uint32_t one_scales = 0x38383838U;
  const MmaResult r = use_register_controls
                          ? mma_register(a, a, a, a, b, b, 0.f, 0.f, 0.f,
                                         0.f, one_scales, one_scales)
                          : mma_immediate(a, a, a, a, b, b, 0.f, 0.f, 0.f,
                                          0.f, one_scales, one_scales);
  float* row = output + threadIdx.x * 4;
  row[0] = r.d0;
  row[1] = r.d1;
  row[2] = r.d2;
  row[3] = r.d3;
}

__device__ __forceinline__ void ldmatrix_x4_probe(
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3,
    const void* smem_ptr) {
  const uint32_t addr =
      static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
      : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x2_probe(
    uint32_t& r0, uint32_t& r1, const void* smem_ptr) {
  const uint32_t addr =
      static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r0), "=r"(r1)
               : "r"(addr));
}

__global__ void scale_layout_probe(float* output, int mode) {
  __shared__ __align__(16) uint8_t a_smem[16 * 32];
  __shared__ __align__(16) uint8_t b_smem[8 * 32];
  const int lane = threadIdx.x & 31;
  for (int i = lane; i < 16 * 32; i += 32) a_smem[i] = 0x11;
  for (int i = lane; i < 8 * 32; i += 32) b_smem[i] = 0x11;
  __syncthreads();

  const int a_row = (lane & 7) + ((lane >> 3) & 1) * 8;
  const int a_col = (lane >> 4) * 16;
  const int b_row = lane & 7;
  const int b_col = ((lane >> 3) & 1) * 16;
  uint32_t a0, a1, a2, a3, b0, b1;
  ldmatrix_x4_probe(a0, a1, a2, a3, a_smem + a_row * 32 + a_col);
  ldmatrix_x2_probe(b0, b1, b_smem + b_row * 32 + b_col);

  const int gid = lane >> 2;
  const int tid = lane & 3;
  const int output_row0 = gid;
  const int output_row1 = gid + 8;
  uint32_t sfa;
  if (mode == 0) {
    const uint8_t row_scales[4] = {0x30, 0x38, 0x40, 0x48};
    const int scale_row = gid + (lane & 1) * 8;
    const uint32_t byte = row_scales[scale_row & 3];
    sfa = byte * 0x01010101U;
  } else {
    sfa = 0x48403830U;
  }
  const MmaResult r = mma_immediate(
      a0, a1, a2, a3, b0, b1, 0.f, 0.f, 0.f, 0.f, sfa,
      0x38383838U);
  output[output_row0 * 8 + tid * 2] = r.d0;
  output[output_row0 * 8 + tid * 2 + 1] = r.d1;
  output[output_row1 * 8 + tid * 2] = r.d2;
  output[output_row1 * 8 + tid * 2 + 1] = r.d3;
}

__global__ void sfb_mapping_probe(float* output) {
  __shared__ __align__(16) uint8_t a_smem[16 * 32];
  __shared__ __align__(16) uint8_t b_smem[8 * 32];
  const int lane = threadIdx.x & 31;
  for (int i = lane; i < 16 * 32; i += 32) a_smem[i] = 0x11;
  for (int i = lane; i < 8 * 32; i += 32) b_smem[i] = 0x11;
  __syncthreads();

  const int a_row = (lane & 7) + ((lane >> 3) & 1) * 8;
  const int a_col = (lane >> 4) * 16;
  const int b_row = lane & 7;
  const int b_col = ((lane >> 3) & 1) * 16;
  uint32_t a0, a1, a2, a3, b0, b1;
  ldmatrix_x4_probe(a0, a1, a2, a3, a_smem + a_row * 32 + a_col);
  ldmatrix_x2_probe(b0, b1, b_smem + b_row * 32 + b_col);

  const int target_lane = blockIdx.x >> 2;
  const int target_byte = blockIdx.x & 3;
  uint32_t sfb = 0x38383838U;
  if (lane == target_lane) {
    sfb = (sfb & ~(0xffU << (target_byte * 8))) |
          (0x40U << (target_byte * 8));
  }
  const MmaResult r = mma_immediate(
      a0, a1, a2, a3, b0, b1, 0.f, 0.f, 0.f, 0.f, 0x38383838U,
      sfb);
  const int gid = lane >> 2;
  const int tid = lane & 3;
  float* matrix = output + blockIdx.x * 16 * 8;
  matrix[gid * 8 + tid * 2] = r.d0;
  matrix[gid * 8 + tid * 2 + 1] = r.d1;
  matrix[(gid + 8) * 8 + tid * 2] = r.d2;
  matrix[(gid + 8) * 8 + tid * 2 + 1] = r.d3;
}

__global__ void mixed_layout_probe(float* output, int b_byte) {
  __shared__ __align__(16) uint8_t a_smem[16 * 32];
  __shared__ __align__(16) uint8_t b_smem[32 * 8];
  const int lane = threadIdx.x & 31;
  for (int i = lane; i < 16 * 32; i += 32) a_smem[i] = 0x38;
  for (int i = lane; i < 32 * 8; i += 32) {
    b_smem[i] = static_cast<uint8_t>(b_byte);
  }
  __syncthreads();

  const int a_row = (lane & 7) + ((lane >> 3) & 1) * 8;
  const int a_col = (lane >> 4) * 16;
  uint32_t a0, a1, a2, a3;
  ldmatrix_x4_probe(a0, a1, a2, a3,
                    a_smem + a_row * 32 + a_col);

  const int gid = lane >> 2;
  const int tid = lane & 3;
  uint32_t b0 = 0;
  uint32_t b1 = 0;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    b0 |= static_cast<uint32_t>(b_smem[(tid * 4 + j) * 8 + gid])
          << (j * 8);
    b1 |= static_cast<uint32_t>(b_smem[(16 + tid * 4 + j) * 8 + gid])
          << (j * 8);
  }

  const MmaResult r = mma_mixed_fp8_fp4(
      a0, a1, a2, a3, b0, b1, 0.f, 0.f, 0.f, 0.f, 0x7f, 0x7f);
  output[gid * 8 + tid * 2] = r.d0;
  output[gid * 8 + tid * 2 + 1] = r.d1;
  output[(gid + 8) * 8 + tid * 2] = r.d2;
  output[(gid + 8) * 8 + tid * 2 + 1] = r.d3;
}

int main() {
  float* device_output = nullptr;
  constexpr int kThreads = 256;
  constexpr int kOutputFloats = 128 * 16 * 8;
  cudaMalloc(&device_output, kOutputFloats * sizeof(float));
  std::vector<float> output(kOutputFloats);
  for (int controls = 0; controls < 2; ++controls) {
    for (int mode = 0; mode < 4; ++mode) {
      probe<<<1, kThreads>>>(device_output, mode, controls);
      const cudaError_t sync_status = cudaDeviceSynchronize();
      if (sync_status != cudaSuccess) {
        std::fprintf(stderr, "launch failed: %s\n",
                     cudaGetErrorString(sync_status));
        return 1;
      }
      cudaMemcpy(output.data(), device_output, output.size() * sizeof(float),
                 cudaMemcpyDeviceToHost);
      int finite = 0;
      float min_value = INFINITY;
      float max_value = -INFINITY;
      for (float value : output) {
        if (std::isfinite(value)) {
          ++finite;
          min_value = std::fmin(min_value, value);
          max_value = std::fmax(max_value, value);
        }
      }
      std::printf("controls=%s mode=%d finite=%d/%zu range=[%g,%g]",
                  controls ? "register" : "immediate", mode, finite,
                  output.size(), min_value, max_value);
      for (int i = 0; i < static_cast<int>(output.size()); ++i) {
        if (!std::isfinite(output[i])) {
          std::printf(" first_nonfinite=(warp=%d,lane=%d,reg=%d,value=%g)",
                      i / 128, (i / 4) & 31, i & 3, output[i]);
          break;
        }
      }
      std::printf("\n");
    }
  }
  for (int mode = 0; mode < 2; ++mode) {
    scale_layout_probe<<<1, 32>>>(device_output, mode);
    cudaDeviceSynchronize();
    cudaMemcpy(output.data(), device_output, 16 * 8 * sizeof(float),
               cudaMemcpyDeviceToHost);
    float max_error = 0.f;
    std::printf("scale_layout mode=%s rows=", mode == 0 ? "row" : "group");
    for (int row = 0; row < 16; ++row) {
      const float expected =
          mode == 0 ? (8.f * static_cast<float>(1 << (row & 3))) : 30.f;
      std::printf("%s%g", row ? "," : "", output[row * 8]);
      for (int col = 0; col < 8; ++col) {
        max_error =
            std::fmax(max_error, std::fabs(output[row * 8 + col] - expected));
      }
    }
    std::printf(" max_error=%g\n", max_error);
  }
  sfb_mapping_probe<<<128, 32>>>(device_output);
  cudaDeviceSynchronize();
  cudaMemcpy(output.data(), device_output, output.size() * sizeof(float),
             cudaMemcpyDeviceToHost);
  std::printf("sfb_mapping");
  for (int target = 0; target < 128; ++target) {
    std::printf("\n lane=%d byte=%d ->", target >> 2, target & 3);
    const float* matrix = output.data() + target * 16 * 8;
    int changed = 0;
    for (int row = 0; row < 16; ++row) {
      for (int col = 0; col < 8; ++col) {
        if (matrix[row * 8 + col] != 16.f) {
          if (changed < 8) {
            std::printf(" (%d,%d:%g)", row, col,
                        matrix[row * 8 + col]);
          }
          ++changed;
        }
      }
    }
    std::printf(" count=%d", changed);
  }
  std::printf("\n");
  // SM120 mxf8f6f4 expects a 4-bit E2M1 value in bits 2..5 of each
  // eight-bit register container (CUTLASS fp4_shift_B).
  for (int b_byte : {0x04, 0x08, 0x0c, 0x28}) {
    mixed_layout_probe<<<1, 32>>>(device_output, b_byte);
    cudaDeviceSynchronize();
    cudaMemcpy(output.data(), device_output, 16 * 8 * sizeof(float),
               cudaMemcpyDeviceToHost);
    float min_value = INFINITY;
    float max_value = -INFINITY;
    for (int i = 0; i < 16 * 8; ++i) {
      min_value = std::fmin(min_value, output[i]);
      max_value = std::fmax(max_value, output[i]);
    }
    std::printf("mixed_layout b=0x%02x range=[%g,%g] row0=", b_byte,
                min_value, max_value);
    for (int col = 0; col < 8; ++col) {
      std::printf("%s%g", col ? "," : "", output[col]);
    }
    std::printf("\n");
  }
  cudaFree(device_output);
  return 0;
}
