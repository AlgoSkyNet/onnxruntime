// Modifications: scaling is moved from masked softmax to the gemm before that.
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cub/cub.cuh>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cuda_common.h"
#include "matmul_nbits.cuh"

using namespace onnxruntime::cuda;
using namespace cub;

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
__device__ __forceinline__ T WarpUniform(T value) {
  struct {
    union {
      T value;
      uint32_t asInt;
    };
  } p;
  p.value = value;
  p.asInt = __shfl_sync(0xffffffff, (unsigned)p.asInt, 0);
  return p.value;
}

/*
__device__ __forceinline__ float AccumulateEightElements(uint32_t values_quant, half scale, uint8_t zp, const half* a) {
  half2 scale_half2 = {scale, scale};
  half zp_adjust = -scale * __short2half_rn(zp);
  half2 zp_adjust2 = {zp_adjust, zp_adjust};
  uint4 vec_a = *(reinterpret_cast<const uint4*>(a));

  half2 element01 = __halves2half2(__uint2half_rn(values_quant & 0xF), __uint2half_rn((values_quant >> 4) & 0xF));
  half2 v0 = element01 * scale_half2 + zp_adjust2;

  half2 element23 = __halves2half2(__uint2half_rn((values_quant >> 8) & 0xF), __uint2half_rn((values_quant >> 12) & 0xF));
  half2 v1 = element23 * scale_half2 + zp_adjust2;

  half2 element45 = __halves2half2(__uint2half_rn((values_quant >> 16) & 0xF), __uint2half_rn((values_quant >> 20) & 0xF));
  half2 v2 = element45 * scale_half2 + zp_adjust2;

  half2 element67 = __halves2half2(__uint2half_rn((values_quant >> 24) & 0xF), __uint2half_rn((values_quant >> 28) & 0xF));
  half2 v3 = element67 * scale_half2 + zp_adjust2;

  v0 = v0 * (*(reinterpret_cast<half2*>(&(vec_a.x))));
  v1 = v1 * (*(reinterpret_cast<half2*>(&(vec_a.y))));
  v2 = v2 * (*(reinterpret_cast<half2*>(&(vec_a.z)))) + v0;
  v3 = v3 * (*(reinterpret_cast<half2*>(&(vec_a.w)))) + v1;
  v3 = v2 + v3;
  return float(v3.x) + float(v3.y);
}
*/

__device__ __forceinline__ float AccumulateEightElements(uint32_t values_quant, half scale, uint8_t zp, const half* a) {
  half2 scale_half2 = {scale, scale};
  half zp_adjust = -scale * __short2half_rn(zp);
  half2 zp_adjust2 = {zp_adjust, zp_adjust};
  uint4 vec_a = *(reinterpret_cast<const uint4*>(a));

  half2 elements[4];
  uint32_t*      h   = reinterpret_cast<uint32_t*>(&elements);

  // First, we extract the i4s and construct an intermediate fp16 number.
  static constexpr uint32_t immLut                = (0xf0 & 0xcc) | 0xaa;
  static constexpr uint32_t BOTTOM_MASK           = 0x000f000f;
  static constexpr uint32_t TOP_MASK              = 0x00f000f0;
  static constexpr uint32_t I4s_TO_F16s_MAGIC_NUM = 0x64006400;

  // Note that the entire sequence only requires 1 shift instruction. This is thanks to the register packing
  // format and the fact that we force our integers to be unsigned, and account for this in the fp16 subtractions.
  // In addition, I exploit the fact that sub and fma have the same throughput in order to convert elt_23 and
  // elt_67 to fp16 without having to shift them to the bottom bits before hand.

  // Shift right by 8 to now consider elt_45 and elt_67. Issue first to hide RAW dependency if we issue
  // immediately before required.
  const uint32_t top_i4s = values_quant >> 8;
  // Extract elt_01 - (i4s & 0x000f000f) | 0x64006400
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                : "=r"(h[0])
                : "r"(values_quant), "n"(BOTTOM_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
  // Extract elt_23 (i4s & 0x00f000f0) | 0x64006400
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                : "=r"(h[1])
                : "r"(values_quant), "n"(TOP_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
  // Extract elt_45 (top_i4s & 0x000f000f) | 0x64006400
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                : "=r"(h[2])
                : "r"(top_i4s), "n"(BOTTOM_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
  // Extract elt_67 (top_i4s & 0x00f000f0) | 0x64006400
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                : "=r"(h[3])
                : "r"(top_i4s), "n"(TOP_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));

  // I use inline PTX below because I am not sure if the compiler will emit float2half instructions if I use the
  // half2 ctor. In this case, I chose performance reliability over code readability.

  // This is the half2 {1032, 1032} represented as an integer.
  static constexpr uint32_t FP16_TOP_MAGIC_NUM = 0x64086408;
  // This is the half2 {1 / 16, 1 / 16} represented as an integer.
  static constexpr uint32_t ONE_SIXTEENTH = 0x2c002c00;
  // This is the half2 {-72, -72} represented as an integer.
  static constexpr uint32_t NEG_72 = 0xd480d480;

  // Finally, we construct the output numbers.
  // Convert elt_01
  asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[0]) : "r"(h[0]), "r"(FP16_TOP_MAGIC_NUM));
  // Convert elt_23
  asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(h[1]) : "r"(h[1]), "r"(ONE_SIXTEENTH), "r"(NEG_72));
  // Convert elt_45
  asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[2]) : "r"(h[2]), "r"(FP16_TOP_MAGIC_NUM));
  // Convert elt_67
  asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(h[3]) : "r"(h[3]), "r"(ONE_SIXTEENTH), "r"(NEG_72));

  half2 v0 = elements[0] * scale_half2 + zp_adjust2;
  half2 v1 = elements[1] * scale_half2 + zp_adjust2;
  half2 v2 = elements[2] * scale_half2 + zp_adjust2;
  half2 v3 = elements[3] * scale_half2 + zp_adjust2;

  v0 = v0 * (*(reinterpret_cast<half2*>(&(vec_a.x))));
  v1 = v1 * (*(reinterpret_cast<half2*>(&(vec_a.y))));
  v2 = v2 * (*(reinterpret_cast<half2*>(&(vec_a.z)))) + v0;
  v3 = v3 * (*(reinterpret_cast<half2*>(&(vec_a.w)))) + v1;
  v3 = v2 + v3;
  return float(v3.x) + float(v3.y);
}

__device__ __forceinline__ float AccumulateEightElements(uint32_t values_quant, float scale, uint8_t zp, const float* a) {
  float4 a_vec_0 = *(reinterpret_cast<const float4*>(a));
  float4 a_vec_1 = *(reinterpret_cast<const float4*>(a + 4));

  float zp_adjust = -scale * zp;
  float v0 = float(values_quant & 0xF) * scale + zp_adjust;
  float v1 = float((values_quant >> 4) & 0xF) * scale + zp_adjust;
  float v2 = float((values_quant >> 8) & 0xF) * scale + zp_adjust;
  float v3 = float((values_quant >> 12) & 0xF) * scale + zp_adjust;
  float v4 = float((values_quant >> 16) & 0xF) * scale + zp_adjust;
  float v5 = float((values_quant >> 20) & 0xF) * scale + zp_adjust;
  float v6 = float((values_quant >> 24) & 0xF) * scale + zp_adjust;
  float v7 = float((values_quant >> 28) & 0xF) * scale + zp_adjust;

  v0 = v0 * a_vec_0.x;
  v1 = v1 * a_vec_0.y;
  v2 = v2 * a_vec_0.z;
  v3 = v3 * a_vec_0.w;
  v4 = v4 * a_vec_1.x + v0;
  v5 = v5 * a_vec_1.y + v1;
  v6 = v6 * a_vec_1.z + v2;
  v7 = v7 * a_vec_1.w + v3;
  return v4 + v5 + v6 + v7;
}

constexpr int kColsPerThreadBlock = 8;
constexpr int kWarpSize = 32;

// kernel for 4bits quantized gemv, i.e., computing A(1,K) x B(K, N)
// B(K, N) is quantized blockwise with 4bits and stored as [N, (K + block_size - 1)/block_size, blob]
// The thread block size is (kWarpSize, kColsPerThreadBlock) and grid size is (N/kColsPerThreadBlock, 1)
// Each thread block computes [1, K] x [kColsPerThreadBlock, (K + block_size - 1)/block_size, blob],
//     i.e., computing kColsPerThreadBlock per block and a warp reduce (1, K) x (K)
template <class T, int block_size>
__global__ void MatMulFloatInt4Kernel(
    T* output,
    const T* a_data,
    const uint8_t* b_data_quant,
    const T* scales_data,
    const uint8_t* zero_points,
    int m,
    int n,
    int k,
    int blocks_per_K) {
  int n_block_id = blockIdx.x;
  int m_id = blockIdx.y;
  int lane_id = threadIdx.x;
  int warp_id = WarpUniform(threadIdx.y);
  int n_id = n_block_id * kColsPerThreadBlock + warp_id;
  int thread_id = warp_id * kWarpSize + lane_id;
  constexpr int k_per_iter = 256;
  int k_iter = k / k_per_iter;

  // blocks_per_k is the number of scales and zero points on the k dim
  const int b_zp_k = (blocks_per_K + 1)/ 2;

  extern __shared__ char shared_buffer[];

  // load scale to shared buffer
  T* b_scale_vec = (T*)shared_buffer;
  uint8_t* b_zp_vec = reinterpret_cast<uint8_t*>(b_scale_vec + kColsPerThreadBlock * blocks_per_K);
  int offset = n_block_id * kColsPerThreadBlock * blocks_per_K;
  for (int i = thread_id; i < kColsPerThreadBlock * blocks_per_K; i += kColsPerThreadBlock * kWarpSize) {
    b_scale_vec[i] = scales_data[offset + i];
  }

  int zp_offset = n_block_id * kColsPerThreadBlock * b_zp_k;
  for (int i = thread_id; i < kColsPerThreadBlock * b_zp_k; i += kColsPerThreadBlock * kWarpSize) {
    b_zp_vec[i] = zero_points != nullptr ? zero_points[zp_offset + i] : uint8_t(0x88);
  }
  __syncthreads();

  a_data += m_id * k;
  b_data_quant += n_id * blocks_per_K * (block_size / 2);

  const int scale_col_offset = warp_id * blocks_per_K;
  const int zp_col_offset = warp_id * b_zp_k;

  float sum = 0.f;
  int k_id = 0;
  for (; k_id < (k & 0xffffff00); k_id += k_per_iter) {
    const int t_k = k_id + (lane_id << 3);  // k index for this thread
    const int t_meta_k = t_k / block_size;  // k index for this thread, points to the scale and zero point
    uint32_t value = *(reinterpret_cast<const uint32_t*>(b_data_quant + (t_k >> 1)));
    T scale = b_scale_vec[scale_col_offset + t_meta_k];
    uint8_t zp = b_zp_vec[zp_col_offset + t_meta_k/2];
    zp = (t_meta_k & 0x01) ? (zp >> 4) : (zp & 0x0f);
    sum += AccumulateEightElements(value, scale, zp, a_data + k_id + (lane_id << 3));
  }

  // handle reminder
  if (k_id + lane_id * 8 < k) {
    const int t_k = k_id + (lane_id << 3);  // k index for this thread
    const int t_meta_k = t_k / block_size;  // k index for this thread, points to the scale and zero point
    uint32_t value = *(reinterpret_cast<const uint32_t*>(b_data_quant + k_iter * 128 + lane_id * 4));
    T scale = b_scale_vec[scale_col_offset + t_meta_k];
    uint8_t zp = b_zp_vec[zp_col_offset + t_meta_k/2];
    zp = (t_meta_k & 0x01) ? (zp >> 4) : (zp & 0x0f);
    sum += AccumulateEightElements(value, scale, zp, a_data + k_id + (lane_id << 3));
  }

  // warp reduction
  for (int i = 16; i > 0; i = i / 2) {
    sum += __shfl_down_sync(0xffffffff, sum, i);
  }

  if (lane_id == 0) {
    output[m_id * n + n_id] = sum;
  }
}

template <class T>
bool TryMatMul4Bits(
    T* output,
    const T* a_data,
    const uint8_t* b_data_quant,
    const T* scales_data,
    const uint8_t* zero_points,
    int m,
    int n,
    int k,
    int block_size,
    int shared_mem_per_block,
    cudaStream_t stream) {
  if (n % kColsPerThreadBlock != 0 || k % 8 != 0 || m > 1) {
    return false;
  }
  dim3 blocks((n + kColsPerThreadBlock - 1) / kColsPerThreadBlock, m);
  dim3 threads(kWarpSize, kColsPerThreadBlock);
  int blocks_per_K = (k + block_size - 1) / block_size;
  int blocks_per_thread_block = blocks_per_K * kColsPerThreadBlock;
  int shared_mem_size = sizeof(T) * blocks_per_thread_block + blocks_per_thread_block / 2;
  if (shared_mem_size > shared_mem_per_block) {
    return false;
  }

  if (16 == block_size) {
    MatMulFloatInt4Kernel<T, 16><<<blocks, threads, shared_mem_size, stream>>>(
        output, a_data, b_data_quant, scales_data, zero_points, m, n, k, blocks_per_K);
  } else if (32 == block_size) {
    MatMulFloatInt4Kernel<T, 32><<<blocks, threads, shared_mem_size, stream>>>(
        output, a_data, b_data_quant, scales_data, zero_points, m, n, k, blocks_per_K);
  } else if (64 == block_size) {
    MatMulFloatInt4Kernel<T, 64><<<blocks, threads, shared_mem_size, stream>>>(
        output, a_data, b_data_quant, scales_data, zero_points, m, n, k, blocks_per_K);
  } else if (128 == block_size) {
    MatMulFloatInt4Kernel<T, 128><<<blocks, threads, shared_mem_size, stream>>>(
        output, a_data, b_data_quant, scales_data, zero_points, m, n, k, blocks_per_K);
  } else {
    ORT_THROW("block size ", block_size, " is not supported");
  }

  return true;
}

template bool TryMatMul4Bits<float>(
    float* output,
    const float* a_data,
    const uint8_t* b_data_quant,
    const float* scales_data,
    const uint8_t* zero_points,
    int m,
    int n,
    int k,
    int block_size,
    int shared_mem_per_block,
    cudaStream_t stream);

template bool TryMatMul4Bits<half>(
    half* output,
    const half* a_data,
    const uint8_t* b_data_quant,
    const half* scales_data,
    const uint8_t* zero_points,
    int m,
    int n,
    int k,
    int block_size,
    int shared_mem_per_block,
    cudaStream_t stream);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
