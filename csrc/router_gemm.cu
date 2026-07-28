// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

template <int V>
__device__ __forceinline__ void load_weight(float const* ptr, float* dst);

template <>
__device__ __forceinline__ void load_weight<8>(float const* ptr, float* dst) {
  float4 a = *reinterpret_cast<float4 const*>(ptr);
  float4 b = *reinterpret_cast<float4 const*>(ptr + 4);
  dst[0] = a.x;
  dst[1] = a.y;
  dst[2] = a.z;
  dst[3] = a.w;
  dst[4] = b.x;
  dst[5] = b.y;
  dst[6] = b.z;
  dst[7] = b.w;
}

template <int B, int M, int EPB, int TG = 1>
__global__ __launch_bounds__(B * TG, 1) void router_gemm_kernel(
    float* out, __nv_bfloat16 const* input, float const* weight) {
  constexpr int H = 3072;
  constexpr int E = 256;
  constexpr int V = 8;
  constexpr int KI = H / (V * B);
  constexpr int W = B / 32;
  constexpr int MG = M / TG;
  int e0 = blockIdx.x * EPB;
  int tid = threadIdx.x % B;
  int m0 = threadIdx.x / B * MG;
  int lane = tid % 32;
  int warp = tid / 32;
  float acc[MG][EPB] = {};
  __shared__ float reduction[M][EPB][W];
  cudaGridDependencySynchronize();
  cudaTriggerProgrammaticLaunchCompletion();
#pragma unroll
  for (int ki = 0; ki < KI; ++ki) {
    int k0 = ki * V * B + tid * V;
    float w[EPB][V];
#pragma unroll
    for (int e = 0; e < EPB; ++e)
      load_weight<V>(weight + (e0 + e) * H + k0, w[e]);
#pragma unroll
    for (int m = 0; m < MG; ++m) {
      uint4 packed = *reinterpret_cast<uint4 const*>(
          input + static_cast<size_t>(m0 + m) * H + k0);
      auto values = reinterpret_cast<__nv_bfloat16 const*>(&packed);
#pragma unroll
      for (int e = 0; e < EPB; ++e)
#pragma unroll
        for (int k = 0; k < V; ++k)
          acc[m][e] += __bfloat162float(values[k]) * w[e][k];
    }
  }
#pragma unroll
  for (int m = 0; m < MG; ++m) {
#pragma unroll
    for (int e = 0; e < EPB; ++e) {
      float sum = acc[m][e];
      sum += __shfl_xor_sync(0xffffffff, sum, 16);
      sum += __shfl_xor_sync(0xffffffff, sum, 8);
      sum += __shfl_xor_sync(0xffffffff, sum, 4);
      sum += __shfl_xor_sync(0xffffffff, sum, 2);
      sum += __shfl_xor_sync(0xffffffff, sum, 1);
      if (lane == 0) reduction[m0 + m][e][warp] = sum;
    }
  }
  __syncthreads();
  for (int i = threadIdx.x; i < M * EPB; i += B * TG) {
    int m = i / EPB;
    int e = i % EPB;
    float sum = 0;
#pragma unroll
    for (int w = 0; w < W; ++w) sum += reduction[m][e][w];
    out[m * E + e0 + e] = sum;
  }
}

template <int B, int M, int EPB, int TG = 1>
void launch(float* out, __nv_bfloat16 const* input, float const* weight,
            cudaStream_t stream) {
  cudaLaunchConfig_t config{};
  config.gridDim = 256 / EPB;
  config.blockDim = B * TG;
  config.stream = stream;
  cudaLaunchAttribute attribute{};
  attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute.val.programmaticStreamSerializationAllowed = 1;
  config.attrs = &attribute;
  config.numAttrs = 1;
  cudaLaunchKernelEx(&config, router_gemm_kernel<B, M, EPB, TG>, out, input,
                     weight);
}

template <int M>
void dispatch(float* out, __nv_bfloat16 const* input, float const* weight,
              cudaStream_t stream) {
  if constexpr (M >= 14 && M % 2 == 0)
    launch<192, M, 2, 2>(out, input, weight, stream);
  else if constexpr (M >= 8 && M <= 12 && M % 2 == 0)
    launch<192, M, 1, 2>(out, input, weight, stream);
  else
    launch<128, M, 1>(out, input, weight, stream);
}

#define CASE(M) \
  case M:       \
    dispatch<M>(out, input, weight, stream); \
    break

void router_gemm(torch::Tensor output, torch::Tensor x, torch::Tensor w) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && output.is_cuda());
  TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && output.is_contiguous());
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(w.scalar_type() == torch::kFloat32);
  TORCH_CHECK(output.scalar_type() == torch::kFloat32);
  TORCH_CHECK(x.size(1) == 3072 && w.size(0) == 256 && w.size(1) == 3072);
  auto out = output.data_ptr<float>();
  auto input = reinterpret_cast<__nv_bfloat16 const*>(x.data_ptr());
  auto weight = w.data_ptr<float>();
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  switch (x.size(0)) {
    CASE(1);
    CASE(2);
    CASE(3);
    CASE(4);
    CASE(5);
    CASE(6);
    CASE(7);
    CASE(8);
    CASE(9);
    CASE(10);
    CASE(11);
    CASE(12);
    CASE(13);
    CASE(14);
    CASE(15);
    CASE(16);
    CASE(17);
    CASE(18);
    CASE(19);
    CASE(20);
    CASE(21);
    CASE(22);
    CASE(23);
    CASE(24);
    CASE(25);
    CASE(26);
    CASE(27);
    CASE(28);
    CASE(29);
    CASE(30);
    CASE(31);
    CASE(32);
    default:
      TORCH_CHECK(false, "router token count must be between 1 and 32");
  }
}

__global__ void qk_rms_kernel(
    __nv_bfloat16 const* qkv, __nv_bfloat16 const* q_weight,
    __nv_bfloat16 const* k_weight, __nv_bfloat16* q_out,
    __nv_bfloat16* k_out, int64_t const* peers, int rank, int epoch,
    float eps) {
  constexpr int Q = 1536;
  constexpr int K = 256;
  constexpr int STRIDE = 2048;
  int token = blockIdx.x;
  int tid = threadIdx.x;
  float qsum = 0;
  float ksum = 0;
  for (int i = tid; i < Q; i += blockDim.x) {
    float value = __bfloat162float(qkv[token * STRIDE + i]);
    qsum += value * value;
  }
  for (int i = tid; i < K; i += blockDim.x) {
    float value = __bfloat162float(qkv[token * STRIDE + Q + i]);
    ksum += value * value;
  }
  for (int offset = 16; offset; offset >>= 1) {
    qsum += __shfl_down_sync(0xffffffff, qsum, offset);
    ksum += __shfl_down_sync(0xffffffff, ksum, offset);
  }
  __shared__ float partial[8][2];
  __shared__ float inverse[2];
  if ((tid & 31) == 0) {
    partial[tid >> 5][0] = qsum;
    partial[tid >> 5][1] = ksum;
  }
  __syncthreads();
  if (tid == 0) {
    qsum = 0;
    ksum = 0;
    for (int warp = 0; warp < 8; ++warp) {
      qsum += partial[warp][0];
      ksum += partial[warp][1];
    }
    auto local = reinterpret_cast<char*>(peers[rank]);
    auto sums = reinterpret_cast<float*>(local + 128);
    sums[token * 2] = qsum / Q;
    sums[token * 2 + 1] = ksum / K;
    __threadfence_system();
    reinterpret_cast<volatile int*>(local)[token] = epoch;
    float total_q = 0;
    float total_k = 0;
    for (int peer = 0; peer < 4; ++peer) {
      auto remote = reinterpret_cast<char*>(peers[peer]);
      auto flag = reinterpret_cast<volatile int*>(remote) + token;
      while (*flag != epoch) __nanosleep(20);
      auto values = reinterpret_cast<volatile float*>(remote + 128);
      total_q += values[token * 2];
      total_k += values[token * 2 + 1];
    }
    inverse[0] = rsqrtf(total_q * 0.25f + eps);
    inverse[1] = rsqrtf(total_k * 0.25f + eps);
  }
  __syncthreads();
  for (int i = tid; i < Q; i += blockDim.x) {
    float value = __bfloat162float(qkv[token * STRIDE + i]);
    float weight = __bfloat162float(q_weight[i]);
    q_out[token * Q + i] = __float2bfloat16(value * inverse[0] * weight);
  }
  for (int i = tid; i < K; i += blockDim.x) {
    float value = __bfloat162float(qkv[token * STRIDE + Q + i]);
    float weight = __bfloat162float(k_weight[i]);
    k_out[token * K + i] = __float2bfloat16(value * inverse[1] * weight);
  }
}

void qk_rms(torch::Tensor qkv, torch::Tensor q_weight, torch::Tensor k_weight,
            torch::Tensor q_out, torch::Tensor k_out, torch::Tensor peers,
            int64_t rank, int64_t epoch, double eps) {
  TORCH_CHECK(qkv.is_cuda() && qkv.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(qkv.is_contiguous() && qkv.size(1) == 2048);
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  qk_rms_kernel<<<qkv.size(0), 256, 0, stream>>>(
      reinterpret_cast<__nv_bfloat16 const*>(qkv.data_ptr()),
      reinterpret_cast<__nv_bfloat16 const*>(q_weight.data_ptr()),
      reinterpret_cast<__nv_bfloat16 const*>(k_weight.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(q_out.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(k_out.data_ptr()),
      peers.data_ptr<int64_t>(), rank, epoch, static_cast<float>(eps));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("forward", &router_gemm);
  module.def("qk_rms", &qk_rms);
}
