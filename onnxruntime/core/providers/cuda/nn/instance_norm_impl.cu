// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include "instance_norm_impl.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
__global__ void _InstanceNormKernel(
    const T* __restrict__ input_data,
    const T* __restrict__ scale,
    const T* __restrict__ bias,
    const T* __restrict__ mean,
    const T* __restrict__ variance,
    const double variance_correction,
    const double epsilon,
    const fast_divmod fdm_HW,
    const fast_divmod fdm_C,
    T* __restrict__ output_data,
    const CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  int nc = fdm_HW.div(id);
  int n, c;
  fdm_C.divmod(nc, n, c);

  // Y = scale * (x - mean) / sqrt (std * std + epsilon) + B
  output_data[id] = scale[c] * (input_data[id] - mean[nc]) / _Sqrt((T)variance_correction + (T)epsilon) + bias[c];
}

template <typename T>
void InstanceNormImpl(
    cudaStream_t stream,
    const T* input_data,
    const T* scale,
    const T* bias,
    const T* mean,
    const T* variance,
    const double variance_correction,
    const double epsilon,
    const fast_divmod& fdm_HW,
    const fast_divmod& fdm_C,
    T* output_data,
    size_t N) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(N) / GridDim::maxThreadsPerBlock));
  _InstanceNormKernel<T><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
      input_data, scale, bias, mean, variance, variance_correction, epsilon, fdm_HW, fdm_C, output_data, (CUDA_LONG)N);
}

#define SPECIALIZED_IMPL(T) \
  template void InstanceNormImpl<T>(cudaStream_t stream, const T* input_data, const T* scale, const T* bias, const T* mean, const T* stddev, const double variance_correction, const double epsilon, const fast_divmod& fdm_HW, const fast_divmod& fdm_C, T* output_data, size_t count);

SPECIALIZED_IMPL(float)
SPECIALIZED_IMPL(double)
// When the input data type is float16, the means and variances will flow in as float32 (special case)
SPECIALIZED_IMPL(half)

}  // namespace cuda
}  // namespace onnxruntime