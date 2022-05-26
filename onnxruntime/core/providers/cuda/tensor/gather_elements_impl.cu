// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/tensor/gather_elements_impl.h"
#include "core/providers/cuda/tensor/scatter_elements_impl.h"
#ifdef ENABLE_TRAINING
#include "orttraining/training_ops/cuda/tensor/gather_elements_grad_impl.h"
#endif

#include "core/providers/cuda/atomic/common.cuh"
#include "core/providers/cuda/cu_inc/common.cuh"

namespace onnxruntime {
namespace cuda {

namespace {
constexpr int threads_per_block = GridDim::maxThreadsPerBlock;
constexpr int thread_worksize = 4;
}  // namespace

template <class T>
struct FuncAssignment {
  __device__ __inline__ void operator()(T* a, const T* b) const { *a = *b; }
};

template <typename T, typename TIndex, bool IsOuterAxis, bool IsGather, typename TFunc>
__global__ void _GatherScatterElements2DKernel(const T* src_data, const TIndex* indices_data, T* output_data,
                                               const int64_t input_dim_along_axis,
                                               const fast_divmod indices_row_size_fmd, const int64_t input_row_size,
                                               const TFunc& func, CUDA_LONG N) {
  CUDA_LONG start = threads_per_block * thread_worksize * blockIdx.x + threadIdx.x;
  CUDA_LONG id;
  T value[thread_worksize];

  if (!IsGather) {
    id = start;
#pragma unroll
    for (int work = 0; work < thread_worksize; ++work) {
      if (id < N) {
        value[work] = src_data[id];
        id += threads_per_block;
      }
    }
  }

  id = start;
#pragma unroll
  for (int work = 0; work < thread_worksize; ++work) {
    if (id < N) {
      int64_t input_offset_along_axis = static_cast<int64_t>(indices_data[id]);
      if (input_offset_along_axis >= -input_dim_along_axis && input_offset_along_axis < input_dim_along_axis) {
        if (input_offset_along_axis < 0) input_offset_along_axis += input_dim_along_axis;
        int64_t input_offset;
        if (IsOuterAxis) {
          input_offset = input_offset_along_axis * input_row_size + indices_row_size_fmd.mod(id);
        } else {
          input_offset = indices_row_size_fmd.div(id) * input_row_size + input_offset_along_axis;
        }
        if (IsGather) {
          func(value + work, src_data + input_offset);
        } else {
          func(output_data + input_offset, value + work);
        }
      }
      id += threads_per_block;
    }
  }

  if (IsGather) {
    id = start;
#pragma unroll
    for (int work = 0; work < thread_worksize; ++work) {
      if (id < N) {
        output_data[id] = value[work];
        id += threads_per_block;
      }
    }
  }
}

template <typename T, typename TIndex, bool IsGather, bool IsStridedIndices, typename TFunc>
__global__ void _GatherScatterElementsKernel(const T* src_data, const TIndex* indices_data, T* output_data,
                                             const int64_t rank, const int64_t input_dim_along_axis,
                                             const int64_t input_stride_along_axis,
                                             const TArray<int64_t> masked_input_strides,
                                             const TArray<fast_divmod> indices_fdms,
                                             const TArray<int64_t> indices_strides, const TFunc& func, CUDA_LONG N) {
  CUDA_LONG start = threads_per_block * thread_worksize * blockIdx.x + threadIdx.x;
  CUDA_LONG id;
  T value[thread_worksize];

  if (!IsGather) {
    id = start;
#pragma unroll
    for (int work = 0; work < thread_worksize; ++work) {
      if (id < N) {
        value[work] = src_data[id];
        id += threads_per_block;
      }
    }
  }

  id = start;
#pragma unroll
  for (int work = 0; work < thread_worksize; ++work) {
    if (id < N) {
      int64_t input_offset = 0;
      int64_t indices_offset = IsStridedIndices ? 0 : id;
      int remain = id;
      int q;
#pragma unroll
      for (auto i = 0; i < indices_fdms.Capacity(); i++) {
        if (i >= rank) {
          break;
        }

        indices_fdms[i].divmod(remain, q, remain);
        input_offset += masked_input_strides[i] * q;
        if (IsStridedIndices) indices_offset += indices_strides[i] * q;
      }

      int64_t input_offset_along_axis = static_cast<int64_t>(indices_data[indices_offset]);
      if (input_offset_along_axis >= -input_dim_along_axis && input_offset_along_axis < input_dim_along_axis) {
        if (input_offset_along_axis < 0) input_offset_along_axis += input_dim_along_axis;
        input_offset += input_offset_along_axis * input_stride_along_axis;

        if (IsGather) {
          func(value + work, src_data + input_offset);
        } else {
          func(output_data + input_offset, value + work);
        }
      }

      id += threads_per_block;
    }
  }

  if (IsGather) {
    id = start;
#pragma unroll
    for (int work = 0; work < thread_worksize; ++work) {
      if (id < N) {
        output_data[id] = value[work];
        id += threads_per_block;
      }
    }
  }
}

#define LAUNCH_GATHER_SCATTER_ELEMENTS_2D_KERNEL(src_data, is_outer_axis, is_gather)                               \
  _GatherScatterElements2DKernel<T, TIndex, is_outer_axis, is_gather, decltype(func)>                              \
      <<<blocksPerGrid, threads_per_block, 0, stream>>>(src_data, indices_data, output_data, input_dim_along_axis, \
                                                        indices_fdms[0], input_row_size, func, N)

template <typename T, typename TIndex>
void GatherElementsImpl(cudaStream_t stream, const int64_t rank, const int64_t axis, const T* input_data,
                        const int64_t input_dim_along_axis, const int64_t input_stride_along_axis,
                        const TArray<int64_t>& masked_input_strides, const TIndex* indices_data,
                        const int64_t indices_size, const TArray<fast_divmod>& indices_fdms,
                        const TArray<int64_t>& indices_strides, T* output_data) {
  CUDA_LONG N = static_cast<CUDA_LONG>(indices_size);
  int blocksPerGrid = static_cast<int>(CeilDiv(N, threads_per_block * thread_worksize));
  auto func = FuncAssignment<T>();
  if (rank == 2) {
    if (axis == 0) {
      int64_t input_row_size = input_stride_along_axis;
      LAUNCH_GATHER_SCATTER_ELEMENTS_2D_KERNEL(input_data, true, true);
    } else {
      int64_t input_row_size = masked_input_strides[0];
      LAUNCH_GATHER_SCATTER_ELEMENTS_2D_KERNEL(input_data, false, true);
    }
    return;
  }

  if (indices_strides.Size() == 0) {
    // Save one divmod in kernel if axis is the last dim.
    int64_t new_rank = rank == axis + 1 ? rank - 1 : rank;
    _GatherScatterElementsKernel<T, TIndex, true, false, decltype(func)>
        <<<blocksPerGrid, threads_per_block, 0, stream>>>(input_data, indices_data, output_data, new_rank,
                                                          input_dim_along_axis, input_stride_along_axis,
                                                          masked_input_strides, indices_fdms, indices_strides, func, N);
    return;
  }

  _GatherScatterElementsKernel<T, TIndex, true, true, decltype(func)><<<blocksPerGrid, threads_per_block, 0, stream>>>(
      input_data, indices_data, output_data, rank, input_dim_along_axis, input_stride_along_axis,
      masked_input_strides, indices_fdms, indices_strides, func, N);
}

template <typename T, typename TIndex, typename TFunc>
Status ScatterElementsImplInternal(cudaStream_t stream, const int64_t rank, const int64_t axis, const T* input_data,
                                   const int64_t input_size, const int64_t input_dim_along_axis,
                                   const int64_t input_stride_along_axis, const TArray<int64_t>& masked_input_strides,
                                   const TIndex* indices_data, const int64_t indices_size,
                                   const TArray<fast_divmod>& indices_fdms, const TArray<int64_t>& indices_strides,
                                   const T* updates_data, T* output_data, const TFunc& func) {
  if (input_data != output_data) {
    CUDA_RETURN_IF_ERROR(
        cudaMemcpyAsync(output_data, input_data, input_size * sizeof(T), cudaMemcpyDeviceToDevice, stream));
  }

  if (indices_size == 0) return Status::OK();

  CUDA_LONG N = static_cast<CUDA_LONG>(indices_size);
  int blocksPerGrid = static_cast<int>(CeilDiv(N, threads_per_block * thread_worksize));
  if (rank == 2) {
    if (axis == 0) {
      int64_t input_row_size = input_stride_along_axis;
      LAUNCH_GATHER_SCATTER_ELEMENTS_2D_KERNEL(updates_data, true, false);
    } else {
      int64_t input_row_size = masked_input_strides[0];
      LAUNCH_GATHER_SCATTER_ELEMENTS_2D_KERNEL(updates_data, false, false);
    }
    return Status::OK();
  }

  if (indices_strides.Size() == 0) {
    // Save one divmod in kernel if axis is the last dim.
    int64_t new_rank = rank == axis + 1 ? rank - 1 : rank;
    _GatherScatterElementsKernel<T, TIndex, false, false, decltype(func)>
        <<<blocksPerGrid, threads_per_block, 0, stream>>>(updates_data, indices_data, output_data, new_rank,
                                                          input_dim_along_axis, input_stride_along_axis,
                                                          masked_input_strides, indices_fdms, indices_strides, func, N);
    return Status::OK();
  }

  _GatherScatterElementsKernel<T, TIndex, false, true, decltype(func)><<<blocksPerGrid, threads_per_block, 0, stream>>>(
      updates_data, indices_data, output_data, rank, input_dim_along_axis, input_stride_along_axis,
      masked_input_strides, indices_fdms, indices_strides, func, N);
  return Status::OK();
}

template <typename T, typename TIndex>
Status ScatterElementsImpl(cudaStream_t stream, const int64_t rank, const int64_t axis, const T* input_data,
                           const int64_t input_size, const int64_t input_dim_along_axis,
                           const int64_t input_stride_along_axis, const TArray<int64_t>& masked_input_strides,
                           const TIndex* indices_data, const int64_t indices_size,
                           const TArray<fast_divmod>& indices_fdms, const TArray<int64_t>& indices_strides,
                           const T* updates_data, T* output_data) {
  return ScatterElementsImplInternal(stream, rank, axis, input_data, input_size, input_dim_along_axis,
                                     input_stride_along_axis, masked_input_strides, indices_data, indices_size,
                                     indices_fdms, indices_strides, updates_data, output_data, FuncAssignment<T>());
}

#define GATHER_SCATTER_ELEMENTS_SPECIALIZED_TINDEX_IMPL(T, TIndex)                                                \
  template void GatherElementsImpl<T, TIndex>(                                                                    \
      cudaStream_t stream, const int64_t rank, const int64_t axis, const T* input_data,                           \
      const int64_t input_dim_along_axis, const int64_t input_stride_along_axis,                                  \
      const TArray<int64_t>& masked_input_strides, const TIndex* indices_data, const int64_t indices_size,        \
      const TArray<fast_divmod>& indices_fdms, const TArray<int64_t>& indices_strides, T* output_data);           \
  template Status ScatterElementsImpl<T, TIndex>(                                                                 \
      cudaStream_t stream, const int64_t rank, const int64_t axis, const T* input_data, const int64_t input_size, \
      const int64_t input_dim_along_axis, const int64_t input_stride_along_axis,                                  \
      const TArray<int64_t>& masked_input_strides, const TIndex* indices_data, const int64_t indices_size,        \
      const TArray<fast_divmod>& indices_fdms, const TArray<int64_t>& indices_strides, const T* updates_data,     \
      T* output_data);

#define GATHER_SCATTER_ELEMENTS_SPECIALIZED_IMPL(T)           \
  GATHER_SCATTER_ELEMENTS_SPECIALIZED_TINDEX_IMPL(T, int32_t) \
  GATHER_SCATTER_ELEMENTS_SPECIALIZED_TINDEX_IMPL(T, int64_t)

// GatherElementsGrad needs atomic_add which supports float types only, so use half, float and double for 16, 32, and 64
// bits data respectively.
GATHER_SCATTER_ELEMENTS_SPECIALIZED_IMPL(int8_t)
GATHER_SCATTER_ELEMENTS_SPECIALIZED_IMPL(half)
GATHER_SCATTER_ELEMENTS_SPECIALIZED_IMPL(float)
GATHER_SCATTER_ELEMENTS_SPECIALIZED_IMPL(double)

#ifdef ENABLE_TRAINING

template <class T>
struct FuncAtomicAdd {
  __device__ __inline__ void operator()(T* a, const T* b) const { atomic_add(a, *b); }
};

template <typename T, typename TIndex>
Status GatherElementsGradImpl(cudaStream_t stream, const int64_t rank, const int64_t axis,
                              const int64_t input_dim_along_axis, const int64_t input_stride_along_axis,
                              const TArray<int64_t>& masked_input_strides, const TIndex* indices_data,
                              const int64_t indices_size, const TArray<fast_divmod>& indices_fdms,
                              const TArray<int64_t>& indices_strides, const T* updates_data, T* output_data) {
  // Give output_data as the input_data parameter by intention,
  // to skip input_data copy, which is not applicable for GatherElementsGrad.
  return ScatterElementsImplInternal(stream, rank, axis, output_data, 0, input_dim_along_axis, input_stride_along_axis,
                                     masked_input_strides, indices_data, indices_size, indices_fdms, indices_strides,
                                     updates_data, output_data, FuncAtomicAdd<T>());
}

#define GATHER_ELEMENTS_GRAD_SPECIALIZED_TINDEX_IMPL(T, TIndex)                                                       \
  template Status GatherElementsGradImpl<T, TIndex>(                                                                  \
      cudaStream_t stream, const int64_t rank, const int64_t axis, const int64_t input_dim_along_axis,                \
      const int64_t input_stride_along_axis, const TArray<int64_t>& masked_input_strides, const TIndex* indices_data, \
      const int64_t indices_size, const TArray<fast_divmod>& indices_fdms, const TArray<int64_t>& indices_strides,    \
      const T* updates_data, T* output_data)

#define GATHER_ELEMENTS_GRAD_SPECIALIZED_SCATTER_ADD_IMPL(T) \
  GATHER_ELEMENTS_GRAD_SPECIALIZED_TINDEX_IMPL(T, int32_t);  \
  GATHER_ELEMENTS_GRAD_SPECIALIZED_TINDEX_IMPL(T, int64_t);

GATHER_ELEMENTS_GRAD_SPECIALIZED_SCATTER_ADD_IMPL(half)
GATHER_ELEMENTS_GRAD_SPECIALIZED_SCATTER_ADD_IMPL(float)
GATHER_ELEMENTS_GRAD_SPECIALIZED_SCATTER_ADD_IMPL(double)

#endif

}  // namespace cuda
}  // namespace onnxruntime
