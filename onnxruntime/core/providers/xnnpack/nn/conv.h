// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "core/framework/allocator.h"
#include "core/providers/cpu/nn/conv_attributes.h"
#include "core/providers/xnnpack/detail/utils.h"
#include "xnnpack.h"

namespace onnxruntime {
class GraphViewer;
class Node;
namespace xnnpack {

class Conv : public OpKernel {
 public:
  enum InputTensors : int {
    IN_X = 0,
    IN_X_SCALE = 1,
    IN_X_ZERO_POINT = 2,
    IN_W = 3,
    IN_W_SCALE = 4,
    IN_W_ZERO_POINT = 5,
    IN_Y_SCALE = 6,
    IN_Y_ZERO_POINT = 7,
    IN_BIAS = 8
  };

  enum OutputTensors : int {
    OUT_Y = 0
  };

 public:
  Conv(const OpKernelInfo& info);

  Status Compute(OpKernelContext* /*context*/) const override;

  // use PrePack to handle the weight layout change as that's not a simple NCHW -> NHWC transpose
  Status PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                 /*out*/ bool& is_packed,
                 /*out*/ PrePackedWeights* prepacked_weights) override;
 private:
  // due to other constraints of this kernel the value of group is either 1 or C, so we can infer that if it's not 1
  // it's a depthwise convolution
  bool IsDepthwise() const { return conv_attrs_.group != 1; }

  ConvAttributes conv_attrs_;
  TensorShapeVector kernel_shape_;
  int64_t C_;
  int64_t M_;
  std::unique_ptr<Tensor> packed_w_;
  const Tensor* B_{nullptr};
  std::optional<std::pair<float, float>> clip_min_max_;

  XnnpackOperator op0_ = nullptr;
  //when xnnpack version has been updated, we can remove the macro
#ifdef XNN_CACHE_ENABLE
  xnn_code_cache code_cache_;
  xnn_caches caches_;
#endif
  QuantParam quant_param_;
  xnn_compute_type conv_type_ = xnn_compute_type_invalid;
};

}  // namespace xnnpack
}  // namespace onnxruntime
