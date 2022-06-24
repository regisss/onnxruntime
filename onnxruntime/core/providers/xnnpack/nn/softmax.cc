// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/op_kernel.h"
#include "core/providers/cpu/math/softmax_shared.h"
#include "core/providers/xnnpack/detail/utils.h"

#include <xnnpack.h>

namespace onnxruntime {
namespace xnnpack {

class Softmax final : public OpKernel {
 public:
  Softmax(const OpKernelInfo& info) : OpKernel{info} {
    const auto& node = info.node();
    opset_ = node.SinceVersion();

    int64_t axis;
    Status status = info.GetAttr<int64_t>("axis", &axis);

    if (status.IsOK()) {
      axis_ = gsl::narrow_cast<int>(axis);
    } else {
      if (opset_ < 13) {
        axis_ = 1;  // opset-12 and below, the default axis value is 1
      } else {
        axis_ = -1;  // opset-13, the default axis value is -1
      }
    }
    log_softmax_ = info.GetKernelDef().OpName() == "LogSoftmax";

    //we have check it in GetCapability
    auto input_defs = node.InputDefs();

    ORT_ENFORCE(GetType(*input_defs[0], kernel_dtype));
    const auto& x_shape = input_defs[0]->Shape();
    if (axis_ == -1) {
      axis_ = x_shape->dim_size() - 1;
    }
    ORT_ENFORCE(axis_ == x_shape->dim_size() - 1, "XNNPACK EP doesn't support softmax when axis is not the last dim:");
    uint32_t channels = gsl::narrow_cast<uint32_t>(x_shape->dim(axis_).dim_value());
    xnn_status xstatus;
    struct xnn_operator* p;
    if (kernel_dtype == ONNX_NAMESPACE::TensorProto_DataType_UINT8) {
      const Tensor* X_zero_point = nullptr;
      const Tensor* Y_zero_point = nullptr;
      const Tensor* X_scale = nullptr;
      const Tensor* Y_scale = nullptr;
      ORT_ENFORCE(info.TryGetConstantInput(InputTensors::IN_X_SCALE, &X_scale),
                  "X_scale input was not constant initializer. XNNPACK EP should not have asked for the node. Node name:",
                  node.Name());
      ORT_ENFORCE(info.TryGetConstantInput(InputTensors::IN_X_ZERO_POINT, &X_zero_point),
                  "X_zero_point input was not constant initializer. XNNPACK EP should not have asked for the node. Node name:",
                  node.Name());
      ORT_ENFORCE(info.TryGetConstantInput(InputTensors::IN_Y_SCALE, &Y_scale),
                  "Y_scale input was not constant initializer. XNNPACK EP should not have asked for the node. Node name:",
                  node.Name());
      ORT_ENFORCE(info.TryGetConstantInput(InputTensors::IN_Y_ZERO_POINT, &Y_zero_point),
                  "Y_zero_point input was not constant initializer. XNNPACK EP should not have asked for the node. Node name:",
                  node.Name());

      ORT_ENFORCE(IsScalarOr1ElementVector(X_scale),
                  "Input x_scale must be a scalar or 1D tensor of size 1");
      ORT_ENFORCE(X_zero_point == nullptr || IsScalarOr1ElementVector(X_zero_point),
                  "input x_zero_point must be a scalar or 1D tensor of size 1 if given");
      ORT_ENFORCE(IsScalarOr1ElementVector(Y_scale),
                  "input y_scale must be a scalar or 1D tensor of size 1");
      ORT_ENFORCE(Y_zero_point == nullptr || IsScalarOr1ElementVector(Y_zero_point),
                  "input y_zero_point must be a scalar or 1D tensor of size 1 if given");

      quant_param_.X_zero_point_value = *(X_zero_point->template Data<uint8_t>());
      quant_param_.X_scale_value = *(X_scale->template Data<float>());
      quant_param_.Y_zero_point_value = *(Y_zero_point->template Data<uint8_t>());
      quant_param_.Y_scale_value = *(Y_scale->template Data<float>());
      xstatus = xnn_create_softmax_nc_qu8(
          channels,
          channels,
          channels,
          quant_param_.X_scale_value,
          gsl::narrow_cast<uint8_t>(quant_param_.Y_zero_point_value),
          quant_param_.Y_scale_value,
          0,  // flags,
          &p);
    } else if (kernel_dtype == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
      xstatus = xnn_create_softmax_nc_f32(
          channels,
          channels,
          channels,
          0,  // flags,
          &p);
    } else {
      ORT_ENFORCE(0, "error kernel type input, expected uint8|float");
    }
    ORT_ENFORCE(xstatus == xnn_status_success, "xnn_create_softmax_nc_f32 failed. Status:", xstatus);
    op0_.reset(p);
  }

  Status Compute(OpKernelContext* ctx) const override;

 private:
  Status ComputeImpl(const Tensor& input, Tensor& output, size_t axis,
                     concurrency::ThreadPool* thread_pool) const;


  Status ComputeImplOpset13(const Tensor& input, Tensor& output, size_t axis,
                            concurrency::ThreadPool* thread_pool, OpKernelContext* ctx) const;

  int axis_;
  int opset_;
  bool log_softmax_;
  int32_t kernel_dtype = 0;
  XnnpackOperator op0_ = nullptr;
  enum InputTensors : int {
    IN_X = 0,
    IN_X_SCALE = 1,
    IN_X_ZERO_POINT = 2,
    IN_Y_SCALE = 3,
    IN_Y_ZERO_POINT = 4,
  };
  QuantParam quant_param_;
};

// compute method of Softmax
Status Softmax::Compute(OpKernelContext* ctx) const {
  const auto* X = ctx->Input<Tensor>(0);
  const auto& X_shape = X->Shape();
  size_t rank = X_shape.NumDimensions();
  auto* Y = ctx->Output(0, X_shape);

  // edge case. one or more dims with value of 0. nothing to do
  if (X_shape.Size() == 0) {
    return Status::OK();
  }

  const size_t axis = static_cast<size_t>(HandleNegativeAxis(axis_, rank));
  concurrency::ThreadPool* thread_pool = ctx->GetOperatorThreadPool();

  if (opset_ < 13) {
    return ComputeImpl(*X, *Y, axis, thread_pool);
  } else {
    return ComputeImplOpset13(*X, *Y, axis, thread_pool, ctx);
  }
}

// opset-12 and below
Status Softmax::ComputeImpl(const Tensor& input, Tensor& output, size_t axis,
                               concurrency::ThreadPool* /*thread_pool*/) const {
  const auto& X_shape = input.Shape();
  const size_t N = X_shape.SizeToDimension(axis);
  // const size_t D = X_shape.SizeFromDimension(axis);
  xnn_status status;
  if (kernel_dtype == ONNX_NAMESPACE::TensorProto_DataType_UINT8) {
    status = xnn_setup_softmax_nc_qu8(
        op0_.get(),
        N,
        input.template Data<uint8_t>(),
        output.template MutableData<uint8_t>(),
        nullptr);
  }
  else {
    status = xnn_setup_softmax_nc_f32(
        op0_.get(),
        N,
        input.template Data<float>(),
        output.template MutableData<float>(),
        nullptr);
  }
  ORT_ENFORCE(status == xnn_status_success, "xnn_setup_softmax_nc_type failed. Status:", status);
  status = xnn_run_operator(op0_.get(), nullptr);
  if (status != xnn_status_success) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "xnn_run_operator returned ", status);
  }
  return Status::OK();
}
// opset-13 and above
Status Softmax::ComputeImplOpset13(const Tensor& input, Tensor& output, size_t axis,
                                      concurrency::ThreadPool* thread_pool, OpKernelContext* /*ctx*/) const {
  return ComputeImpl(input, output, axis, thread_pool);
}

// Register an alternate version of this kernel that supports the channels_last
// attribute in order to consume and produce NHWC tensors.
ONNX_OPERATOR_VERSIONED_KERNEL_EX(Softmax, kOnnxDomain, 1, 11, kXnnpackExecutionProvider,
                                  KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                                  Softmax);
ONNX_OPERATOR_VERSIONED_KERNEL_EX(QLinearSoftmax, kMSDomain, 1, 11, kXnnpackExecutionProvider,
                                  KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<uint8_t>()),
                                  Softmax);
ONNX_OPERATOR_KERNEL_EX(Softmax, kOnnxDomain, 12, kXnnpackExecutionProvider,
                        KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                        Softmax);

}  // namespace xnnpack
}  // namespace onnxruntime
