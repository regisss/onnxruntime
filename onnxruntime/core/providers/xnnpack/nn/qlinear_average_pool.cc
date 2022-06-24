// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/op_kernel.h"
#include "core/providers/cpu/nn/pool_attributes.h"
#include "core/providers/xnnpack/detail/utils.h"
#include "xnnpack.h"
// to sanity check output shape
#include "core/framework/tensorprotoutils.h"

namespace onnxruntime {
namespace xnnpack {

class QLinearAveragePool : public OpKernel {
 public:
  QLinearAveragePool(const OpKernelInfo& info);

  Status Compute(OpKernelContext* context) const override;

 private:
  const PoolAttributes pool_attrs_;
  TensorShapeVector output_dims_;

  XnnpackOperator op0_ = nullptr;
  std::optional<std::pair<uint8_t, uint8_t>> clip_min_max_;
  enum InputTensors : int {
    IN_X = 0,
    IN_X_SCALE = 1,
    IN_X_ZERO_POINT = 2,
    IN_Y_SCALE = 3,
    IN_Y_ZERO_POINT = 4,
  };
  QuantParam quant_param_;
};

QLinearAveragePool::QLinearAveragePool(const OpKernelInfo& info)
    : OpKernel(info),
      pool_attrs_{info, "QLinearAveragePool", info.node().SinceVersion()} {
  const auto& node = info.node();
  uint32_t input_padding_top = gsl::narrow<uint32_t>(pool_attrs_.pads[0]);
  uint32_t input_padding_left = gsl::narrow<uint32_t>(pool_attrs_.pads[1]);
  uint32_t input_padding_bottom = gsl::narrow<uint32_t>(pool_attrs_.pads[2]);
  uint32_t input_padding_right = gsl::narrow<uint32_t>(pool_attrs_.pads[3]);

  uint32_t pooling_height = gsl::narrow<uint32_t>(pool_attrs_.kernel_shape[0]);
  uint32_t pooling_width = gsl::narrow<uint32_t>(pool_attrs_.kernel_shape[1]);
  uint32_t stride_height = gsl::narrow<uint32_t>(pool_attrs_.strides[0]);
  uint32_t stride_width = gsl::narrow<uint32_t>(pool_attrs_.strides[1]);

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

  // get values from any fusion with an activation
  if (std::string activation; info.GetAttr<std::string>("activation", &activation).IsOK()) {
    if (activation == "Clip" || activation == "Relu") {
      std::vector<uint8_t> activation_params;
      /*
      // min/max could be from Clip or Relu
      if (info.GetAttrs<uint8_t>("activation_params", activation_params).IsOK()) {
        if (activation_params.size() == 2) {
          clip_min_max_ = {activation_params[0], activation_params[1]};
        }
      }*/
    }
  }

  uint8_t output_min = clip_min_max_ ? clip_min_max_->first : 0;
  uint8_t output_max = clip_min_max_ ? clip_min_max_->second : 255;

  uint32_t flags = 0;
  if (pool_attrs_.auto_pad == AutoPadType::SAME_UPPER) {
    flags |= XNN_FLAG_TENSORFLOW_SAME_PADDING;
  }

  // input is NHWC and we only support input with 4 dims. we checked C, H, W were all known in the op support checker
  const auto& X_arg = *Node().InputDefs()[0];
  const auto& X_shape = *X_arg.Shape();
  int64_t H = X_shape.dim(1).dim_value();
  int64_t W = X_shape.dim(2).dim_value();
  int64_t C = X_shape.dim(3).dim_value();

  // create NCHW shape to calculate most of the output shape. 'N' is set in Compute.
  TensorShapeVector input_shape{1, C, H, W};
  auto pads = pool_attrs_.pads;
  auto nchw_output_dims = pool_attrs_.SetOutputSize(input_shape, C, &pads);
  output_dims_ = {-1, nchw_output_dims[2], nchw_output_dims[3], nchw_output_dims[1]};

  // TEMPORARY sanity check. If C, H and W are known, the output shape should have been able to be inferred, with the
  // exception of the batch size. Can be removed once we've run more models using xnnpack QLinearAveragePool.
  auto inferred_output_shape = utils::GetTensorShapeFromTensorShapeProto(*Node().OutputDefs()[0]->Shape());
  ORT_ENFORCE(inferred_output_shape[1] == output_dims_[1] &&
                  inferred_output_shape[2] == output_dims_[2] &&
                  inferred_output_shape[3] == output_dims_[3],
              "Shape mismatch between inferred value and calculated value.");

  xnn_status status;
  struct xnn_operator* p;
  status = xnn_create_average_pooling2d_nhwc_qu8(input_padding_top, input_padding_right,
                                                 input_padding_bottom, input_padding_left,
                                                 pooling_height, pooling_width,
                                                 stride_height, stride_width,
                                                 C, C, C,  // channels, input_pixel_stride, output_pixel_stride
                                                 quant_param_.X_zero_point_value,
                                                 quant_param_.X_scale_value,
                                                 quant_param_.Y_zero_point_value,
                                                 quant_param_.Y_scale_value,
                                                 output_min, output_max, flags, &p);
  ORT_ENFORCE(status == xnn_status_success, "xnn_create_average_pooling2d_nhwc_f32 failed. Status:", status);

  op0_.reset(p);
}

Status QLinearAveragePool::Compute(OpKernelContext* context) const {
  const auto& X = *context->Input<Tensor>(0);
  const auto& X_shape = X.Shape();

  int64_t N = X_shape[0];
  int64_t H = X_shape[1];
  int64_t W = X_shape[2];

  // set the N dim to the correct value
  TensorShapeVector output_dims{output_dims_};
  output_dims[0] = N;
  Tensor* Y = context->Output(0, output_dims);

  // empty input
  if (Y->Shape().Size() == 0) {
    return Status::OK();
  }

  xnn_status status = xnn_setup_average_pooling2d_nhwc_qu8(op0_.get(), N, H, W,
                                                           X.Data<uint8_t>(), Y->MutableData<uint8_t>(),
                                                           nullptr /*threadpool */);  // TBD: how to handle threading

  if (status != xnn_status_success) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "xnn_setup_average_pooling2d_nhwc_f32 returned ", status);
  }

  status = xnn_run_operator(op0_.get(), nullptr);
  if (status != xnn_status_success) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "xnn_run_operator returned ", status);
  }

  return Status::OK();
}
ONNX_OPERATOR_TYPED_KERNEL_EX(
    QLinearAveragePool,
    kMSDomain,
    1,
    uint8_t,
    kXnnpackExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<uint8_t>()),
    QLinearAveragePool);

}  // namespace xnnpack
}  // namespace onnxruntime
