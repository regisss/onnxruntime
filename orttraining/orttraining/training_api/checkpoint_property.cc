// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "onnx/defs/tensor_proto_util.h"
#include "core/common/inlined_containers.h"
#include "core/platform/path_lib.h"
#include "core/platform/env.h"
#include "core/framework/tensorprotoutils.h"
#include "orttraining/training_api/include/checkpoint_property.h"

namespace onnxruntime {
namespace training {
namespace api {

namespace {

template <typename T>
void ParsePropertyFromTensorProto(const ONNX_NAMESPACE::TensorProto& tensor_proto,
                                  std::string& name,
                                  PropertyDataType& value) {
  std::vector<int64_t> tensor_shape_vec = utils::GetTensorShapeFromTensorProto(tensor_proto);
  int64_t expected_num_elements = 1;
  for (auto& d : tensor_shape_vec) {
    expected_num_elements *= d;
  }
  ORT_ENFORCE(expected_num_elements == 1, "Only scalar value support for checkpoint property.");
  Path model_path;
  InlinedVector<T> data_vector(1);
  T* p = data_vector.data();
  ORT_THROW_IF_ERROR(utils::UnpackTensor<T>(tensor_proto, model_path, p, expected_num_elements));
  name = tensor_proto.name();
  value = data_vector[0];
}

}  // namespace

void PropertyBag::AddProperty(const ONNX_NAMESPACE::TensorProto& tensor_proto) {
  ORT_ENFORCE(named_properties_.find(tensor_proto.name()) == named_properties_.end(),
              "Duplicated property named ", tensor_proto.name());

  if (!IsSupportedDataType(tensor_proto.data_type())) {
    ORT_THROW("Failed to add property from tensorproto: float, int64_t and std::string data types supported only.");
  }

  auto data_type = tensor_proto.data_type();
  std::string prop_name;
  PropertyDataType prop_value;
  switch (data_type) {
    case ONNX_NAMESPACE::TensorProto::FLOAT: {
      ParsePropertyFromTensorProto<float>(tensor_proto, prop_name, prop_value);
      break;
    }
    case ONNX_NAMESPACE::TensorProto::STRING: {
      ParsePropertyFromTensorProto<std::string>(tensor_proto, prop_name, prop_value);
      break;
    }
    case ONNX_NAMESPACE::TensorProto::INT64: {
      ParsePropertyFromTensorProto<int64_t>(tensor_proto, prop_name, prop_value);
      break;
    }
    default:
      ORT_THROW("Unsupported input data type of ", data_type);
  }

  named_properties_.insert({prop_name, prop_value});
}

}  // namespace api
}  // namespace training
}  // namespace onnxruntime
