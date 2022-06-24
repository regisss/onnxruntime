// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "utils.h"
#include <unordered_map>

#include "core/framework/tensorprotoutils.h"
#include "core/graph/indexed_sub_graph.h"
#include "core/graph/node_attr_utils.h"

#include "core/providers/shared/node_unit/node_unit.h"
#include "onnx/defs/attr_proto_util.h"
#include "core/common/safeint.h"
namespace onnxruntime {
namespace xnnpack {

bool GetType(const NodeArg& node_arg, int32_t& type) {
  type = ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED;
  const auto* type_proto = node_arg.TypeAsProto();
  if (!type_proto || !type_proto->has_tensor_type() || !type_proto->tensor_type().has_elem_type()) {
    LOGS_DEFAULT(WARNING) << "NodeArg [" << node_arg.Name() << "] has no input type";
    return false;
  }

  type = type_proto->tensor_type().elem_type();
  return true;
}

bool GetShape(const NodeArg& node_arg, Shape& shape) {
  shape.clear();
  const auto* shape_proto = node_arg.Shape();

  if (!shape_proto) {
    LOGS_DEFAULT(WARNING) << "NodeArg [" << node_arg.Name() << "] has no shape info";
    return false;
  }

  // uses 0 for dynamic dimension, which is the default value for dim.dim_value()
  for (const auto& dim : shape_proto->dim())
    shape.push_back(SafeInt<uint32_t>(dim.dim_value()));

  return true;
}

QuantizedOpType GetQuantizedOpType(const NodeUnit& node_unit) {
  const auto& op_type = node_unit.OpType();
  if (node_unit.UnitType() == NodeUnit::Type::QDQGroup) {
    if (op_type == "Conv")
      return QuantizedOpType::QDQConv;
    else if (op_type == "MaxPool")
      return QuantizedOpType::QDQMaxPool;
    else if (op_type == "AveragePool")
      return QuantizedOpType::QDQAvgPool;
    else if (op_type == "Softmax")
      return QuantizedOpType::QDQSoftmax;
  } else if (node_unit.OpType() == "QLinearConv") {
    return QuantizedOpType::QLinearConv;
  }
  return QuantizedOpType::Unknown;
}

bool IsPaddingTypeSupported(AutoPadType auto_pad) {
  return auto_pad == AutoPadType::NOTSET ||
         auto_pad == AutoPadType::VALID ||
         auto_pad == AutoPadType::SAME_UPPER;
}

typedef std::string ONNXOpType;

static std::unordered_map<QuantizedOpType, ONNXOpType> QDQType_Onnxtype_map = {
    {QuantizedOpType::QDQConv, "QLinearConv"},
    {QuantizedOpType::QDQAvgPool, "QLinearAveragePool"},
    {QuantizedOpType::QDQSoftmax, "QLinearSoftmax"},
    {QuantizedOpType::QDQMaxPool, "QLinearMaxPool"},
};

std::unique_ptr<IndexedSubGraph::MetaDef> FuseQDQGroup(const NodeUnit& unit_node) {
  QuantizedOpType qtype = GetQuantizedOpType(unit_node);
  // create a ComputeCapability for QDQ node.
  std::unique_ptr<IndexedSubGraph::MetaDef> metadef = std::make_unique<IndexedSubGraph::MetaDef>();
  IndexedSubGraph::MetaDef& def = *metadef;
  if (QDQType_Onnxtype_map.count(qtype) == 0) {
    return metadef;
  }
  // inputs
  const auto& inputs = unit_node.Inputs();
  def.name = QDQType_Onnxtype_map[qtype];
  if (qtype == QuantizedOpType::QDQConv) {
    // registration
    def.domain = kMSInternalNHWCDomain;  // node.Domain();  // should always be kMSInternalNHWCDomain
    def.since_version = 10;              // onnx schema version
    if (inputs.size() > 2) {             // with bias
      def.inputs.reserve(9);
    } else {
      def.inputs.reserve(8);
    }
    // x x-scale x-zp w w-scale w-zp
    std::for_each(inputs.cbegin(), inputs.cbegin() + 2,
                  [&def](const NodeUnitIODef& arg) {
                    // keep the number of inputs the same by inserting an empty string for a missing optional input
                    def.inputs.push_back(arg.node_arg.Name());
                    const auto& quant_param = arg.quant_param.value();
                    def.inputs.push_back(quant_param.scale.Name());
                    def.inputs.push_back(quant_param.zero_point ? quant_param.zero_point->Name() : "");
                  });
    // y-scale y-zeropoint
    const auto& y_quant_param = unit_node.Outputs()[0].quant_param.value();
    def.inputs.push_back(y_quant_param.scale.Name());
    def.inputs.push_back(y_quant_param.zero_point ? y_quant_param.zero_point->Name() : "");
    // bias
    if (inputs.size() > 2) {
      def.inputs.push_back(inputs[2].node_arg.Name());
    }
  } else if (qtype == QuantizedOpType::QDQAvgPool || qtype == QuantizedOpType::QDQSoftmax) {
    // registration
    def.domain = kMSDomain;  // should always be kMSDomain
    def.since_version = 1;   // onnx schema version
    //x xsc xzp ysc yzp
    def.inputs.reserve(5);
    // x x-scale x-zp
    std::for_each(inputs.cbegin(), inputs.cend(),
                  [&def](const NodeUnitIODef& arg) {
                    // keep the number of inputs the same by inserting an empty string for a missing optional input
                    def.inputs.push_back(arg.node_arg.Name());
                    const auto& quant_param = arg.quant_param.value();
                    def.inputs.push_back(quant_param.scale.Name());
                    def.inputs.push_back(quant_param.zero_point ? quant_param.zero_point->Name() : "");
                  });
    // y-scale y-zeropoint
    const auto& y_quant_param = unit_node.Outputs()[0].quant_param.value();
    def.inputs.push_back(y_quant_param.scale.Name());
    def.inputs.push_back(y_quant_param.zero_point ? y_quant_param.zero_point->Name() : "");
    //we used the nhwc schema here, and avgpool is layout sensitive
    if (qtype == QuantizedOpType::QDQAvgPool){
      def.attributes["channels_last"] = utils::MakeAttribute(std::string("channels_last"), int64_t(1));
    }
  } else {
    // QDQMaxPool?
  }
  // outputs
  for (const auto& out : unit_node.Outputs()) {
    def.outputs.push_back(out.node_arg.Name());
  }

  // attributes
  // copy existing and add the activation info
  def.attributes.insert(unit_node.GetNode().GetAttributes().begin(),unit_node.GetNode().GetAttributes().end());
  return metadef;
}

// Fuse activation with node. Currently Conv and MaxPool are supported.
std::unique_ptr<IndexedSubGraph::MetaDef> FuseActivation(const Node& node, const Node& activation,
                                                         const GraphViewer& graph) {
  std::unique_ptr<IndexedSubGraph::MetaDef> metadef = std::make_unique<IndexedSubGraph::MetaDef>();
  IndexedSubGraph::MetaDef& def = *metadef;

  // we use the op type/domain to match the static xnnpack Conv or MaxPool kernel
  // registration
  def.name = node.OpType();
  def.domain = node.Domain();  // should always be kMSInternalNHWCDomain
  def.since_version = node.SinceVersion();

  // inputs
  const auto& inputs = node.InputDefs();
  def.inputs.reserve(inputs.size());
  std::for_each(inputs.cbegin(), inputs.cend(),
                [&def](const NodeArg* arg) {
                  // keep the number of inputs the same by inserting an empty string for a missing optional input
                  def.inputs.push_back(arg ? arg->Name() : "");
                });

  // outputs
  def.outputs.push_back(activation.OutputDefs()[0]->Name());

  // attributes
  // copy existing and add the activation info
  def.attributes = node.GetAttributes();

  // use infinity as the default as that's what xnnpack uses if min/max are not set
  float min = -INFINITY;
  float max = INFINITY;

  const auto& activation_type = activation.OpType();
  if (activation_type == "Clip") {
    min = std::numeric_limits<float>::min();
    max = std::numeric_limits<float>::max();
    bool min_max_are_attributes = activation.SinceVersion() == 1 || activation.SinceVersion() == 6;

    if (min_max_are_attributes) {
      ProtoHelperNodeContext nc(activation);
      OpNodeProtoHelper info(&nc);
      min = info.GetAttrOrDefault<float>("min", min);
      max = info.GetAttrOrDefault<float>("max", max);
    } else {
      const auto& clip_inputs = activation.InputDefs();
      const auto num_inputs = clip_inputs.size();

      const auto update_value = [&](size_t idx, float& value_to_set) {
        if (num_inputs > idx) {
          const NodeArg& arg = *clip_inputs[idx];
          if (arg.Exists()) {
            const auto& value = *graph.GetConstantInitializer(arg.Name(), true);
            // these should never be in external data as it makes no sense to put scalars there.
            ORT_ENFORCE(utils::HasExternalData(value) == false,
                        "External data is not supported for the scalar min/max Clip values");

            value_to_set = utils::HasRawData(value)
                               ? *reinterpret_cast<const float*>(value.raw_data().data())
                               : value.float_data()[0];
          }
        }
      };

      update_value(1, min);
      update_value(2, max);
    }
  } else if (activation_type == "Relu") {
    min = 0.f;
  } else {
    ORT_NOT_IMPLEMENTED("No support for fusion of ", node.OpType(), " with ", activation_type);
  }

  InlinedVector<float> activation_params{min, max};
  def.attributes.insert({"activation", utils::MakeAttribute("activation", activation_type)});
  def.attributes.insert({"activation_params", utils::MakeAttribute("activation_params", activation_params)});

  return metadef;
}

}  // namespace xnnpack
}  // namespace onnxruntime
