// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>

#include "core/framework/op_kernel.h"
#include "core/graph/indexed_sub_graph.h"
#include "core/providers/common.h"

#include "xnnpack.h"

namespace onnxruntime {
class GraphViewer;
class NodeUnit;
namespace xnnpack {

// from xnnpack subgraph.h
enum xnn_compute_type {
  xnn_compute_type_invalid = 0,
  xnn_compute_type_fp32,
  xnn_compute_type_fp16,
  xnn_compute_type_qc8,
  xnn_compute_type_qs8,
  xnn_compute_type_qu8,
  /*
  xnn_compute_type_fp32_to_fp16,
  xnn_compute_type_fp32_to_qs8,
  xnn_compute_type_fp32_to_qu8,
  xnn_compute_type_fp16_to_fp32,
  xnn_compute_type_qs8_to_fp32,
  xnn_compute_type_qu8_to_fp32,*/
};

struct QuantParam {
  uint8_t X_zero_point_value = 0;
  uint8_t W_zero_point_value = 0;
  uint8_t Y_zero_point_value = 0;

  float X_scale_value = 0;
  float W_scale_value = 0;
  const float *W_scale_arr = 0;
  float Y_scale_value = 0;
};

using Shape = std::vector<uint32_t>;
enum class QuantizedOpType : uint8_t {
  QLinearConv,
  QLinearMaxPool,
  QlinearAvgPool,
  // QDQ operator
  QDQConv,
  QDQMaxPool,
  QDQAvgPool,
  QDQSoftmax,
  Unknown,
};

QuantizedOpType GetQuantizedOpType(const NodeUnit& node_unit);

// forward declaration for this EP's namespace.
template <typename T>
KernelCreateInfo BuildKernelCreateInfo();

struct XnnpackOperatorDeleter {
  void operator()(struct xnn_operator* p) const {
    if (p != nullptr) {
      // Ignore returned value because it fails only when xnnpack wasn't initialized
      xnn_delete_operator(p);
    }
  }
};

bool IsPaddingTypeSupported(AutoPadType auto_pad);

using XnnpackOperator = std::unique_ptr<struct xnn_operator, XnnpackOperatorDeleter>;

std::unique_ptr<IndexedSubGraph::MetaDef> FuseActivation(const Node& conv, const Node& activation,
                                                         const GraphViewer& graph);
std::unique_ptr<IndexedSubGraph::MetaDef> FuseQDQGroup(const NodeUnit& unit_node);

bool GetType(const NodeArg& node_arg, int32_t& type);
bool GetShape(const NodeArg& node_arg, Shape& shape);
bool ParseQuantParamFromInfoByOrder(const OpKernelInfo& info, std::vector<int32_t> scale_zp_indexs, QuantParam& quant_param_, int32_t zptype);
}  // namespace xnnpack
}  // namespace onnxruntime
