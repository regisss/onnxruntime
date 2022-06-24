// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "node_support_checker.h"


namespace onnxruntime {
class NodeUnit;
class GraphViewer;

namespace xnnpack {
// check to see if an ONNX NCHW Conv node is supported by this implementation. the first input and output will be
// converted to NHWC by ORT.
bool IsConvOnnxNodeSupported(const onnxruntime::NodeUnit& nchw_nodeunit, const onnxruntime::GraphViewer& graph);
// check to see if an ONNX NCHW Conv node is supported by this implementation. the first input and output will be
// converted to NHWC by ORT.
bool IsMaxPoolOnnxNodeSupported(const onnxruntime::NodeUnit& nchw_nodeunit, const onnxruntime::GraphViewer& graph);
bool IsAveragePoolOnnxNodeSupported(const onnxruntime::NodeUnit& nodeunit, const onnxruntime::GraphViewer& /*graph*/);
bool IsSoftmaxOnnxNodeSupported(const onnxruntime::NodeUnit& nodeunit, const onnxruntime::GraphViewer& /*graph*/);

}  // namespace xnnpack
}  // namespace onnxruntime
