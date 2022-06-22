/******************************************************************************
 * Copyright (C) 2021 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************
 */
#include "export.h"
#include <torch/csrc/api/include/torch/version.h>
#include <string>

#if ((TORCH_VERSION_MAJOR == 1) && (TORCH_VERSION_MINOR == 12))
#include <torch/csrc/onnx/onnx.h>
#include <map>
#include <memory>
namespace torch {
namespace jit {
extern std::string pretty_print_onnx(
    const std::shared_ptr<Graph>& graph,
    const std::map<std::string, at::Tensor>& initializers,
    int64_t onnx_opset_version,
    bool defer_weight_export,
    ::torch::onnx::OperatorExportTypes operator_export_type,
    bool google_printer,
    bool keep_initializers_as_inputs,
    const std::map<std::string, int>& custom_opsets,
    bool add_node_names);

}
} // namespace torch

#else
#include <torch/csrc/jit/serialization/export.h>
#endif // Torch 1.12 specific hack. Remove on PT 1.13 upgrade

namespace serialize {

constexpr int64_t kONNXOpsetVersion = 8;
std::string GraphToProtoString(const GraphPtr& graph) {
  return torch::jit::pretty_print_onnx(
      graph,
      {},
      kONNXOpsetVersion,
      true,
      ::torch::onnx::OperatorExportTypes::ONNX_ATEN_FALLBACK,
      true,
      true,
      {},
      true);
}

} // namespace serialize
