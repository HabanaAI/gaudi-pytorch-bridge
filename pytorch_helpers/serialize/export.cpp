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

#include <torch/csrc/jit/serialization/export.h>

#include <string>
#include "export.h"

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
