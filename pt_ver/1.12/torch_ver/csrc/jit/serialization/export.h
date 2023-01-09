/*******************************************************************************
 * Copyright (C) 2022 Habana Labs, Ltd. an Intel Company
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

#pragma once

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/onnx/onnx.h>
#include <map>
#include <memory>

namespace torch::jit {

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

} // namespace torch::jit
