/*******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
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

#include <memory>
#include <string>
#include <vector>

#include <torch/csrc/jit/ir/ir.h>
#include "backend/synapse_helpers/layout_utils.h"

namespace habana {
namespace graph {

using IVal = torch::jit::IValue;
using IValPtrShared = std::shared_ptr<IVal>;
using CValPtr = const torch::jit::Value*;
using ValueIvalueMap = std::unordered_map<CValPtr, IValPtrShared>;

struct SymIntData {
  std::vector<int64_t> values;
};

using InputPatchFnPtr = std::function<void(
    c10::SmallVectorImpl<torch::jit::IValue*>&,
    c10::SmallVectorImpl<habana::graph::SymIntData>&,
    std::vector<c10::IValue>&)>;
using InputPatchPair = std::pair<InputPatchFnPtr, std::vector<int64_t>>;

struct DynamicGraphMetaData {
  torch::jit::Stack ds_stack;
  std::unordered_map<int64_t, habana::graph::SymIntData>
      ds_tensor_to_scalar_map;
  std::vector<InputPatchPair> ds_input_patching_list;
  std::vector<size_t> remove_input_indexes;
};

int64_t GetSymintValue(torch::jit::Stack&, uint64_t);
std::string GetDynamicTensorName(const std::string&, synTensorType type);
template <typename T>
std::vector<T> GetH2DTensorHostData(at::Tensor& tensor);

} // namespace graph
} // namespace habana
