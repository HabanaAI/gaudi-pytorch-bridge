/**
* Copyright (c) 2021-2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#pragma once

#include <memory>
#include <queue>
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
  c10::SmallVector<int64_t, 8> lookup_data;
};

struct LaunchDynamicShapes {
  std::vector<at::Tensor> ds_tensors;
  std::vector<std::vector<int64_t>> patch_values;
};

struct DynamicPatchingData {
  std::queue<LaunchDynamicShapes> launch_shapes;
};
using InputPatchFnPtr = std::function<void(
    c10::SmallVectorImpl<torch::jit::IValue*>&,
    c10::SmallVectorImpl<habana::graph::SymIntData>&,
    c10::SmallVectorImpl<std::vector<int64_t>>&,
    c10::SmallVectorImpl<std::vector<std::pair<int64_t, int64_t>>>&,
    std::vector<c10::IValue>&,
    LaunchDynamicShapes&)>;
using InputPatchPair = std::pair<InputPatchFnPtr, std::vector<int64_t>>;

struct DynamicGraphMetaData {
  torch::jit::Stack ds_stack;
  std::unordered_map<int64_t, habana::graph::SymIntData>
      ds_tensor_to_scalar_map;
  std::map<int64_t, std::vector<int64_t>> ds_tensor_to_tensor_map;
  std::map<int64_t, std::vector<std::pair<int64_t, int64_t>>> ds_mixed_map;
  std::vector<InputPatchPair> ds_input_patching_list;
  std::vector<size_t> remove_input_indexes;
  std::vector<torch::jit::Node*> negative_size_nodes;
  bool static_fallback;
};

int64_t GetSymintValue(torch::jit::Stack&, uint64_t);
std::string GetDynamicTensorName(const std::string&, synTensorType type);
template <typename T>
std::vector<T> GetH2DTensorHostData(at::Tensor& tensor);

} // namespace graph
} // namespace habana
