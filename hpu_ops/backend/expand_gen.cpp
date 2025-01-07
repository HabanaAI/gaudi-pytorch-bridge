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
#include <ATen/ExpandUtils.h>
#include "hpu_ops/expand.h"

namespace habana {
InferOutputMetaRetType ExpandOp::OutputShapeInf(
    const torch::jit::Stack& inputs) {
  auto self = inputs[0].toTensor();

  std::vector<int64_t> expandedSizes;
  std::vector<int64_t> expandedStrides;
  InferOutputMetaRetType out;
  if (inputs[1].isIntList()) {
    auto size = inputs[1].toIntList();
    std::tie(expandedSizes, expandedStrides) = at::inferExpandGeometry(
        self.sizes(), self.strides(), at::IntArrayRef(size.vec()));

    habana_helpers::recalc_strides(expandedStrides, expandedSizes);
    out.AddShapeTensor(TensorMetaData(
        expandedSizes,
        expandedStrides,
        self.scalar_type(),
        self.suggest_memory_format()));
  } else {
    auto expand_shape = inputs[1].toTensor();
    expandedSizes = expand_shape.sizes().vec();
    expandedStrides = HabanaOperator::CalculateStrides(
        expandedSizes, self.suggest_memory_format());
  }

  out.AddOutputTensor(TensorMetaData(
      expandedSizes,
      expandedStrides,
      self.scalar_type(),
      self.suggest_memory_format()));
  return out;
}

void ExpandOp::AddNode(
    synapse_helpers::graph& graph,
    [[maybe_unused]] const at::Stack& inputs) {
  const auto& metadata = GetOutputMetaData(0);
  auto final_result_index =
      metadata.persistent ? c10::make_optional<int>(0) : c10::nullopt;
  auto broadcast = BroadcastHelper(
      graph, syn_in(0), metadata.shape, metadata.dtype, final_result_index);
  syn_out(0) = std::move(broadcast);
}

OutputMetaDataVector ExpandMeta(const at::Stack& stack) {
  const auto& self = stack_tensor(stack, 0);
  auto size = stack.at(1).toIntList();
  std::vector<int64_t> expandedSizes;
  std::vector<int64_t> expandedStrides;
  std::tie(expandedSizes, expandedStrides) = at::inferExpandGeometry(
      self.sizes(), self.strides(), at::IntArrayRef(size.vec()));

  OutputMetaDataVector meta(1);
  meta.at(0).shape = expandedSizes;
  meta.at(0).dtype = self.scalar_type();
  return meta;
}

ExpandOp::ExpandOp(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, "broadcast", scalar_type, {0}, {}, {}, false) {
  SetOutputMetaFn(ExpandMeta);
}
} // namespace habana

static const auto& ExpandKernelRegistry = habana::KernelRegistry().add(
    "aten::expand",
    KERNEL_FN_GLOBAL(habana::ExpandOp));
