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

#include "hpu_ops/unique.h"

namespace habana {

UniqueEager::UniqueEager(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, {}, scalar_type, {0, 0, 0}, {}, {}, false) {
  SetOutputMetaFn(Unique2Meta);
}

std::vector<synapse_helpers::tensor> UniqueCommon(
    OpBackend* op,
    synapse_helpers::graph& graph,
    UniqueParams_t self_params,
    synTensor self_synin,
    c10::optional<int> final_result_index_0,
    c10::optional<int> final_result_index_1,
    [[maybe_unused]] c10::optional<int> final_result_index_2) {
  int elements = self_params.numel;
  std::vector<int64_t> output_shape{elements};
  std::vector<int64_t> valid_count_shape{1};
  ns_UniqueKernel::ParamsV2 params = {};
  params.sorted = self_params.sorted;
  params.returnCounts = 0;
  params.returnInverse = self_params.inverted;
  params.dim = -5; /// NOTE - arbitrary value based on the documentation
  std::vector<synTensor> inputs = {self_synin};
  auto guid = get_guid_with_precision("unique_fwd", self_params.dtype);
  auto shape_tensor_dtype =
      (common::IsInt64Supported() ? c10::ScalarType::Long
                                  : c10::ScalarType::Int);

  std::vector<NodeAttr::NodeOutputAttr> nodeOutputAttrs = {
      {output_shape, self_params.dtype, final_result_index_0},
      {valid_count_shape, shape_tensor_dtype, final_result_index_1}};

  if (self_params.inverted) {
    nodeOutputAttrs.push_back(
        {output_shape, c10::ScalarType::Long, final_result_index_2});
  }
  return OpBackend::BuildNode(
      op, graph, {guid, inputs, nodeOutputAttrs, &params, sizeof(params)});
}

void UniqueEager::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  auto sorted = stack.at(1).toBool();
  auto inverted = stack.at(2).toBool();

  UniqueParams_t self_params;
  self_params.dtype = self.scalar_type();
  self_params.sizes = self.sizes().vec();
  self_params.numel = self.numel();
  self_params.sorted = sorted;
  self_params.inverted = inverted;

  auto unique = UniqueCommon(this, graph, self_params, syn_in(0), 0, 1, 2);

  syn_out(0) = std::move(unique.at(0));
  syn_out(1) = std::move(unique.at(1));
  if (inverted) {
    syn_out(2) = std::move(unique.at(2));
  }
}
} // namespace habana

static const auto& UniqueKernelRegistry = habana::KernelRegistry().add(
    "hpu::_unique_eager",
    KERNEL_FN_GLOBAL(habana::UniqueEager));
