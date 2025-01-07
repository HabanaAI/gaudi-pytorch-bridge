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
#include "habana_kernels/basic_kernels.h"
#include "hpu_ops/hpu_op_helper.h"

namespace sh = synapse_helpers;

namespace habana {

HPU_OP_BACKEND(_StridedInsert_Backend)

struct StridedInsert_Backend : _StridedInsert_Backend {
  StridedInsert_Backend(int device_id, c10::ScalarType scalar_type)
      : _StridedInsert_Backend(
            device_id,
            "strided_insert",
            scalar_type,
            {},
            {0},
            {},
            false) {}
};

void _StridedInsert_Backend::AddNode(sh::graph& graph, const at::Stack& stack) {
  size_t size = 0;
  PARAMS_STUB(synStridedOpParams);
  StridedInsertOperator::compute_params(*this, *params, stack, graph);

  auto num_syn_inputs = GetSynInputs().size();
  std::vector<synTensor> syn_inputs;
  syn_inputs.reserve(num_syn_inputs);
  for (size_t i = 0; i < num_syn_inputs; ++i) {
    syn_inputs.emplace_back(syn_in(i));
  }

  const auto& outshape = stack_tensor(stack, 0).sizes();
  auto result = BuildOp(
      graph,
      guid_,
      std::move(syn_inputs),
      {{outshape, ScalarType(), 0}},
      params.get(),
      size);
  syn_out(0) = std::move(result[0]);
}

static const auto& kr_strided_insert = KernelRegistry().REGISTER_HPU_BACKEND(
    "hpu::strided_insert_",
    StridedInsert_Backend);

} // namespace habana