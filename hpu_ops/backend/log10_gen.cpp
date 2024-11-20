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
#include "generated/backend/_foreach_log10.h"
#include "generated/backend/log10.h"

namespace habana {

void ForeachLog10::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto& tensors = stack[0].toTensorList();
  for (auto i = 0u; i < tensors.size(); ++i) {
    const auto& tensor = tensors[i];
    const at::ScalarType dtype = isIntegralType(tensor.scalar_type(), true)
        ? torch::kFloat32
        : tensor.scalar_type();
    auto out = BuildOp(
        graph,
        get_guid_with_precision("log10_fwd", dtype),
        {syn_in(i)},
        {{{tensor.sizes()}, dtype, i}});
    syn_out(i) = std::move(out[0]);
  }
}
} // namespace habana
