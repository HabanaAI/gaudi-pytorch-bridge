/**
 * Copyright (c) 2021-2025 Intel Corporation
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

#include "backend/habana_operator.h"
#include "generated/backend/view.h"

namespace habana {
void ViewDtype::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  HABANA_ASSERT(
      stack[0].isTensor(), "Input arg 0 expected to be tensor");
  HABANA_ASSERT(
      stack[1].isScalar(), "Input arg 1 needs to be of scalar type");
  auto self = stack[0].toTensor();
  auto dtype = stack[1].toScalarType();
  auto sizes = self.sizes().vec();
  auto size_ratio =
      float(scalarTypeToTypeMeta(dtype).itemsize()) / self.element_size();
  sizes[self.dim() - 1] /= size_ratio;
  auto result = OpBackend::BuildNode(
      this,
      graph,
      {"reinterpret_cast",
       {syn_in(0)},
       {{sizes, dtype, 0}}});
  syn_out(0) = std::move(result[0]);
}

} // namespace habana
