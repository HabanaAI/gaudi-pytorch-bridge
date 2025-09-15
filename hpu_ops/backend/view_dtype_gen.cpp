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

#include "generated/backend/view.h"

namespace habana {

OutputMetaDataVector ViewDtypeMeta(const at::Stack& stack) {
  const auto& self = stack[0].toTensor();
  const auto dtype = stack[1].toScalarType();
  auto sizes = self.sizes().vec();
  const auto size_ratio =
      static_cast<float>(scalarTypeToTypeMeta(dtype).itemsize()) /
      self.element_size();
  sizes[self.dim() - 1] /= size_ratio;
  OutputMetaData meta;
  meta.dtype = dtype;
  meta.shape = sizes;
  return {meta};
}

void ViewDtype::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  HABANA_ASSERT(stack[0].isTensor(), "Input arg 0 expected to be tensor");
  HABANA_ASSERT(stack[1].isScalar(), "Input arg 1 needs to be of scalar type");
  auto meta = ViewDtypeMeta(stack)[0];
  auto result = OpBackend::BuildNode(
      this,
      graph,
      {"reinterpret_cast", {syn_in(0)}, {{meta.shape, meta.dtype, 0}}});
  syn_out(0) = std::move(result[0]);
}

} // namespace habana
