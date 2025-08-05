/**
 * Copyright (c) 2025 Intel Corporation
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

#include "hpu_ops/hpu_op_helper.h"
#include "hpu_ops/op_backend.h"

namespace habana {
struct UniqueDimParams_t {
  c10::ScalarType dtype;
  std::vector<int64_t> sizes;
  int64_t numel;
  int64_t dim;
  bool sorted;
  bool return_inverted;
  bool return_counts;
};
struct UniqueDimEager : OpBackend {
  UniqueDimEager(int device_id, c10::ScalarType scalar_type);
  void AddNode(synapse_helpers::graph&, const at::Stack&) override;
};

OutputMetaDataVector UniqueDimMeta(const at::Stack& stack) {
  const auto& self = stack_tensor(stack, 0);
  const auto output_shape = self.sizes().vec();
  const auto dim = get_dim_in_tpc_order(stack.at(1).toInt(), self.dim());
  const auto param_shape = std::vector<int64_t>{output_shape.at(dim)};
  std::vector<int64_t> valid_count_shape{1};
  OutputMetaDataVector meta(4);
  meta.at(0).shape = output_shape;
  meta.at(0).dtype = self.scalar_type();
  meta.at(1).shape = valid_count_shape;
  meta.at(1).dtype = at::ScalarType::Long;
  meta.at(2).shape = param_shape;
  meta.at(2).dtype = at::ScalarType::Long;
  meta.at(3).shape = param_shape;
  meta.at(3).dtype = at::ScalarType::Long;
  return meta;
}
} // namespace habana
