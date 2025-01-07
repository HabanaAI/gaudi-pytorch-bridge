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

#include "hpu_ops/hpu_op_helper.h"
#include "hpu_ops/op_backend.h"

namespace habana {
struct UniqueParams_t {
  c10::ScalarType dtype;
  std::vector<int64_t> sizes;
  int64_t numel;
  bool sorted;
  bool inverted;
};
struct UniqueEager : OpBackend {
  UniqueEager(int device_id, c10::ScalarType scalar_type);
  void AddNode(synapse_helpers::graph&, const at::Stack&) override;
};

OutputMetaDataVector Unique2Meta(const at::Stack& stack) {
  const auto& self = stack_tensor(stack, 0);
  int elements = self.numel();
  auto inputShape = self.sizes().vec();
  auto dtype = self.scalar_type();
  std::vector<int64_t> output_shape{elements};
  std::vector<int64_t> valid_count_shape{1};
  OutputMetaDataVector meta(3);
  meta.at(0).shape = output_shape;
  meta.at(0).dtype = dtype;
  meta.at(1).shape = valid_count_shape;
  meta.at(1).dtype = at::ScalarType::Int;
  meta.at(2).shape = output_shape;
  meta.at(2).dtype = at::ScalarType::Long;
  return meta;
}
} // namespace habana
