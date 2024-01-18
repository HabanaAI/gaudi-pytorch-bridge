/*******************************************************************************
 * Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
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
