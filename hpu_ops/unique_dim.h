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
  auto inputShape = self.sizes().vec();
  auto dtype = self.scalar_type();
  auto output_shape = self.sizes().vec();
  int64_t dim = -stack.at(1).toInt() + self.dim() - 1;
  auto param_shape = std::vector<int64_t>{output_shape.at(dim)};
  std::vector<int64_t> valid_count_shape{1};
  OutputMetaDataVector meta(4);
  meta.at(0).shape = output_shape;
  meta.at(0).dtype = dtype;
  meta.at(1).shape = valid_count_shape;
  meta.at(1).dtype = at::ScalarType::Long;
  meta.at(2).shape = param_shape;
  meta.at(2).dtype = at::ScalarType::Long;
  meta.at(3).shape = param_shape;
  meta.at(3).dtype = at::ScalarType::Long;
  return meta;
}
} // namespace habana