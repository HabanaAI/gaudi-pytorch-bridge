/******************************************************************************
 * Copyright (C) 2021-2023 Habana Labs, Ltd. an Intel Company
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

#include "generated/backend/argmax.h"
#include "generated/backend/argmin.h"
#include "hpu_ops/backend/reduction_template.h"

namespace {
auto output_type() {
  return GET_ENV_FLAG_NEW(PT_ENABLE_INT64_SUPPORT) ? c10::ScalarType::Long
                                                   : c10::ScalarType::Int;
}
} // namespace
namespace habana {

sizes_vec ArgMinMaxOutputShape(const at::Stack& stack) {
  const torch::Tensor& self = stack_tensor(stack, 0);

  auto dim = stack.at(1);
  auto is_dim_none = dim.isNone();
  auto dim_vec =
      is_dim_none ? std::vector<int64_t>{} : std::vector<int64_t>{dim.toInt()};

  const bool keepdim = stack.at(2).toBool();
  auto shape = ReductionOutputShape(self, dim_vec, keepdim);

  return {shape};
}

void ArgMinMax::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  const bool keepdim = stack.at(2).toBool();

  auto shape = ArgMinMaxOutputShape(stack)[0];
  auto dtype = output_type();
  auto dim = stack.at(1);
  auto is_dim_none = dim.isNone();

  auto dim_vec =
      is_dim_none ? std::vector<int64_t>{} : std::vector<int64_t>{dim.toInt()};

  auto op = HandleReductionDimAndKeepdim(
      this,
      graph,
      self,
      {syn_in(0)},
      dim_vec,
      keepdim,
      guid_,
      {{shape, dtype, 0}});

  syn_out(0) = std::move(op[0]);
}
} // namespace habana
