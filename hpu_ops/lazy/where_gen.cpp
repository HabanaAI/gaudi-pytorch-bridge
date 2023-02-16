/*******************************************************************************
 * Copyright (C) 2022-2023 Habana Labs, Ltd. an Intel Company
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

#include "backend/synapse_helpers/device_helpers.h"
#include "generated/lazy/where.h"
#include "habana_helpers/dtype_helpers.h"

namespace habana {
FALLBACK_CHECK(
    WhereFallbackCheck,
    const at::Tensor& condition,
    const at::Tensor& self,
    const at::Tensor& other) {
  if (condition.scalar_type() != torch::kBool) {
    return false;
  }

  // After type promotion, it should pick one of these guids
  //  where_fwd_i8
  //  where_fwd_i32
  //  where_fwd_bf16
  //  where_fwd_f32
  //  where_fwd_f16 only for Gaudi2/Gaudi3/Greco
  auto result_type = at::result_type(self, other);
  switch (result_type) {
    case torch::kBool:
    case torch::kInt32:
    case torch::kInt64:
    case torch::kBFloat16:
    case torch::kFloat32:
      return true;
    case torch::kHalf: {
      return synapse_helpers::device_supports_fp16(
          synapse_helpers::HPURegistrar::get_device().type());
    }
    default:
      return false;
  }
}

template <>
WhereFrontend<at::Tensor>::WhereFrontend(
    const std::string& qualstring,
    const std::vector<at::IValue>& stack,
    const std::function<sizes_vec(const at::Stack&)>& out_shapes_fn)
    : habana_lazy::LazyOp<at::Tensor>(qualstring, stack, out_shapes_fn) {
  const auto& self = stack_tensor(stack, 1);
  const auto& other = stack_tensor(stack, 2);
  set_scalar_types({at::result_type(self, other)});
}

template <>
at::Tensor WhereFrontend<at::Tensor>::get_result_overrideable() {
  return {};
}

template <>
WhereFrontend<at::Tensor&>::WhereFrontend(
    const std::string& qualstring,
    const std::vector<at::IValue>& stack,
    const std::function<sizes_vec(const at::Stack&)>& out_shapes_fn)
    : habana_lazy::LazyOp<at::Tensor&>(qualstring, stack, out_shapes_fn) {}

template <>
at::Tensor& WhereFrontend<at::Tensor&>::get_result_overrideable() {
  return get_inputs().back().toTensor();
}

} // namespace habana
