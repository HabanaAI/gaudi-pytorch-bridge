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

#include "generated/backend/floor_divide.h"
#include "habana_kernels/binary_kernels.h"
#include "pytorch_helpers/habana_helpers/pt_version_check.h"

// Except bfloat16, all other types are computed in following type
#define COMMON_COMPUTATION_TYPE_TPC c10::ScalarType::Float

namespace habana {

void FloorDivideOperator::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  _TORCH_WARN_ONCE(
      "floor_divide is deprecated, and will be removed in a future version of pytorch."
      "It currently rounds toward 0 (like the \'trunc\' function NOT \'floor\')."
      "This results in incorrect rounding for negative values."
      "To keep the current behavior, use torch.div(a, b, rounding_mode=\'trunc\'),"
      "or for actual floor division, use torch.div(a, b, rounding_mode=\'floor\'). (function operator())");

  const at::Tensor self = stack_tensor(stack, 0);
  const auto other = stack.at(1);
  const std::string rounding_mode = "floor";

  const at::ScalarType& final_result_type = other.isScalar()
      ? at::result_type(self, other.toScalar())
      : at::result_type(self, other.toTensor());

  const at::ScalarType& computation_type =
      (c10::ScalarType::BFloat16 == final_result_type)
      ? c10::ScalarType::BFloat16
      : COMMON_COMPUTATION_TYPE_TPC;

  auto shape_out = other.isScalar()
      ? self.sizes().vec()
      : BinaryOperator::compute_output_shape(self, other.toTensor());

  auto divOp = BuildOp(
      graph,
      get_guid_with_precision("div_fwd", computation_type),
      {syn_in(0), syn_in(1)},
      {{shape_out, computation_type}});

  auto makeIntegerOp = BuildOp(
      graph,
      get_guid_with_precision(rounding_mode + "_fwd", computation_type),
      {divOp.at(0).get()},
      {{shape_out, computation_type, 0}});

  syn_out(0) = std::move(makeIntegerOp[0]);
}
} // namespace habana
