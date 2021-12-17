/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/hpu_op.h"
#include "habana_kernels/binary_kernels.h"

// Except bfloat16, all other types are computed in following type
#define COMMON_COMPUTATION_TYPE_TPC c10::ScalarType::Float

// For use in div_rounding_mode
#define StrModeTruncate "trunc"

namespace habana {

void FloorDivideOperator::AddNode(
    synapse_helpers::graph& graph,
    at::Stack& stack,
    const std::vector<bool>& is_output_persistent_list) {
  _TORCH_WARN_ONCE(
      "floor_divide is deprecated, and will be removed in a future version of pytorch."
      "It currently rounds toward 0 (like the \'trunc\' function NOT \'floor\')."
      "This results in incorrect rounding for negative values."
      "To keep the current behavior, use torch.div(a, b, rounding_mode=\'trunc\'),"
      "or for actual floor division, use torch.div(a, b, rounding_mode=\'floor\'). (function operator())");

  const at::Tensor self = stack_tensor(stack, 0);
  const at::Tensor other = stack_tensor(stack, 1);
  std::string rounding_mode = StrModeTruncate;
  std::vector<at::Tensor> tensors = {self, other};

  const at::ScalarType& final_result_type = at::result_type(self, other);

  const at::ScalarType& computation_type =
      (c10::ScalarType::BFloat16 == final_result_type)
      ? c10::ScalarType::BFloat16
      : COMMON_COMPUTATION_TYPE_TPC;
  const std::string opStringSuffix =
      "_fwd_" + habana_helpers::name_suffix_from_type(computation_type);

  auto shape_out = BinaryOperator::compute_output_shape(self, other);

  auto divOp = BuildOp(
      graph,
      "div" + opStringSuffix,
      {syn_in(0), syn_in(1)},
      {{shape_out, computation_type}});

  auto makeIntegerOp = BuildOp(
      graph,
      rounding_mode + opStringSuffix,
      {divOp.at(0).get()},
      {{shape_out, computation_type, is_output_persistent_list[0], 0}});

  syn_out(0) = std::move(makeIntegerOp[0]);
}
} // namespace habana
