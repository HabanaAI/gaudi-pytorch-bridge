/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "generated/log_sigmoid_backward.h"
#include "generated/log_sigmoid_forward.h"
#include "hpu_op_helper.h"

namespace habana {

sizes_vec LogSigmoidfwdOutputShape(const at::Stack& stack) {
  const torch::Tensor& self = stack_tensor(stack, 0);
  return {self.sizes().vec(), self.sizes().vec()};
}

void LogSigmoidForward::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto& inputshape = stack_tensor(stack, 0).sizes();

  // negitive(input)
  auto neg = BuildOp(
      graph,
      "neg_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {syn_in(0)},
      {{inputshape, ScalarType()}});

  // zero constant
  auto zero = ConstantHelper(graph, 0, ScalarType());

  // max(neg(input), 0)
  auto max_vec = BuildOp(
      graph,
      "max_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {neg[0].get(), zero.get()},
      {{inputshape, ScalarType()}});

  // neg(max(neg(input), 0))
  auto buffer_neg = BuildOp(
      graph,
      "neg_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {max_vec[0].get()},
      {{inputshape, ScalarType()}});

  // exp(neg(max(neg(input), 0)))
  auto buffer_left = BuildOp(
      graph,
      "exp_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {buffer_neg[0].get()},
      {{inputshape, ScalarType()}});

  // sub(neg(input), max(neg(input), 0))
  auto buffer_right_input = BuildOp(
      graph,
      "sub_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {neg[0].get(), max_vec[0].get()},
      {{inputshape, ScalarType()}});

  // exp(buffer_right_input)
  auto buffer_right = BuildOp(
      graph,
      "exp_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {buffer_right_input[0].get()},
      {{inputshape, ScalarType()}});

  // add(left_buffer, right_buffer)
  auto buffer = BuildOp(
      graph,
      "add_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {buffer_right[0].get(), buffer_left[0].get()},
      {{inputshape, ScalarType(), 1}});

  // log(buffer)
  auto log = BuildOp(
      graph,
      "log_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {buffer[0].get()},
      {{inputshape, ScalarType()}});

  // add(log, max_vec)
  auto max_vec_log = BuildOp(
      graph,
      "add_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {max_vec[0].get(), log[0].get()},
      {{inputshape, ScalarType()}});

  // neg( add(log, max_vec))
  auto result = BuildOp(
      graph,
      "neg_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {max_vec_log[0].get()},
      {{inputshape, ScalarType(), 0}});

  syn_out(0) = std::move(result[0]);
  syn_out(1) = std::move(buffer[0]);
}

void LogSigmoidBackward::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto& inputshape = stack_tensor(stack, 1).sizes();

  auto zero_vec = ConstantHelper(graph, 0, ScalarType(), inputshape);

  // input < zero_vec
  auto mask = BuildOp(
      graph,
      "less_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {syn_in(1), zero_vec.get()},
      {{inputshape, ScalarType()}});

  // one vector
  auto one_vec = ConstantHelper(graph, 1, ScalarType(), inputshape);

  // neg(one_vec)
  auto one_vec_neg = BuildOp(
      graph,
      "neg_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {one_vec.get()},
      {{inputshape, ScalarType()}});

  // where(mask, neg(one_vec), zero_vec)
  auto max_deriv_vec = BuildOp(
      graph,
      "where_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {mask[0].get(), one_vec_neg[0].get(), zero_vec.get()},
      {{inputshape, ScalarType()}});

  // where(mask, one_vec, neg(one_vec))
  auto sign_vec = BuildOp(
      graph,
      "where_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {mask[0].get(), one_vec.get(), one_vec_neg[0].get()},
      {{inputshape, ScalarType()}});

  // sub(buffer, one_vec)
  auto sub = BuildOp(
      graph,
      "sub_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {syn_in(2), one_vec.get()},
      {{inputshape, ScalarType()}});

  // sub(buffer, one_vec) / buffer
  auto o_div = BuildOp(
      graph,
      "div_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {sub[0].get(), syn_in(2)},
      {{inputshape, ScalarType()}});

  // mult(sing_vec, (sub(buffer, one_vec) / buffer))
  auto o_mult = BuildOp(
      graph,
      MULT_GUID + habana_helpers::name_suffix_from_type(ScalarType()),
      {sign_vec[0].get(), o_div[0].get()},
      {{inputshape, ScalarType()}});

  // max_drive_vec + (sing_vec * (sub(buffer, one_vec) / buffer))
  auto o_add = BuildOp(
      graph,
      "add_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {max_deriv_vec[0].get(), o_mult[0].get()},
      {{inputshape, ScalarType()}});

  // neg(o_add)
  auto o_neg = BuildOp(
      graph,
      "neg_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {o_add[0].get()},
      {{inputshape, ScalarType()}});

  // mult(neg(o_add), grad_output)
  auto output = BuildOp(
      graph,
      MULT_GUID + habana_helpers::name_suffix_from_type(ScalarType()),
      {o_neg[0].get(), syn_in(0)},
      {{inputshape, ScalarType(), 0}});

  syn_out(0) = std::move(output[0]);
}
} // namespace habana
