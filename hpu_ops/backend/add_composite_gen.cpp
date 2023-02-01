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

#include "hpu_ops/common/add_composite_gen.h"
#include "generated/backend/addcdiv.h"
#include "generated/backend/addcmul.h"

constexpr int inp_idx = 0; // index of input(self)
constexpr int oth1_idx = 1; // index of other1
constexpr int oth2_idx = 2; // index of other2
constexpr int val_idx = 3; // index of value

namespace habana {

sizes_vec AddCOpsOutputShape(const at::Stack& stack) {
  const torch::Tensor& self = stack_tensor(stack, inp_idx);
  const torch::Tensor& other1 = stack_tensor(stack, oth1_idx);
  const torch::Tensor& other2 = stack_tensor(stack, oth2_idx);
  auto tmp = at::infer_size(self.sizes(), other1.sizes());
  return {at::infer_size(tmp, other2.sizes())};
}

void AddCOpBE::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  const at::Tensor self = stack_tensor(stack, inp_idx);
  const at::Tensor other1 = stack_tensor(stack, oth1_idx);
  const at::Tensor other2 = stack_tensor(stack, oth2_idx);

  std::vector<synapse_helpers::tensor> mul, variable_op;

  // variable_op_inputs = {other1, other2}
  std::vector<synTensor> variable_op_inputs{syn_in(oth1_idx), syn_in(oth2_idx)};
  // if necessary (i.e, if value != 1 in y = input + value * (other1 op other2)
  // where op = mul or div) do multiplication with value
  if (!stack.at(val_idx).isNone()) {
    mul = BuildOp( // other1 * value
        graph,
        MULT_GUID + habana_helpers::name_suffix_from_type(ScalarType()),
        {syn_in(oth1_idx), syn_in(val_idx)},
        {{other1.sizes(), ScalarType()}});
    // variable_op_inputs = {other1 * value, other2}
    variable_op_inputs = {mul[0].get(), syn_in(oth2_idx)};
  }

  // Based on the guid_, do mult/div/other binary op
  auto outsize_variable_op = at::infer_size(other1.sizes(), other2.sizes());
  variable_op = BuildOp(
      graph, guid_, variable_op_inputs, {{outsize_variable_op, ScalarType()}});

  // Finally add op with self
  std::vector<synTensor> add_op_inputs{syn_in(inp_idx), variable_op[0].get()};
  auto outshape = AddCOpsOutputShape(stack)[0];

  auto add_op = BuildOp(
      graph,
      "add_" + habana_helpers::name_suffix_from_type(ScalarType()),
      add_op_inputs,
      {{outshape, ScalarType(), 0}});

  // output
  syn_out(0) = std::move(add_op[0]);
}

} // namespace habana
