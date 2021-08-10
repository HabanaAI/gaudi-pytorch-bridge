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

namespace habana {
sizes_vec HabanaOperatorHelper::BinaryOutputShape(
    const torch::Tensor& self,
    const torch::Tensor& other) {
  return {at::infer_size(self.sizes(), other.sizes())};
}

void BinaryOp::AddNode(
    synapse_helpers::graph& graph,
    at::Stack& stack,
    const std::vector<bool>& is_output_persistent_list) {
  const at::Tensor self = stack_tensor(stack, 0);
  const at::Tensor other = stack_tensor(stack, 1);
  const at::ScalarType& result_type = at::result_type(self, other);

  bool do_alpha_mul =
      ScalarInputs().size() and ScalarInputs().at(ScalarId()).toFloat() != 1.;
  int cast_index = -1;
  if (self.scalar_type() != result_type and
      other.scalar_type() == result_type) {
    cast_index = 0;
  } else if (
      self.scalar_type() == result_type and
      other.scalar_type() != result_type) {
    cast_index = 1;
  }

  std::vector<synTensor> binaryop_inputs{syn_in(0), syn_in(1)};
  std::vector<synapse_helpers::tensor> constant, mul, cast;

  if (do_alpha_mul) {
    size_t size = 0;
    const at::Scalar& val = ScalarInputs().at(ScalarId());
    PARAMS_STUB(ns_ConstantKernel::Params);
    if (result_type == c10::ScalarType::Int) {
      get<int>(params->constant) = val.to<int>();
    } else {
      get<float>(params->constant) = val.to<float>();
    }

    constant = BuildOp(
        graph,
        "constant_" + habana_helpers::name_suffix_from_type(result_type),
        {},
        {{1, result_type}},
        params.get(),
        size);
    mul = BuildOp(
        graph,
        "mult_fwd_" + habana_helpers::name_suffix_from_type(result_type),
        {syn_in(1), constant[0].get()},
        {{stack_tensor(stack, 1).sizes(), result_type}});
    binaryop_inputs = {syn_in(0), mul[0].get()};
  }

  if (cast_index >= 0) {
    const std::string& cast_from = habana_helpers::name_suffix_from_type(
        stack_tensor(stack, cast_index).scalar_type());
    const std::string& cast_to =
        habana_helpers::name_suffix_from_type(result_type);
    // Suffix the promoted type
    guid_ = guid_.substr(0, guid_.find_last_of('_') + 1) + cast_to;

    // Insert cast on the input with lower dtype
    cast = BuildOp(
        graph,
        "cast_" + cast_from + "_to_" + cast_to,
        {binaryop_inputs.at(cast_index)},
        {{stack_tensor(stack, cast_index).sizes(), result_type}});

    binaryop_inputs.at(cast_index) = cast.at(0).get();
  }

  auto outshape = HabanaOperatorHelper::BinaryOutputShape(self, other)[0];

  auto op = BuildOp(
      graph,
      guid_,
      binaryop_inputs,
      {{outshape, result_type, is_output_persistent_list[0], true}});

  syn_out(0) = std::move(op[0]);
}
} // namespace habana
