/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

//#include "generated/addcdiv.h"
#include "generated/addcmul.h"

namespace habana {

sizes_vec AddCOpsOutputShape(const at::Stack& stack, bool) {
  const torch::Tensor& self = stack_tensor(stack, 0);
  const torch::Tensor& other1 = stack_tensor(stack, 1);
  const torch::Tensor& other2 = stack_tensor(stack, 2);
  auto tmp = at::infer_size(self.sizes(), other1.sizes());
  return {at::infer_size(tmp, other2.sizes())};
}

void AddCOpOut::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  const at::Tensor self = stack_tensor(stack, 0);
  const at::Tensor other1 = stack_tensor(stack, 1);
  const at::Tensor other2 = stack_tensor(stack, 2);

  std::vector<synapse_helpers::tensor> mul, constant, variable_op;
  std::vector<synTensor> vectSynTensor{syn_in(0), syn_in(1), syn_in(2)};

  std::vector<synTensor> variable_op_inputs{
      vectSynTensor.at(1), vectSynTensor.at(2)};
  // if scalar mul if required, only then do cast and do the mult
  if (ScalarInputs().size() and
      ScalarInputs().at(ScalarId()[0]).toFloat() != 1.) {
    size_t size = 0;
    const at::Scalar& scalarVal = ScalarInputs().at(ScalarId()[0]);
    PARAMS_STUB(ns_ConstantKernel::Params);
    if (scalarVal.isIntegral(/*include bools*/ false)) {
      get<int>(params->constant) = scalarVal.to<int>();
    } else {
      get<float>(params->constant) = scalarVal.to<float>();
    }

    // Cast the scalar into target "ScalarType()" and create a tensor with
    // scalar's value
    constant = BuildOp(
        graph,
        "constant_" + habana_helpers::name_suffix_from_type(ScalarType()),
        {},
        {{1, ScalarType()}},
        params.get(),
        size);

    // Do the multiplication with Generated tensor
    mul = BuildOp(
        graph,
        MULT_GUID + habana_helpers::name_suffix_from_type(ScalarType()),
        {vectSynTensor.at(1), constant[0].get()},
        {{other1.sizes(), ScalarType()}});
    variable_op_inputs = {mul[0].get(), vectSynTensor.at(2)};

  } // if(ScalarInputs().size()

  // Based on the guid_, do mult/div/other binary op
  auto outsize_variable_op = at::infer_size(other1.sizes(), other2.sizes());
  variable_op = BuildOp(
      graph, guid_, variable_op_inputs, {{outsize_variable_op, ScalarType()}});

  // Finally add op with self
  std::vector<synTensor> add_op_inputs{
      vectSynTensor.at(0), variable_op[0].get()};
  auto outshape = AddCOpsOutputShape(stack, true)[0];

  auto add_op = BuildOp(
      graph,
      "add_" + habana_helpers::name_suffix_from_type(ScalarType()),
      add_op_inputs,
      {{outshape, ScalarType(), 0}});

  // output
  syn_out(0) = std::move(add_op[0]);
}

} // namespace habana
