/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "generated/gelu.h"
#include "generated/gelu_backward.h"
#include "hpu_op_helper.h"
#include "op_backend.h"

namespace habana {
std::shared_ptr<void> FillGeluParams(
    const at::Stack& stack,
    size_t& size,
    int approx_index) {
  PARAMS_STUB(ns_GeluKernel::Params);
  if (GET_ENV_FLAG_NEW(PT_HPU_FORCE_TANH_FOR_GELU)) {
    params->approximation = true;
    return params;
  } else {
    params->approximation = stack.at(approx_index).to<std::string>() == "tanh";
    return params;
  }
}

std::shared_ptr<void> FillGeluFwdParams(const at::Stack& stack, size_t& size) {
  return FillGeluParams(stack, size, 1 /*Approximation Index in Fwd pass*/);
}

std::shared_ptr<void> FillGeluBwdParams(const at::Stack& stack, size_t& size) {
  return FillGeluParams(stack, size, 2 /*Approximation Index in Bwd pass*/);
}

std::vector<synapse_helpers::tensor> GeluCommonFunc(
    OpBackend* op,
    synapse_helpers::graph& graph,
    std::vector<synTensor> input,
    const at::IntArrayRef outshape,
    std::shared_ptr<void> params,
    size_t size,
    c10::optional<int> final_result_index = c10::nullopt) {
  if (synapse_helpers::HPURegistrar::get_device().type() ==
      synDeviceType::synDeviceGreco) {
    return OpBackend::BuildNode(
        op,
        graph,
        {"gelu_fwd_" + habana_helpers::name_suffix_from_type(op->ScalarType()),
         std::move(input),
         {{outshape, op->ScalarType(), final_result_index}}});
  }

  return OpBackend::BuildNode(
      op,
      graph,
      {"gelu_fwd_" + habana_helpers::name_suffix_from_type(op->ScalarType()),
       std::move(input),
       {{outshape, op->ScalarType(), final_result_index},
        {outshape, op->ScalarType()}},
       params.get(),
       size});
}

void Gelu::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  const auto& outshape = stack_tensor(stack, 0).sizes();
  size_t size = 0;
  auto params = FillGeluFwdParams(stack, size);
  auto Gelu =
      GeluCommonFunc(this, graph, {syn_in(0)}, outshape, params, size, 0);
  syn_out(0) = std::move(Gelu[0]);
}

void GeluBwd::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  const auto& outshape = stack_tensor(stack, 0).sizes();
  std::vector<synapse_helpers::tensor> t_retain;
  size_t size = 0;
  auto params = FillGeluBwdParams(stack, size);
  auto Gelu = GeluCommonFunc(this, graph, {syn_in(1)}, outshape, params, size);

#if 0
  bool force_tanh_path;
  if (GET_ENV_FLAG_NEW(PT_HPU_FORCE_TANH_FOR_GELU)) {
    force_tanh_path = true;
  } else {
    force_tanh_path = stack.at(2).to<std::string>() == "tanh";
  }
  if (force_tanh_path) {
    // Formula: tf.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))
    // step 1: x*x*x
    auto pow_3 = BuildOp(
        graph,
        "mult_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
        {syn_in(1), syn_in(1)},
        {{outshape, ScalarType()}});
    pow_3 = BuildOp(
        graph,
        "mult_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
        {syn_in(1), pow_3[0].get()},
        {{outshape, ScalarType()}});
    // step 2:  0.044715 * tf.pow(x, 3)
    double val_1 = 0.044715;
    auto constant = ConstantHelper(graph, val_1, ScalarType(), outshape);
    auto mul_1 = BuildOp(
        graph,
        "mult_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
        {pow_3[0].get(), constant.get()},
        {{outshape, ScalarType()}});
    // step 3: Add (x + 0.044715 * tf.pow(x, 3))
    auto add_1 = BuildOp(
        graph,
        "add_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
        {syn_in(1), mul_1[0].get()},
        {{outshape, ScalarType()}});
    // step 4: np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))
    double sqrt_1 = M_2_SQRTPI * M_SQRT1_2;
    auto constant_1 = ConstantHelper(graph, sqrt_1, ScalarType(), outshape);
    auto mul_2 = BuildOp(
        graph,
        "mult_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
        {add_1[0].get(), constant_1.get()},
        {{outshape, ScalarType()}});
    // step 5: tf.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))
    t_retain = BuildOp(
        graph,
        "tanh_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
        {mul_2[0].get()},
        {{outshape, ScalarType()}});
  } else {
    // Formula: erf(x/1.4142135623730951)
    double val = 1.4142135623730951;
    auto constant = ConstantHelper(graph, val, ScalarType(), outshape);
    auto div = BuildOp(
        graph,
        "div_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
        {syn_in(1), constant.get()},
        {{outshape, ScalarType()}});
    t_retain = BuildOp(
        graph,
        "erf_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
        {div[0].get()},
        {{outshape, ScalarType()}});
  }
#endif
  auto Gelu_bwd = BuildOp(
      graph,
      "gelu_bwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {syn_in(0), syn_in(1), Gelu[1].get()},
      {{outshape, ScalarType(), 0}},
      params.get(),
      size);

  syn_out(0) = std::move(Gelu_bwd[0]);
}
} // namespace habana
