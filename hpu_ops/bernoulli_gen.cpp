
/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/bernoulli.h"
#include "habana_kernels/random_gen_kernels.h"

namespace habana {
std::shared_ptr<void> FillBernoulliParams(size_t& size) {
  PARAMS_STUB(ns_RandomBernoulli::Params);
  return params;
}

template <>
LazyBernoulliOutFrontend<at::Tensor&>::LazyBernoulliOutFrontend(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&, bool)>& out_shapes_fn)
    : habana_lazy::LazyOp<at::Tensor&>(qualstring, inputs, out_shapes_fn) {
  // Generators can't be represented in JIT graph
  // https://github.com/pytorch/pytorch/issues/64005
  LazyBernoulliOutFrontend<at::Tensor&>::get_inputs().at(1) =
      habana_lazy::get_tensor_for_scalar(
          inputs[1].toDouble(), inputs[0].toTensor().options());
  LazyBernoulliOutFrontend<at::Tensor&>::get_inputs().at(2) =
      get_seed_tensor_hpu(inputs.at(2).toOptional<at::Generator>());
}

template <>
at::Tensor& LazyBernoulliOutFrontend<at::Tensor&>::get_result_overrideable() {
  return stack_tensor(LazyBernoulliOutFrontend<at::Tensor&>::get_inputs(), 0);
}

template <typename T>
LazyBernoulliFrontend<T>::LazyBernoulliFrontend(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&, bool)>& out_shapes_fn)
    : habana_lazy::LazyOp<T>(qualstring, inputs, out_shapes_fn) {
  LazyBernoulliFrontend<T>::get_inputs().at(1) =
      habana_lazy::get_tensor_for_scalar(
          inputs[1].toDouble(), inputs[0].toTensor().options());
  LazyBernoulliFrontend<T>::get_inputs().back() =
      get_seed_tensor_hpu(inputs.back().toOptional<at::Generator>());
}
template <typename T>
T LazyBernoulliFrontend<T>::get_result_overrideable() {
  return stack_tensor(LazyBernoulliFrontend<T>::get_inputs(), 0);
}
template struct LazyBernoulliFrontend<at::Tensor&>;
template struct LazyBernoulliFrontend<at::Tensor>;

void Bernoulli::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  // When p is a scalar, convert to a tensor since tpc kernel takes probability
  // as the first and only input
  auto outshape = stack_tensor(stack, 0).sizes();
  // For "outplace" and "out" variant, self tensor is the probability input
  int p_index = IsInplace() ? 1 : 0;
  // For "tensor_out" and "float_out" variant
  if (!IsInplace() &&
      c10::isFloatingType(stack.at(1).toTensor().scalar_type())) {
    p_index = 1;
  }
  int seed_index = p_index + 1;

  auto bcastOp = BuildOp(
      graph,
      "broadcast_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {syn_in(p_index)},
      {{outshape, ScalarType()}});
  size_t size = 0;
  auto params = FillBernoulliParams(size);

  auto dest_type = ScalarType();
  dest_type = ScalarType() == c10::ScalarType::Float ? c10::ScalarType::Int
                                                     : c10::ScalarType::Short;
  auto op = BuildOp(
      graph,
      guid_,
      {bcastOp.at(0).get(), syn_in(seed_index)},
      {{outshape, dest_type}},
      params.get(),
      size);
  auto castOp =
      CastHelper(graph, op.at(0).get(), outshape, dest_type, ScalarType(), 0);
  syn_out(0) = std::move(castOp);
}
} // namespace habana
