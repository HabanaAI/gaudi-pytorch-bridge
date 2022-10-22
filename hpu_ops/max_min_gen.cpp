/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/max.h"
#include "generated/min.h"
#include "reduction_template.h"

namespace habana {

sizes_vec MinMaxOutputShape(const at::Stack& stack) {
  const torch::Tensor& self = stack_tensor(stack, 0);
  auto dim = stack.at(1).toInt();
  auto keepdim = stack.at(2).toBool();
  auto shapes = ReductionOutputShape(self, dim, keepdim)[0];
  return {shapes, shapes};
}

template <>
LazyMinMax<std::tuple<at::Tensor, at::Tensor>>::LazyMinMax(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&)>& out_shapes_fn)
    : habana_lazy::LazyOp<std::tuple<at::Tensor, at::Tensor>>(
          qualstring,
          inputs,
          out_shapes_fn,
          -1) {}

template <>
std::tuple<at::Tensor, at::Tensor> LazyMinMax<
    std::tuple<at::Tensor, at::Tensor>>::get_result_overrideable() {
  auto inputs = get_inputs();
  auto t = inputs.at(0).toTensor();
  auto out_shape = MinMaxOutputShape(inputs)[0];
  at::Tensor min = habana_lazy::empty_hpu_lazy(
      out_shape, t.options(), t.suggest_memory_format(), false);
  at::Tensor min_indices = habana_lazy::empty_hpu_lazy(
      out_shape,
      t.options().dtype(c10::ScalarType::Long),
      t.suggest_memory_format(),
      false);
  return {min, min_indices};
}

std::shared_ptr<void> FillMinMaxParams(const at::Stack& stack, size_t& size) {
  PARAMS_STUB(ns_Reduction::Params);
  auto dim = stack.at(1).toInt();
  dim = (dim >= 0) ? static_cast<int>(stack.at(0).toTensor().dim()) - 1 - dim
                   : -(dim + 1);

  params->reductionDimension = dim;
  return params;
}

void MinMaxOut::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  auto dim = stack.at(1).toInt();
  auto keepdim = stack.at(2).toBool();

  auto shape = MinMaxOutputShape(stack)[0];

  auto reduce_max = HandleReductionDimAndKeepdim(
      this,
      graph,
      self,
      {syn_in(0)},
      dim,
      keepdim,
      guid_,
      {{shape, ScalarType(), 0}, {shape, c10::ScalarType::Int, 1}});

  syn_out(0) = std::move(reduce_max[0]);
  syn_out(1) = std::move(reduce_max[1]);
}

void MinMaxNoDim::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto self = stack.at(0).toTensor();

  auto shape = AllAnyOutputShape(stack)[0];

  auto min_max = HandleReductionDimAndKeepdim(
      this,
      graph,
      self,
      {syn_in(0)},
      {},
      false,
      guid_,
      {{shape, ScalarType(), 0}, {shape, c10::ScalarType::Int}});

  syn_out(0) = std::move(min_max[0]);
}

} // namespace habana
