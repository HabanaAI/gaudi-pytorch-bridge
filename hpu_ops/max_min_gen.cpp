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

static void DummyOutput(
    synapse_helpers::graph& graph,
    PytorchKernelContextPtr& p_context_,
    bool persistent,
    bool external) {
  p_context_->syn_outputs_.emplace_back(habana_helpers::create_tensor(
      p_context_->pt_outputs_.at(1), graph, persistent, external));
}

void MinMaxOut::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  auto dim = stack.at(1).toInt();
  auto keepdim = stack.at(2).toBool();
  auto shape = MinMaxOutputShape(stack)[0];

  std::vector<NodeAttr::NodeOutputAttr> output_attrs{
      {shape, ScalarType(), 0}, {shape, c10::ScalarType::Int, 1}};
  std::vector<NodeAttr::NodeOutputAttr> output_attrs_greco{
      {shape, ScalarType(), 0}};

  const bool greco_device = is_Greco_device();

  if (greco_device) {
    p_context_->syn_outputs_.pop_back();

    DummyOutput(
        graph,
        p_context_,
        IsOutputPersistent(1),
        GetOutputMetaData(1).external);
  }
  auto reduce_max = HandleReductionDimAndKeepdim(
      this,
      graph,
      self,
      {syn_in(0)},
      dim,
      keepdim,
      guid_,
      greco_device ? output_attrs_greco : output_attrs);

  if (!is_Greco_device()) {
    syn_out(1) = std::move(reduce_max[1]);
  }
  syn_out(0) = std::move(reduce_max[0]);
}

void MaxDimOp::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  auto dim = stack.at(1).toInt();
  bool keepdim = stack.at(2).toBool();
  auto outshape = MinMaxOutputShape(stack)[0];
  size_t size = 0;
  const auto& params = FillMinMaxParams(stack, size);
  if (self.dim() == 0) {
    auto res = BuildOp(
        graph,
        "memcpy_" + habana_helpers::name_suffix_from_type(ScalarType()),
        {syn_in(0)},
        {{outshape, ScalarType(), 0}});
    syn_out(0) = std::move(res[0]);
    syn_out(1) = std::move(res[0]);
  } else {
    if (keepdim) {
      auto max_dim = BuildOp(
          graph,
          guid_,
          {syn_in(0)},
          {{outshape, ScalarType(), 0}, {outshape, c10::ScalarType::Int, 1}},
          params.get(),
          size);

      syn_out(0) = std::move(max_dim[0]);
      syn_out(1) = std::move(max_dim[1]);
    } else {
      auto shape = self.sizes().vec();
      dim = c10::maybe_wrap_dim(dim, self.dim(), true);
      shape[dim] = 1;

      auto max_dim = BuildOp(
          graph,
          guid_,
          {syn_in(0)},
          {{shape, ScalarType()}, {shape, c10::ScalarType::Int}},
          params.get(),
          size);
      auto max =
          ReshapeHelper(graph, max_dim[0].get(), outshape, ScalarType(), 0);
      auto max_indices = ReshapeHelper(
          graph, max_dim[1].get(), outshape, c10::ScalarType::Int, 1);
      syn_out(0) = std::move(max);
      syn_out(1) = std::move(max_indices);
    }
  }
}
} // namespace habana
