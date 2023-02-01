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

#include "generated/backend/max.h"
#include "generated/backend/min.h"
#include "hpu_ops/backend/reduction_template.h"

namespace habana {

sizes_vec MinMaxOutputShape(const at::Stack& stack) {
  const torch::Tensor& self = stack_tensor(stack, 0);
  int64_t dim = stack.at(1).toInt();
  bool keepdim = stack.at(2).toBool();
  auto shapes = ReductionOutputShape(self, int64_t(dim), keepdim)[0];
  return {shapes, shapes};
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
