/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include <bitset>
#include "generated/hpu_op.h"

namespace habana {
sizes_vec MinMaxFlattenInputOutputShape(const at::Stack&, bool) {
  return {{}};
}

template <>
LazyMinMax<::std::tuple<at::Tensor, at::Tensor>>::LazyMinMax(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&, bool)>& out_shapes_fn)
    : habana_lazy::LazyOp<::std::tuple<at::Tensor, at::Tensor>>(
          qualstring,
          inputs,
          out_shapes_fn,
          -1) {}

template <>
::std::tuple<at::Tensor, at::Tensor> LazyMinMax<
    ::std::tuple<at::Tensor, at::Tensor>>::get_result_overrideable() {
  const auto& inputs =
      habana_lazy::LazyOp<::std::tuple<at::Tensor, at::Tensor>>::get_inputs();
  const auto& t = inputs.at(0).toTensor();
  auto shape = MinMaxOutputShape(inputs)[0];
  auto values = habana_lazy::empty_hpu_lazy(
      shape, t.options(), t.suggest_memory_format(), false);
  auto indices = habana_lazy::empty_hpu_lazy(
      shape, t.options().dtype(at::kLong), t.suggest_memory_format(), false);
  return ::std::tuple<at::Tensor, at::Tensor>(values, indices);
}

sizes_vec MinMaxOutputShape(const at::Stack& stack, bool) {
  const torch::Tensor& self = stack_tensor(stack, 0);
  auto dim = stack.at(1).toInt();
  auto keepdim = stack.at(2).toBool();
  std::vector<int64_t> shape{self.sizes().vec()};
  // Negative dimension
  if (dim < 0) {
    dim += self.dim();
  }

  if (keepdim) {
    shape[dim] = 1;
  } else {
    shape.erase(shape.begin() + dim);
  }
  return {shape, shape};
}

std::shared_ptr<void> FillMinMaxParams(const at::Stack& stack, size_t& size) {
  PARAMS_STUB(ns_Reduction::Params);
  auto dim = stack.at(1).toInt();
  dim = (dim >= 0)
      ? static_cast<int>(stack.at(0).toTensor().dim()) - 1 - stack.at(1).toInt()
      : -(dim + 1);

  params->reductionDimension = dim;
  return params;
}

void MinMaxFlattenInput::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  size_t size = 0;
  PARAMS_STUB(ns_Reduction::Params);
  auto reshape_outshape = self.numel();

  auto reshape = BuildOp(
      graph, "reshape", {syn_in(0)}, {{reshape_outshape, ScalarType()}});

  // This Node is used in "aten::max(Tensor self) -> Tensor"
  auto max_out = BuildOp(
      graph,
      guid_,
      {reshape[0].get()},
      {{1, ScalarType(), 0}, {1, c10::ScalarType::Int}},
      params.get(),
      size);

  syn_out(0) = std::move(max_out[0]);
}

void MinMax::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  auto dim = stack.at(1).toInt();
  auto keepdim = stack.at(2).toBool();

  auto outshape = self.sizes().vec();
  const auto shape = ComputeOutputShapes(stack, true)[0];
  // Negative dimension
  dim = c10::maybe_wrap_dim(dim, self.dim(), true);
  outshape[dim] = 1;
  size_t size = 0;
  const auto& params = FillParams(stack, size);
  auto dtype = c10::ScalarType::Int;

  if (!keepdim) {
    auto reduce_max = BuildOp(
        graph,
        guid_,
        {syn_in(0)},
        {{outshape, ScalarType()}, {outshape, dtype}},
        params.get(),
        size);

    auto reshape1 = BuildOp(
        graph, "reshape", {reduce_max[0].get()}, {{shape, ScalarType(), 0}});

    auto reshape2 =
        BuildOp(graph, "reshape", {reduce_max[1].get()}, {{shape, dtype, 1}});

    syn_out(0) = std::move(reshape1[0]);
    syn_out(1) = std::move(reshape2[0]);
  } else {
    auto reduce_max = BuildOp(
        graph,
        guid_,
        {syn_in(0)},
        {{outshape, ScalarType(), 0}, {outshape, dtype, 1}},
        params.get(),
        size);

    syn_out(0) = std::move(reduce_max[0]);
    syn_out(1) = std::move(reduce_max[1]);
  }
}

} // namespace habana
