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

template <>
LazyMin<::std::tuple<at::Tensor, at::Tensor>>::LazyMin(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&, bool)>& out_shapes_fn)
    : habana_lazy::LazyOp<::std::tuple<at::Tensor, at::Tensor>>(
          qualstring,
          inputs,
          out_shapes_fn,
          -1) {}

template <>
::std::tuple<at::Tensor, at::Tensor> LazyMin<
    ::std::tuple<at::Tensor, at::Tensor>>::get_result_overrideable() {
  const auto& inputs =
      habana_lazy::LazyOp<::std::tuple<at::Tensor, at::Tensor>>::get_inputs();
  const auto& t = inputs.at(0).toTensor();
  auto shape = MinOutputShape(inputs)[0];
  auto values = habana_lazy::empty_hpu_lazy(
      shape, t.options(), t.suggest_memory_format(), false);
  auto indices = habana_lazy::empty_hpu_lazy(
      shape, t.options().dtype(at::kLong), t.suggest_memory_format(), false);
  return ::std::tuple<at::Tensor, at::Tensor>(values, indices);
}

sizes_vec MinOutputShape(const at::Stack& stack, bool) {
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

std::shared_ptr<void> FillMinParams(const at::Stack& stack, size_t& size) {
  PARAMS_STUB(ns_Reduction::Params);
  auto dim = stack.at(1).toInt();
  dim = (dim >= 0)
      ? static_cast<int>(stack.at(0).toTensor().dim()) - 1 - stack.at(1).toInt()
      : -(dim + 1);

  params->reductionDimension = dim;
  return params;
}

void Min::AddNode(
    synapse_helpers::graph& graph,
    at::Stack& stack,
    const std::vector<bool>& is_output_persistent_list) {
  auto self = stack.at(0).toTensor();
  auto dim = stack.at(1).toInt();

  std::vector<int64_t> outshape{self.sizes().vec()};
  // Negative dimension
  if (dim < 0) {
    dim += self.dim();
  }
  outshape[dim] = 1;

  size_t size = 0;
  const auto& params = FillMinParams(stack, size);
  auto dtype = c10::ScalarType::Int;

  auto reduce_min = BuildOp(
      graph,
      "reduce_min_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {syn_in(0)},
      {{outshape, ScalarType(), is_output_persistent_list[0], 0},
       {outshape, dtype, is_output_persistent_list[1], 1}},
      params.get(),
      size);

  syn_out(0) = std::move(reduce_min[0]);
  syn_out(1) = std::move(reduce_min[1]);
}
} // namespace habana
