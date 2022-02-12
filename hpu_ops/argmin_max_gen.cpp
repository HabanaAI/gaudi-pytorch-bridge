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
#include "habana_kernels/reduction_kernels.h"

namespace habana {

template <>
LazyArgmin<at::Tensor>::LazyArgmin(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&, bool)>& out_shapes_fn)
    : habana_lazy::LazyOp<at::Tensor>(qualstring, inputs, out_shapes_fn, -1) {}

template <>
at::Tensor LazyArgmin<at::Tensor>::get_result_overrideable() {
  const auto& inputs = habana_lazy::LazyOp<at::Tensor>::get_inputs();
  const auto& t = inputs.at(0).toTensor();
  auto shape = ArgMinMaxOutputShape(inputs)[0];
  return habana_lazy::empty_hpu_lazy(
      shape, t.options().dtype(at::kLong), t.suggest_memory_format(), false);
}

std::shared_ptr<void> FillArgMinMaxParams(
    const at::Stack& stack,
    size_t& size) {
  PARAMS_STUB(ns_Reduction::Params);
  auto r_dim = stack.at(1).isNone() ? 0 : stack.at(1).toInt();
  auto ndim = stack.at(0).toTensor().dim();
  r_dim = c10::maybe_wrap_dim(r_dim, ndim, true);
  auto dim = stack.at(1).isNone() ? 0 : ndim - 1 - r_dim;

  params->reductionDimension = dim;
  return params;
}

sizes_vec ArgMinMaxOutputShape(const at::Stack& stack, bool) {
  const torch::Tensor& self = stack_tensor(stack, 0);
  auto isEmptyDim = stack.at(1).isNone();

  const bool keepdim = stack.at(2).toBool();
  std::vector<int64_t> shape{self.sizes().vec()};
  if (isEmptyDim && !keepdim) {
    return {{}};
  } else if (isEmptyDim && keepdim) {
    std::vector<int64_t> shape(self.dim(), 1);
    return {shape};
  } else {
    std::vector<int64_t> dim{stack.at(1).toInt()};
    shape = ReduceOperator::compute_output_shape(self, dim, keepdim);
    return {shape};
  }
}

void ArgMinMax::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  auto isEmptyDim = stack.at(1).isNone();
  const bool keepdim = stack.at(2).toBool();

  auto dim = isEmptyDim
      ? 0
      : c10::maybe_wrap_dim(stack.at(1).toInt(), self.dim(), true);

  auto shape = ArgMinMaxOutputShape(stack)[0];
  std::vector<int64_t> outshape{self.sizes().vec()};

  size_t size = 0;
  const auto& params = FillArgMinMaxParams(stack, size);
  auto dtype = torch::kInt;
  int64_t shape_val = 1;

  // If dim is None, find max/min from the flattened input tensor.
  if (isEmptyDim) {
    for (size_t i = 0; i < outshape.size(); ++i) {
      shape_val *= outshape[i];
    } // flattened input shape

    std::vector<int64_t> output_shape{shape_val};
    auto reshape =
        BuildOp(graph, "reshape", {syn_in(0)}, {{output_shape, ScalarType()}});
    if (!keepdim) {
      auto op = BuildOp(
          graph,
          guid_,
          {reshape[0].get()},
          {{shape, dtype, 0}},
          params.get(),
          size);
      syn_out(0) = std::move(op[0]);
    } else {
      std::vector<int64_t> op_shape{1};
      auto op = BuildOp(
          graph,
          guid_,
          {reshape[0].get()},
          {{op_shape, dtype}},
          params.get(),
          size);
      auto output =
          BuildOp(graph, "reshape", {op[0].get()}, {{shape, dtype, 0}});
      syn_out(0) = std::move(output[0]);
    }
  } else if (!keepdim) { // reduce dim when keepdim is false using reshape.
    outshape[dim] = 1;
    auto op = BuildOp(
        graph, guid_, {syn_in(0)}, {{outshape, dtype}}, params.get(), size);
    auto reshape =
        BuildOp(graph, "reshape", {op[0].get()}, {{shape, dtype, 0}});

    syn_out(0) = std::move(reshape[0]);
  } else { // Direct TPC kernel call when keepdim is true.
    auto op = BuildOp(
        graph, guid_, {syn_in(0)}, {{shape, dtype, 0}}, params.get(), size);
    syn_out(0) = std::move(op[0]);
  }
}
} // namespace habana
