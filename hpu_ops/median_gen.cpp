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
#include "hpu_op_helper.h"

namespace habana {
constexpr size_t selfPositionInArgList = 0;
constexpr size_t reduction_axisInArgList = 1;
constexpr size_t keepdimInArgList = 2;
constexpr size_t ksmallest = 0;

template <>
LazyMedian<at::Tensor>::LazyMedian(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&, bool)>& out_shapes_fn)
    : habana_lazy::LazyOp<at::Tensor>(qualstring, inputs, out_shapes_fn, -1) {}

template <>
at::Tensor LazyMedian<at::Tensor>::get_result_overrideable() {
  const auto& inputs = habana_lazy::LazyOp<at::Tensor>::get_inputs();
  const auto& t = inputs.at(selfPositionInArgList).toTensor();
  auto shape = MedianOutputShape(inputs)[selfPositionInArgList];
  return habana_lazy::empty_hpu_lazy(
      shape, t.options(), t.suggest_memory_format(), false);
}

template <>
LazyMediandim<::std::tuple<at::Tensor, at::Tensor>>::LazyMediandim(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&, bool)>& out_shapes_fn)
    : habana_lazy::LazyOp<::std::tuple<at::Tensor, at::Tensor>>(
          qualstring,
          inputs,
          out_shapes_fn,
          -1) {}

template <>
::std::tuple<at::Tensor, at::Tensor> LazyMediandim<
    ::std::tuple<at::Tensor, at::Tensor>>::get_result_overrideable() {
  const auto& inputs =
      habana_lazy::LazyOp<::std::tuple<at::Tensor, at::Tensor>>::get_inputs();
  const auto& t = inputs.at(selfPositionInArgList).toTensor();
  auto shape = MediandimOutputShape(inputs)[selfPositionInArgList];
  auto values = habana_lazy::empty_hpu_lazy(
      shape, t.options(), t.suggest_memory_format(), false);
  auto indices = habana_lazy::empty_hpu_lazy(
      shape, t.options().dtype(at::kLong), t.suggest_memory_format(), false);
  return ::std::tuple<at::Tensor, at::Tensor>(values, indices);
}

sizes_vec MedianOutputShape(const at::Stack& stack, bool) {
  static_cast<void>(stack);
  return {{}};
}

sizes_vec MediandimOutputShape(const at::Stack& stack, bool) {
  auto self = stack.at(selfPositionInArgList).toTensor();
  auto self_size = self.sizes().vec();
  int64_t reduction_axis = c10::maybe_wrap_dim(
      stack[reduction_axisInArgList].toInt(), self.dim(), /*wrap_scalar=*/true);

  bool keep_dim = stack[keepdimInArgList].toBool();
  std::vector<int64_t> outshape = {self_size};
  if (keep_dim)
    outshape[reduction_axis] = 1;
  else {
    std::vector<int64_t>::iterator itr = outshape.begin() + reduction_axis;
    outshape.erase(itr);
  }
  return {outshape, outshape};
}

void Median::AddNode(
    synapse_helpers::graph& graph,
    at::Stack& stack,
    const std::vector<bool>& is_output_persistent_list) {
  auto self = stack_tensor(stack, selfPositionInArgList);
  auto self_size = self.sizes().vec();
  int64_t reduction_axis = 0;
  if (stack.size() > 1) {
    reduction_axis = c10::maybe_wrap_dim(
        stack[reduction_axisInArgList].toInt(),
        self.dim(),
        /*wrap_scalar=*/true);
  }

  std::vector<synapse_helpers::tensor> reshaped_inp;
  std::vector<int64_t> reshape_size = {self.numel()};
  std::vector<int64_t> reshape_outshape = {reshape_size};

  if (stack.size() == 1) {
    reshaped_inp = BuildOp(
        graph,
        "reshape",
        {syn_in(selfPositionInArgList)},
        {{reshape_outshape, ScalarType(), false}});
  }

  synBeamParams Topk_params{};
  std::vector<int64_t> topk_outshape;

  if (stack.size() == 1) {
    topk_outshape = {self.numel()};
    Topk_params.bsw = self.numel();
    Topk_params.axis = reduction_axis;
    Topk_params.bottomK = ksmallest;
  } else {
    topk_outshape = self_size;
    Topk_params.bsw = self_size[reduction_axis];
    Topk_params.axis = self.ndimension() - reduction_axis - 1;
    Topk_params.bottomK = ksmallest;
  }

  std::vector<synapse_helpers::tensor> Topk;
  if (stack.size() == 1) {
    Topk = BuildOp(
        graph,
        "topk",
        {reshaped_inp[0].get()},
        {{topk_outshape[0], ScalarType()}, {topk_outshape[0], ScalarType()}},
        &Topk_params,
        sizeof(Topk_params));
  } else {
    Topk = BuildOp(
        graph,
        "topk",
        {syn_in(selfPositionInArgList)},
        {{topk_outshape, ScalarType()}, {topk_outshape, ScalarType()}},
        &Topk_params,
        sizeof(Topk_params));
  }

  synSliceParams Slice_params{};
  std::vector<int64_t> Slice_outshape;

  if (stack.size() == 1) {
    Slice_outshape.push_back(1);
    Slice_params.axes[0] = reduction_axis;
    Slice_params.starts[0] = self.numel() / 2;
    Slice_params.ends[0] = self.numel() / 2;
    Slice_params.steps[0] = 1;
  } else {
    Slice_outshape = {self_size};
    Slice_outshape[reduction_axis] = 1;

    for (int idx = 0; idx < self.ndimension(); ++idx) {
      if (idx == (self.ndimension() - reduction_axis - 1)) {
        Slice_params.axes[idx] = idx;
        Slice_params.starts[idx] = self_size[self.ndimension() - idx - 1] / 2;
        Slice_params.ends[idx] = self_size[self.ndimension() - idx - 1] / 2;
      } else {
        Slice_params.axes[idx] = idx;
        Slice_params.starts[idx] = 0;
        Slice_params.ends[idx] = self_size[self.ndimension() - idx - 1];
      }
      Slice_params.steps[idx] = 1;
    }
  }

  auto median_value = BuildOp(
      graph,
      "slice",
      {Topk[0].get()},
      {{Slice_outshape, ScalarType(), is_output_persistent_list[0], 0}},
      &Slice_params,
      sizeof(Slice_params));
  syn_out(0) = std::move(median_value[0]);

  if (stack.size() > 1) {
    auto median_index = BuildOp(
        graph,
        "slice",
        {Topk[1].get()},
        {{Slice_outshape, ScalarType(), is_output_persistent_list[0], 1}},
        &Slice_params,
        sizeof(Slice_params));
    syn_out(1) = std::move(median_index[0]);
  }
}
} // namespace habana
