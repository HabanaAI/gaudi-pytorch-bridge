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
#include "generated/backend/topk.h"
#include "habana_kernels/index_kernels.h"

namespace habana {

sizes_vec TopkOutputShape(const at::Stack& stack) {
  const torch::Tensor& self = stack_tensor(stack, 0);
  auto k = stack.at(1).isScalar() ? stack.at(1).toInt()
                                  : stack.at(1).toTensor().sizes().vec()[0];
  auto dim_ = stack.at(2).isNone() ? self.dim() : stack.at(2).toInt();
  auto dim = at::maybe_wrap_dim(dim_, self.dim(), /*wrap_scalar=*/true);

  std::vector<int64_t> shape = self.sizes().vec();
  if (shape.size() > 0) {
    shape[dim] = k;
  }
  return {{shape, shape}};
}

void Topk::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  auto k = stack.at(1).isScalar() ? stack.at(1).toInt()
                                  : stack.at(1).toTensor().sizes().vec()[0];
  auto dim_ = stack.at(2).isNone() ? self.dim() : stack.at(2).toInt();
  auto dim = at::maybe_wrap_dim(dim_, self.dim(), /*wrap_scalar=*/true);
  bool largest = stack.at(3).isNone() ? false : stack.at(3).toBool();

  TORCH_CHECK(
      k >= 0 && k <= (self.dim() > 0 ? self.size(dim) : 1),
      "selected index k out of range");
  TORCH_CHECK(
      !(self.dtype() == c10::ScalarType::BFloat16),
      "BFloat16 is not supported");

  auto outshape = TopkOutputShape(stack)[0];

  auto env1 = std::getenv("ENABLE_EXPERIMENTAL_FLAGS");
  bool enable_experimental_flags =
      (env1 != nullptr) && (absl::string_view{env1} != "0");

  auto env2 = std::getenv("ENABLE_TOPK_IN_CGUID");
  bool enable_topk_in_cguid =
      (env2 != nullptr) && (absl::string_view{env2} != "0");

  /*
     To support dynamic shape, the TPC kernel inputs needs to be
     {values_tensor, indices_tensor, null, k_tensor}. The following code
     create 3 additional ops to create the indices tensor: arrnage op ->
     reshape op -> repeat op
  */

  std::vector<synapse_helpers::tensor> result{};

  if (isMetaMode() || graph.is_dynamic_graph()) {
    std::vector<synTensor> syn_inputs{syn_in(0)};
    if (enable_experimental_flags && enable_topk_in_cguid) {
      syn_inputs.emplace_back(nullptr);
      syn_inputs.emplace_back(nullptr);
      CreateShapeTensorInput(graph, ScalarType(), {k}, syn_inputs);

      ns_TopkNodeV2::ParamsV4 params;
      params.axis = self.dim() - dim - 1;
      params.bottomK = !largest;
      params.isVcData = false;
      params.kType = K_TENSOR_SHAPE;
      // Add topk op
      result = BuildOp(
          graph,
          "topk",
          {std::move(syn_inputs)},
          {{outshape, ScalarType(), 0}, {outshape, c10::ScalarType::Int, 1}},
          &params,
          sizeof(params));
    } else {
      torch::jit::Stack temp_stack;
      const auto input_shape = self.sizes();
      int start = 0;
      int limit = input_shape[dim];
      int step = 1;

      // Add arange op - the input tensor for the arange op is also the
      // output tensor
      std::vector<synTensor> range_in{};
      int input_output_depth =
          ArangeOperator::GetOutputSize(start, limit, step);
      std::vector<int64_t> input_output_sizes_vec{input_output_depth};
      c10::IntArrayRef input_output_shape(
          input_output_sizes_vec.data(), input_output_sizes_vec.size());

      ns_RangeKernel::Params arrangeparam{};
      arrangeparam.start.i = start;
      arrangeparam.limit.i = limit;
      arrangeparam.delta.i = step;
      // Allocate idst if its not added from frontend.
      std::vector<int64_t> sizes_vec{step, limit, start};
      c10::IntArrayRef idst_sizes(sizes_vec.data(), sizes_vec.size());
      CreateShapeTensorInput(
          graph,
          c10::ScalarType::Int,
          idst_sizes,
          range_in,
          INPUT_DESCRIBING_SHAPE_TENSOR);
      auto range_op = BuildOp(
          graph,
          "range_i32",
          range_in,
          {{input_output_shape, c10::ScalarType::Int}},
          &arrangeparam,
          sizeof(arrangeparam));

      // Add reshape op
      auto reshaped_shape = std::vector<int64_t>(self.ndimension(), 1);
      reshaped_shape[dim] = limit;
      auto reshape_op = ReshapeHelper(
          graph, range_op[0].get(), reshaped_shape, c10::ScalarType::Int);

      // Add repeat op
      std::vector<int64_t> repeats = input_shape.vec();
      repeats[dim] = 1;
      int64_t repeat_size = repeats.size();
      ns_TileKernel::ParamsV2 tileparams{};
      for (int64_t i = 0; i < repeat_size; ++i) {
        tileparams.repeat[repeat_size - i - 1] = repeats[i];
      }

      std::vector<synTensor> repeat_in = {reshape_op.get()};
      CreateShapeTensorInput(
          graph,
          c10::ScalarType::Int,
          repeats,
          repeat_in,
          INPUT_DESCRIBING_SHAPE_TENSOR);

      std::vector<int64_t> repeat_outshape(repeats.size());
      int64_t num_new_dimensions = repeats.size() - reshaped_shape.size();
      std::vector<int64_t> padded_size(num_new_dimensions, 1);
      padded_size.insert(
          padded_size.end(), reshaped_shape.begin(), reshaped_shape.end());
      for (size_t i = 0; i < repeats.size(); ++i) {
        repeat_outshape[i] = padded_size[i] * repeats[i];
      }

      auto repeat_op = BuildOp(
          graph,
          "tile_fwd_i32",
          repeat_in,
          {{repeat_outshape, c10::ScalarType::Int}},
          &tileparams,
          sizeof(tileparams));
      syn_inputs.emplace_back(repeat_op[0].get());
      syn_inputs.emplace_back(nullptr);
      CreateShapeTensorInput(graph, ScalarType(), {k}, syn_inputs);

      synBeamParams params;
      params.bsw = k;
      params.axis = self.dim() - dim - 1;
      params.bottomK = !largest;

      // Add topk op
      result = BuildOp(
          graph,
          "topk",
          {std::move(syn_inputs)},
          {{outshape, ScalarType(), 0}, {outshape, c10::ScalarType::Int, 1}},
          &params,
          sizeof(params));
    }
  } else {
    if (enable_experimental_flags && enable_topk_in_cguid) {
      ns_TopkNodeV2::ParamsV4 params;
      params.axis = self.dim() - dim - 1;
      params.bottomK = !largest;
      params.bsw = k;
      params.kType = K_TENSOR_NONE;
      // Add topk op
      result = BuildOp(
          graph,
          "topk",
          {syn_in(0)},
          {{outshape, ScalarType(), 0}, {outshape, c10::ScalarType::Int, 1}},
          &params,
          sizeof(params));
    } else {
      synBeamParams params;
      params.bsw = k;
      params.axis = self.dim() - dim - 1;
      params.bottomK = !largest;

      // Add topk op
      result = BuildOp(
          graph,
          "topk",
          {syn_in(0)},
          {{outshape, ScalarType(), 0}, {outshape, c10::ScalarType::Int, 1}},
          &params,
          sizeof(params));
    }
  }
  syn_out(0) = std::move(result[0]);
  syn_out(1) = std::move(result[1]);
}
} // namespace habana
