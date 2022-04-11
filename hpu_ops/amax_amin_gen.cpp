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
#include "reduction_op_util.h"

namespace habana {
sizes_vec AmaxAminOutputShape(const at::Stack& stack, bool) {
  const torch::Tensor& self = stack_tensor(stack, 0);
  std::vector<int64_t> dim = stack.at(1).toIntList().vec();
  const bool keepdim = stack.at(2).toBool();
  std::vector<int64_t> compute_shape =
      ReduceOperator::compute_output_shape(self, dim, keepdim);
  return {compute_shape};
}

sizes_vec AminmaxOutputShape(const at::Stack& stack, bool) {
  const torch::Tensor& self = stack_tensor(stack, 0);
  auto dim = stack.at(1);
  auto is_dim_none = dim.isNone();
  const bool keepdim = stack.at(2).toBool();

  std::vector<int64_t> shape{self.sizes().vec()};
  if (is_dim_none && !keepdim) {
    return {{}, {}};
  } else if (is_dim_none && keepdim) {
    std::vector<int64_t> shape(self.dim(), 1);
    return {{shape}, {shape}};
  } else {
    std::vector<int64_t> dim_vec{dim.toInt()};
    shape = ReduceOperator::compute_output_shape(self, dim_vec, keepdim);
    return {{shape}, {shape}};
  }
}

void AmaxAmin::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto new_shape = AmaxAminOutputShape(stack)[0];

  auto self = stack.at(0).toTensor();

  const bool keepdim = stack.at(2).toBool();
  auto self_shape = self.sizes().vec();
  auto dim = stack.at(1).toIntVector();

  auto AmaxAmin = HandleReductionDimAndKeepdim(
      this,
      graph,
      {syn_in(0)},
      dim,
      keepdim,
      guid_,
      self_shape,
      new_shape,
      {{{}, ScalarType()}, {self_shape, ScalarType()}});

  syn_out(0) = std::move(AmaxAmin.at(0));
}

static std::vector<synapse_helpers::tensor> AminmaxOutput(
    OpBackend* op,
    synapse_helpers::graph& graph,
    std::vector<synTensor> input,
    const at::ScalarType& dtype,
    const at::IntArrayRef outshape,
    std::shared_ptr<void> params,
    size_t size,
    c10::optional<int> final_idx1 = c10::nullopt,
    c10::optional<int> final_idx2 = c10::nullopt) {
  std::vector<synapse_helpers::tensor> amin_max;

  auto amin = OpBackend::BuildNode(
      op,
      graph,
      {"reduce_min_fwd",
       input,
       {{outshape, dtype, final_idx1}, {outshape, dtype}},
       params.get(),
       size});
  amin_max.push_back(std::move(amin[0]));
  auto amax = OpBackend::BuildNode(
      op,
      graph,
      {"reduce_max_fwd",
       input,
       {{outshape, dtype, final_idx2}, {outshape, dtype}},
       params.get(),
       size});
  amin_max.push_back(std::move(amax[0]));
  return amin_max;
}

static std::vector<synapse_helpers::tensor> AminmaxCommon(
    OpBackend* op,
    synapse_helpers::graph& graph,
    std::vector<synTensor> input_tensor,
    const at::ScalarType& dtype,
    const at::IntArrayRef output_shape,
    const torch::Tensor self,
    const int64_t dim,
    const bool keepdim,
    const bool is_dim_none,
    std::shared_ptr<void> params,
    size_t size,
    c10::optional<int> final_idx1 = c10::nullopt,
    c10::optional<int> final_idx2 = c10::nullopt) {
  auto input_shape = self.sizes().vec();
  std::vector<synapse_helpers::tensor> out_tensor;

  // If dim is None, find max/min from the flattened input tensor.
  if (is_dim_none) {
    // flattened input shape
    auto reshape = OpBackend::BuildReshape(
        op, graph, input_tensor[0], {self.numel()}, dtype);

    if (!keepdim) {
      auto min_max = AminmaxOutput(
          op,
          graph,
          {reshape.get()},
          dtype,
          output_shape,
          params,
          size,
          final_idx1,
          final_idx2);
      out_tensor.push_back(std::move(min_max[0]));
      out_tensor.push_back(std::move(min_max[1]));

    } else {
      auto min_max =
          AminmaxOutput(op, graph, {reshape.get()}, dtype, {1}, params, size);

      auto reshape_min = OpBackend::BuildReshape(
          op, graph, min_max[0].get(), output_shape, dtype, final_idx1);

      auto reshape_max = OpBackend::BuildReshape(
          op, graph, min_max[1].get(), output_shape, dtype, final_idx2);

      out_tensor.push_back(std::move(reshape_min));
      out_tensor.push_back(std::move(reshape_max));
    }
  } else if (!keepdim) { // reduce dim when keepdim is false using reshape.
    input_shape[dim] = 1;
    auto min_max = AminmaxOutput(
        op, graph, input_tensor, dtype, input_shape, params, size);

    auto reshape_min = OpBackend::BuildReshape(
        op, graph, min_max[0].get(), output_shape, dtype, final_idx1);

    auto reshape_max = OpBackend::BuildReshape(
        op, graph, min_max[1].get(), output_shape, dtype, final_idx2);

    out_tensor.push_back(std::move(reshape_min));
    out_tensor.push_back(std::move(reshape_max));
  } else { // Direct TPC kernel call when keepdim is true.
    auto min_max = AminmaxOutput(
        op,
        graph,
        input_tensor,
        dtype,
        output_shape,
        params,
        size,
        final_idx1,
        final_idx2);
    out_tensor.push_back(std::move(min_max[0]));
    out_tensor.push_back(std::move(min_max[1]));
  }
  return out_tensor;
}

void Aminmax::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  auto is_dim_none = stack.at(1).isNone();
  const bool keepdim = stack.at(2).toBool();
  auto input_shape = self.sizes().vec();

  auto dim = is_dim_none
      ? 0
      : c10::maybe_wrap_dim(stack.at(1).toInt(), self.dim(), true);

  size_t size = 0;
  const auto& params = FillParams(stack, size);
  const auto output_shape = ComputeOutputShapes(stack, true)[0];
  synTensor input_tensor = syn_in(0);

  if ((self.scalar_type() == torch::kBool) ||
      (self.scalar_type() == torch::kInt8)) {
    auto cast = CastHelper(
        graph, input_tensor, input_shape, self.scalar_type(), torch::kInt32);
    input_tensor = cast.get();

    auto amin_max = AminmaxCommon(
        this,
        graph,
        {input_tensor},
        torch::kInt32,
        output_shape,
        self,
        dim,
        keepdim,
        is_dim_none,
        params,
        size);
    auto cast1 = CastHelper(
        graph, amin_max[0].get(), output_shape, torch::kInt32, torch::kBool, 0);
    auto cast2 = CastHelper(
        graph, amin_max[1].get(), output_shape, torch::kInt32, torch::kBool, 1);

    syn_out(0) = std::move(cast1);
    syn_out(1) = std::move(cast2);
  } else {
    auto amin_max = AminmaxCommon(
        this,
        graph,
        {input_tensor},
        self.scalar_type(),
        output_shape,
        self,
        dim,
        keepdim,
        is_dim_none,
        params,
        size,
        0,
        1);
    syn_out(0) = std::move(amin_max[0]);
    syn_out(1) = std::move(amin_max[1]);
  }
}
} // namespace habana