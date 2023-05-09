/*******************************************************************************
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

#include "backend/helpers/lowering_util.h"
#include "generated/backend/std.h"
#include "generated/backend/std_mean.h"
#include "generated/backend/var.h"
#include "generated/backend/var_mean.h"
#include "hpu_ops/backend/reduction_template.h"

namespace habana {

sizes_vec StdVarComputeOutShape(const at::Stack& stack) {
  const torch::Tensor& self = stack_tensor(stack, 0);
  auto dim =
      stack.at(1).isNone() ? std::vector<int64_t>{} : stack.at(1).toIntVector();
  int ndims = self.sizes().vec().size();
  LoweringUtil::SortAndRemoveDuplicateDims(dim, ndims);
  const bool keepdim = stack.at(3).toBool();
  return ReductionOutputShape(self, dim, keepdim);
}

sizes_vec StdVarMeanComputeOutShape(const at::Stack& stack) {
  auto outshape = StdVarComputeOutShape(stack)[0];
  return {outshape, outshape};
}

std::vector<synapse_helpers::tensor> StdVarCommonFunc(
    OpBackend* op,
    synapse_helpers::graph& graph,
    const at::Tensor& self,
    const bool keepdim,
    const at::IntArrayRef dims,
    std::vector<synTensor> input,
    const int correction,
    const std::vector<NodeAttr::NodeOutputAttr>& output_attr,
    const bool take_sqrt,
    const bool mean_op) {
  auto input_shape = self.sizes().vec();
  int ndims = input_shape.size();
  auto dim = dims.vec();

  // checking dim is continuous to avoid reduce_sum
  LoweringUtil::SortAndRemoveDuplicateDims(dim, ndims);
  auto num_dim = dim.size();
  const bool is_arrdim = num_dim > 1;
  int divisor = num_dim == 0 ? self.numel() : 1;
  bool dim_continuous = false;
  for (auto i = 0u; i < num_dim && is_arrdim; i++) {
    if (dim[i] == i) {
      dim_continuous = true;
    } else {
      dim_continuous = false;
      break;
    }
  }
  const bool enable_reduce_sum = is_arrdim && !dim_continuous;
  const bool is_bf16 = op->ScalarType() == torch::kBFloat16;
  int min_dim = num_dim == 0 ? 0 : dim[0];

  for (unsigned i = 0; i < dim.size(); i++)
    divisor *= input_shape[dims[i]];

  divisor = divisor - correction;
  std::vector<synapse_helpers::tensor> sum;
  std::vector<synapse_helpers::tensor> outputs;

  // when keepdim is false there will be incompatible input sizes for the
  // sub node so keepdim is set as true for mean and it is reshaped at the end.
  auto mean_out = HandleReductionDimAndKeepdim(
      op,
      graph,
      self,
      input,
      dim,
      true,
      "reduce_mean_fwd_" +
          habana_helpers::name_suffix_from_type(op->ScalarType()),
      {output_attr[1]});

  auto difference = OpBackend::BuildNode(
      op,
      graph,
      {"sub_fwd_" + habana_helpers::name_suffix_from_type(op->ScalarType()),
       {input[0], mean_out[0].get()},
       {{input_shape, op->ScalarType()}}});

  input_shape[min_dim] = 1;
  // Using reduction squares only in first reduction
  // followed by reduce_sum for rest of the dims.
  // example: input_shape [8,3,2,2] with dim=[0,2,3] only dim=0 is passed to
  // reduction_helper, output_shape [1,3,2,2]
  auto sum_square_input = is_bf16
      ? *HandleReductionDtype(op, graph, self, difference[0].get(), at::kFloat)
      : std::move(difference[0]);
  auto sum_square = HandleReductionDimAndKeepdim(
      op,
      graph,
      self,
      {sum_square_input.get()},
      enable_reduce_sum ? min_dim : dims,
      enable_reduce_sum ? true : keepdim,
      "reduce_sum_square_fwd_f32",
      {{enable_reduce_sum ? input_shape : output_attr[0].sizes}});

  // Flattening of contiguous axes using reshape is handled
  // in HandleReductionDimAndKeepdim
  sum.emplace_back(std::move(sum_square[0]));

  // Since input shape is calculated from self tensor
  // input_shape remains to be [8,3,2,2] but required input_shape [1,3,2,2] and
  // dim becomes [2,3] so there will output shape mismatch if all dims are
  // passed collectively. Hence individual dims are passed
  if (enable_reduce_sum) {
    for (size_t i = 1; i < dims.size(); i++) {
      input_shape[dims[i]] = 1;
      auto reduce_sum = HandleReductionDimAndKeepdim(
          op,
          graph,
          self,
          {sum.back().get()},
          dims[i],
          true,
          "reduce_sum_fwd_f32",
          {{input_shape}});
      sum.emplace_back(std::move(reduce_sum[0]));
    }

    if (!keepdim) {
      auto reshape_tensor = OpBackend::BuildReshape(
          op, graph, sum.back().get(), output_attr[0].sizes, at::kFloat);
      sum.emplace_back(std::move(reshape_tensor));
    }
  }

  auto divisor_tensor =
      OpBackend::BuildConstant(op, graph, divisor, at::kFloat);

  auto reciprocal = OpBackend::BuildNode(
      op, graph, {"reciprocal_fwd_f32", {divisor_tensor.get()}, {{1}}});

  auto div = OpBackend::BuildNode(
      op,
      graph,
      {"mult_fwd_f32",
       {sum.back().get(), reciprocal[0].get()},
       (take_sqrt || is_bf16)
           ? std::vector<NodeAttr::NodeOutputAttr>{{output_attr[0].sizes}}
           : std::vector<NodeAttr::NodeOutputAttr>{output_attr[0]}});
  sum.emplace_back(std::move(div[0]));

  if (is_bf16) {
    auto cast_f32 = OpBackend::BuildCast(
        op,
        graph,
        sum.back().get(),
        output_attr[0].sizes,
        at::kFloat,
        torch::kBFloat16,
        take_sqrt ? c10::nullopt : (c10::optional<int>)0);
    sum.emplace_back(std::move(cast_f32));
  }
  if (take_sqrt) {
    auto sqrt = OpBackend::BuildNode(
        op,
        graph,
        {"sqrt_fwd_" + habana_helpers::name_suffix_from_type(op->ScalarType()),
         {sum.back().get()},
         {output_attr[0]}});

    outputs.emplace_back(std::move(sqrt[0]));
  } else {
    outputs.emplace_back(std::move(sum.back()));
  }

  if (mean_op)
    outputs.emplace_back(std::move(mean_out[0]));

  return outputs;
}

void Var::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  auto dim =
      stack.at(1).isNone() ? std::vector<int64_t>{} : stack.at(1).toIntVector();
  const int correction = stack.at(2).isNone() ? 0 : stack.at(2).toInt();
  const bool keepdim = stack.at(3).toBool();

  auto out_shape = ReductionOutputShape(self, dim, keepdim)[0];
  auto mean_shape = ReductionOutputShape(self, dim, true)[0];

  auto var = StdVarCommonFunc(
      this,
      graph,
      self,
      keepdim,
      dim,
      {syn_in(0)},
      correction,
      {{out_shape, ScalarType(), 0}, {mean_shape, ScalarType()}},
      false, /*take_sqrt*/
      false /*mean_out_required*/);

  syn_out(0) = std::move(var.at(0));
}

void VarMean::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  auto dim =
      stack.at(1).isNone() ? std::vector<int64_t>{} : stack.at(1).toIntVector();
  const int correction = stack.at(2).isNone() ? 0 : stack.at(2).toInt();
  const bool keepdim = stack.at(3).toBool();

  auto out_shape = ReductionOutputShape(self, dim, keepdim)[0];
  auto mean_shape = ReductionOutputShape(self, dim, true)[0];

  std::vector<NodeAttr::NodeOutputAttr> output_attrs{
      {out_shape, ScalarType(), 0}};

  if (keepdim) {
    output_attrs.push_back({mean_shape, ScalarType(), 1});
    auto var_mean = StdVarCommonFunc(
        this,
        graph,
        self,
        keepdim,
        dim,
        {syn_in(0)},
        correction,
        output_attrs,
        false, /*take_sqrt*/
        true /*mean_out_required*/);

    syn_out(0) = std::move(var_mean[0]);
    syn_out(1) = std::move(var_mean[1]);
  } else {
    output_attrs.push_back({mean_shape, ScalarType()});

    auto var_mean = StdVarCommonFunc(
        this,
        graph,
        self,
        keepdim,
        dim,
        {syn_in(0)},
        correction,
        output_attrs,
        false, /*take_sqrt*/
        true /*mean_out_required*/);

    auto reshape =
        ReshapeHelper(graph, var_mean[1].get(), out_shape, ScalarType(), 1);

    syn_out(0) = std::move(var_mean[0]);
    syn_out(1) = std::move(reshape);
  }
}

void Std::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  auto dim =
      stack.at(1).isNone() ? std::vector<int64_t>{} : stack.at(1).toIntVector();
  const int correction = stack.at(2).isNone() ? 0 : stack.at(2).toInt();
  const bool keepdim = stack.at(3).toBool();

  auto out_shape = ReductionOutputShape(self, dim, keepdim)[0];
  auto mean_shape = ReductionOutputShape(self, dim, true)[0];

  auto std = StdVarCommonFunc(
      this,
      graph,
      self,
      keepdim,
      dim,
      {syn_in(0)},
      correction,
      {{out_shape, ScalarType(), 0}, {mean_shape, ScalarType()}},
      true, /*take_sqrt*/
      false /*mean_out_required*/);

  syn_out(0) = std::move(std.at(0));
}

void StdMean::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  auto dim =
      stack.at(1).isNone() ? std::vector<int64_t>{} : stack.at(1).toIntVector();
  const int correction = stack.at(2).isNone() ? 0 : stack.at(2).toInt();
  const bool keepdim = stack.at(3).toBool();

  auto out_shape = ReductionOutputShape(self, dim, keepdim)[0];
  auto mean_shape = ReductionOutputShape(self, dim, true)[0];

  std::vector<NodeAttr::NodeOutputAttr> output_attrs{
      {out_shape, ScalarType(), 0}};

  if (keepdim) {
    output_attrs.push_back({mean_shape, ScalarType(), 1});

    auto std_mean = StdVarCommonFunc(
        this,
        graph,
        self,
        keepdim,
        dim,
        {syn_in(0)},
        correction,
        output_attrs,
        true, /*take_sqrt*/
        true /*mean_out_required*/);

    syn_out(0) = std::move(std_mean[0]);
    syn_out(1) = std::move(std_mean[1]);
  } else {
    output_attrs.push_back({mean_shape, ScalarType()});
    auto std_mean = StdVarCommonFunc(
        this,
        graph,
        self,
        keepdim,
        dim,
        {syn_in(0)},
        correction,
        output_attrs,
        true, /*take_sqrt*/
        true /*mean_out_required*/);

    auto reshape =
        ReshapeHelper(graph, std_mean[1].get(), out_shape, ScalarType(), 1);

    syn_out(0) = std::move(std_mean[0]);
    syn_out(1) = std::move(reshape);
  }
}
} // namespace habana
