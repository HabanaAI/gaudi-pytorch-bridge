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

OutputMetaDataVector StdVarMeta(const at::Stack& stack) {
  const torch::Tensor& self = stack_tensor(stack, 0);
  std::vector<int64_t> dims;
  bool keepdim = false;
  if (!stack.at(1).isBool()) {
    dims = stack.at(1).isNone() ? std::vector<int64_t>{}
                                : stack.at(1).toIntVector();
    keepdim = stack.at(3).toBool();
  }
  int ndims = self.sizes().vec().size();
  LoweringUtil::SortAndRemoveDuplicateDims(dims, ndims);

  OutputMetaData meta;
  meta.dtype = self.scalar_type();
  meta.shape = ReductionOutputShape(self, dims, keepdim)[0];
  return {meta};
}

OutputMetaDataVector StdVarMeanMeta(const at::Stack& stack) {
  const torch::Tensor& self = stack_tensor(stack, 0);
  auto meta = StdVarMeta(stack)[0];
  return {meta, meta};
}

static int prepareDivisor(
    const at::Tensor& self,
    const at::IntArrayRef dims,
    const int correction) {
  const auto input_shape = self.sizes().vec();
  const auto dimsVec = dims.vec();
  const auto num_dim = dimsVec.size();

  int divisor{1};
  switch (num_dim) {
    case 0:
      divisor = self.numel();
      break;
    case 1:
      divisor = input_shape.size() == 0 ? 1 : input_shape[dims.front()];
      break;
    default:
      for (unsigned i = 0; i < dimsVec.size() && i < dims.size() &&
           dims[i] < static_cast<int64_t>(input_shape.size());
           i++) {
        divisor *= input_shape[dims[i]];
      }
      break;
  }

  return divisor - correction;
}

// checking dim is continuous to avoid reduce_sum
static bool needsReduceSum(std::vector<int64_t> dimsVec) {
  const auto num_dim = dimsVec.size();

  for (auto i = 0u; i < num_dim; i++) {
    if (dimsVec[i] != i) {
      return true;
    }
  }

  return false;
}

static at::IntArrayRef prepareDims(std::vector<int64_t>& dimsVec, int ndims) {
  LoweringUtil::SortAndRemoveDuplicateDims(dimsVec, ndims);
  return at::IntArrayRef(dimsVec.data(), dimsVec.size());
}

std::vector<synapse_helpers::tensor> StdVarCommonFunc(
    OpBackend* op,
    synapse_helpers::graph& graph,
    const at::Tensor& self,
    const bool keepdim,
    at::IntArrayRef dims,
    std::vector<synTensor> input,
    const int correction,
    const std::vector<NodeAttr::NodeOutputAttr>& output_attr,
    const bool take_sqrt,
    const bool mean_op) {
  auto input_shape = self.sizes().vec();
  if (input_shape.size() == 0) {
    input_shape.push_back(1);
  }
  const int ndims = input_shape.size();
  auto dimsVec = dims.vec();

  dims = prepareDims(dimsVec, ndims);
  const bool enable_reduce_sum = needsReduceSum(dimsVec);
  const bool is_bf16 = op->ScalarType() == torch::kBFloat16;
  const int min_dim = (dimsVec.size() == 0) ? 0 : dimsVec.front();
  std::vector<synapse_helpers::tensor> sum;
  std::vector<synapse_helpers::tensor> outputs;

  // when keepdim is false there will be incompatible input sizes for the
  // sub node so keepdim is set as true for mean and it is reshaped at the end.
  auto mean_out = HandleReductionDimAndKeepdim(
      op,
      graph,
      self,
      input,
      dimsVec,
      true,
      get_guid_with_precision("reduce_mean_fwd", op->ScalarType()),
      {output_attr[1]});

  auto difference = OpBackend::BuildNode(
      op,
      graph,
      {get_guid_with_precision("sub_fwd", op->ScalarType()),
       {input.front(), mean_out.front().get()},
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
  const int divisor = prepareDivisor(self, dims, correction);
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
        {get_guid_with_precision("sqrt_fwd", op->ScalarType()),
         {sum.back().get()},
         {output_attr[0]}});

    outputs.emplace_back(std::move(sqrt[0]));
  } else {
    outputs.emplace_back(std::move(sum.back()));
  }

  if (mean_op) {
    outputs.emplace_back(std::move(mean_out[0]));
  }

  return outputs;
}

void Var::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  auto dim =
      stack.at(1).isNone() ? std::vector<int64_t>{} : stack.at(1).toIntVector();
  const int correction = stack.at(2).isNone() ? 0 : stack.at(2).toInt();
  const bool keepdim = stack.at(3).toBool();

  auto meta = StdVarMeta(stack)[0];
  auto mean_shape = ReductionOutputShape(self, dim, true)[0];

  auto var = StdVarCommonFunc(
      this,
      graph,
      self,
      keepdim,
      dim,
      {syn_in(0)},
      correction,
      {{meta.shape, meta.dtype, 0}, {mean_shape, meta.dtype}},
      false, /*take_sqrt*/
      false /*mean_out_required*/);

  syn_out(0) = std::move(var.at(0));
}

void VarMean::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto self = stack.at(0).toTensor();

  std::vector<int64_t> dim;
  int correction = 0;
  bool keepdim = false;
  if (stack.at(1).isBool()) {
    // this argument is for 'unbiased', convert its value for 'correction'
    correction = static_cast<int>(stack.at(1).toBool());
  } else {
    dim = stack.at(1).isNone() ? std::vector<int64_t>{}
                               : stack.at(1).toIntVector();
    correction = stack.at(2).isNone() ? 0 : stack.at(2).toInt();
    keepdim = stack.at(3).toBool();
  }

  auto meta = StdVarMeanMeta(stack);
  auto mean_shape = ReductionOutputShape(self, dim, true)[0];
  c10::optional<int> finalIndex =
      keepdim ? c10::make_optional<int>(1) : c10::nullopt;
  std::vector<NodeAttr::NodeOutputAttr> output_attrs{
      {meta[0].shape, meta[0].dtype, 0},
      {mean_shape, meta[1].dtype, finalIndex}};

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

  if (keepdim) {
    syn_out(1) = std::move(var_mean[1]);
  } else {
    auto reshape = ReshapeHelper(
        graph, var_mean[1].get(), meta[1].shape, meta[1].dtype, 1);
    syn_out(1) = std::move(reshape);
  }
}

void Std::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  auto dim =
      stack.at(1).isNone() ? std::vector<int64_t>{} : stack.at(1).toIntVector();
  const int correction = stack.at(2).isNone() ? 0 : stack.at(2).toInt();
  const bool keepdim = stack.at(3).toBool();

  auto meta = StdVarMeta(stack)[0];
  auto mean_shape = ReductionOutputShape(self, dim, true)[0];

  auto std = StdVarCommonFunc(
      this,
      graph,
      self,
      keepdim,
      dim,
      {syn_in(0)},
      correction,
      {{meta.shape, meta.dtype, 0}, {mean_shape, meta.dtype}},
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

  auto meta = StdVarMeanMeta(stack);
  auto mean_shape = ReductionOutputShape(self, dim, true)[0];
  c10::optional<int> finalIndex =
      keepdim ? c10::make_optional<int>(1) : c10::nullopt;
  std::vector<NodeAttr::NodeOutputAttr> output_attrs{
      {meta[0].shape, meta[0].dtype, 0},
      {mean_shape, meta[1].dtype, finalIndex}};

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

  if (keepdim) {
    syn_out(1) = std::move(std_mean[1]);
  } else {
    auto reshape = ReshapeHelper(
        graph, std_mean[1].get(), meta[1].shape, meta[1].dtype, 1);
    syn_out(1) = std::move(reshape);
  }
}
} // namespace habana
