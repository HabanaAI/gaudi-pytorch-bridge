/**
 * Copyright (c) 2021-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "backend/habana_operator.h"
#include "backend/helpers/lowering_util.h"
#include "generated/backend/std.h"
#include "generated/backend/std_mean.h"
#include "generated/backend/var.h"
#include "generated/backend/var_mean.h"
#include "hpu_ops/backend/reduction_template.h"
#include "pytorch_helpers/habana_helpers/conversion.h"

namespace habana {

std::vector<int64_t> fillDims(const at::Tensor& self) {
  std::vector<int64_t> dims;
  auto input_size = self.dim();
  dims.reserve(safe_convert<size_t>(input_size));
  for (auto i = 0; i < input_size; i++) {
    dims.push_back(i);
  }
  return dims;
}

std::vector<int64_t> getDimsVector(
    const at::Stack& stack,
    const at::Tensor& self) {
  // the second condition is true when no other arguments except input are
  // passed to e.g. var_mean
  return stack.at(1).isNone() || stack.at(1).isBool() ||
          stack.at(1).toIntVector().empty()
      ? fillDims(self)
      : stack.at(1).toIntVector();
}

OutputMetaDataVector StdVarMeta(const at::Stack& stack) {
  const torch::Tensor& self = stack_tensor(stack, 0);
  auto dims = getDimsVector(stack, self);
  bool keepdim = false;
  if (!stack.at(1).isBool()) {
    keepdim = stack.at(3).toBool();
  }
  int ndims = safe_convert<int>(self.sizes().size());
  LoweringUtil::SortAndRemoveDuplicateDims(dims, ndims);

  OutputMetaDataVector metaVec;
  metaVec.reserve(2);
  auto& meta = metaVec.emplace_back();
  meta.dtype = self.scalar_type();
  meta.shape = ReductionOutputShape(self, dims, keepdim)[0];
  return metaVec;
}

OutputMetaDataVector StdVarMeanMeta(const at::Stack& stack) {
  auto metaVec = StdVarMeta(stack);
  metaVec.push_back(metaVec[0]);
  return metaVec;
}

std::vector<synapse_helpers::tensor> slice_size_helper(
    OpBackend* op,
    synapse_helpers::graph& graph,
    const at::Tensor& self,
    const std::vector<synTensor>& input,
    const std::vector<int64_t>& dims) {
  auto intermediate_shape = self.sizes().vec();
  auto rank = intermediate_shape.size();
  std::vector<synapse_helpers::tensor> slice_axis_output;
  auto slice_output = input;

  if (rank == 1) {
    return OpBackend::BuildNode(
        op, graph, {"size_i32", input, {{{1}, c10::ScalarType::Int}}});
  }

  for (uint64_t i = 0; i < rank; i++) {
    if (std::find(dims.begin(), dims.end(), i) == dims.end()) {
      intermediate_shape[i] = 1;

      synSliceAxisParamsV2 slice_params{};
      slice_params.axis = safe_convert<unsigned int>(rank - i - 1);
      slice_params.begin = 0;
      slice_params.end = 1;

      slice_axis_output = OpBackend::BuildNode(
          op,
          graph,
          {"slice_axis",
           slice_output,
           {{intermediate_shape, self.scalar_type()}},
           &slice_params,
           sizeof(slice_params)});
      slice_output[0] = slice_axis_output[0].get();
    }
  }

  return OpBackend::BuildNode(
      op,
      graph,
      {"size_i32", {slice_output[0]}, {{{1}, c10::ScalarType::Int}}});
}

// checking if dim is continuous to avoid reduce_sum
static bool needsReduceSum(std::vector<int64_t> dimsVec) {
  if (dimsVec.size() == 1) {
    return false;
  }
  for (auto i = 0U; i < dimsVec.size() - 1; i++) {
    // If difference between two next elements is different than one, then dims
    // are not consecutive
    if (dimsVec[i + 1] - dimsVec[i] != 1) {
      return true;
    }
  }
  return false;
}

std::vector<synapse_helpers::tensor> StdVarCommonFunc(
    OpBackend* op,
    synapse_helpers::graph& graph,
    const at::Tensor& self,
    const bool keepdim,
    at::IntArrayRef dims,
    std::vector<synTensor> input,
    const double correction,
    const std::vector<NodeAttr::NodeOutputAttr>& output_attr,
    const bool take_sqrt,
    const bool mean_op) {
  auto input_shape = self.sizes().vec();
  if (input_shape.empty()) {
    input_shape.push_back(1);
  }
  const size_t ndims = input_shape.size();
  auto dimsVec = dims.vec();
  LoweringUtil::SortAndRemoveDuplicateDims(
      dimsVec, safe_convert<int64_t>(ndims));

  const bool enable_reduce_sum = needsReduceSum(dimsVec);
  const int64_t min_dim = (dimsVec.empty()) ? 0 : dimsVec.front();
  std::vector<synapse_helpers::tensor> outputs;

  // when keepdim is false there will be incompatible input sizes for the
  // sub node so keepdim is set as true for mean and it is reshaped at the end.
  auto mean = HandleReduction(
      op,
      graph,
      input[0],
      "reduce_mean_multi_dim_fwd",
      dimsVec,
      safe_convert<int64_t>(ndims),
      true,
      {output_attr[1]});

  using namespace std::literals;

  auto difference = OpBackend::BuildNode(
      op,
      graph,
      {get_guid_with_precision("sub_fwd"sv, op->ScalarType()),
       {input[0], mean[0].get()},
       {{input_shape, op->ScalarType()}}});

  // Using reduction squares only in first reduction
  // followed by reduce_sum for rest of the dims so the input is squared only
  // once example: input_shape [8,3,2,2] with dim=[0,2,3] only dim=0 is passed
  // to reduction_helper, output_shape [1,3,2,2]
  auto reduce_sum_square_output_shape =
      CalculateReductionMultiDimAndKeepdimOutputSize(
          input_shape,
          enable_reduce_sum ? std::vector<int64_t>{min_dim} : dimsVec,
          enable_reduce_sum ? true : keepdim);

  auto sum_square = HandleReduction(
      op,
      graph,
      difference[0].get(),
      "reduce_sum_square_multi_dim_fwd",
      enable_reduce_sum ? std::vector<int64_t>{min_dim} : dimsVec,
      safe_convert<int64_t>(ndims),
      enable_reduce_sum ? true : keepdim,
      {{reduce_sum_square_output_shape, output_attr[0].dtype}});

  auto sum_square_final = enable_reduce_sum
      ? HandleReduction(
            op,
            graph,
            sum_square[0].get(),
            "reduce_sum_multi_dim_fwd",
            std::vector(dimsVec.begin() + 1, dimsVec.end()),
            safe_convert<int64_t>(ndims),
            keepdim,
            {{output_attr[0].sizes, output_attr[0].dtype}})
      : std::move(sum_square);

  std::vector<synapse_helpers::tensor> reciprocal;
  auto slice_size_output = slice_size_helper(op, graph, self, input, dimsVec);
  if (correction != 0.0) {
    auto correction_tensor =
        OpBackend::BuildConstant(op, graph, correction, at::kFloat);
    auto diff = OpBackend::BuildNode(
        op,
        graph,
        {get_guid_with_precision("sub_fwd"sv, op->ScalarType()),
         {slice_size_output[0].get(), correction_tensor.get()},
         {{{1}, op->ScalarType()}}});
    auto zero = OpBackend::BuildConstant(op, graph, 0.0, op->ScalarType());
    auto max = OpBackend::BuildNode(
        op,
        graph,
        {get_guid_with_precision("max_fwd"sv, op->ScalarType()),
         {diff.at(0).get(), zero.get()},
         {{{1}, op->ScalarType()}}});
    reciprocal = OpBackend::BuildNode(
        op, graph, {"reciprocal_fwd_f32", {max.at(0).get()}, {{1}}});
  } else {
    reciprocal = OpBackend::BuildNode(
        op,
        graph,
        {"reciprocal_fwd_f32", {slice_size_output.at(0).get()}, {{1}}});
  }

  auto div = OpBackend::BuildNode(
      op,
      graph,
      {"mult_fwd_f32",
       {sum_square_final[0].get(), reciprocal[0].get()},
       (take_sqrt)
           ? std::vector<NodeAttr::NodeOutputAttr>{{output_attr[0].sizes}}
           : std::vector<NodeAttr::NodeOutputAttr>{output_attr[0]}});

  if (take_sqrt) {
    auto sqrt = OpBackend::BuildNode(
        op,
        graph,
        {get_guid_with_precision("sqrt_fwd"sv, op->ScalarType()),
         {div[0].get()},
         {output_attr[0]}});

    outputs.emplace_back(std::move(sqrt[0]));
  } else {
    outputs.emplace_back(std::move(div[0]));
  }

  if (mean_op) {
    if (keepdim) {
      outputs.emplace_back(std::move(mean[0]));
    } else {
      const auto rank = output_attr[1].sizes.size();
      if (dimsVec.size() == output_attr[1].sizes.size() || rank == 0) {
        auto squeeze = OpBackend::BuildNode(
            op,
            graph,
            {get_guid_with_precision("squeeze"sv, output_attr[1].dtype),
             {mean[0].get()},
             {{output_attr[1].sizes, output_attr[1].dtype, 1}}});
        outputs.emplace_back(std::move(squeeze[0]));

      } else {
        std::vector<synapse_helpers::tensor> intermediate_syn_helpers;
        intermediate_syn_helpers.emplace_back(std::move(mean[0]));

        // Need to have dims sorted from the highest to the lowest
        std::sort(dimsVec.begin(), dimsVec.end(), std::greater<int64_t>());
        auto temp_size = output_attr[1].sizes.vec();
        for (size_t i = 0; i < dimsVec.size(); ++i) {
          const auto current_rank = temp_size.size();
          const auto dim = dimsVec[i];
          auto dim_synapse_order = safe_convert<unsigned int>(
              current_rank - safe_convert<size_t>(dim) - 1);
          synAxisParams params{dim_synapse_order};

          temp_size.erase(temp_size.begin() + safe_convert<ptrdiff_t>(dim));
          NodeAttr::NodeOutputAttr out_attr{
              temp_size,
              output_attr[1].dtype,
              (i == dimsVec.size() - 1) ? c10::make_optional<int>(1)
                                        : std::nullopt};
          intermediate_syn_helpers.emplace_back(
              std::move(
                  OpBackend::BuildNode(
                      op,
                      graph,
                      {get_guid_with_precision(
                           "squeeze"sv, output_attr[1].dtype),
                       {intermediate_syn_helpers.back().get()},
                       {out_attr},
                       &params,
                       sizeof(params)})[0]));
        }
        outputs.emplace_back(std::move(intermediate_syn_helpers.back()));
      }
    }
  }

  return outputs;
}

// correction must be 0 or 1, but ivalue can be float/double/int/None.
double getCorrectionValue(c10::IValue ivalue) {
  if (ivalue.isNone()) {
    return 1.0;
  }
  if (ivalue.isInt()) {
    return static_cast<double>(ivalue.toInt());
  }
  // computation for floating point
  return ivalue.toDouble();
}

SharedMetaDataVector VarStdCommonSharedMeta(
    const at::Stack& stack,
    const bool keepdim,
    const double correction,
    const bool take_sqrt,
    const bool mean_op) {
  auto self = stack.at(0).toTensor();
  auto dtype = self.scalar_type();
  auto inRank = self.dim();

  SharedMetaDataVector out;
  out.reserve(10 + (2 * inRank));

  auto& meanSharedMeta = out.emplace_back("reduce_mean_multi_dim_fwd");
  meanSharedMeta.inputs_data.emplace_back(inRank, dtype);
  meanSharedMeta.outputs_data.emplace_back(inRank, dtype);

  auto& sub1SharedMeta = out.emplace_back("sub_fwd");
  sub1SharedMeta.inputs_data.emplace_back(inRank, dtype);
  sub1SharedMeta.inputs_data.emplace_back(1, dtype);
  sub1SharedMeta.outputs_data.emplace_back(inRank, dtype);

  auto& sumSquareSharedMeta =
      out.emplace_back("reduce_sum_square_multi_dim_fwd");
  sumSquareSharedMeta.inputs_data.emplace_back(inRank, dtype);
  sumSquareSharedMeta.outputs_data.emplace_back(inRank, dtype);

  auto dims = getDimsVector(stack, self);
  const bool enable_reduce_sum = needsReduceSum(dims);
  if (enable_reduce_sum) {
    auto& sumSquareFinalSharedMeta =
        out.emplace_back("reduce_sum_multi_dim_fwd");
    sumSquareFinalSharedMeta.inputs_data.emplace_back(inRank, dtype);
    sumSquareFinalSharedMeta.outputs_data.emplace_back(inRank, dtype);
  }

  // slice size helper
  if (inRank != 1) {
    auto& sliceAxisSharedMeta = out.emplace_back("slice_axis");
    sliceAxisSharedMeta.inputs_data.emplace_back(inRank, dtype);
    sliceAxisSharedMeta.outputs_data.emplace_back(1, dtype);
    for (auto i = 0U; i < inRank; ++i) {
      out.push_back(sliceAxisSharedMeta);
    }
  }
  auto& sz = out.emplace_back("size");
  sz.inputs_data.emplace_back(inRank, dtype);
  sz.outputs_data.emplace_back(1, c10::ScalarType::Int);

  if (correction != 0.0) {
    auto& sub2SharedMeta = out.emplace_back("sub_fwd");
    sub2SharedMeta.inputs_data.emplace_back(1, dtype);
    sub2SharedMeta.inputs_data.emplace_back(1, dtype);
    sub2SharedMeta.outputs_data.emplace_back(1, dtype);
  }

  auto& reciprocalSharedMeta = out.emplace_back("reciprocal_fwd");
  reciprocalSharedMeta.inputs_data.emplace_back(1, c10::ScalarType::Float);
  reciprocalSharedMeta.outputs_data.emplace_back(1, c10::ScalarType::Float);

  auto& multSharedMeta = out.emplace_back("mult_fwd");
  multSharedMeta.inputs_data.emplace_back(inRank, c10::ScalarType::Float);
  multSharedMeta.inputs_data.emplace_back(1, c10::ScalarType::Float);
  multSharedMeta.outputs_data.emplace_back(inRank, c10::ScalarType::Float);

  if (take_sqrt) {
    auto& sqrtSharedMeta = out.emplace_back("sqrt_fwd");
    sqrtSharedMeta.inputs_data.emplace_back(inRank, c10::ScalarType::Float);
    sqrtSharedMeta.outputs_data.emplace_back(inRank, c10::ScalarType::Float);
  }

  if (mean_op && !keepdim) {
    for (auto i = 0U; i < inRank; ++i) {
      auto& squeezeSharedMeta = out.emplace_back("squeeze");
      squeezeSharedMeta.inputs_data.emplace_back(inRank - i, dtype);
      squeezeSharedMeta.outputs_data.emplace_back(inRank - i - 1, dtype);
    }
  }

  return out;
}

double getVarMeanCorrection(const at::Stack& stack) {
  if (stack.at(1).isBool()) {
    // this argument is for 'unbiased', convert its value for 'correction'
    return static_cast<int>(stack.at(1).toBool());
  }
  return getCorrectionValue(stack.at(2));
}

bool getVarMeanKeepdim(const at::Stack& stack) {
  if (stack.at(1).isBool()) {
    return false;
  }
  return stack.at(3).toBool();
}

SharedMetaDataVector VarSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode /*unused*/) {
  const double correction = getCorrectionValue(stack.at(2));
  const bool keepdim = stack.at(3).toBool();
  return VarStdCommonSharedMeta(stack, keepdim, correction, false, false);
}

SharedMetaDataVector VarMeanSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode /*unused*/) {
  double correction = getVarMeanCorrection(stack);
  bool keepdim = getVarMeanKeepdim(stack);
  return VarStdCommonSharedMeta(stack, keepdim, correction, false, true);
}

SharedMetaDataVector StdSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode /*unused*/) {
  const double correction = getCorrectionValue(stack.at(2));
  const bool keepdim = stack.at(3).toBool();
  return VarStdCommonSharedMeta(stack, keepdim, correction, true, false);
}

SharedMetaDataVector StdMeanSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode /*unused*/) {
  const double correction = getCorrectionValue(stack.at(2));
  const bool keepdim = stack.at(3).toBool();
  return VarStdCommonSharedMeta(stack, keepdim, correction, true, true);
}

void Var::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  auto dims = getDimsVector(stack, self);
  const double correction = getCorrectionValue(stack.at(2));
  const bool keepdim = stack.at(3).toBool();

  auto meta = StdVarMeta(stack)[0];
  auto mean_shape = ReductionOutputShape(self, dims, true)[0];

  auto var = StdVarCommonFunc(
      this,
      graph,
      self,
      keepdim,
      dims,
      {syn_in(0)},
      correction,
      {{meta.shape, meta.dtype, 0}, {mean_shape, meta.dtype}},
      false, /*take_sqrt*/
      false /*mean_out_required*/);

  syn_out(0) = std::move(var.at(0));
}

void VarMean::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto self = stack.at(0).toTensor();

  double correction = getVarMeanCorrection(stack);
  bool keepdim = getVarMeanKeepdim(stack);

  auto dims = getDimsVector(stack, self);

  auto meta = StdVarMeanMeta(stack);
  auto mean_shape = ReductionOutputShape(self, dims, true)[0];

  std::optional<int> finalIndex =
      keepdim ? c10::make_optional<int>(1) : std::nullopt;
  std::vector<NodeAttr::NodeOutputAttr> output_attrs{
      {meta[0].shape, meta[0].dtype, 0},
      {mean_shape, meta[1].dtype, finalIndex}};

  auto var_mean = StdVarCommonFunc(
      this,
      graph,
      self,
      keepdim,
      dims,
      {syn_in(0)},
      correction,
      output_attrs,
      false, /*take_sqrt*/
      true /*mean_out_required*/);
  syn_out(0) = std::move(var_mean[0]);
  syn_out(1) = std::move(var_mean[1]);
}

void Std::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  auto dims = getDimsVector(stack, self);
  const double correction = getCorrectionValue(stack.at(2));
  const bool keepdim = stack.at(3).toBool();

  auto meta = StdVarMeta(stack)[0];
  auto mean_shape = ReductionOutputShape(self, dims, true)[0];

  auto std = StdVarCommonFunc(
      this,
      graph,
      self,
      keepdim,
      dims,
      {syn_in(0)},
      correction,
      {{meta.shape, meta.dtype, 0}, {mean_shape, meta.dtype}},
      true, /*take_sqrt*/
      false /*mean_out_required*/);

  syn_out(0) = std::move(std.at(0));
}

void StdMean::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  auto dims = getDimsVector(stack, self);
  const double correction = getCorrectionValue(stack.at(2));
  const bool keepdim = stack.at(3).toBool();

  auto meta = StdVarMeanMeta(stack);
  auto mean_shape = ReductionOutputShape(self, dims, true)[0];
  std::optional<int> finalIndex =
      keepdim ? c10::make_optional<int>(1) : std::nullopt;
  std::vector<NodeAttr::NodeOutputAttr> output_attrs{
      {meta[0].shape, meta[0].dtype, 0},
      {mean_shape, meta[1].dtype, finalIndex}};

  auto std_mean = StdVarCommonFunc(
      this,
      graph,
      self,
      keepdim,
      dims,
      {syn_in(0)},
      correction,
      output_attrs,
      true, /*take_sqrt*/
      true /*mean_out_required*/);
  syn_out(0) = std::move(std_mean[0]);
  syn_out(1) = std::move(std_mean[1]);
}
} // namespace habana
