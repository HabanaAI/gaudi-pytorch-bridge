/**
 * Copyright (c) 2021-2026 Intel Corporation
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

#include "generated/backend/nansum.h"
#include "habana_kernels/reduction_kernels.h"
#include "hpu_ops/backend/reduction_template.h"
#include "hpu_ops/shared_meta_common.h"

namespace habana {

OutputMetaDataVector NanSumIntListMeta(const at::Stack& stack) {
  const torch::Tensor& self = stack_tensor(stack, 0);
  std::vector<int64_t> dim;
  if (!stack.at(1).isNone()) {
    dim = stack.at(1).toIntVector();
  }
  const bool keepdim = stack.at(2).toBool();

  OutputMetaDataVector metaVec(1);
  auto& meta = metaVec.front();
  meta.dtype =
      stack.at(3).toOptional<at::ScalarType>().value_or(self.scalar_type());
  meta.shape = ReduceOperator::compute_output_shape(self, dim, keepdim);
  return metaVec;
}

SharedMetaDataVector NanSumSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode /*unused*/) {
  const auto& self = stack_tensor(stack, 0);
  auto computeDtype =
      stack.at(3).toOptional<at::ScalarType>().value_or(self.scalar_type());
  if (c10::isIntegralType(computeDtype, true)) {
    computeDtype = c10::ScalarType::Int;
  }
  const auto inputRank = self.dim();
  int64_t outputRank = 1;
  const bool keepDim = stack.at(2).toBool();
  if (!keepDim) {
    if (!stack.at(1).isNone()) {
      auto dims = stack.at(1).toIntVector().size();
      outputRank = dims > 0 ? inputRank - static_cast<int64_t>(dims) : 1;
    }
  } else {
    outputRank = inputRank;
  }

  if (outputRank <= 0) {
    outputRank = 1;
  }

  SharedMetaTensor commonTensor = {inputRank, computeDtype};

  // Don't create a shared meta for a constant node, as it would disable SAG
  // flow. In fact, this node is always created with a shape {1}, which is
  // supported by SAG.

  SharedMetaDataVector nansumSharedMetaVec;
  nansumSharedMetaVec.reserve(3);

  auto& isNanSharedMeta = nansumSharedMetaVec.emplace_back("isnan_fwd");
  isNanSharedMeta.inputs_data = {commonTensor};
  isNanSharedMeta.outputs_data = {{inputRank, c10::ScalarType::Char}};

  auto& whereSharedMeta = nansumSharedMetaVec.emplace_back("where_fwd");
  whereSharedMeta.inputs_data = {
      isNanSharedMeta.outputs_data[0], commonTensor, commonTensor};
  whereSharedMeta.outputs_data = {commonTensor};

  auto& reduceSharedMeta =
      nansumSharedMetaVec.emplace_back("reduce_sum_multi_dim_fwd");
  reduceSharedMeta.inputs_data = {commonTensor};
  reduceSharedMeta.outputs_data.emplace_back(outputRank, computeDtype);

  return nansumSharedMetaVec;
}

void NansumList::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto meta = NanSumIntListMeta(stack)[0];
  auto self = stack.at(0).toTensor();
  const auto& inputShape = self.sizes();
  auto inputType = self.scalar_type();
  std::vector<int64_t> dim;
  if (!stack.at(1).isNone()) {
    dim = stack.at(1).toIntVector();
  }
  auto keepDim = stack.at(2).toBool();
  auto params = FillReductionParams(self.dim(), dim, keepDim);

  auto compute_type =
      c10::isIntegralType(meta.dtype, true) ? c10::ScalarType::Int : meta.dtype;

  std::optional<synapse_helpers::tensor> castedInput = std::nullopt;
  if (habana_helpers::getInternalDtype(compute_type) !=
      habana_helpers::getInternalDtype(inputType)) {
    castedInput = OpBackend::BuildCast(
        this, graph, syn_in(0), inputShape, inputType, compute_type);
  }
  auto* input = castedInput.has_value() ? castedInput.value().get() : syn_in(0);

  using namespace std::literals;
  // isNan on input
  auto is_nan = BuildOp(
      graph,
      get_guid_with_precision("isnan_fwd"sv, compute_type),
      {input},
      {{inputShape, c10::ScalarType::Char}});

  auto zero_constant = ConstantHelper(graph, 0.0F, compute_type);

  // where on is_nan
  auto where = BuildOp(
      graph,
      get_guid_with_precision("where_fwd"sv, compute_type),
      {is_nan[0].get(), zero_constant.get(), input},
      {{inputShape, compute_type}});

  const bool is_cast_not_required =
      habana_helpers::getInternalDtype(compute_type) ==
      habana_helpers::getInternalDtype(meta.dtype);
  NodeAttr::NodeOutputAttr out_attr = {meta.shape, compute_type};
  if (is_cast_not_required) {
    out_attr.final_result_index = 0;
  }

  auto reduce_sum = BuildOp(
      graph,
      get_guid_with_precision("reduce_sum_multi_dim_fwd"sv, compute_type),
      {where[0].get()},
      {out_attr},
      &params,
      sizeof(params));

  if (is_cast_not_required) {
    syn_out(0) = std::move(reduce_sum[0]);
  } else {
    auto castOut = OpBackend::BuildCast(
        this,
        graph,
        reduce_sum[0].get(),
        meta.shape,
        compute_type,
        meta.dtype,
        0);
    syn_out(0) = std::move(castOut);
  }
}
} // namespace habana
