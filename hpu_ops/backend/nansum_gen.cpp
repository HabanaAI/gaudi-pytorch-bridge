/*******************************************************************************
 * Copyright (C) 2021-2024 Habana Labs, Ltd. an Intel Company
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

#include "generated/backend/nansum.h"
#include "habana_kernels/reduction_kernels.h"
#include "hpu_ops/backend/reduction_template.h"

namespace habana {

OutputMetaDataVector NanSumIntListMeta(const at::Stack& stack) {
  const torch::Tensor& self = stack_tensor(stack, 0);
  std::vector<int64_t> dim;
  if (!stack.at(1).isNone()) {
    dim = stack.at(1).toIntVector();
  }
  const bool keepdim = stack.at(2).toBool();

  OutputMetaData meta;
  meta.dtype =
      stack.at(3).toOptional<at::ScalarType>().value_or(self.scalar_type());
  meta.shape = ReduceOperator::compute_output_shape(self, dim, keepdim);
  return {meta};
}

void NansumList::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto meta = NanSumIntListMeta(stack)[0];
  auto self = stack.at(0).toTensor();
  const auto& inputShape = self.sizes();
  auto inputType = self.scalar_type();
  std::vector<int64_t> dim;
  if (!stack.at(1).isNone())
    dim = stack.at(1).toIntVector();

  auto keepDim = stack.at(2).toBool();
  auto params = FillReductionParams(self.dim(), dim, keepDim);

  auto compute_type =
      c10::isIntegralType(meta.dtype, true) ? c10::ScalarType::Int : meta.dtype;

  c10::optional<synapse_helpers::tensor> castedInput = c10::nullopt;
  if (habana_helpers::getInternalDtype(compute_type) !=
      habana_helpers::getInternalDtype(inputType)) {
    castedInput = OpBackend::BuildCast(
        this, graph, syn_in(0), inputShape, inputType, compute_type);
  }
  auto input = castedInput.has_value() ? castedInput.value().get() : syn_in(0);

  // isNan on input
  auto is_nan = BuildOp(
      graph,
      get_guid_with_precision("isnan_fwd", compute_type),
      {input},
      {{inputShape, c10::ScalarType::Char}});

  auto zero_constant = ConstantHelper(graph, 0.0f, compute_type, inputShape);

  // where on is_nan
  auto where = BuildOp(
      graph,
      get_guid_with_precision("where_fwd", compute_type),
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
      get_guid_with_precision("reduce_sum_multi_dim_fwd", compute_type),
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
