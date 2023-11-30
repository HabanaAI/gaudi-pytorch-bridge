/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "generated/backend/nansum.h"
#include "habana_kernels/reduction_kernels.h"
#include "hpu_ops/backend/reduction_template.h"

#define guidReducesum "reduce_sum_fwd"

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
  bool keepdim = stack.at(2).toBool();
  auto guid = get_guid_with_precision(guidReducesum, meta.dtype);

  c10::optional<synapse_helpers::tensor> castedInput = c10::nullopt;
  if (habana_helpers::getInternalDtype(meta.dtype) !=
      habana_helpers::getInternalDtype(inputType)) {
    castedInput = OpBackend::BuildCast(
        this, graph, syn_in(0), inputShape, inputType, meta.dtype);
  }
  auto input = castedInput.has_value() ? castedInput.value().get() : syn_in(0);

  // isNan on input
  auto is_nan = BuildOp(
      graph,
      get_guid_with_precision("isnan_fwd", meta.dtype),
      {input},
      {{inputShape, c10::ScalarType::Char}});

  auto zero_constant = ConstantHelper(graph, 0.0f, meta.dtype, inputShape);

  // where on is_nan
  auto where = BuildOp(
      graph,
      get_guid_with_precision("where_fwd", meta.dtype),
      {is_nan[0].get(), zero_constant.get(), input},
      {{inputShape, meta.dtype}});

  auto reduce_sum = HandleReductionDimAndKeepdim(
      this,
      graph,
      self,
      {where[0].get()},
      dim,
      keepdim,
      guid,
      {{meta.shape, meta.dtype, 0}});
  syn_out(0) = std::move(reduce_sum[0]);
}
} // namespace habana
