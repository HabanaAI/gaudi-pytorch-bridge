/*******************************************************************************
 * Copyright (C) 2022-2024 Habana Labs, Ltd. an Intel Company
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

#include "backend/synapse_helpers/device_helpers.h"
#include "generated/backend/where.h"
#include "habana_helpers/dtype_helpers.h"

namespace habana {

OutputMetaDataVector WhereMeta(const at::Stack& stack) {
  auto cond = stack_tensor(stack, 0);
  auto self = stack_tensor(stack, 1);
  auto other = stack_tensor(stack, 2);

  const auto& condSizes = cond.sizes();
  const auto& selfSizes = self.sizes();
  const auto& otherSizes = other.sizes();

  OutputMetaData meta{};

  meta.dtype = at::result_type(self, other);
  meta.shape = at::infer_size(at::infer_size(condSizes, selfSizes), otherSizes);

  return {meta};
}

void WhereBackend::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto shape = WhereMeta(stack)[0].shape;
  const auto& self = stack_tensor(stack, 1);
  const auto& other = stack_tensor(stack, 2);

  c10::optional<const at::IValue*> output = IsOutputAvailable()
      ? c10::make_optional<const at::IValue*>(&stack.back())
      : c10::nullopt;

  auto dtype_helper =
      habana_helpers::DTypeHelper::binary_op_with_type_promotion(
          {stack.at(1), stack.at(2)}, output, false);

  c10::ScalarType result_type =
      habana_helpers::getInternalDtype(dtype_helper.get_result_dtype());

  std::vector<synapse_helpers::tensor> cast;
  std::vector<synTensor> inputs = {syn_in(0), syn_in(1), syn_in(2)};

  if (habana_helpers::getInternalDtype(self.scalar_type()) != result_type) {
    cast.emplace_back(BuildCast(
        this, graph, syn_in(1), self.sizes(), self.scalar_type(), result_type));
    inputs[1] = cast[0].get();
  } else if (
      habana_helpers::getInternalDtype(other.scalar_type()) != result_type) {
    cast.emplace_back(BuildCast(
        this,
        graph,
        syn_in(2),
        other.sizes(),
        other.scalar_type(),
        result_type));
    inputs[2] = cast[0].get();
  }

  update_guid_dtype(guid_, result_type);

  auto result =
      BuildOp(graph, guid_, std::move(inputs), {{shape, result_type, 0}});
  syn_out(0) = std::move(result[0]);
}

FALLBACK_CHECK(
    WhereFallbackCheck,
    const at::Tensor& condition,
    const at::Tensor& self,
    const at::Tensor& other) {
  if (condition.scalar_type() != torch::kBool) {
    return false;
  }

  auto result_type = at::result_type(self, other);
  switch (result_type) {
    case torch::kBool:
    case torch::kInt32:
    case torch::kBFloat16:
    case torch::kFloat32:
    case torch::kFloat64:
    case torch::kUInt8:
    case torch::kInt16:
    case torch::kInt8:
      return true;
    // When Int64 isn't supported kInt64 is actually of type Int32
    case torch::kInt64:
      return true;
    case torch::kHalf: {
      return synapse_helpers::device_supports_fp16(
          HPURegistrar::get_device().type());
    }
    default:
      return false;
  }
}

} // namespace habana
