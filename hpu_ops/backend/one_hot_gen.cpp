/******************************************************************************
 * Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
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
#include "generated/backend/one_hot.h"

namespace {
constexpr int64_t DEFAULT_NUM_OF_CLASSES = -1;
}

namespace habana {
int64_t calculateNumberOfClasses(const at::Stack& stack) {
  const auto num_classes = stack.at(1).toInt();

  TORCH_CHECK(
      num_classes != DEFAULT_NUM_OF_CLASSES, "Number of classes cannot be -1");

  return num_classes;
}

OutputMetaDataVector OneHotMeta(const at::Stack& stack) {
  auto input = stack_tensor(stack, 0);
  OutputMetaData meta;
  const auto num_classes = calculateNumberOfClasses(stack);
  meta.shape = input.sizes().vec();
  meta.shape.push_back(num_classes);
  meta.dtype = input.scalar_type();

  return {meta};
}

void OneHot::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  const auto num_classes = calculateNumberOfClasses(stack);
  const auto meta = OutputMeta(stack)[0];
  auto input = stack_tensor(stack, 0);
  const std::string guid{"one_hot_fwd"};
  std::optional<synapse_helpers::tensor> cast{};

  ns_OneHotKernel::Params oneHotParams{
      .axis = 0,
      .depth = static_cast<int>(num_classes),
      .on_value = 1.0f,
      .off_value = 0.0f};

  auto output_type = c10::ScalarType::Float;

  if (!isIntegralType(meta.dtype, true)) {
    output_type = meta.dtype;
    c10::ScalarType target_type = c10::ScalarType::Short;

    if (output_type == at::kHalf || output_type == at::kFloat) {
      target_type = c10::ScalarType::Int;
    }
    cast = BuildCast(
        this, graph, syn_in(0), input.sizes(), output_type, target_type);
  }

  auto input_feature_map = (cast.has_value()) ? cast->get() : syn_in(0);
  auto result = BuildOp(
      graph,
      get_guid_with_precision(guid, output_type),
      {input_feature_map},
      {{meta.shape, output_type}},
      &oneHotParams,
      sizeof(oneHotParams));

  // If output datatype of one_hot op and Int datatype map
  // to the same precision datatype in TPC, cast is not needed.
  // Different datatypes can map to same precision.
  // For example, if INT64 is not supported, both Long and INT
  // map to i32 precision in TPC.
  auto from_dtype = habana_helpers::GetPrecisionString(output_type);
  auto to_dtype = habana_helpers::GetPrecisionString(c10::ScalarType::Int);
  if (from_dtype != to_dtype) {
      syn_out(0) = BuildCast(
          this,
          graph,
          result[0].get(),
          meta.shape,
          output_type,
          c10::ScalarType::Int,
          0);
  } else {
      syn_out(0) = std::move(result.at(0));
  }
}
} // namespace habana
