/******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
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

sizes_vec OneHotOutputShape(const at::Stack& stack) {
  auto input = stack_tensor(stack, 0);
  const auto num_classes = calculateNumberOfClasses(stack);
  auto out_shape = input.sizes().vec();
  out_shape.push_back(num_classes);

  return {out_shape};
}

void OneHot::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  const auto num_classes = calculateNumberOfClasses(stack);
  const auto output_shape = OneHotOutputShape(stack)[0];
  auto input = stack_tensor(stack, 0);
  const std::string guid{"one_hot_fwd_"};
  std::optional<synapse_helpers::tensor> cast{};

  ns_OneHotKernel::Params oneHotParams{
      .axis = 0,
      .depth = static_cast<int>(num_classes),
      .on_value = 1.0f,
      .off_value = 0.0f};

  auto output_type = c10::ScalarType::Float;

  if (ScalarType() != c10::ScalarType::Int) {
    output_type = ScalarType();
    c10::ScalarType target_type = c10::ScalarType::Short;

    if (output_type == at::kHalf || output_type == at::kFloat) {
      target_type = c10::ScalarType::Int;
    }
    cast =
        CastHelper(graph, syn_in(0), input.sizes(), output_type, target_type);
  }

  auto input_feature_map = (cast.has_value()) ? cast->get() : syn_in(0);
  auto result = BuildOp(
      graph,
      guid + habana_helpers::name_suffix_from_type(output_type),
      {input_feature_map},
      {{output_shape, output_type}},
      &oneHotParams,
      sizeof(oneHotParams));

  syn_out(0) = CastHelper(
      graph,
      result[0].get(),
      output_shape,
      output_type,
      c10::ScalarType::Int,
      0);
}
} // namespace habana
