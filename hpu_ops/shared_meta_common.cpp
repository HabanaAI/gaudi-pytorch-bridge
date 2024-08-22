/******************************************************************************
 * Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
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

#include "hpu_ops/shared_meta_common.h"

namespace habana {

SharedMetaDataVector Input0SharedMeta(
    const at::Stack& stack,
    const std::string& guid) {
  const auto& input = stack_tensor(stack, 0);

  SharedMetaData meta{guid};
  meta.inputs_data = {{input.dim(), input.scalar_type()}};
  meta.outputs_data = {meta.inputs_data[0]};

  return {meta};
}

SharedMetaDataVector Input0ToOut0And1SharedMeta(
    const at::Stack& stack,
    const std::string& guid) {
  const auto& input = stack_tensor(stack, 0);

  SharedMetaData meta{guid};
  SharedMetaTensor inOutTensor = {input.dim(), input.scalar_type()};
  meta.inputs_data = {inOutTensor};
  meta.outputs_data = {inOutTensor, inOutTensor};

  return {meta};
}

SharedMetaDataVector AdaptiveBwdSharedMeta(
    const at::Stack& stack,
    const std::string& guid) {
  const auto& grad = stack_tensor(stack, 0);
  const auto& input = stack_tensor(stack, 1);

  SharedMetaData meta{guid};
  meta.inputs_data = {
      {grad.dim(), grad.scalar_type()}, {input.dim(), input.scalar_type()}};
  meta.outputs_data = {{input.dim(), grad.scalar_type()}};

  return {meta};
}

SharedMetaDataVector AvgPoolBwdSharedMeta(
    const at::Stack& stack,
    const std::string& guid) {
  const auto& grad = stack_tensor(stack, 0);
  const auto& input = stack_tensor(stack, 1);

  SharedMetaData meta{guid};
  meta.inputs_data = {{grad.dim(), grad.scalar_type()}};
  meta.outputs_data = {{input.dim(), input.scalar_type()}};

  return {meta};
}

SharedMetaDataVector FillCumSumProdSharedMeta(
    const at::Stack& stack,
    const std::string& guid) {
  const auto& input = stack_tensor(stack, 0);
  at::ScalarType dtype =
      stack.at(2).isNone() ? input.scalar_type() : stack.at(2).toScalarType();

  if (habana_helpers::is_downcast_to_int_needed(dtype))
    dtype = at::ScalarType::Int;
  else if (dtype == at::ScalarType::Double)
    dtype = at::ScalarType::Float;
  else if (
      dtype == at::ScalarType::Bool || dtype == at::ScalarType::Char ||
      dtype == at::ScalarType::Byte)
    dtype = at::ScalarType::Int;

  SharedMetaData meta{guid};
  meta.inputs_data = {{input.dim(), dtype}};
  meta.outputs_data = {{input.dim(), dtype}};

  return {meta};
}

SharedMetaDataVector IsFiniteInfNanSharedMeta(
    const at::Stack& stack,
    const std::string& guid) {
  const auto& input = stack_tensor(stack, 0);
  auto dtype = input.scalar_type();
  auto rank = input.dim();

  if (c10::isIntegralType(dtype, true))
    dtype = c10::ScalarType::Int;

  SharedMetaData meta{guid};
  meta.inputs_data = {{rank, dtype}};
  meta.outputs_data = {{rank, torch::kBool}};

  return {meta};
}

SharedMetaDataVector RoundingSharedMeta(
    const at::Stack& stack,
    const std::string& guid) {
  auto input = stack.at(0).toTensor();
  auto rank = input.dim();
  auto dtype = input.scalar_type();

  SharedMetaData roundingMeta;
  roundingMeta.guid = c10::isIntegralType(dtype, true) ? "identity" : guid;

  SharedMetaTensor inOutTensor = {rank, dtype};
  roundingMeta.inputs_data = {inOutTensor};
  roundingMeta.outputs_data = {inOutTensor};
  return {roundingMeta};
}

} // namespace habana
