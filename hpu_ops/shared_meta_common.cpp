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

} // namespace habana
