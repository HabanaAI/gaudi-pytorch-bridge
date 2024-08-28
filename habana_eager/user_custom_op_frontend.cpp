/*******************************************************************************
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
#include "habana_eager/ops/eager_op.h"
#include "include/habanalabs/hpu_custom_op_pt2.h"

namespace habana {
namespace custom_op {

std::vector<at::Tensor> UserCustomOpDescriptor::execute(
    const std::vector<c10::IValue>& inputs) {
  std::vector<std::vector<int64_t>> output_shapes;
  std::vector<at::ScalarType> output_dtypes;
  for (const auto& meta : output_meta_fn_(inputs)) {
    output_shapes.push_back(meta.shape);
    output_dtypes.push_back(meta.dtype);
  }
  habana::eager::EagerOp<std::vector<at::Tensor>> hpu_op{
      schema_, inputs, std::move(output_shapes)};
  hpu_op.set_scalar_types(std::move(output_dtypes));
  return hpu_op.call();
}

} // namespace custom_op
} // namespace habana
