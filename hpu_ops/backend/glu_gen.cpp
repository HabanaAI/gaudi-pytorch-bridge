/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "generated/backend/glu.h"
#include "generated/backend/glu_backward.h"

namespace habana {

sizes_vec GluOutputShape(const at::Stack& stack) {
  auto out_shape = stack.at(0).toTensor().sizes().vec();
  const int64_t axis = stack.at(1).toInt();
  auto dim = (axis >= 0) ? axis : stack.at(0).toTensor().dim() + axis;
  out_shape[dim] = out_shape[dim] / 2;
  return {out_shape};
}

sizes_vec GluBwdOutputShape(const at::Stack& stack) {
  auto out_shape = stack.at(1).toTensor().sizes().vec();
  return {out_shape};
}

std::shared_ptr<void> FillGluParams(
    const at::Stack& stack,
    size_t& size,
    int dim_index) {
  auto self = stack_tensor(stack, dim_index - 1);
  int dim_ = stack.at(dim_index).toInt();
  auto dim = at::maybe_wrap_dim(dim_, self.dim(), /*wrap_scalar=*/true);
  PARAMS_STUB(ns_GatherKernel::Params);
  params->axis = dim;
  return params;
}
std::shared_ptr<void> FillGluFwdParams(const at::Stack& stack, size_t& size) {
  return FillGluParams(stack, size, 1 /*dim_index FWD*/);
}

std::shared_ptr<void> FillGluBwdParams(const at::Stack& stack, size_t& size) {
  return FillGluParams(stack, size, 2 /*dim_index BWD*/);
}

} // namespace habana
