/******************************************************************************
 * Copyright (C) 2021-2023 Habana Labs, Ltd. an Intel Company
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

#include "hpu_ops/common/add_composite_gen.h"
#include "generated/backend/addcdiv.h"
#include "generated/backend/addcmul.h"

namespace habana {

OutputMetaDataVector AddCOpsMeta(const at::Stack& stack) {
  const torch::Tensor& self = stack_tensor(stack, inp_idx);
  const torch::Tensor& other1 = stack_tensor(stack, oth1_idx);
  const torch::Tensor& other2 = stack_tensor(stack, oth2_idx);
  auto tmp = at::infer_size(self.sizes(), other1.sizes());
  OutputMetaData meta;
  meta.dtype = self.scalar_type();
  meta.shape = at::infer_size(tmp, other2.sizes());
  return {meta};
}

std::shared_ptr<void> FillAddCompositeParams(
    const at::Stack& stack,
    enum modes mode_t,
    size_t& size) {
  PARAMS_STUB(ns_BinaryWithAlphaKernel::Params);
  auto out_scalar_type = stack[inp_idx].toTensor().scalar_type();

  // if alpha is not equal to 1 then it is passed as tensor (4th input),
  // otherwise as params
  if (c10::isFloatingType(out_scalar_type))
    params->alpha.f = 1.0f;
  else
    params->alpha.i = 1;

  if (mode_t == mul)
    params->mode = BinaryWithAlphaMode_t::BINARY_WITH_ALPHA_MODE_CMUL;
  else if (mode_t == div)
    params->mode = BinaryWithAlphaMode_t::BINARY_WITH_ALPHA_MODE_CDIV;

  return params;
}

std::shared_ptr<void> FillAddcmulParams(const at::Stack& stack, size_t& size) {
  return FillAddCompositeParams(stack, mul, size);
}

std::shared_ptr<void> FillAddcdivParams(const at::Stack& stack, size_t& size) {
  return FillAddCompositeParams(stack, div, size);
}

} // namespace habana
