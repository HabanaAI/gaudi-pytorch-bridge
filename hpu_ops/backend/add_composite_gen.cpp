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
    BinaryWithAlphaMode_t mode,
    size_t& size) {
  PARAMS_STUB(ns_BinaryWithAlphaKernel::Params);
  auto out_scalar_type = stack[inp_idx].toTensor().scalar_type();

  params->mode = mode;
  // if alpha is not equal to 1 then it is passed as tensor (4th input),
  // otherwise as params
  auto val =
      stack[val_scalar_idx].isScalar() ? stack[val_scalar_idx].toScalar() : 1;
  if (c10::isFloatingType(out_scalar_type)) {
    get<float>(params->alpha) = val.to<float>();
  } else {
    get<int>(params->alpha) = val.to<int>();
  }

  return params;
}

std::shared_ptr<void> FillAddcmulParams(const at::Stack& stack, size_t& size) {
  return FillAddCompositeParams(
      stack, BinaryWithAlphaMode_t::BINARY_WITH_ALPHA_MODE_CMUL, size);
}

std::shared_ptr<void> FillAddcdivParams(const at::Stack& stack, size_t& size) {
  return FillAddCompositeParams(
      stack, BinaryWithAlphaMode_t::BINARY_WITH_ALPHA_MODE_CDIV, size);
}

} // namespace habana
