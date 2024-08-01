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

#include "generated/lazy/addcdiv.h"
#include "generated/lazy/addcmul.h"
namespace habana {

static void convert_scalar_val_to_tensor(at::Stack& inputs) {
  auto self = inputs.at(0).toTensor();
  auto value = inputs.at(3).toScalar();
  at::Tensor valueTensor;
  if (!value.equal(1))
    valueTensor =
        habana_lazy::get_tensor_for_scalar(value.to<double>(), self.options());

  c10::optional<at::Tensor> valueTensorOpt = c10::make_optional(valueTensor);
  inputs.at(3) = valueTensorOpt;
}

HPU_OP_FRONTEND_CUSTOM_CTOR_ONLY(habana_lazy::LazyOp, AddCOpFE, at::Tensor&) {
  convert_scalar_val_to_tensor(get_inputs());
}

HPU_OP_FRONTEND_CUSTOM_CTOR_ONLY(habana_lazy::LazyOp, AddCOpFE, at::Tensor) {
  convert_scalar_val_to_tensor(get_inputs());
}

} // namespace habana
