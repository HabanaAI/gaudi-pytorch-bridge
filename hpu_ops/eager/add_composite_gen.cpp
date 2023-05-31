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
#include "hpu_ops/common/add_composite_gen.h"
#include "generated/eager/addcdiv.h"
#include "generated/eager/addcmul.h"
namespace habana {

static void convert_scalar_val_to_tensor(at::Stack& inputs) {
  auto self = inputs[inp_idx].toTensor();
  auto value = inputs[val_scalar_idx].toScalar();
  at::Tensor valueTensor;
  // The scalar to tensor conversion in eager frontend is a workaround to the
  // problem reported here: https://jira.habana-labs.com/browse/SW-147394
  if (!value.equal(1))
    valueTensor = at::scalar_tensor(value, self.options());

  c10::optional<at::Tensor> valueTensorOpt = c10::make_optional(valueTensor);
  inputs[val_scalar_idx] = valueTensorOpt;
}

HPU_OP_FRONTEND_CUSTOM_CTOR_ONLY(eager::EagerOp, AddCOpFE, at::Tensor&) {
  convert_scalar_val_to_tensor(get_inputs());
}

HPU_OP_FRONTEND_CUSTOM_CTOR_ONLY(eager::EagerOp, AddCOpFE, at::Tensor) {
  convert_scalar_val_to_tensor(get_inputs());
}

} // namespace habana
