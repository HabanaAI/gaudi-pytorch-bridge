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
static void fill_optionals(at::Stack& inputs) {
  /* According to schema_args in hpu_op.yaml the Tensor "value" should be placed
  in the 4th position and the Tensor "scalar_value" in 5th position Initially,
  the Tensor "scalar_value" is in the 4th position, so we need to add Tensor
  value=None before the Tensor "scalar_value" to move it on the correct position
  */
  inputs.insert(inputs.begin() + val_tensor_idx, c10::nullopt);
}

HPU_OP_FRONTEND_CUSTOM_CTOR_ONLY(eager::EagerOp, AddCOpFE, at::Tensor&) {
  fill_optionals(get_inputs());
}

HPU_OP_FRONTEND_CUSTOM_CTOR_ONLY(eager::EagerOp, AddCOpFE, at::Tensor) {
  fill_optionals(get_inputs());
}

} // namespace habana
