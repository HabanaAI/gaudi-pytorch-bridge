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

#include "hpu_ops/common/arange_gen.h"
#include "generated/eager/arange.h"
#include "habana_eager/ops/eager_op.h"
#include "habana_helpers/dtype_helpers.h"
#include "hpu_ops/hpu_op_helper.h"

namespace habana {
HPU_OP_FRONTEND_CUSTOM_CTOR(eager::EagerOp, ArangeFE, -1, at::Tensor&) {
  m_inputs = {
      inputs[0].toScalar(),
      inputs[1].toScalar(),
      inputs[2].toScalar(),
      {},
      {},
      inputs[3],
  };
  validate_inputs(m_inputs);
}

HPU_OP_FRONTEND_CREATE_RESULT_ONLY(eager::EagerOp, ArangeFE, at::Tensor&) {
  HABANA_ASSERT(false, "Shouldn't be reachable");
  return eager::EagerOp<at::Tensor&>::get_result_overrideable();
}
} // namespace habana