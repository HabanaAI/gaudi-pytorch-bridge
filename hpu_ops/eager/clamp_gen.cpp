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

#include <c10_ver/core/SymIntArrayRef.h>
#include "generated/eager/clamp.h"
#include "generated/eager/clamp_max.h"
#include "generated/eager/clamp_min.h"
#include "habana_eager/ops/eager_op.h"
namespace habana {
HPU_OP_FRONTEND_CUSTOM_CTOR_ONLY(eager::EagerOp, ClampFE, at::Tensor) {}
HPU_OP_FRONTEND_CUSTOM_CTOR_ONLY(eager::EagerOp, ClampFE, at::Tensor&) {}
} // namespace habana
