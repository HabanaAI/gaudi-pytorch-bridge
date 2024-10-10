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
#pragma once
#include "habana_eager/ops/eager_op.h"
#include "hpu_ops/common/reduction_template.h"

namespace habana {
HPU_REDUCTION_TEMPLATE_FRONTEND(eager::EagerOp)
} // namespace habana
