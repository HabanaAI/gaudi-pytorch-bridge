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
#include "generated/eager/addcdiv.h"
#include "generated/eager/addcmul.h"
namespace habana {
HPU_OP_FRONTEND_CUSTOM_CTOR_ONLY(eager::EagerOp, AddCOpFE, at::Tensor&) {}
} // namespace habana
