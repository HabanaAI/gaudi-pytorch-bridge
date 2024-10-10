/******************************************************************************
 * Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
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
#include "generated/eager/convolution_backward_overrideable.h"
#include "hpu_ops/hpu_op_helper.h"

namespace habana {

HPU_OP_FRONTEND_CUSTOM_CTOR_ONLY(
    eager::EagerOp,
    ConvolutionBackwardOverrideableFE,
    ::std::tuple<at::Tensor, at::Tensor, at::Tensor>) {}

} // namespace habana
