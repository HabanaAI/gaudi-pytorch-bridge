/*******************************************************************************
 * Copyright (C) 2022 Habana Labs, Ltd. an Intel Company
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

#include <ATen/core/boxing/KernelFunction.h>
#include <ATen/native/CPUFallback.h>

namespace at_ver::native {

template <c10::KernelFunction::BoxedKernelFunction* fallback_fn, class Op>
using call_fallback_fn_symint =
    at::native::call_fallback_fn_symint<fallback_fn, Op>;

} // namespace at_ver::native
