/*******************************************************************************
 * Copyright (C) 2020-2024 Habana Labs, Ltd. an Intel Company
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

#include <torch/csrc/jit/ir/ir.h>
#include "backend/habana_operator.h"

namespace habana_lazy {
at::IntArrayRef getDimsForLayout5d(
    habana::LayoutFormat channel_order,
    habana::LayoutFormat current_order);

at::IntArrayRef getDimsForLayout(
    habana::LayoutFormat channel_order,
    habana::LayoutFormat current_order);
}; // namespace habana_lazy
