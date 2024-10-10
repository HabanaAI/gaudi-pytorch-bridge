/******************************************************************************
 * Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
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
#include "generated/backend/replication_pad1d.h"
#include "generated/backend/replication_pad1d_backward.h"
#include "generated/backend/replication_pad2d.h"
#include "generated/backend/replication_pad2d_backward.h"
#include "generated/backend/replication_pad3d.h"
#include "generated/backend/replication_pad3d_backward.h"

namespace habana {
enum PadType : int8_t { pad1D = 0, pad2D, pad3D };
sizes_vec ComputePadOutputShape(const at::Stack& stack, PadType padType);
std::shared_ptr<void> FillPadFwdBwdParams(
    const at::Stack& stack,
    PadType padType,
    size_t& size,
    bool backward);

} // namespace habana