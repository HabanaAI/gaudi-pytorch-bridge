/*******************************************************************************
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

#include <ATen/core/TensorBody.h>

namespace habana {
namespace eager {
at::Tensor& set_(
    at::Tensor& self,
    at::Storage source,
    at::SymInt storage_offset,
    at::SymIntArrayRef size,
    at::SymIntArrayRef stride);
} // namespace eager
} // namespace habana
