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
#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/TensorBody.h>

namespace habana {
namespace eager {
at::Tensor nonzero_eager(const at::Tensor& self);
at::Tensor& nonzero_out_eager(const at::Tensor& self, at::Tensor& out);
} // namespace eager
} // namespace habana
