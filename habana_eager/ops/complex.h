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
#include "habana_kernels/lazy_kernels_declarations.h"

namespace habana {
namespace eager {

at::Tensor complex_hpu(const at::Tensor& real, const at::Tensor& imag);
} // namespace eager
} // namespace habana
