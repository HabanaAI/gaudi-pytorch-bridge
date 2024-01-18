/*******************************************************************************
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

namespace habana {
namespace eager {
std::tuple<at::Tensor, at::Tensor> _unique_eager(
    const at::Tensor& self,
    bool sorted,
    bool return_inverse);
} // namespace eager
} // namespace habana
