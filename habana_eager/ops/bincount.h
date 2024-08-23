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
at::Tensor bincount_eager(
    const at::Tensor& self,
    const c10::optional<at::Tensor>& weights,
    int64_t minlength);
} // namespace eager
} // namespace habana
