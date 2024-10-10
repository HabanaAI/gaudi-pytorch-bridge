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
 *******************************************************************************/

#include <ATen/core/Generator.h>

namespace habana {
namespace detail {
at::Generator& getDefaultHPUGenerator();
at::Generator createHPUGenerator();
} // namespace detail
uint32_t get_seed_hpu(const c10::optional<at::Generator>& gen);
at::Tensor get_seed_tensor_hpu(const c10::optional<at::Generator>& gen);
} // namespace habana
