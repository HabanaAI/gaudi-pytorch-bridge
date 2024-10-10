/******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 ******************************************************************************
 */

#pragma once

#include <cstdint>

namespace habana {
namespace profile {

namespace memory {
void recordAllocation(
    uint64_t addr,
    uint64_t size,
    uint64_t total_allocated,
    uint64_t total_reserved);
void recordDeallocation(
    uint64_t addr,
    uint64_t total_allocated,
    uint64_t total_reserved);
bool enabled();
}; // namespace memory

}; // namespace profile
}; // namespace habana