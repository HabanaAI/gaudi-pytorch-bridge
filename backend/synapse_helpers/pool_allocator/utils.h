/*******************************************************************************
 * Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
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
#include <synapse_api_types.h>
#include "PoolAllocator.h"

namespace synapse_helpers {
namespace pool_allocator {

static const uint64_t kInvalidBinNum = -1;
// The largest bin'd chunk size is 256 << 21 = 512MB.
static const uint64_t kNumBins = 21;
static const size_t kMinAllocationBits = 8;
static const size_t kMinAllocationSize = 1 << kMinAllocationBits;

// workaround only
void set_device_deallocation(bool flag);
bool get_device_deallocation();
// --

void print_device_memory_stats(synDeviceId deviceID);

} // namespace pool_allocator
} // namespace synapse_helpers
