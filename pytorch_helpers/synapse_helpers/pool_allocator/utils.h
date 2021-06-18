/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#pragma once
#include <synapse_api_types.h>
#include <synapse_helpers/device.h>
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

size_t block_align(size_t n);
void print_device_memory_stats(synDeviceId deviceID);

} // namespace pool_allocator
} // namespace synapse_helpers
