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

// workaround only
void set_device_deallocation(bool flag);
bool get_device_deallocation();
// --

void print_device_memory_stats(synDeviceId deviceID);

} // namespace pool_allocator
} // namespace synapse_helpers
