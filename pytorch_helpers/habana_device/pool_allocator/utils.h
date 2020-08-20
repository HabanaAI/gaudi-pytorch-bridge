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
#include <ATen/ATen.h>
#include <c10/core/Allocator.h>
#include <synapse_api_types.h>
#include <synapse_helpers/device.h>
#include <synapse_helpers/habana_tensor.h>
#include "PoolAllocator.h"

namespace at {
namespace habana {
namespace pool_allocator {

// workaround only
void set_device_deallocation(bool flag);
bool get_device_deallocation();
// --

size_t block_align(size_t n);
void print_device_memory_stats(synDeviceId deviceID);

} // namespace pool_allocator
} // namespace habana
} // namespace at