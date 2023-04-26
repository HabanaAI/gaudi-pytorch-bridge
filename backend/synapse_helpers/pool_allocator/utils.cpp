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
#include <synapse_api.h>

#include <habana_helpers/logging.h>
#include "PoolAllocator.h"
#include "utils.h"

namespace synapse_helpers::pool_allocator {

// fix me - Synapse dev map is nullified before free.
// There is a random failure in synapse when memory is freed.
// This flags ensures, we stop freeing once we encounter
//   an error till issue is resolved.
// All device buffers are released during device release
//   like other resource deallocations in synapse
static bool null_dev_map_found = false;

void set_device_deallocation(bool flag) {
  null_dev_map_found = flag;
}

bool get_device_deallocation() {
  return null_dev_map_found;
}

void print_device_memory_stats(synDeviceId deviceID) {
  uint64_t free_mem, total_mem;
  auto status = synDeviceGetMemoryInfo(deviceID, &free_mem, &total_mem);
  if (synStatus::synSuccess != status) {
    PT_DEVMEM_FATAL(
        Logger::formatStatusMsg(status),
        "POOL:: Cannot obtain device memory size.");
  }
  PT_DEVMEM_DEBUG(
      "POOL:: Device memory size: total= ", total_mem, " free = ", free_mem);
}

} // namespace synapse_helpers::pool_allocator
