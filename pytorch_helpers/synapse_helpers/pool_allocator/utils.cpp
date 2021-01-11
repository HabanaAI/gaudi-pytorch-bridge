/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include <synapse_api.h>

#include <habana_helpers/logging.h>
//#include "../HPUAllocator.h"
//#include "../HPUCheck.h"
//#include "../HPUGuardImpl.h"
//#include "../hpu_cached_devices.h"
#include "CoalescedPoolAllocator.h"
#include "PoolAllocator.h"
#include "utils.h"

namespace synapse_helpers {
namespace pool_allocator {

// fix me - synpase dev map is nullified before free.
// There is a random failure in synapse when memory is freed.
// This flags ensures, we stop freeing once we encounter
//   an error till issue is resolved.
// All device buffers are released during device release
//   like other resource dealloations in synapse
static bool null_dev_map_found = false;

void set_device_deallocation(bool flag) {
  null_dev_map_found = flag;
}

bool get_device_deallocation() {
  return null_dev_map_found;
}

size_t block_align(size_t n) {
  return (n + DEFAULT_ALIGNMENT - 1) & ~(DEFAULT_ALIGNMENT - 1);
}

void print_device_memory_stats(synDeviceId deviceID) {
  uint64_t free_mem, total_mem;
  auto status = synDeviceGetMemoryInfo(deviceID, &free_mem, &total_mem);
  if (synStatus::synSuccess != status) {
    PT_DEVICE_FATAL(
        "POOL:: Cannot obtain device memory size. Status: ", status);
  }
  PT_DEVICE_DEBUG(
      "POOL:: Device memory size: total= ", total_mem, " free = ", free_mem);
}

} // namespace pool_allocator
} // namespace synapse_helpers
