/**
* Copyright (c) 2021-2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
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
