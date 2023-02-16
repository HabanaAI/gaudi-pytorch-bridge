/*******************************************************************************
 * Copyright (C) 2022-2023 Habana Labs, Ltd. an Intel Company
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

#include "misc_utils.h"
#include <ATen/Tensor.h>
#include "habana_helpers/logging.h"

namespace habana {

bool IsHostMemoryThresholdReached() {
  // PT_HPU_HOST_MEMORY_THRESHOLD_PERCENT - Maximum percentage of total memory
  // beyond which PyTorch Bridge will remove cached recipes
  uint32_t host_memory_threshold_percent =
      GET_ENV_FLAG_NEW(PT_HPU_HOST_MEMORY_THRESHOLD_PERCENT);

  if (host_memory_threshold_percent) {
    struct sysinfo si;
    sysinfo(&si);
    uint64_t totalram_bytes = si.totalram;
    uint64_t freeram_avail_bytes = si.freeram;
    uint64_t host_memory_used_bytes = totalram_bytes - freeram_avail_bytes;
    uint64_t host_memory_threshold_bytes =
        (totalram_bytes * host_memory_threshold_percent) / 100;
    if (host_memory_used_bytes > host_memory_threshold_bytes) {
      static bool warned_once = false;
      if (!warned_once) {
        warned_once = true;
        PT_BRIDGE_WARN(
            "Cache eviction started HostMemoryThresholdReached host_memory_used_bytes = ",
            host_memory_used_bytes,
            "host_memory_threshold_bytes",
            host_memory_threshold_bytes);
      }
      return true;
    }
  }

  return false;
}

} // namespace habana
