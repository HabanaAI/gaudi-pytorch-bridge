
#include "misc_utils.h"
#include <ATen/Tensor.h>

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
      return true;
    }
  }

  return false;
}

} // namespace habana

namespace habana_lazy {

bool IsCollective(const c10::Symbol& symbol) {
  static c10::Symbol hccl_namepsace =
      c10::Symbol::fromQualString("namespaces::hccl");
  return symbol.ns() == hccl_namepsace;
}

} // namespace habana_lazy
