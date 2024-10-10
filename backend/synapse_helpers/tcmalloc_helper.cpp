#include "backend/synapse_helpers/tcmalloc_helper.h"
#include <stdlib.h>
#include "pytorch_helpers/habana_helpers/logging.h"

namespace synapse_helpers {
// Function to release free memory using tcmalloc library
void ReleaseFreeMemory() {
  static ReleaseFreeMemoryFunc releaseFreeMemory =
      reinterpret_cast<ReleaseFreeMemoryFunc>(
          dlsym(RTLD_DEFAULT, "MallocExtension_ReleaseFreeMemory"));
  if (releaseFreeMemory) {
    PT_DYNAMIC_SHAPE_DEBUG("MallocExtension_ReleaseFreeMemory called");
    releaseFreeMemory();
  } else {
    PT_DYNAMIC_SHAPE_WARN("MallocExtension_ReleaseFreeMemory was not linked");
  }
}
} // namespace synapse_helpers
