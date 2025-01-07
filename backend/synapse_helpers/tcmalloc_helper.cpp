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
