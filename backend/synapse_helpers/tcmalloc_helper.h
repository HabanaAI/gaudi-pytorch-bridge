#pragma once

#include <dlfcn.h>
#include <memory>

using ReleaseFreeMemoryFunc = void (*)(void);
namespace synapse_helpers {
// Function to release free memory using tcmalloc library
void ReleaseFreeMemory();
} // namespace synapse_helpers
