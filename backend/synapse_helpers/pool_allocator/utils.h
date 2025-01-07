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
