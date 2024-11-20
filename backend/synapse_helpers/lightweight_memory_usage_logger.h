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
#ifndef DISABLE_MEMORY_MONITORING
namespace synapse_helpers::pool_allocator {
class CoalescedStringentPooling;
}

namespace synapse_helpers::LightweightMemoryMonitor {
void setDevice(const synapse_helpers::pool_allocator::CoalescedStringentPooling*
                   allocator_ptr);
void resetDevice();
} // namespace synapse_helpers::LightweightMemoryMonitor
#define MEMORY_MONITORING_SET_DEVICE(device) \
  synapse_helpers::LightweightMemoryMonitor::setDevice(device);
#define MEMORY_MONITORING_RESET_DEVICE \
  synapse_helpers::LightweightMemoryMonitor::resetDevice();
#elif
#define MEMORY_MONITORING_SET_DEVICE(device)
#define MEMORY_MONITORING_RESET_DEVICE
#endif
