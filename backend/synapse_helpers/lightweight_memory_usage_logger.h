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
