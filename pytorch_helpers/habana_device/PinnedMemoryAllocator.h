/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#pragma once
#include <ATen/ATen.h>
#include <c10/core/Allocator.h>
#include <synapse_api_types.h>
#include <synapse_helpers/device.h>

namespace at {
namespace habana {

at::Allocator* getPinnedMemoryAllocator();
bool PinnedMemoryAllocator_is_pinned(void *ptr);

class PinnedMemoryAllocator final : public at::Allocator {
 public:
  PinnedMemoryAllocator();
  ~PinnedMemoryAllocator();
  at::DataPtr allocate(size_t size) const override;
  at::DeleterFnPtr raw_deleter() const override;
  static void deleter(void *ptr);

  // user must manually set active device before calling allocator functions
  static synDeviceId allocator_active_device_id;
};

} // namespace habana
} // namespace at
