/*******************************************************************************
 * Copyright (C) 2020-2024 Habana Labs, Ltd. an Intel Company
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
#include <ATen/ATen.h>
#include <c10/core/Allocator.h>
#include <synapse_api_types.h>
#include "backend/synapse_helpers/device.h"
#include "pytorch_helpers/habana_helpers/pt_version_check.h"

namespace habana {

at::Allocator* getPinnedMemoryAllocator();
bool PinnedMemoryAllocator_is_pinned(void* ptr);

class PinnedMemoryAllocator final : public at::Allocator {
 public:
  PinnedMemoryAllocator();
  ~PinnedMemoryAllocator();
  at::DeleterFnPtr raw_deleter() const override;
  static void deleter(void* ptr);

#if IS_PYTORCH_AT_LEAST(2, 3)
  at::DataPtr allocate(size_t size) override;
  void copy_data(void* dest, const void* src, std::size_t count) const override;
#else
  at::DataPtr allocate(size_t size) const override;
#endif
  // user must manually set active device before calling allocator functions
  static synDeviceId allocator_active_device_id;
};

} // namespace habana
