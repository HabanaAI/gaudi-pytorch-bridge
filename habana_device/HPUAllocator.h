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
#include <synapse_helpers/habana_tensor.h>

namespace at {
namespace habana {

// user must manually set active device before calling allocator functions
extern synDeviceId allocator_active_device_id;

class HabanaAllocator {
 public:
  void* malloc(size_t num_bytes);
  void free(void* ptr);
};

class HPUDeviceAllocator final : public at::Allocator {
 public:
  at::DataPtr allocate(size_t size) const override;
  at::DeleterFnPtr raw_deleter() const override;
};

} // namespace habana
} // namespace at

namespace habana_helpers {
class HabanaAllocator : public synapse_helpers::device_allocator {
 public:
  HabanaAllocator(synDeviceId);

  void reset() override;
  void release() override;
  void* alloc(size_t size) override;
  void free(void* ptr) override;

 private:
  synDeviceId device_id{synapse_helpers::device::INVALID_ID};
};
} // namespace habana_helpers
