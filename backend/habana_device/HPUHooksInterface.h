/*******************************************************************************
 * Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
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
#include <ATen/detail/HPUHooksInterface.h>
#include "pytorch_helpers/habana_helpers/pt_version_check.h"

namespace habana {

struct HPUHooks : public at::HPUHooksInterface {
  HPUHooks(at::HPUHooksArgs){};

#if IS_PYTORCH_AT_LEAST(2, 6)
  void init() const override;
#else
  void initHPU() const override;
#endif
  bool hasHPU() const override;
  const at::Generator& getDefaultHPUGenerator(
      at::DeviceIndex device_index = -1) const override;
  at::Device getDeviceFromPtr(void* data) const override;
  bool isPinnedPtr(const void* data) const override;
  at::Allocator* getPinnedMemoryAllocator() const override;
  bool hasPrimaryContext(at::DeviceIndex device_index) const override;
};
} // namespace habana
