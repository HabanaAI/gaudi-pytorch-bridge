/******************************************************************************
 * Copyright (C) 2021 Habana Labs, Ltd. an Intel Company
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

// #include "absl/container/flat_hash_map.h"
#include <mutex>
#include "backend/habana_device/hpu_cached_devices.h"
#include "backend/synapse_helpers/device_types.h"
#include "imedia_proxy.h"

namespace torch_hpu {
class PytMediaProxy final : public IMediaProxy {
 public:
  explicit PytMediaProxy(int device_id);
  ~PytMediaProxy() override;

  uintptr_t allocatePersistentBuffer(size_t size) override;
  void freePersistentBuffer(uintptr_t buffer) override;
  uintptr_t allocateFrameworkHostOutputTensor(
      habana_helpers::TensorShape shape,
      torch::ScalarType dtype) override;
  uintptr_t allocateFrameworkDeviceOutputTensor(
      habana_helpers::TensorShape shape,
      torch::ScalarType dtype) override;
  void freeFrameworkOutputTensor(uint64_t addr) override;
  synDeviceId getSynDeviceId() override;
  synStreamHandle getComputeStream() override;
  torch::Tensor getFrameworkOutputTensor(uintptr_t addr) override;

 private:
  std::mutex m_mutex;
  std::unordered_map<uintptr_t, void*> buffer_to_address_;
  std::unordered_map<uintptr_t, torch::Tensor> buffer_to_output_tensor_;
  int device_id_;
};
} // namespace torch_hpu
