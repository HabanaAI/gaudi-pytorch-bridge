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
