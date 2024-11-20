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

#include <torch/torch.h>
#include "backend/helpers/tensor_shape.h"

namespace torch_hpu {
class IMediaProxy {
 public:
  virtual ~IMediaProxy() = default;
  virtual uintptr_t allocatePersistentBuffer(size_t size) = 0;
  virtual void freePersistentBuffer(uintptr_t buffer) = 0;
  virtual uintptr_t allocateFrameworkDeviceOutputTensor(
      habana_helpers::TensorShape shape,
      torch::ScalarType dtype) = 0;
  virtual uintptr_t allocateFrameworkHostOutputTensor(
      habana_helpers::TensorShape shape,
      torch::ScalarType dtype) = 0;
  virtual void freeFrameworkOutputTensor(uint64_t addr) = 0;
  virtual synDeviceId getSynDeviceId() = 0;
  virtual synStreamHandle getComputeStream() = 0;
  virtual torch::Tensor getFrameworkOutputTensor(uintptr_t addr) = 0;
};
} // namespace torch_hpu