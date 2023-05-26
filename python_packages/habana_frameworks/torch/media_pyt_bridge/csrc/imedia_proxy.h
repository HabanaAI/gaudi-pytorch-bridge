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