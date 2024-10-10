/*******************************************************************************
 * Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
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
#include <ATen/Tensor.h>

#include "pytorch_helpers/habana_helpers/logging.h"

#include <atomic>
#include <future>

namespace habana {

namespace eager {

/**
 * @brief
 * Habana eager (non-lazy path) tensor implementation.
 *
 * This class provides a backend tensor for the lowering flow in
 * non-lazy eager. The non-lazy eager kernels require to send the input and
 * output tensors to the pipelined lowering flow in PT Habana bridge. The input
 * and output tensors need to be disassociated with the frontend tensors
 * available to PT framework, to avoid race conditions on tensor metadata
 * content. It also cleanly separates the backend container that is sent to
 * synapse lowering. The container needs to ensure the following -
 * - Fast creation. This is done from the main thread in the context of op
 * frontend and must take very low time to create.
 * - Must conserve the tensor metadata present at the time of op frontend call.
 * - Must ensure the tensor storage lifetime is extended till the lowering flow.
 * - TBD: Provide additional service to manage view inputs/outputs.
 *
 * Detailed description of the design is in
 * https://confluence.habana-labs.com/display/Frameworks/Eager+mode#Eagermode-ChangeRequiredforPT2.0
 *
 */

/**
 * @brief
 * HbEagerTensorPool: This is the backend tensor pool. This is currently based
 * on the following logic -
 * - Purpose:
 *   Create a pool of tensors, and use them as backend tensor during lowering of
 *   eager ops
 *   Note: at::Tensor is based on UndefinedTensorImpl, and doing a
 *         shallow copy on them isn't good enough as the tensor still remains
 * defined() = false. Tensors ops [example:stride()] on UndefinedTensorImpl
 * causes an error. Given this, we create at::empty tensors in the pool that are
 * 1 element tensors on the host side and a shallow copy on these TensorImpl
 * serves the purpose.
 * - Lifetime:
 *   Creation: During pool initialization
 *   Use as backend tensor: shallow copy from src, which results in extra
 *    refcount on the storageImpl Backend tensor lifetime: Backend tensor
 * remains alive till the execution of the op is done in the device. At the
 * point, the event sync after the op recipe execution in device is triggered,
 * and as part of the callback from the event, the backend tensor is released.
 * - Pool management:
 *   The pool is created with an initial number of at::empty({}) tensors.
 * There are two pools of tensors, one created in background by a separate
 * thread. This main thread switches to the already filled up pool and triggers
 * the refilling of the alternate pool in the other thread.
 */
class HbEagerTensorPool {
 public:
  static HbEagerTensorPool& getInstance() {
    static thread_local HbEagerTensorPool instance;
    return instance;
  }

 public:
  static at::Tensor get_backend_tensor(const at::Tensor& frontend_tensor);

 private:
  HbEagerTensorPool();
  ~HbEagerTensorPool() = default;
  HbEagerTensorPool(const HbEagerTensorPool&) = delete;
  HbEagerTensorPool& operator=(const HbEagerTensorPool&) = delete;

  void extend_empty_tensor_pool();
  at::Tensor get_tensor();

 private:
  std::deque<at::Tensor> tensor_pool_;
  std::deque<at::Tensor> tensor_pool_other_;
  std::future<void> handle_;

  const size_t pool_size_{GET_ENV_FLAG_NEW(PT_HPU_EAGER_TENSOR_POOL_SIZE)};
};

} // namespace eager
} // namespace habana
