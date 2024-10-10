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

#include "eager_tensor.h"
#include "backend/backend_meta.h"
#include "backend/synapse_helpers/env_flags.h"
#include "pytorch_helpers/habana_helpers/pt_version_check.h"

namespace habana {
namespace eager {

HbEagerTensorPool::HbEagerTensorPool() {
  extend_empty_tensor_pool();
}

void HbEagerTensorPool::extend_empty_tensor_pool() {
  handle_ = std::async(std::launch::async, [this]() {
    for (size_t i = 0; i < pool_size_; ++i)
      tensor_pool_other_.push_front(at::empty({}, c10::nullopt));
  });
}

at::Tensor HbEagerTensorPool::get_tensor() {
  if (tensor_pool_.empty()) {
    handle_.wait();
    std::swap(tensor_pool_, tensor_pool_other_);
    extend_empty_tensor_pool();
  }

  auto t = tensor_pool_.back();
  tensor_pool_.pop_back();
  return t;
}

/** - Note on time:
 *  TBD: Measure the cost for shallow_copy_from(), possibly optimize by copying
 *  only subset of what it copies.
 */
at::Tensor HbEagerTensorPool::get_backend_tensor(
    const at::Tensor& frontend_tensor) {
  std::chrono::steady_clock::time_point t_start;
  const bool take_timestamp =
      GET_ENV_FLAG_NEW(PT_HPU_ENABLE_EAGER_TENSOR_TIMESTAMP);
  if (take_timestamp) {
    t_start = std::chrono::steady_clock::now();
  }
  auto backend_tensor = getInstance().get_tensor();
  // get extra meta to force allocation of BackendMetadata
  // to ensure it is shared between FE and BE tensor
  get_tensor_extra_meta(frontend_tensor);
  HABANA_ASSERT(
      backend_tensor.defined(), "Undefined eager pool backend tensor");
  // Shallow copy from frontend_tensor. Updates the TensorImpl metadata
  // (size/stride/...) and increases the refcount by pointing to the same
  // storageImpl.

  auto const_id = INVALID_CONST_ID;
  auto is_frontend_tensor_const = habana::is_tensor_const(frontend_tensor);
  if (is_frontend_tensor_const) {
    const_id = habana::get_tensor_const_id(frontend_tensor);
  }

  backend_tensor.unsafeGetTensorImpl()->shallow_copy_from(
      frontend_tensor.getIntrusivePtr());

  if (!habana::is_tensor_const(backend_tensor) and is_frontend_tensor_const) {
    habana::set_tensor_const(
        backend_tensor, is_frontend_tensor_const, const_id);
  }

  HABANA_ASSERT(
      backend_tensor.unsafeGetTensorImpl() !=
          frontend_tensor.unsafeGetTensorImpl(),
      "HbEagerTensorPool::get_backend_tensor backend and frontend tensor TensorImpl needs "
      "to be different.");
  HABANA_ASSERT(
      backend_tensor.is_alias_of(frontend_tensor),
      "HbEagerTensorPool::get_backend_tensor backend and frontend tensor must share same "
      "storage.");
  HABANA_ASSERT(
      !backend_tensor.unsafeGetTensorImpl()->pyobj_slot()->owns_pyobj(),
      "HbEagerTensorPool::get_backend_tensor backend tensor shouldn't own pyobj");
  if (take_timestamp) {
    auto t_end = std::chrono::steady_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start)
            .count();
    // TBD: Move this to another interface that reports aggregated average
    PT_BRIDGE_DEBUG(
        "Time for HbEagerTensorPool::get_backend_tensor = ", duration);
  }
  return backend_tensor;
}

} // namespace eager
} // namespace habana
