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
#include "backend/synapse_helpers/env_flags.h"
#include "pytorch_helpers/habana_helpers/pt_version_check.h"

namespace habana {
namespace eager {

HbEagerTensorPool* HbEagerTensorPool::instance_ = nullptr;

at::Tensor HbEagerTensorPool::get_tensor() {
  if (tensor_pool.empty()) {
    handle.wait();
    std::swap(tensor_pool, tensor_pool_other);
    handle = std::async(
        std::launch::async, &HbEagerTensorPool::extend_empty_tensor_pool, this);
  }

  auto t = tensor_pool.back();
  tensor_pool.pop_back();
  return t;
}

/** - Note on time:
 *  TBD: Measure the cost for shallow_copy_from(), possibly optimize by copying
 *  only subset of what it copies.
 */
at::Tensor HbEagerTensorPool::get_backend_tensor(
    const at::Tensor& frontend_tensor) {
  std::chrono::steady_clock::time_point t_start;
  if (take_timestamp) {
    t_start = std::chrono::steady_clock::now();
  }
  auto backend_tensor = get_tensor();
  HABANA_ASSERT(
      backend_tensor.defined(), "Undefined eager pool backend tensor");
  // Shallow copy from frontend_tensor. Updates the TensorImpl metadata
  // (size/stride/...) and increases the refcount by pointing to the same
  // storageImpl.
  backend_tensor.unsafeGetTensorImpl()->shallow_copy_from(
      frontend_tensor.getIntrusivePtr());

  HABANA_ASSERT(
      backend_tensor.unsafeGetTensorImpl() !=
          frontend_tensor.unsafeGetTensorImpl(),
      "HbEagerTensorPool::get_backend_tensor backend and frontend tensor TensorImpl needs "
      "to be different.");
  HABANA_ASSERT(
      backend_tensor.is_alias_of(frontend_tensor),
      "HbEagerTensorPool::get_backend_tensor backend and frontend tensor must share same "
      "storage.");
#if IS_PYTORCH_OLDER_THAN(2, 0)
  HABANA_ASSERT(
      !backend_tensor.unsafeGetTensorImpl()->owns_pyobj(),
      "HbEagerTensorPool::get_backend_tensor backend tensor shouldn't own pyobj");
#else
  HABANA_ASSERT(
      !backend_tensor.unsafeGetTensorImpl()->pyobj_slot()->owns_pyobj(),
      "HbEagerTensorPool::get_backend_tensor backend tensor shouldn't own pyobj");
#endif
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

bool HbEagerTensorPool::is_view(__attribute__((unused))
                                at::Tensor& backend_tensor) {
  HABANA_ASSERT(0, "HbEagerTensorPool::is_view Unimplemented");
  return false;
}

at::Tensor HbEagerTensorPool::get_base_tensor(at::Tensor& backend_tensor) {
  HABANA_ASSERT(0, "HbEagerTensorPool::get_base_tensor Unimplemented");

  // TBD: Can we use the _base() from TensorBase? Does it rely on autograd mode
  // alone?
  HABANA_ASSERT(
      is_view(backend_tensor),
      "HbEagerTensorPool::get_base_tensor called on a non-view tensor");
  at::Tensor base;
  // The base will be a 1D tensor, that will be created on the complete storage
  // pointed to by the backend_tensor.
  return base;
}
} // namespace eager
} // namespace habana
