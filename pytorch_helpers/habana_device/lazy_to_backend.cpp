/******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
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

#include "backend/lazy_to_backend.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/lazy_executor.h"

bool lazy_to_backend::is_const_tensor(const at::Tensor& tensor) {
  const auto& hb_tensor = habana_lazy::GetHbInternalTensorImpl(tensor);
  return hb_tensor->IsConstTensor();
}

void* lazy_to_backend::host_ptr_for_const_tensor(const at::Tensor& tensor) {
  const auto& hb_tensor = habana_lazy::GetHbInternalTensorImpl(tensor);
  return hb_tensor->get_host_ptr();
}

std::tuple<synapse_helpers::layouts::MemoryPermutation, bool> lazy_to_backend::
    get_memory_permutation(const at::Tensor& tensor) {
  // It should be handled in SW-122018
  if (GET_ENV_FLAG_NEW(PT_HPU_EAGER_OPS)) {
    PT_EAGER_DEBUG(
        "Skipping permutations for EagerOp with duplicate inputs...");
    return {synapse_helpers::layouts::MemoryPermutation{}, false};
  }
  auto hb_weight_impl = habana_lazy::GetHbInternalTensorImpl(tensor);
  if (hb_weight_impl)
    return {
        hb_weight_impl->GetMemoryPermutation(),
        hb_weight_impl->GetDontAllowPermutation()};
  return {synapse_helpers::layouts::MemoryPermutation{}, false};
}

bool lazy_to_backend::is_lazy_inference_call_context() {
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    if (!habana_lazy::isDeviceInLoweringMode()) {
      // Lazy mode shape inference call, early return without execution
      return true;
    }
  }
  return false;
}
