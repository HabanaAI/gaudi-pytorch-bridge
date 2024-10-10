/*******************************************************************************
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
#pragma once

#include "habana_kernels/lazy_kernels.h"
#include "habana_kernels/template_helpers.h"
#include "hpu_ops/hpu_op_helper.h"

namespace habana {

inline size_t CopyTensors(
    at::TensorList tl,
    std::vector<at::Tensor>& tensors_copy) {
  std::copy(tl.begin(), tl.end(), std::back_inserter(tensors_copy));
  return tl.size();
}

inline size_t CopyTensors(
    c10::ArrayRef<at::TensorList> array,
    std::vector<at::Tensor>& tensors_copy) {
  size_t count = 0;
  for (auto&& v : array) {
    count += CopyTensors(v, tensors_copy);
  }
  return count;
}

template <class T, class U>
void runInplaceMaybeWithAccThread(const char* op, T&& lazy_op, U result) {
  if (habana_lazy::AccThread::Get().CanUseAccThread()) {
    PT_LAZY_PARALLEL_ACC_DEBUG("Running ", op, " in accumulation thread");
    std::vector<at::Tensor> tensors_copy;
    auto count = CopyTensors(result, tensors_copy);
    scheduleAccTask(std::move(lazy_op), std::move(tensors_copy));
    MAYBE_FLUSH_OP(count);
  } else {
    return lazy_op.call(result);
  }
}

} // namespace habana
