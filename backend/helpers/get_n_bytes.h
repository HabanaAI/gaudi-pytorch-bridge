/*******************************************************************************
 * Copyright (C) 2021-2023 Habana Labs, Ltd. an Intel Company
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
#include <ATen/core/Tensor.h>
#include <c10/core/TensorImpl.h>
#include "common/utils.h"
#include "habana_helpers/logging.h"
namespace habana_helpers {

inline bool is_downcast_to_int_needed(at::ScalarType dtype) {
  return !common::IsInt64Supported() && dtype == at::ScalarType::Long;
}

static inline size_t calculate_nbytes(
    size_t num_bytes,
    const caffe2::TypeMeta d_type) {
  auto s_type = c10::typeMetaToScalarType(d_type);
  if (habana_helpers::is_downcast_to_int_needed(s_type)) {
    PT_LAZY_DEBUG("GetNBytes() called for tensor with dtype 'Long'!");
    // As our allocation is for 'Int'
    num_bytes /= 2;
  }
  if (s_type == c10::ScalarType::Double) {
    PT_LAZY_DEBUG("GetNBytes() called for tensor with dtype 'Double'!");
    // As our allocation is for 'Float'
    num_bytes /= 2;
  }
  return num_bytes;
}

// Use GetNBytes instead of nbytes
inline size_t GetNBytes(c10::StorageImpl* impl, const caffe2::TypeMeta d_type) {
  size_t num_bytes = calculate_nbytes(impl->nbytes(), d_type);
  return num_bytes;
}

inline size_t GetNBytes(at::TensorImpl* impl) {
  size_t num_bytes = calculate_nbytes(impl->storage().nbytes(), impl->dtype());
  return num_bytes;
}

inline size_t GetNBytes(const at::Tensor& tensor) {
  size_t num_bytes = calculate_nbytes(tensor.nbytes(), tensor.dtype());
  return num_bytes;
}

inline size_t GetNBytes(at::Tensor& tensor) {
  size_t num_bytes = calculate_nbytes(tensor.nbytes(), tensor.dtype());
  return num_bytes;
}

} // namespace habana_helpers
