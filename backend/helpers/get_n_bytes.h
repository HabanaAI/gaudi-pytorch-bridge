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
