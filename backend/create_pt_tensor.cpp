/******************************************************************************
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
#include "backend/create_pt_tensor.h"
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_strided.h>
#include <c10/core/ScalarTypeToTypeMeta.h>
#include <c10/core/TensorImpl.h>
#include "habana_helpers/logging_pt.h"

namespace habana {
StorageLessWrapperTensorImpl::StorageLessWrapperTensorImpl(
    const at::Tensor& rep,
    at::optional<caffe2::TypeMeta> data_type)
    : TensorImpl(
          c10::DispatchKeySet(c10::DispatchKey::HPU),
          data_type.has_value() ? data_type.value() : rep.dtype(),
          rep.device()) {}

StorageLessWrapperTensorImpl::StorageLessWrapperTensorImpl(
    at::optional<caffe2::TypeMeta> data_type)
    : TensorImpl(
          c10::DispatchKeySet(c10::DispatchKey::HPU),
          data_type.value(),
          at::kHPU) {}

void StorageLessWrapperTensorImpl::release_resources() {}

bool StorageLessWrapperTensorImpl::has_storage() const {
  return false;
}

const at::Storage& StorageLessWrapperTensorImpl::storage() const {
  TORCH_CHECK(0, "StorageLessWrapperTensorImpl tensors do not have storage");
}

static bool alwaysAllocOnDevice() {
  static bool allocOnDevice = GET_ENV_FLAG_NEW(HABANA_USE_PERSISTENT_TENSOR);
  return allocOnDevice;
}

} // namespace habana

at::Tensor habana::nonPersistentTensor(
    const at::Tensor& input,
    at::IntArrayRef size,
    const at::TensorOptions& options,
    at::optional<c10::MemoryFormat> optional_memory_format,
    at::optional<caffe2::TypeMeta> data_type) {
  static_cast<void>(options);
  auto t = at::detail::make_tensor<habana::StorageLessWrapperTensorImpl>(
      input, data_type);
  t.unsafeGetTensorImpl()->set_sizes_contiguous(size);

  if (optional_memory_format.has_value()) {
    t.unsafeGetTensorImpl()->empty_tensor_restride(
        optional_memory_format.value_or(c10::MemoryFormat::Contiguous));
  } else {
    auto memory_format = input.options().memory_format_opt().value_or(
        c10::MemoryFormat::Contiguous);
    t.unsafeGetTensorImpl()->empty_tensor_restride(memory_format);
  }

  PT_SYNHELPER_DEBUG("Allocating non persistent tensor: size = ", size);
  return t;
}

at::Tensor habana::nonPersistentTensor(
    at::IntArrayRef size,
    at::IntArrayRef strides,
    at::optional<c10::MemoryFormat> optional_memory_format,
    at::optional<caffe2::TypeMeta> data_type) {
  auto t =
      at::detail::make_tensor<habana::StorageLessWrapperTensorImpl>(data_type);
  t.unsafeGetTensorImpl()->set_sizes_and_strides(size, strides);
  t.unsafeGetTensorImpl()->empty_tensor_restride(
      optional_memory_format.value_or(c10::MemoryFormat::Contiguous));
  PT_SYNHELPER_DEBUG("Allocating non persistent tensor: size = ", size);
  return t;
}

at::Tensor habana::nonPersistentTensor(
    const at::Tensor& input,
    at::IntArrayRef size,
    at::IntArrayRef strides,
    const at::TensorOptions& options,
    at::optional<c10::MemoryFormat> optional_memory_format,
    at::optional<caffe2::TypeMeta> data_type) {
  auto t = at::detail::make_tensor<habana::StorageLessWrapperTensorImpl>(
      input, data_type);
  t.unsafeGetTensorImpl()->set_sizes_and_strides(size, strides);

  at::MemoryFormat memory_format = at::MemoryFormat::Contiguous;
  if (optional_memory_format.has_value()) {
    switch (*optional_memory_format) {
      case at::MemoryFormat::ChannelsLast:
      case at::MemoryFormat::ChannelsLast3d:
        memory_format = *optional_memory_format;
        break;
      default:
        memory_format =
            options.memory_format_opt().value_or(at::MemoryFormat::Contiguous);
        break;
    }
  }
  t.unsafeGetTensorImpl()->empty_tensor_restride(memory_format);

  PT_SYNHELPER_DEBUG("Allocating non persistent tensor: size = ", size);
  return t;
}

at::Tensor habana::createPTTensor(const at::Tensor& input, bool is_persistent) {
  at::Tensor t;
  if (is_persistent || alwaysAllocOnDevice()) {
    t = at::empty(
        input.sizes(), input.options(), input.suggest_memory_format());
  } else {
    t = habana::nonPersistentTensor(
        input, input.sizes(), input.options(), input.suggest_memory_format());
  }

  return t;
}

at::Tensor habana::createPTTensor(
    const at::Tensor& input,
    at::IntArrayRef size,
    const at::TensorOptions& options,
    bool is_persistent) {
  at::Tensor t;
  if (is_persistent || alwaysAllocOnDevice()) {
    t = at::empty(size, options, input.suggest_memory_format());
  } else {
    t = habana::nonPersistentTensor(
        input,
        size,
        options,
        input.suggest_memory_format(),
        options.dtype_opt());
  }

  return t;
}

at::Tensor habana::createPTTensor(
    const at::Tensor& input,
    at::IntArrayRef size,
    const at::TensorOptions& options,
    at::optional<c10::MemoryFormat> optional_memory_format,
    bool is_persistent) {
  at::Tensor t;
  if (is_persistent || alwaysAllocOnDevice()) {
    t = at::empty(
        size,
        options,
        optional_memory_format.value_or(c10::MemoryFormat::Contiguous));
  } else {
    t = habana::nonPersistentTensor(
        input,
        size,
        options,
        optional_memory_format.value_or(c10::MemoryFormat::Contiguous));
  }

  return t;
}

at::Tensor habana::createPTTensor(
    const at::Tensor& input,
    at::IntArrayRef size,
    at::IntArrayRef strides,
    const at::TensorOptions& options,
    at::optional<c10::MemoryFormat> optional_memory_format,
    bool is_persistent) {
  at::Tensor t;
  if (is_persistent || alwaysAllocOnDevice()) {
    t = at::empty_strided(size, strides, options);
  } else {
    t = habana::nonPersistentTensor(
        input,
        size,
        strides,
        options,
        optional_memory_format.value_or(c10::MemoryFormat::Contiguous));
  }

  return t;
}

at::Tensor habana::createPTTensor(
    const at::Tensor& input,
    at::IntArrayRef size,
    const at::TensorOptions& options,
    at::optional<c10::MemoryFormat> optional_memory_format,
    c10::ScalarType data_type,
    bool is_persistent) {
  at::Tensor t;
  HABANA_ASSERT(c10::ScalarType::Undefined != data_type, "undefined dtype");
  if (is_persistent || alwaysAllocOnDevice()) {
    t = at::empty(
        size,
        input.options().dtype(data_type),
        optional_memory_format.value_or(c10::MemoryFormat::Contiguous));
  } else {
    t = habana::nonPersistentTensor(
        input,
        size,
        options,
        optional_memory_format.value_or(c10::MemoryFormat::Contiguous),
        scalarTypeToTypeMeta(data_type));
  }

  return t;
}
