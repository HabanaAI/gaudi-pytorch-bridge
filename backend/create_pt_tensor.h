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
#pragma once
#include <ATen/Tensor.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/TensorOptions.h>

namespace habana {
struct StorageLessWrapperTensorImpl : public c10::TensorImpl {
  explicit StorageLessWrapperTensorImpl(
      const at::Tensor& rep,
      at::optional<caffe2::TypeMeta> data_type = c10::nullopt);

  explicit StorageLessWrapperTensorImpl(
      at::optional<caffe2::TypeMeta> data_type = c10::nullopt);
  void release_resources() override;

  bool has_storage() const override;

  const at::Storage& storage() const override;
};

at::Tensor nonPersistentTensor(
    const at::Tensor& input,
    at::IntArrayRef size,
    const at::TensorOptions& options = {},
    at::optional<c10::MemoryFormat> optional_memory_format = c10::nullopt,
    at::optional<caffe2::TypeMeta> data_type = c10::nullopt);

at::Tensor nonPersistentTensor(
    at::IntArrayRef size,
    at::IntArrayRef strides,
    at::optional<c10::MemoryFormat> optional_memory_format = c10::nullopt,
    at::optional<caffe2::TypeMeta> data_type = c10::nullopt);

at::Tensor nonPersistentTensor(
    const at::Tensor& input,
    at::IntArrayRef size,
    at::IntArrayRef strides,
    const at::TensorOptions& options = {},
    at::optional<c10::MemoryFormat> optional_memory_format = c10::nullopt,
    at::optional<caffe2::TypeMeta> data_type = c10::nullopt);

at::Tensor createPTTensor(const at::Tensor& input, bool is_persistent);

at::Tensor createPTTensor(
    const at::Tensor& input,
    at::IntArrayRef size,
    const at::TensorOptions& options,
    bool is_persistent);

at::Tensor createPTTensor(
    const at::Tensor& input,
    at::IntArrayRef size,
    const at::TensorOptions& options,
    at::optional<c10::MemoryFormat> optional_memory_format,
    bool is_persistent);

at::Tensor createPTTensor(
    const at::Tensor& input,
    at::IntArrayRef size,
    const at::TensorOptions& options,
    at::optional<c10::MemoryFormat> optional_memory_format,
    c10::ScalarType data_type,
    bool is_persistent);

at::Tensor createPTTensor(
    const at::Tensor& input,
    at::IntArrayRef size,
    at::IntArrayRef strides,
    const at::TensorOptions& options,
    at::optional<c10::MemoryFormat> optional_memory_format,
    bool is_persistent);
} // namespace habana
