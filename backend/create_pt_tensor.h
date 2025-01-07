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
