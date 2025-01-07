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
#include <c10/core/TensorImpl.h>
#include "backend/helpers/get_n_bytes.h"
#include "hpu_lazy_tensors.h"
#include "tensor_impl.h"
#pragma once
namespace habana_lazy {

// Checks whether a c10::optional<Tensor> is defined.
inline bool IsDefined(const c10::optional<at::Tensor>& tensor) {
  return tensor.has_value() && tensor.value().defined();
}

// Creates an ATen tensor with HbLazy type id from an HbLazyTensor.
at::Tensor AtenFromHbLazyTensor(
    HbLazyTensor&& HbLazy_tensor,
    c10::optional<synTensorType> tensor_type,
    c10::optional<c10::IntArrayRef> size,
    c10::optional<c10::IntArrayRef> stride,
    c10::optional<c10::MemoryFormat> mem_format);
at::Tensor AtenFromHbLazyTensor(
    const HbLazyTensor& HbLazy_tensor,
    c10::optional<synTensorType> tensor_type,
    c10::optional<c10::IntArrayRef> size,
    c10::optional<c10::IntArrayRef> stride,
    c10::optional<c10::MemoryFormat> mem_format);
at::Tensor AtenFromHbLazyTensor(
    HbLazyTensor&& HbLazy_tensor,
    const c10::Storage& lazy_storage,
    c10::DispatchKeySet key_set,
    c10::optional<synTensorType> tensor_type,
    c10::optional<c10::IntArrayRef> size,
    c10::optional<c10::IntArrayRef> stride,
    c10::optional<c10::MemoryFormat> mem_format);
at::Tensor AtenInternalHbTensor(
    c10::Storage&& storage,
    const caffe2::TypeMeta& data_type,
    c10::optional<synTensorType> tensor_type,
    c10::optional<c10::IntArrayRef> size,
    c10::optional<c10::IntArrayRef> stride,
    c10::optional<c10::MemoryFormat> mem_format);

HbLazyTensorImpl* GetHbLazyTensorImpl(const at::Tensor& tensor);

HbInternalTensorImpl* GetHbInternalTensorImpl(const at::Tensor& tensor);

// If tensor is an HbLazy tensor type, returns the HbLazyTensor embedded within
// it, otherwise creates a new HbLazy tensor type with tensor as data.
HbLazyTensor GetOrCreateHbLazyTensor(
    const at::Tensor& tensor,
    const c10::Device& device = c10::kHPU);

HbLazyTensor GetOrCreateHbLazyTensor(
    const c10::optional<at::Tensor>& tensor,
    const c10::Device& device);

// Extracts the HbLazyTensor out of our version of at::Tensor. Throws an
// exception if tensor is not an HbLazy tensor.
HbLazyTensor GetHbLazyTensor(
    const at::Tensor& tensor,
    bool get_updated = true,
    bool handle_collective = true);

HbLazyTensor SyncAndGetHbLazyTensor(
    const at::Tensor& tensor,
    bool get_updated = true,
    bool handle_collective = true);

int64_t GetHbLazyTensorId(
    const at::Tensor& tensor,
    bool get_updated = true,
    bool handle_collective = true);

c10::optional<HbLazyTensor> TryGetHbLazyTensor(
    const at::Tensor& tensor,
    bool get_updated = true,
    bool handle_collective = true,
    bool is_size_strides_update = true);

void MarkTensorAsOutputFromCollectiveOp(const at::Tensor& tensor);

bool IsHbLazyTensor(const at::Tensor& tensor);

ir::Value GetIrValueForNone();

ir::Value GetIrValueForScalar(const c10::Scalar& scalar);

at::Tensor CreateHbLazyTensor(
    at::Tensor tensor,
    const c10::optional<at::Device>& device);

ir::Value GetIrValueForListConstruct(
    const ir::ValueList& values,
    bool optional = false);

void* GetLazyTensorDataPtr(const at::Tensor& t);
} // namespace habana_lazy
