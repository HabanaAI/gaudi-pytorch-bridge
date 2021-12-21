/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "hpu_lazy_tensors.h"
#include "tensor_impl.h"
#pragma once
namespace habana_lazy {

// Checks whether a c10::optional<Tensor> is defined.
inline bool IsDefined(const c10::optional<at::Tensor>& tensor) {
  return tensor.has_value() && tensor.value().defined();
}

at::Tensor HbLazyToAtenTensor(
    HbLazyTensor HbLazy_tensor,
    const at::TensorOptions& tensor_options);

// Creates an ATen tensor with HbLazy type id from an HbLazyTensor.
at::Tensor AtenFromHbLazyTensor(HbLazyTensor HbLazy_tensor);
at::Tensor AtenInternalHbTensor(
    c10::Storage&& storage,
    const caffe2::TypeMeta& data_type);

HbLazyTensorImpl* GetHbLazyTensorImpl(const at::Tensor& tensor);

HbInternalTensorImpl* GetHbInternalTensorImpl(const at::Tensor& tensor);

// If tensor is an HbLazy tensor type, returns the HbLazyTensor embedded within
// it, otherwise creates a new HbLazy tensor type with tensor as data.
HbLazyTensor GetOrCreateHbLazyTensor(
    const at::Tensor& tensor,
    const c10::Device& device = c10::kHPU);

void setTensorAsInputNode(HbLazyTensor hl_tensor);

HbLazyTensor GetOrCreateHbLazyTensor(
    const c10::optional<at::Tensor>& tensor,
    const c10::Device& device);

// Extracts the HbLazyTensor out of our version of at::Tensor. Throws an
// exception if tensor is not an HbLazy tensor.
HbLazyTensor GetHbLazyTensor(const at::Tensor& tensor);

c10::optional<HbLazyTensor> TryGetHbLazyTensor(const at::Tensor& tensor);

bool IsHbLazyTensor(const at::Tensor& tensor);

ir::Value GetIrValueForNone();

ir::Value GetIrValueForScalar(const c10::Scalar& scalar);
at::Tensor CreateHbLazyTensor(
    at::Tensor tensor,
    const c10::optional<at::Device>& device);
c10::optional<at::Device> GetHblazyDevice(const at::Tensor& tensor);

ir::Value GetIrValueForListConstruct(
    const ir::ValueList& values,
    bool optional = false);

std::vector<at::Tensor> HpuGetFallbackTensorList(
    const std::vector<at::Tensor>& tensors);
void HpuGatherLazyFallbackTensorList(
    const std::vector<at::Tensor>& tensors,
    std::vector<HbLazyTensor>& tensors_to_execute);
void HpuGatherLazyFallbackOptTensorList(
    const std::vector<c10::optional<at::Tensor>>& tensors,
    std::vector<HbLazyTensor>& tensors_to_execute);
const std::vector<c10::optional<at::Tensor>> HpuGetFallbackOptTensorList(
    const std::vector<c10::optional<at::Tensor>>& tensors);
c10::List<c10::optional<at::Tensor>> HpuGetFallbackOptTensorList(
    const c10::List<c10::optional<at::Tensor>>& tensors);

at::Tensor CreateHpuTensor(
    const at::Tensor& tensor,
    const c10::optional<c10::Device>& device);
std::vector<at::Tensor> CreateHpuTensors(
    const std::vector<at::Tensor>& tensors,
    const c10::optional<c10::Device>& device);

void HpuUpdateTensors(
    std::vector<at::Tensor>& dst_tensors,
    std::vector<at::Tensor>& src_tensors,
    const std::vector<size_t>& indices);

c10::optional<c10::Device> GetHpuDevice(const at::Tensor& tensor);
c10::optional<c10::Device> GetHpuDevice(
    const c10::optional<at::Tensor>& tensor);
c10::optional<c10::Device> GetHpuDevice(const at::TensorList& tensors);
c10::optional<c10::Device> GetHpuDevice(
    const at::TensorOptions& tensor_options);
c10::optional<c10::Device> GetHpuDevice(const c10::Device& device);
c10::optional<c10::Device> GetHpuDevice(
    const c10::optional<c10::Device>& device);
void* GetLazyTensorDataPtr(const at::Tensor& t);
} // namespace habana_lazy
