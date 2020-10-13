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
// namespace habana_lazy
namespace habana_lazy {

at::Tensor HbLazyToAtenTensor(
    HbLazyTensor HbLazy_tensor,
    const at::TensorOptions& tensor_options);

// Creates an ATen tensor with HbLazy type id from an HbLazyTensor.
at::Tensor AtenFromHbLazyTensor(
    HbLazyTensor HbLazy_tensor,
    c10::Storage&& storage);
at::Tensor AtenFromHbLazyTensor(HbLazyTensor HbLazy_tensor);

HbLazyTensorImpl* GetHbLazyTensorImpl(const at::Tensor& tensor);

// If tensor is an HbLazy tensor type, returns the HbLazyTensor embedded within
// it, otherwise creates a new HbLazy tensor type with tensor as data.
HbLazyTensor GetOrCreateHbLazyTensor(
    const at::Tensor& tensor,
    const c10::Device& device);

void setTensorAsInputNode(HbLazyTensor hl_tensor);

HbLazyTensor GetOrCreateHbLazyTensor(
    const c10::optional<at::Tensor>& tensor,
    const c10::Device& device);

// Extracts the HbLazyTensor out of our version of at::Tensor. Throws an
// exception if tensor is not an HbLazy tensor.
HbLazyTensor GetHbLazyTensor(const at::Tensor& tensor);

c10::optional<HbLazyTensor> TryGetHbLazyTensor(const at::Tensor& tensor);

bool IsHbLazyTensor(const at::Tensor& tensor);

Value GetIrValueForScalar(const c10::Scalar& scalar);
at::Tensor CreateHbLazyTensor(
    at::Tensor tensor,
    const c10::optional<at::Device>& device);
c10::optional<at::Device> GetHblazyDevice(const at::Tensor& tensor);
} // namespace habana_lazy
