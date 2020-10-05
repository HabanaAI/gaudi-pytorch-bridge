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
#include <ATen/Tensor.h>
#include <torch/csrc/jit/ir/ir.h>
#include "habana_helpers/tensor_utils.h"

using namespace habana_lazy;

HbContextArena* HbContextArena::Get() {
  static HbContextArena* arena = new HbContextArena();
  return arena;
};

void HbContextArena::RegisterTensor(std::shared_ptr<Data> data) {
  HbContext* devctx = GetHbContext(data->device);
  devctx->tensors_data.emplace(data->unique_id, data);
}

void HbContextArena::UnregisterTensor(Data* data) {
  HbContext* devctx = GetHbContext(data->device);
  devctx->tensors_data.erase(data->unique_id);
}

std::vector<HbLazyTensor> HbContextArena::GetLiveTensors(
    const c10::Device* device) {
  std::vector<HbLazyTensor> tensors;
  auto fn = [&](HbContext* devctx) {
    for (auto& uid_wptr : devctx->tensors_data) {
      std::shared_ptr<Data> data = uid_wptr.second.lock();
      if (data != nullptr) {
        tensors.push_back(HbLazyTensor(std::move(data)));
      }
    }
  };
  ForAllHbContexts(fn, device);
  return tensors;
}

std::vector<HbContext*> HbContextArena::GetAllHbContexts() {
  std::vector<HbContext*> all_device_contexts;
  all_device_contexts.reserve(mp_device_contexts.size());
  for (auto& device_contexts : mp_device_contexts) {
    all_device_contexts.push_back(device_contexts.second);
  }
  return all_device_contexts;
}

void HbContextArena::ForAllHbContexts(
    const std::function<void(HbContext*)>& fn,
    const c10::Device* device) {
  if (device == nullptr) {
    for (auto devctx : GetAllHbContexts()) {
      fn(devctx);
    }
  } else {
    fn(GetHbContext(*device));
  }
}

HbContext* HbContextArena::GetHbContext(const c10::Device& device) {
  auto it = mp_device_contexts.find(device);
  if (it == mp_device_contexts.end()) {
    it = mp_device_contexts.emplace(device, new HbContext()).first;
  }
  return it->second;
}

HbLazyTensor::HbLazyTensor(const at::Tensor& tensor, const c10::Device& device)
    : mp_data(std::make_shared<Data>(tensor, device)) {}

HbLazyTensor::HbLazyTensor(
    Value ir_value,
    const at::Device& device,
    c10::optional<at::ScalarType> logical_element_type)
    : mp_data(std::make_shared<Data>(
          std::move(ir_value),
          device,
          logical_element_type)) {
  // TODO : TryLimitGraphSize();
}

HbLazyTensor::HbLazyTensor(std::shared_ptr<Data> data)
    : mp_data(std::move(data)) {}

HbLazyTensor HbLazyTensor::Create(
    const at::Tensor& tensor,
    const c10::Device& device) {
  HbLazyTensor habana_tensor(tensor, device);
  HbContextArena::Get()->RegisterTensor(habana_tensor.data_ptr());
  return habana_tensor;
}

HbLazyTensor HbLazyTensor::Create(
    Value ir_value,
    const at::Device& device,
    c10::optional<at::ScalarType> logical_element_type) {
  HbLazyTensor hb_tensor(std::move(ir_value), device, logical_element_type);
  HbContextArena::Get()->RegisterTensor(hb_tensor.data_ptr());
  return hb_tensor;
}
at::Tensor CopyTensor(const at::Tensor& ref) {
  return ref.to(ref.options(), /*non_blocking=*/false, /*copy=*/true);
}

at::Tensor HbLazyTensor::ToTensor(bool detached) {
  at::Tensor tensor;
  c10::optional<at::Tensor> tensor_data = CurrentTensorData();
  if (!tensor_data) {
    // TODO:: Will need to check if we need to activate this path
    // We arent allocation any new memory to tensors which isnt coming via At
    // calls
    // so this case shouldnt arise
    return tensor;
  } else {
    tensor = *tensor_data;
    if (detached) {
      if (data()->ir_value) {
        // If we have other authoritive sources, just drop our reference and
        // transfer it to the caller.
        data()->tensor_data = c10::nullopt;
      } else {
        // Otherwise we need to make a copy to prevent the caller changing our
        // version.
        tensor = CopyTensor(tensor);
      }
    }
  }
  return tensor;
}

void HbLazyTensor::AssignIrValue(habana_lazy::Value ir_value) const {
  data()->ir_value = std::move(ir_value);
}

habana_lazy::Value HbLazyTensor::CurrentIrValue() const {
  return data()->ir_value;
}

void* HbLazyTensor::CurrentHabanaData() const {
  return data()->data_ptr;
}

habana_lazy::Value HbLazyTensor::GetIrValue() const {
  habana_lazy::Value ir_value = CurrentIrValue();
  if (ir_value) {
    return ir_value;
  }
  void* device_data = CurrentHabanaData();
  if (device_data != nullptr) {
    // In case of tensor node, we do not clear the device data when we set the
    // IR node. This because we want further calls to GetIrValue() to fetch the
    // same IR node, and not create new ones (even though the lowering context
    // will still collapse them all into a single Habana parameter op). So call
    // which wants the device data will still find it, w/out having to fetch it
    // via a computation on device
    AssignIrValue(CreateTensorNode(device_data, /*read_only=*/false));
    return data()->ir_value;
  }
  c10::optional<at::Tensor> tensor_data = CurrentTensorData();
  AssignIrValue(GetIrValueForTensor(*tensor_data, GetDevice()));
  return data()->ir_value;
}

void HbLazyTensor::SetTensorData(at::Tensor tensor_data) {
  data()->tensor_data = std::move(tensor_data);
}

c10::optional<at::Tensor> HbLazyTensor::CurrentTensorData() const {
  return data()->tensor_data;
}

const c10::Device& HbLazyTensor::GetDevice() const {
  return data()->device;
}

void HbLazyTensor::SetScalarType(
    c10::optional<at::ScalarType> logical_element_type) {
  data()->logical_element_type = logical_element_type;
}

void HbLazyTensor::SetTensor(at::Tensor tensor) {
  SetTensorData(tensor);
  AssignIrValue(habana_lazy::Value());
}

Data* HbLazyTensor::data() const {
  return mp_data.get();
}

c10::ScalarType HbLazyTensor::dtype() const {
  if (data()->tensor_data) {
    return data()->tensor_data->scalar_type();
  } else
    return c10::ScalarType::Float;
}
habana_lazy::Value HbLazyTensor::CreateTensorNode(void* data, bool read_only)
    const {
  return CurrentIrValue();
  // data->SetInfo(std::make_shared<DeviceDataInfo>(GetUniqueId(), read_only));
  // return ir::MakeNode<ir::ops::DeviceData>(std::move(data));
}

habana_lazy::Value HbLazyTensor::GetIrValueForTensor(
    const at::Tensor& tensor,
    const c10::Device& device) const {
  bool read_only = false;
  void* data = tensor.data_ptr();
  return CreateTensorNode(std::move(data), read_only);
}

HbLazyTensor HbLazyTensor::CreateHbLazyTensor(
    c10::IntArrayRef size,
    at::Scalar fill_value,
    const at::Device& device,
    at::ScalarType scalar_type) {
  habana_lazy::Value val;
  // Creating a dummy IR::Value right now
  // After Vaibhav's update, we should plug in utility to create IR
  // from metadata(commented line)
  return Create(
      // GetIrValueForScalar(fill_value, shape, device)
      val,
      device,
      scalar_type);
}