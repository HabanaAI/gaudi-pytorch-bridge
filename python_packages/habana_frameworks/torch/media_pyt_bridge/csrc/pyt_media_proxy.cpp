/******************************************************************************
 * Copyright (C) 2021 Habana Labs, Ltd. an Intel Company
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
#include "pyt_media_proxy.h"
#include <torch/torch.h>
#include "backend/habana_device/HPUStream.h"
#include "backend/habana_device/hpu_cached_devices.h"
#include "habana_helpers/logging.h"
#include "pytorch_helpers/lazy_to_backend.cpp"

namespace torch_hpu {

PytMediaProxy::PytMediaProxy(int device_id) : device_id_(device_id) {}

PytMediaProxy::~PytMediaProxy() {
  if (!habana::HPURegistrar::get_hpu_registrar().is_initialized()) {
    return; // Nothing to do
  }
  auto& device = habana::HPURegistrar::get_device(device_id_);
  // print unrelease memory details
  for (auto elem : buffer_to_address_) {
    PT_BRIDGE_WARN("Unreleased buffer found, address = ", elem.first);
    device.get_device_memory().free(elem.second);
  }
  for (auto elem : buffer_to_output_tensor_) {
    PT_BRIDGE_WARN("Unreleased tensor found, address = ", elem.first);
  }
}

uintptr_t PytMediaProxy::allocatePersistentBuffer(size_t size) {
  PT_BRIDGE_DEBUG("allocatePersistentBuffer size = ", size);
  void* address{nullptr};
  auto& device = habana::HPURegistrar::get_device(device_id_);
  device.get_device_memory().malloc(&address, size);
  auto real_address = reinterpret_cast<uintptr_t>(
      device.syn_device().get_fixed_address(address));
  std::unique_lock<std::mutex> lock(m_mutex);
  auto iterator_emplaced_pair =
      buffer_to_address_.emplace(real_address, address);
  HABANA_ASSERT(iterator_emplaced_pair.second);
  return real_address;
}

void PytMediaProxy::freePersistentBuffer(uintptr_t real_address) {
  std::unique_lock<std::mutex> lock(m_mutex);
  PT_BRIDGE_DEBUG("freePersistentBuffer addr = ", std::hex, real_address);
  auto it = buffer_to_address_.find(real_address);
  HABANA_ASSERT(it != buffer_to_address_.end());
  auto& device = habana::HPURegistrar::get_device(device_id_);
  device.get_device_memory().free(it->second);
  buffer_to_address_.erase(it);
}

uintptr_t PytMediaProxy::allocateFrameworkHostOutputTensor(
    habana_helpers::TensorShape shape,
    torch::ScalarType dtype) {
  torch::Tensor tensor = torch::empty(shape.get_dims(), dtype);
  auto tensor_data_ptr = reinterpret_cast<uintptr_t>(tensor.data_ptr());
  std::unique_lock<std::mutex> lock(m_mutex);
  auto iterator_emplaced_pair =
      buffer_to_output_tensor_.emplace(tensor_data_ptr, tensor);
  PT_BRIDGE_DEBUG(
      "allocateFrameworkHostOutputTensor addr = ", std::hex, tensor_data_ptr);
  HABANA_ASSERT(iterator_emplaced_pair.second);
  return iterator_emplaced_pair.first->first;
}

uintptr_t PytMediaProxy::allocateFrameworkDeviceOutputTensor(
    habana_helpers::TensorShape shape,
    torch::ScalarType dtype) {
  at::TensorOptions hb_options = at::TensorOptions(torch::kHPU);
  hb_options = hb_options.dtype(dtype);

  torch::Tensor tensor;
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 0) {
    tensor = torch::empty(
        shape.get_dims(), hb_options, c10::MemoryFormat::Contiguous);
  } else {
    // empty_hpu_lazy is used instead of torch::empty because from mediapipe
    // thread torch::empty may trigger marksteps. This will create some
    // inconsistency in live tensors and might lead to complications.
    tensor = habana_lazy::empty_hpu_lazy(
        shape.get_dims(), hb_options, c10::MemoryFormat::Contiguous, true);
  }

  auto& device = habana::HPURegistrar::get_device(device_id_);
  auto tensor_data_ptr = reinterpret_cast<uintptr_t>(
      device.syn_device().get_fixed_address(tensor.data_ptr()));
  HABANA_ASSERT(tensor_data_ptr != synapse_helpers::device_nullptr);
  std::unique_lock<std::mutex> lock(m_mutex);
  auto iterator_emplaced_pair =
      buffer_to_output_tensor_.emplace(tensor_data_ptr, tensor);
  PT_BRIDGE_DEBUG(
      "allocateFrameworkDeviceOutputTensor addr = ", std::hex, tensor_data_ptr);
  HABANA_ASSERT(iterator_emplaced_pair.second);
  return iterator_emplaced_pair.first->first;
}

void PytMediaProxy::freeFrameworkOutputTensor(uint64_t addr) {
  std::unique_lock<std::mutex> lock(m_mutex);
  PT_BRIDGE_DEBUG("freeFrameworkOutputTensor addr = ", std::hex, addr);
  auto it = buffer_to_output_tensor_.find(addr);
  HABANA_ASSERT(it != buffer_to_output_tensor_.end());
  auto tensor = std::move(it->second);
  buffer_to_output_tensor_.erase(addr);
}

synDeviceId PytMediaProxy::getSynDeviceId() {
  auto& device = habana::HPURegistrar::get_device(device_id_);
  return device.id();
}

synStreamHandle PytMediaProxy::getComputeStream() {
  auto& device = habana::HPURegistrar::get_device(device_id_);
  auto hpu_stream = c10::hpu::getDefaultHPUStream(device.id());
  return static_cast<synStreamHandle>(
      (void*)device.syn_device().get_stream(hpu_stream.id()));
}

torch::Tensor PytMediaProxy::getFrameworkOutputTensor(uintptr_t addr) {
  std::unique_lock<std::mutex> lock(m_mutex);
  PT_BRIDGE_DEBUG("getFrameworkOutputTensor addr = ", std::hex, addr);
  auto it = buffer_to_output_tensor_.find(addr);
  HABANA_ASSERT(it != buffer_to_output_tensor_.end());
  auto tensor = std::move(it->second);
  buffer_to_output_tensor_.erase(it);
  return tensor;
}

} // namespace torch_hpu
