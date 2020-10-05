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
#include "habana_bridge/kernel/hpu_habana_launch_op_pt.h"
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
    ir::Value ir_value,
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

void HbLazyTensor::setTensorSize(std::vector<int64_t> sizes) {
  data()->sizes = sizes;
}
HbLazyTensor HbLazyTensor::Create(
    ir::Value ir_value,
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

void HbLazyTensor::AssignIrValue(ir::Value ir_value) const {
  data()->ir_value = std::move(ir_value);
}

ir::Value& HbLazyTensor::CurrentIrValue() const {
  return data()->ir_value;
}

void* HbLazyTensor::CurrentHabanaData() const {
  return data()->data_ptr;
}

ir::Value HbLazyTensor::GetIrValue() const {
  ir::Value ir_value = CurrentIrValue();
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
  AssignIrValue(ir::Value());
  setPtrDataIrToData();
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
ir::Value HbLazyTensor::CreateTensorNode(void* data, bool read_only)
    const {
  return CurrentIrValue();
  // data->SetInfo(std::make_shared<DeviceDataInfo>(GetUniqueId(), read_only));
  // return ir::MakeNode<ir::ops::DeviceData>(std::move(data));
}

void HbLazyTensor::setPtrDataIrToData() {
  if (mp_data.get())
    mp_data->ir_value.m_data_ptr = mp_data;
}

ir::Value HbLazyTensor::createIrValueFromData() {
  ir::Value v{data_ptr()};
  return v;
}

ir::Value HbLazyTensor::GetIrValueForTensor(
    const at::Tensor& tensor,
    const c10::Device& device) const {
  bool read_only = false;
  void* data = nullptr;
  // We have storageless tensors and should support creating nodes from them
  if (tensor.has_storage()) {
    data = tensor.data_ptr();
  }
  return CreateTensorNode(std::move(data), read_only);
}

HbLazyTensor HbLazyTensor::CreateHbLazyTensor(
    c10::IntArrayRef size,
    at::Scalar fill_value,
    const at::Device& device,
    at::ScalarType scalar_type) {
  ir::Value val;
  // Creating a dummy IR::Value right now
  // After Vaibhav's update, we should plug in utility to create IR
  // from metadata(commented line)
  HbLazyTensor hb_tensor = Create(
      // GetIrValueForScalar(fill_value, shape, device)
      val,
      device,
      scalar_type);

  // We keep a weak pointer in our IR back to data pointer of lazy tensor
  // This needs to be updated here or we can push it in the constructor
  // Keeping it here for now so that its not implicitly set
  hb_tensor.setPtrDataIrToData();
  // Setup the size information in the data of Lazy tensor
  hb_tensor.setTensorSize(size.vec());
  return hb_tensor;
}

/************************************************************************
 * @brief Returns indices of tensors corresponding to tensors with valid IR
 * values which feeds in to RunPostOrder
 ************************************************************************/
std::vector<int> HbLazyTensor::CollectSyncTensors(
    const std::vector<HbLazyTensor>& tensors) const {
  std::vector<int> indices = {};
  for (size_t i = 0; i < tensors.size(); ++i) {
    auto ir_value = tensors[i].CurrentIrValue();
    if (ir_value) {
      indices.push_back(i);
    }
  }
  return indices;
}

/************************************************************************
 * @brief Computes post order list of NodePtrs. This function should be
 *executed at the trigger points. The output would be consumed during the
 *conversion of lazy IR to JIT IR.
 * @param[in] tensors - vector Hb Lazy Tensors
 * @param[in] indices - vector of indices corresponding to input tensors with
 *valid IR values
 * @param[out] po_data - Post Ordered vector of tensors
 ************************************************************************/
habana_lazy::PostOrderData HbLazyTensor::RunPostOrder(
    const std::vector<HbLazyTensor>& tensors,
    std::vector<int> indices) {
  habana_lazy::PostOrderData po_data;
  std::vector<ir::NodePtr> p_roots;
  p_roots.reserve(indices.size());
  for (auto index : indices) {
    auto ir_value = tensors.at(index).CurrentIrValue();
    if (ir_value) {
      p_roots.push_back(ir_value.mp_node);
      // update output list
      po_data.outputs.push_back(ir_value);
    }
  }

  ir::Utils::ComputePostOrder(
      p_roots, &po_data.emission_map, po_data.post_order);
  ir::Utils::ComputePostOrderInputs(po_data.inputs, po_data.post_order);

  return po_data;
}

c10::optional<at::Tensor> HbLazyTensor::GetHbLazyTensorData() {
  // Generate the tensor data if its not been generated yet
  if (data()->ir_value && !CurrentTensorData()) {
    applyPendingGraph();
  }
  return data()->tensor_data;
}

void HbLazyTensor::applyPendingGraph() {
  // Ensure that the graph execution has taken place so taht the tensors
  // requested have the data required updated in them. This is usually done
  // before sync points in execution
  if (!CurrentTensorData()) {
    std::vector<HbLazyTensor> tensors({*this});
    // SyncTensorsGraph(&tensors, {}, /*wait=*/true, /*sync_xla_data=*/false);
  }
}

std::vector<HbLazyTensor> HbLazyTensor::GetLiveTensors(
    const c10::Device* device) {
  return HbContextArena::Get()->GetLiveTensors(device);
}

void HbLazyTensor::SyncTensorsGraph(
    std::vector<HbLazyTensor>* tensors,
    absl::Span<const std::string> devices) {
  SyncTensorsGraphInternal(tensors, devices);
}

void HbLazyTensor::SyncLiveTensorsGraph(
    const c10::Device* device,
    absl::Span<const std::string> devices) {
  auto tensors = GetLiveTensors(device);
  SyncTensorsGraph(&tensors, devices);
}

void HbLazyTensor::SyncTensorsGraphInternal(
    std::vector<HbLazyTensor>* tensors,
    absl::Span<const std::string> devices) {
  // TODO Get graph and stack
  // auto op = std::make_shared<HabanaLaunchOpPT>(graph, false);
  // op->run(stack);
}
