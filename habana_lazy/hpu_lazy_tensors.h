/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#pragma once
#include <ATen/Tensor.h>
#include <c10/core/Device.h>
#include <torch/csrc/jit/ir/ir.h>
#include <unordered_set>
#include "ir.h"

// TODO : Dummy IR used as placeholder, replace with actual IR and move to IR
// file
// namespace habana_lazy
namespace habana_lazy {
enum LayoutFormat { kNHWC = 0, kNCHW = 1, kHWCK = 2, kANY = 3, kINVALID = 4 };
struct Data {
  Data(at::Tensor tensor_data, const c10::Device& device)
      : data_ptr(nullptr),
        device(c10::Device(c10::DeviceType::HABANA, 0)),
        logical_element_type(tensor_data.scalar_type()),
        tensor_data(std::move(tensor_data)),
        unique_id(0) {}
  Data(
      Value ir_value,
      const at::Device& device,
      c10::optional<at::ScalarType> logical_element_type)
      : data_ptr(nullptr),
        ir_value(std::move(ir_value)),
        device(device),
        logical_element_type(logical_element_type),
        unique_id(0) {}
  ~Data(){};
  void* data_ptr;
  habana_lazy::Value ir_value;
  LayoutFormat tensor_layout;
  c10::Device device;
  c10::optional<at::ScalarType> logical_element_type;
  c10::optional<at::Tensor> tensor_data;
  const int unique_id;
};

class HbLazyTensor {
 public:
  // This is the core Lazy tensor data structure where all the tensor data is
  // held. The Habana Lazy tensor is nothing more than a shared pointer to a
  // Data object.
  static HbLazyTensor Create(
      const at::Tensor& tensor,
      const c10::Device& device);

  static HbLazyTensor Create(
      Value ir_value,
      const at::Device& device,
      c10::optional<at::ScalarType> logical_element_type);
  // Creates an empty/null tensor.
  HbLazyTensor() = default;
  HbLazyTensor(const at::Tensor& tensor, const c10::Device& device);
  HbLazyTensor(
      Value ir_value,
      const at::Device& device,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);
  HbLazyTensor(std::shared_ptr<Data> data);
  at::Tensor ToTensor(bool detached);
  bool is_null() const {
    return data_ptr() == nullptr;
  }
  // int size(int dim) const;
  void SetTensor(at::Tensor tensor);
  void SetTensorData(at::Tensor tensor_data);
  void AssignIrValue(habana_lazy::Value ir_value) const;
  habana_lazy::Value GetIrValueForTensor(
      const at::Tensor& tensor,
      const c10::Device& device) const;
  c10::ScalarType dtype() const;
  c10::optional<c10::ScalarType> dtype_optional() const;
  // Set logical_element_type which is visible to upstream PyTorch.
  void SetScalarType(c10::optional<c10::ScalarType> logical_element_type);
  const c10::Device& GetDevice() const;
  // Retrieves the current IR Node, or nullptr in case no active IR Node is
  // available.
  habana_lazy::Value CurrentIrValue() const;
  habana_lazy::Value GetIrValue() const;
  c10::optional<at::Tensor> CurrentTensorData() const;
  void* CurrentHabanaData() const;
  // Applies the queue of operations in preparation for using the data.
  // void ApplyPendingGraph();
  // static void MarkStep(const c10::Device& device);
  // Retrieves the PyTorch CPU tensors behind the Habana Lazy tensors IR
  // operations. All the tensors must be on the same device.
  // static std::vector<at::Tensor> GetTensors(std::vector<HbLazyTensor>*
  // tensors);
  Data* data() const;
  std::shared_ptr<Data> data_ptr() const {
    return mp_data;
  }
  static HbLazyTensor CreateHbLazyTensor(
      c10::IntArrayRef size,
      at::Scalar fill_value,
      const at::Device& device,
      at::ScalarType scalar_type);
  habana_lazy::Value CreateTensorNode(void* data, bool read_only) const;

 private:
  std::shared_ptr<Data> mp_data;
};

// The HbContextArena holds per device live information and statistics,
// among which the Habana tensors which are currently alive in the system. This
// is used to create computation checkpoints in order to flush pending
// operations and ensure the same computations are created during the
// training loops.
struct HbContext {
  std::unordered_map<int, std::weak_ptr<Data>> tensors_data;
  habana_lazy::Value seed_ir_value;
};

class HbContextArena {
 public:
  static HbContextArena* Get();
  void RegisterTensor(std::shared_ptr<Data> data);
  void UnregisterTensor(Data* data);
  std::vector<HbLazyTensor> GetLiveTensors(const c10::Device* device);
  void MarkStep(const c10::Device& device){};

 private:
  std::vector<HbContext*> GetAllHbContexts();
  void ForAllHbContexts(
      const std::function<void(HbContext*)>& fn,
      const c10::Device* device);
  HbContext* GetHbContext(const c10::Device& device);
  std::unordered_map<c10::Device, HbContext*> mp_device_contexts;
};

} // namespace habana_lazy
