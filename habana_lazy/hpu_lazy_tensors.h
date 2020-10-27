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
#include <absl/types/span.h>
#include <c10/core/Device.h>
#include <torch/csrc/jit/ir/ir.h>
#include <unordered_set>
#include "ir.h"
#include "ir_utils.h"

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
        original_element_type(tensor_data.scalar_type()),
        unique_id(GetNextTensorId()) {}
  Data(const c10::Device& device)
      : data_ptr(nullptr),
        device(c10::Device(c10::DeviceType::HABANA, 0)),
        unique_id(GetNextTensorId()) {}
  Data(
      ir::Value ir_value,
      const at::Device& device,
      c10::optional<at::ScalarType> logical_element_type)
      : data_ptr(nullptr),
        ir_value(std::move(ir_value)),
        device(device),
        logical_element_type(logical_element_type),
        original_element_type(logical_element_type.value()),
        unique_id(GetNextTensorId()) {}
  ~Data(){};
  int64_t GetNextTensorId() {
    static std::atomic<int64_t>* id_generator = new std::atomic<int64_t>(1);
    return id_generator->fetch_add(1);
  }
  void* data_ptr;
  ir::Value ir_value;
  LayoutFormat tensor_layout;
  c10::Device device;
  c10::optional<at::ScalarType> logical_element_type;
  c10::optional<at::Tensor> tensor_data;
  at::ScalarType original_element_type;
  const int64_t unique_id = 0;
  std::vector<int64_t> sizes;
};

struct PostOrderData {
  ir::NodePtrList post_order;
  ir::Utils::EmissionMap emission_map;
  ir::ValueList inputs;
  ir::ValueList outputs;
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
      ir::Value ir_value,
      const at::Device& device,
      c10::optional<at::ScalarType> logical_element_type);
  // Creates an empty/null tensor.
  HbLazyTensor() = default;
  HbLazyTensor(const at::Tensor& tensor, const c10::Device& device);
  HbLazyTensor(const c10::Device& device);
  HbLazyTensor(
      ir::Value ir_value,
      const at::Device& device,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);
  HbLazyTensor(std::shared_ptr<Data> data);
  at::Tensor ToTensor(bool detached);
  bool is_null() const {
    return data_ptr() == nullptr;
  }
  // int size(int dim) const;
  void SetTensor(at::Tensor tensor);
  void setTensorSize(std::vector<int64_t> sizes);
  // Sets up a pointer from IR in data ptr back to data ptr
  // its cyclic in nature, being managed by weak pointer in IR
  void setPtrDataIrToData();
  ir::Value createIrValueFromData();
  void SetTensorData(at::Tensor tensor_data);
  void AssignIrValue(ir::Value ir_value) const;
  ir::Value GetIrValueForTensor(
      const at::Tensor& tensor,
      const c10::Device& device) const;
  c10::ScalarType dtype() const;
  c10::optional<c10::ScalarType> dtype_optional() const;
  // Set logical_element_type which is visible to upstream PyTorch.
  void SetScalarType(c10::optional<c10::ScalarType> logical_element_type);
  const c10::Device& GetDevice() const;
  // Retrieves the current IR Node, or nullptr in case no active IR Node is
  // available.
  ir::Value& CurrentIrValue() const;
  ir::Value GetIrValue() const;
  c10::optional<at::Tensor> CurrentTensorData() const;
  void setTensorOriginalType(c10::ScalarType type);
  c10::ScalarType getTensorOriginalType();
  void* CurrentHabanaData() const;
  // Applies the queue of operations in preparation for using the data.
  void applyPendingGraph();
  c10::optional<at::Tensor> GetHbLazyTensorData();
  static void MarkStep(const c10::Device& device);
  // Retrieves the PyTorch CPU tensors behind the Habana Lazy tensors IR
  // operations. All the tensors must be on the same device.
  // static std::vector<at::Tensor> GetTensors(std::vector<HbLazyTensor>*
  // tensors);
  static HbLazyTensor CreateHbLazyTensor(
      c10::IntArrayRef size,
      at::Scalar fill_value,
      const at::Device& device,
      at::ScalarType scalar_type);
  ir::Value CreateTensorNode(void* data, bool read_only) const;
  static std::vector<int> CollectSyncTensors(
      const std::vector<HbLazyTensor>& tensors);
  static PostOrderData RunPostOrder(
      const std::vector<HbLazyTensor>& tensors,
      std::vector<int> indices);

  // Retrieves the set of tensors which are currently live in the system,
  // for the given device. If device is nullptr, the live tensors for all
  // devices will be returned.
  static std::vector<HbLazyTensor> GetLiveTensors(const c10::Device* device);

  static void SyncTensorsGraph(
      std::vector<HbLazyTensor>* tensors,
      absl::Span<const std::string> devices);

  static void SyncLiveTensorsGraph(
      const c10::Device* device,
      absl::Span<const std::string> devices);

 private:
  Data* data() const;
  std::shared_ptr<Data> data_ptr() const {
    return mp_data;
  }
  std::shared_ptr<Data> mp_data;

  static void SyncTensorsGraphInternal(
      std::vector<HbLazyTensor>* tensors,
      absl::Span<const std::string> devices);
};

// The HbContextArena holds per device live information and statistics,
// among which the Habana tensors which are currently alive in the system.
// This is used to create computation checkpoints in order to flush pending
// operations and ensure the same computations are created during the
// training loops.
struct HbContext {
  std::unordered_map<int, std::weak_ptr<Data>> tensors_data;
  ir::Value seed_ir_value;
};

class HbContextArena {
 public:
  static HbContextArena* Get();
  void RegisterTensor(std::shared_ptr<Data> data);
  void UnregisterTensor(Data* data);
  std::vector<HbLazyTensor> GetLiveTensors(const c10::Device* device);
  void MarkStep(const c10::Device& device);

 private:
  std::vector<HbContext*> GetAllHbContexts();
  void ForAllHbContexts(
      const std::function<void(HbContext*)>& fn,
      const c10::Device* device);
  HbContext* GetHbContext(const c10::Device& device);
  std::unordered_map<c10::Device, HbContext*> mp_device_contexts;
};

} // namespace habana_lazy
