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
#include <chrono>
#include <future>
#include <thread>
#include <unordered_set>

#include <ATen/Tensor.h>
#include <c10/core/Device.h>
#include <torch/csrc/jit/ir/ir.h>

#include "habana_device/hpu_cached_devices.h"
#include "habana_helpers/misc_utils.h"
#include "ir.h"
#include "ir_utils.h"
#include "view.h"

enum LazyTensorExecutionStatus {
  kUN_REGISTERED = 0,
  kREGISTERED,
  kEXECUTING,
  kEXECUTION_COMPLETE,
  kINPUT
};

// TODO : Dummy IR used as placeholder, replace with actual IR and move to IR
// file
// namespace habana_lazy

namespace habana_lazy {
struct Data {
  Data(at::Tensor tensor_data, const c10::Device& device)
      : data_ptr(nullptr),
        device(c10::Device(c10::DeviceType::HPU, 0)),
        logical_element_type(tensor_data.scalar_type()),
        tensor_data(std::move(tensor_data)),
        original_element_type(tensor_data.scalar_type()),
        unique_id(GetNextTensorId()) {
    static_cast<void>(device);
  }

  Data(const c10::Device& device)
      : data_ptr(nullptr),
        device(c10::Device(c10::DeviceType::HPU, 0)),
        unique_id(GetNextTensorId()) {
    static_cast<void>(device);
  }
  Data(
      ir::Value ir_value,
      const at::Device& device,
      c10::optional<at::ScalarType> logical_element_type)
      : data_ptr(nullptr),
        ir_value(std::move(ir_value)),
        device(c10::Device(c10::DeviceType::HPU, 0)),
        logical_element_type(logical_element_type),
        original_element_type(logical_element_type.value()),
        unique_id(GetNextTensorId()) {
    static_cast<void>(device);
  }
  ~Data();
  int64_t GetNextTensorId() {
    static std::atomic<int64_t>* id_generator = new std::atomic<int64_t>(1);
    return id_generator->fetch_add(1);
  }

  void* data_ptr;
  ir::Value ir_value;
  LayoutFormat tensor_layout = kNCHW;
  c10::Device device;
  c10::optional<at::ScalarType> logical_element_type;
  c10::optional<at::Tensor> tensor_data;
  c10::optional<at::Tensor> cpu_tensor_data;
  bool sbs_live_tensor = false;
  bool sbs_compare_tensor = true;
  int sbs_tensor_version = 0;
  at::ScalarType original_element_type;
  const int64_t unique_id = 0;
  std::vector<int64_t> sizes;
  LazyTensorExecutionStatus execution_status = kUN_REGISTERED;
  ir::LazyView parent_view;
  int num_views = 0;
  // Version counter tracks the number of times we use tensor as output
  // if its zero, that means this tensor hasnt been output in any op
  // IMPORTANT : this is used per graph right now, that means for each lazy
  // graph generated it will be reset to 0. We are only tracking version for
  // that particular graph execution.
  int version = 0;
}; // namespace habana_lazy

struct HbLazyFrontEndInfoToBackend {
  void set_optimized_lazy_eager_key(const size_t key) {
    optimized_lazy_eager_key = key;
  }

  void set_lazy_op_name(const std::string& name) {
    op_name = name;
    std::replace(op_name.begin(), op_name.end(), ':', '_');
  }

  size_t get_optimized_lazy_eager_key() {
    return optimized_lazy_eager_key;
  }

  std::string get_lazy_op_name() {
    return op_name;
  }

 private:
  size_t optimized_lazy_eager_key = 0;
  std::string op_name = getHabanaLazyGraphName();
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
  void SetCPUTensorData(at::Tensor tensor_data);
  void SetSBSLiveTensorIndication(bool live);
  bool GetSBSLiveTensorIndication() const;
  void SetSBSCompareIndication(bool compare);
  bool GetSBSCompareIndication() const;
  void UpdateSBSTensorVersion();
  int GetSBSTensorVersion() const;
  const c10::optional<at::Tensor>& GetCPUTensorData() const;
  void AssignIrValue(ir::Value ir_value) const;
  c10::ScalarType dtype() const;
  c10::optional<c10::ScalarType> dtype_optional() const;
  // Set logical_element_type which is visible to upstream PyTorch.
  void SetScalarType(c10::optional<c10::ScalarType> logical_element_type);
  const c10::Device& GetDevice() const;
  const std::vector<int64_t>& GetSizes() const;
  // Retrieves the current IR Node, or nullptr in case no active IR Node is
  // available.
  ir::Value& CurrentIrValue() const;
  ir::Value GetIrValue() const;
  c10::optional<at::Tensor> CurrentTensorData() const;
  void setTensorOriginalType(c10::ScalarType type);
  c10::ScalarType getTensorOriginalType() const;
  void* CurrentHabanaData() const;
  // Applies the queue of operations in preparation for using the data.
  void applyPendingGraph();
  c10::optional<at::Tensor> GetHbLazyTensorData();

  // Static methods
  static at::Tensor Process0DTensor(std::shared_ptr<Data>& d);
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
  ir::Value CreateTensorNode() const;
  static std::vector<int> CollectSyncTensors(
      const std::vector<HbLazyTensor>& tensors);
  static ir::PostOrderData RunPostOrder(
      const std::vector<HbLazyTensor>& tensors,
      std::vector<int> indices);

  // Retrieves the set of tensors which are currently live in the system,
  // for the given device. If device is nullptr, the live tensors for all
  // devices will be returned.
  static std::vector<HbLazyTensor> GetLiveTensors(const c10::Device* device);

  static void SyncTensorsGraph(
      std::vector<HbLazyTensor>* tensors,
      std::shared_ptr<HbLazyFrontEndInfoToBackend> lazyFrontEndInfo = nullptr);
  static void SyncTensorsGraphFast(
      std::vector<HbLazyTensor>* tensors,
      std::vector<ir::Value>& input_values,
      std::shared_ptr<HbLazyFrontEndInfoToBackend> lazyFrontEndInfo = nullptr);

  static void SyncLiveTensorsGraph(
      const c10::Device* device,
      bool use_cached_graph,
      std::shared_ptr<HbLazyFrontEndInfoToBackend> lazy_front_end_info);

  static void StepMarker(
      const std::string& device_str = {},
      std::shared_ptr<HbLazyFrontEndInfoToBackend> lazy_front_end_info =
          nullptr);
  static void StepMarkerBind(const std::string& device_str = {});
  static void InitiateBucketRefinement();
  static void SetDynamicMode();

  static void RunSavedGraph(const std::string& device_str);
  static void ExecuteCachedGraph();

  static void* lazyTensorDataPtr(const at::Tensor& t);

  void ShallowCopyTo(HbLazyTensor* dest) const;

  int64_t getTensorUniqueId() const {
    if (mp_data.get()) {
      return mp_data.get()->unique_id;
    } else
      return -1;
  }

  std::shared_ptr<Data> getDataPtr() const {
    return mp_data;
  }

  // returns true if we have already created an aten tensor with storage and
  // attached
  bool isStorageAttached();
  c10::TensorImpl* getAttachedTensorImpl() const;
  c10::optional<at::Tensor> CurrentTensorAttached() const {
    return data()->tensor_data;
  }
  ir::Value GetIrValueForTensor(
      const at::Tensor& tensor,
      const c10::Device& device) const;
  void addView(ir::LazyView view) {
    // WE will support multiple views in future , but for now a single one is
    // supported
    TORCH_CHECK(
        data()->num_views == 0,
        "Trying to create a duplicate view on Lazy tensor");
    data()->parent_view = std::move(view);
    data()->num_views++;
  }
  c10::optional<ir::LazyView> getView() const {
    if (data()->num_views)
      return c10::make_optional(data()->parent_view);
    else
      return c10::nullopt;
  }
  void updateVersion() {
    data()->version++;
  }
  void resetVersionCounter() {
    data()->version = 0;
  }
  int getVersion() const {
    return data()->version;
  }
  void SetTensorLayout(LayoutFormat layout) {
    data()->tensor_layout = layout;
  }
  LayoutFormat GetTensorLayout() const {
    return data()->tensor_layout;
  }

 private:
  Data* data() const;
  std::shared_ptr<Data> mp_data;
  std::shared_ptr<Data> data_ptr() const {
    return mp_data;
  }

  void ClearAndAssignNewIrValue();
  static void SyncTensorsGraphInternal(
      std::vector<HbLazyTensor>* tensors,
      std::shared_ptr<HbLazyFrontEndInfoToBackend> lazyFrontEndInfo = nullptr);
  static void SyncTensorsGraphInternalFast(
      std::vector<HbLazyTensor>* tensors,
      std::vector<ir::Value>& input_values,
      std::shared_ptr<HbLazyFrontEndInfoToBackend> lazyFrontEndInfo = nullptr);
  static bool switch_dynamic_mode;

  // The following handle is used to keep track of refinement thread.
  // Every invocation of StepMarker first checks whether a refinement thread
  // is running and creates one only when there is no refinement thread
  // running.
  static std::future<bool> refinement_handle_;
};

// The HbContextArena holds per device live information and statistics,
// among which the Habana tensors which are currently alive in the system.
// This is used to create computation checkpoints in order to flush pending
// operations and ensure the same computations are created during the
// training loops.
struct HbContext {
  std::map<int64_t, std::weak_ptr<Data>> tensors_data;
  ir::Value seed_ir_value;
};

class HbContextArena {
 public:
  static HbContextArena* Get();
  void RegisterTensor(std::shared_ptr<Data> data);
  void UnregisterTensor(Data* data);
  std::vector<HbLazyTensor> GetLiveTensors(const c10::Device* device);
  void MarkStep(const c10::Device& device);
  std::recursive_mutex& GetMutex() {
    return m_mtx;
  }
  HbContext* GetHbContext(const c10::Device& device);
  HbContext* GetHbContext();

 private:
  std::vector<HbContext*> GetAllHbContexts();
  void ForAllHbContexts(
      const std::function<void(HbContext*)>& fn,
      const c10::Device* device);
  std::unordered_map<c10::Device, HbContext*> mp_device_contexts;
  std::recursive_mutex m_mtx;
};

inline c10::Device SynapseDeviceToAtenDevice(
    const synapse_helpers::device& device) {
  return c10::Device(at::kHPU, device.id());
}

inline c10::Device GetDeviceOrCurrent(const std::string& device_str) {
  if (device_str.empty()) {
    return SynapseDeviceToAtenDevice(
        synapse_helpers::HPURegistrar::get_device());
  }

  return c10::Device(device_str);
}

inline std::string GetCurrentThreadDevice() {
  return SynapseDeviceToAtenDevice(synapse_helpers::HPURegistrar::get_device())
      .str();
}

} // namespace habana_lazy
