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
#include <unordered_set>

#include <ATen/Tensor.h>
#include <c10/core/Device.h>
#include <torch/csrc/jit/ir/ir.h>

#include "habana_device/hpu_cached_devices.h"
#include "habana_helpers/misc_utils.h"
#include "habana_helpers/tensor_utils.h"
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
      ir::Value&& ir_value,
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
  std::string sbs_tensor_name = "";
  at::ScalarType original_element_type;
  const int64_t unique_id = 0;
  SmallSizeVec sizes;
  bool is_broadcastable = false;
  LazyTensorExecutionStatus execution_status = kUN_REGISTERED;
  // is_executing flag is set to true if this tensor is part of launch
  // thread. Reset after launch is completed
  bool is_executing = false;
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

  bool get_is_optimized_lazy_eager() {
    return is_optimized_lazy_eager;
  }

  void set_is_optimized_lazy_eager(bool flag) {
    is_optimized_lazy_eager = flag;
  }

  void set_is_broadcasting_op(bool flag) {
    is_broadcastable = flag;
  }

  std::vector<ir::Value>& get_input_values() {
    return input_values;
  }

  void set_input_values(std::vector<ir::Value>& input_vals) {
    input_values = input_vals;
  }

  bool get_is_hccl_send_mark_step() {
    return is_hccl_send_mark_step;
  }

  void set_is_hccl_send_mark_step(bool flag) {
    is_hccl_send_mark_step = flag;
  }

 private:
  // The value 0 of optimized_lazy_eager_key is used to indicate the unhandled
  // cases in optimized lazy eager so that no cache entry is prepared in
  // optimized lazy cache for such cases.
  size_t optimized_lazy_eager_key = 0;
  std::string op_name = getHabanaLazyGraphName();
  bool is_optimized_lazy_eager = false;
  std::vector<ir::Value> input_values{};
  bool is_hccl_send_mark_step = false;
  bool is_broadcastable = false;
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
      ir::Value&& ir_value,
      const at::Device& device,
      c10::optional<at::ScalarType> logical_element_type);
  // Creates an empty/null tensor.
  HbLazyTensor() = default;
  HbLazyTensor(const HbLazyTensor& other) = default;
  HbLazyTensor(HbLazyTensor&& other) = default;
  HbLazyTensor& operator=(const HbLazyTensor& other) = default;
  HbLazyTensor& operator=(HbLazyTensor&& other) = default;
  HbLazyTensor(const at::Tensor& tensor, const c10::Device& device);
  HbLazyTensor(const c10::Device& device);
  HbLazyTensor(
      ir::Value&& ir_value,
      const at::Device& device,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);
  HbLazyTensor(std::shared_ptr<Data> data);

  at::Tensor ToTensor(bool detached);
  bool is_null() const {
    return data_ptr() == nullptr;
  }
  // int size(int dim) const;
  void SetTensor(at::Tensor tensor);
  void setTensorSize(c10::IntArrayRef sizes);
  // Sets up a pointer from IR in data ptr back to data ptr
  // its cyclic in nature, being managed by weak pointer in IR
  void setPtrDataIrToData();
  ir::Value createIrValueFromData();
  void SetTensorData(at::Tensor tensor_data);
  c10::optional<at::Tensor> GetTensorData();
  void SetCPUTensorData(at::Tensor tensor_data);
  void SetSBSLiveTensorIndication(bool live);
  bool GetSBSLiveTensorIndication() const;
  void SetSBSCompareIndication(bool compare);
  bool GetSBSCompareIndication() const;
  void UpdateSBSTensorVersion();
  int GetSBSTensorVersion() const;
  void SetSBSTensorName(const std::string& name);
  std::string FetchSBSTensorName() const;
  const c10::optional<at::Tensor>& GetCPUTensorData() const;
  void AssignIrValue(ir::Value ir_value) const;
  c10::ScalarType dtype() const;
  c10::optional<c10::ScalarType> dtype_optional() const;
  // Set logical_element_type which is visible to upstream PyTorch.
  void SetScalarType(c10::optional<c10::ScalarType> logical_element_type);
  const c10::Device& GetDevice() const;
  const SmallSizeVec& GetSizes() const;
  // Retrieves the current IR Node, or nullptr in case no active IR Node is
  // available.
  ir::Value& CurrentIrValue() const;
  ir::Value GetIrValue() const;
  c10::optional<at::Tensor> CurrentTensorData() const;
  void setTensorOriginalType(c10::ScalarType type);
  c10::ScalarType getTensorOriginalType() const;
  void* CurrentHabanaData() const;
  bool IsExecutionInProgress() const;
  void SetExecutionInProgress() const;
  // Applies the queue of operations in preparation for using the data.
  void applyPendingGraph();
  c10::optional<at::Tensor> GetHbLazyTensorData();
  c10::optional<at::Tensor> GetHbLazyTensorDataForMedia();

  // Static methods
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

  static void SyncTensorsGraph(
      std::vector<HbLazyTensor>* tensors,
      std::shared_ptr<HbLazyFrontEndInfoToBackend> lazyFrontEndInfo = nullptr,
      bool async = false,
      bool collect_sync_tensors = true,
      synEventHandle event_handle = nullptr,
      synapse_helpers::hpuStream_t event_stream = 0,
      bool event_flag = false);

  static void SyncLiveTensorsGraph(
      const c10::Device* device,
      std::shared_ptr<HbLazyFrontEndInfoToBackend> lazy_front_end_info,
      std::vector<HbLazyTensor> out_hb_lazy_tensor = {},
      bool async = false,
      synEventHandle event_handle = nullptr,
      synapse_helpers::hpuStream_t event_stream = 0,
      bool event_flag = false,
      bool is_allreduce = false,
      std::set<int64_t> bucket_id = {},
      std::set<int64_t> bucket_recent_id = {});

  static void IterStepMarker();

  static void StepMarker(
      const std::string& device_str = {},
      std::shared_ptr<HbLazyFrontEndInfoToBackend> lazy_front_end_info =
          nullptr,
      std::vector<HbLazyTensor> out_hb_lazy_tensor = {},
      bool async = false /* Wait for launch thread to finish for internal MS */,
      synEventHandle event_handle = nullptr,
      synapse_helpers::hpuStream_t event_stream = 0,
      bool event_flag = false,
      bool is_allreduce = false,
      std::set<int64_t> bucket_id = {},
      std::set<int64_t> bucket_recent_id = {});
  static void StepMarkerBind(const std::string& device_str = {});
  static void StepMarkerFinish();
  static void InitiateBucketRefinement();
  static void SetDynamicMode();

  static void ExecuteCachedGraph(
      std::shared_ptr<torch::jit::Graph> graph,
      size_t hash,
      size_t graphKey,
      std::string opStrs,
      ir::ValueList& input_vals,
      ir::ValueList& output_vals,
      std::vector<habana_lazy::HbLazyTensor> hblazy_tensors,
      bool is_cached);

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
      std::shared_ptr<HbLazyFrontEndInfoToBackend> lazyFrontEndInfo = nullptr,
      bool async = false,
      bool collect_sync_tensors = true,
      synEventHandle event_handle = nullptr,
      synapse_helpers::hpuStream_t event_stream = 0,
      bool event_flag = false);
  static bool switch_dynamic_mode;
};

// The HbContextArena holds per device live information and statistics,
// among which the Habana tensors which are currently alive in the system.
// This is used to create computation checkpoints in order to flush pending
// operations and ensure the same computations are created during the
// training loops.
struct HbContext {
  std::map<int64_t, std::weak_ptr<Data>> tensors_data;
  std::map<int64_t, std::weak_ptr<Data>> tensors_data_opt;
  ir::Value seed_ir_value;
};

class HbContextArena {
 public:
  static HbContextArena* Get();
  void RegisterTensor(std::shared_ptr<Data> data);
  void UnregisterTensor(Data* data);
  std::weak_ptr<Data>& GetTensorDataPtrFromHbContext(Data* data);
  std::vector<HbLazyTensor> GetLiveTensors(
      const c10::Device* device,
      bool is_allreduce = false,
      std::set<int64_t> bucket_recent_id = {});
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
