/*******************************************************************************
 * Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
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

#pragma once
#include <c10/core/Device.h>
#include "backend/helpers/layout.h"
#include "backend/helpers/tensor_utils.h"
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

enum StridedOPType {
  kStridedOpDefault = 0,
  kStridedOpView,
  kStridedOpSlice,
  kStridedOpTranspose,
  kStridedOpT,
  kStridedOpPermute,
  kStridedOpSqueeze,
  kStridedOpUnsqueeze,
  kStridedOpExpand,
  kStridedOpIdentity,
  kStridedOpViewDtype,
  kStridedOpSqueezeDims
};

enum ViewStatus { kViewRead = 0, kViewWrite = 1, kEvaluated };

struct StridedOpSliceParams {
  int64_t dim;
  int64_t start;
  int64_t end;
  int64_t step;
};

struct StridedOpTransposeParams {
  int64_t dim0;
  int64_t dim1;
};

struct StridedOpSqueezeParams {
  int64_t dim;
};

struct StridedOpExpandParams {
  bool implicit = false;
};

union OpParams {
  StridedOpSliceParams slice_param;
  StridedOpTransposeParams transpose_param;
  StridedOpSqueezeParams squeeze_param;
  StridedOpExpandParams expand_param;
  OpParams(){};
};

struct StrideParams {
  // storing the tensor helps to retain extend the lifetime of tensor until all
  // the views have expired
  // base is used as node input for torch.as_strided. For rest of the view like
  // ops like view, select, slice, transpose etc we should the parent. This is
  // because only for as_strided the following relation holds true b =
  // torch.as_strided(a) c = as_strided(b) this is same as c = as_strided(a)
  // with the composite stride, size and offset params
  at::Tensor base;
  at::Tensor parent;
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;
  int64_t offset;
  int64_t parent_id;
  StridedOPType optype;
  OpParams params;
  ViewStatus viewStatus = kViewRead;
  size_t write_cnt = 0;

  size_t Size() const {
    size_t size = sizeof(*this);
    size += sizes.size() * sizeof(decltype(sizes)::value_type);
    size += strides.size() * sizeof(decltype(strides)::value_type);
    return size;
  }
};
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
  int64_t GetNextTensorId();

  bool is_random_seed_tensor = false;
  void* data_ptr;
  ir::Value ir_value;
  habana::LayoutFormat tensor_layout = habana::LayoutFormat::NCHW;
  c10::Device device;
  c10::optional<at::ScalarType> logical_element_type;
  c10::optional<at::Tensor> tensor_data;
  c10::optional<at::Tensor> cpu_tensor_data;
  c10::optional<c10::SmallVector<at::Tensor, 3>> tensor_shallow_copy;
  c10::optional<StrideParams> stride_params;
  c10::optional<at::Tensor> recent_base;
  bool sbs_live_tensor = false;
  bool sbs_compare_tensor = true;
  int sbs_tensor_version = 0;
  std::string sbs_tensor_name = "";
  bool collective = false;
  at::ScalarType original_element_type;
  const int64_t unique_id = 0;
  SmallSizeVec sizes;
  bool is_broadcastable = false;
  LazyTensorExecutionStatus execution_status = kUN_REGISTERED;
  // is_executing flag is set to true if this tensor is part of launch
  // thread. Reset after launch is completed
  bool is_executing = false;
  std::atomic<unsigned int> is_op_acc{0};
  ir::LazyView parent_view;
  int num_views = 0;
  // Version counter tracks the number of times we use tensor as output
  // if its zero, that means this tensor hasnt been output in any op
  // IMPORTANT : this is used per graph right now, that means for each lazy
  // graph generated it will be reset to 0. We are only tracking version for
  // that particular graph execution.
  int version = 0;
  std::atomic<int64_t> running_cntr = -1; // -1 is invalid tensor ID.
  // used for carrying constant tensor metadata from aten::tensor to lazy to
  // backend tensor
  bool is_const_tensor = false;
  int const_id = -1;
};

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

  void set_input_values(std::vector<ir::Value>&& input_vals) {
    input_values = std::move(input_vals);
  }

  bool get_is_hccl_send_mark_step() {
    return is_hccl_send_mark_step;
  }

  void set_is_hccl_send_mark_step(bool flag) {
    is_hccl_send_mark_step = flag;
  }

  std::vector<uint64_t> get_lazy_eager_op_input_uids() {
    return lazy_eager_op_input_uids;
  }

  size_t get_lazy_eager_op_num_of_uids() {
    return lazy_eager_op_num_of_uids;
  }

  void set_lazy_eager_op_input_uids(std::vector<uint64_t>&& uids) {
    lazy_eager_op_input_uids = std::move(uids);
    lazy_eager_op_num_of_uids = lazy_eager_op_input_uids.size();
  }

  std::vector<std::vector<int64_t>> get_out_shapes() {
    return out_shapes;
  }

  void set_out_shapes(std::vector<std::vector<int64_t>> shapes) {
    out_shapes = shapes;
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
  std::vector<uint64_t> lazy_eager_op_input_uids{};
  size_t lazy_eager_op_num_of_uids = 0;
  std::vector<std::vector<int64_t>> out_shapes{};
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
  ir::Value createIrValueFromData() const {
    return {data_ptr()};
  }
  void SetTensorDataNullOpt();
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
  void SetCollective();
  void ClearCollective();
  bool IsCollective() const;
  void ClearStrideParams();
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
  const ir::Value& GetIrValue() const;
  ir::Value& IrSetNode(ir::NodePtr node, size_t index = 0) const;
  c10::optional<at::Tensor> CurrentTensorData() const;
  void setTensorOriginalType(c10::ScalarType type);
  c10::ScalarType getTensorOriginalType() const;
  void* CurrentHabanaData() const;
  bool IsExecutionInProgress() const;
  void SetExecutionInProgress() const;
  void ResetExecutionInProgress() const;
  bool IsOpAccumulationInProgress() const;
  void SetOpAccumulationInProgress() const;
  // Applies the queue of operations in preparation for using the data.
  void applyPendingGraph();
  /* Produces underlying data Tensor.
   * If data tensor is currently being evaluated then this function blocks until
   * this calculation is completed. If this HbLazyTensor is not yet calculated,
   * this function issues a MarkStep.
   */
  at::Tensor EvaluateTensorData(bool sync_acc_thread = true);
  void ValidateTensorData() const;
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
  // Assigns tensor to an Input IR node if it is not assigned to any node,
  // otherwise assets.
  void IrInitAsInputNode() const;
  // Unconditionally assigns tensor to an Input IR node. Preserves data pointer.
  void IrReconnectAsInputNode() const;
  static std::vector<int> CollectSyncTensors(
      const std::vector<HbLazyTensor>& tensors);
  static ir::PostOrderData RunPostOrder(
      const std::vector<HbLazyTensor>& tensors,
      std::vector<int> indices);

  static void SyncTensorsGraph(
      std::vector<HbLazyTensor>* tensors,
      std::shared_ptr<HbLazyFrontEndInfoToBackend> lazyFrontEndInfo = nullptr,
      bool async = false,
      bool collect_sync_tensors = true);

  static void SyncLiveTensorsGraph(
      const c10::Device* device,
      std::shared_ptr<HbLazyFrontEndInfoToBackend> lazy_front_end_info,
      std::vector<HbLazyTensor> out_hb_lazy_tensor = {},
      bool async = false,
      bool is_allreduce = false,
      std::vector<HbLazyTensor> bucket_hl_t = {},
      std::set<int64_t> bucket_recent_id = {});

  static void IterStepMarker();

  static void StepMarker(
      const std::string& device_str = {},
      std::shared_ptr<HbLazyFrontEndInfoToBackend> lazy_front_end_info =
          nullptr,
      std::vector<HbLazyTensor> out_hb_lazy_tensor = {},
      bool async = false /* Wait for launch thread to finish for internal MS */,
      bool is_allreduce = false,
      std::vector<HbLazyTensor> bucket_hl_t = {},
      std::set<int64_t> bucket_recent_id = {});
  static void StepMarkerBind(
      const std::string& device_str = {},
      bool sync = false);
  static void StepMarkerFinish(bool wait_only = false);
  static void InitiateBucketRefinement();
  static void SetDynamicMode();

  static void ExecuteCachedGraph(
      std::shared_ptr<torch::jit::Graph> graph,
      size_t hash,
      size_t graphKey,
      std::string opStrs,
      std::vector<habana_lazy::HbLazyTensor> hblazy_tensors_in,
      std::vector<habana_lazy::HbLazyTensor> hblazy_tensors_out,
      std::vector<habana_lazy::HbLazyTensor> hbt_last_out_used_as_inputs,
      const std::unordered_map<int64_t, c10::optional<at::Generator>>&
          seed_tensors_generator_map,
      bool is_cached,
      uint64_t launch_jobid);

  static void* lazyTensorDataPtr(const at::Tensor& t);

  void ShallowCopyTo(HbLazyTensor* dest) const;

  bool IsHpuGraphOutTensor() {
    return is_hpugraph_out_tensor;
  }

  void SetHpuGraphOutTensor(bool flag) {
    is_hpugraph_out_tensor = flag;
  }

  int64_t getTensorUniqueId() const {
    if (mp_data.get()) {
      return mp_data.get()->unique_id;
    } else
      return -1;
  }

  int64_t getTensorRunningId() const {
    if (mp_data.get()) {
      return mp_data.get()->running_cntr;
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

  void SetTensorLayout(habana::LayoutFormat layout) {
    data()->tensor_layout = layout;
  }
  habana::LayoutFormat GetTensorLayout() const {
    return data()->tensor_layout;
  }

  void SetIsStrided(bool flag) {
    is_strided = flag;
  }

  bool GetIsStrided() const {
    return is_strided;
  }

  bool created_as_zero_size_tensor = false;

  bool IsConstTensor() const {
    return data()->is_const_tensor;
  }

  int GetConstTensorId() const {
    return data()->const_id;
  }
  void SetIsConstTensor(bool is_const, int const_id) {
    data()->is_const_tensor = is_const;
    data()->const_id = const_id;
    if (is_const) {
      HABANA_ASSERT(
          const_id != -1, "LazyTensor: Constant tensor can not have id as -1");
    }
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
      bool collect_sync_tensors = true);

  bool is_hpugraph_out_tensor = false;
  static bool switch_dynamic_mode;
  bool is_strided = false;
};

struct Holder {
  Holder(HbLazyTensor&& lt, bool flag) : tensor(lt), marker(flag) {}
  HbLazyTensor tensor;
  bool marker;
};

struct Snapshot {
  std::vector<HbLazyTensor> tensors;
};

class StaleLazyTensorKeeper {
 public:
  static StaleLazyTensorKeeper& getInstance() {
    static StaleLazyTensorKeeper instance;
    return instance;
  }

  void add(HbLazyTensor&& t) {
    Holder holder(std::move(t), false);
    std::lock_guard<std::mutex> lock(mutex);
    kept_alive.push_back(std::move(holder));
  }

  void mark_end_of_accumulation() {
    Holder holder(HbLazyTensor(), true);
    std::lock_guard<std::mutex> lock(mutex);
    kept_alive.push_back(std::move(holder));
  }

  std::shared_ptr<Snapshot> extract_snapshot();

 private:
  std::mutex mutex;
  std::list<Holder> kept_alive;
};

// The HbContextArena holds per device live information and statistics,
// among which the Habana tensors which are currently alive in the system.
// This is used to create computation checkpoints in order to flush pending
// operations and ensure the same computations are created during the
// training loops.
struct HbContext {
  absl::flat_hash_map<int64_t, std::weak_ptr<Data>> tensors_data;
  std::vector<int64_t> tensors_data_opt_order;
  ir::Value seed_ir_value;
  void clear_tensors_data() {
    tensors_data_opt.clear();
    tensors_data_opt_order.clear();
    tensors_data_del.clear();
  }

  size_t tensors_data_size() {
    return tensors_data_opt.size();
  }

  void insert(int64_t unique_id, std::weak_ptr<Data> m_data_ptr) {
    bool exists = tensors_data_opt.count(unique_id) != 0;
    exists |= (tensors_data_del.count(unique_id) != 0);
    tensors_data_opt[unique_id] = m_data_ptr;
    if (exists) {
      auto pos = std::find(
          tensors_data_opt_order.begin(),
          tensors_data_opt_order.end(),
          unique_id);
      if (pos != tensors_data_opt_order.end()) {
        tensors_data_opt_order.erase(pos);
      }
    }
    tensors_data_opt_order.emplace_back(unique_id);
  }

  void erase(int64_t unique_id) {
    tensors_data_del.insert(unique_id);
    tensors_data_opt.erase(unique_id);
  }

  std::shared_ptr<Data> getDataPtr(int64_t unique_id) {
    if (tensors_data_opt.count(unique_id) != 0) {
      return tensors_data_opt.at(unique_id).lock();
    } else {
      return nullptr;
    }
  }

 private:
  absl::flat_hash_map<int64_t, std::weak_ptr<Data>> tensors_data_opt;
  absl::flat_hash_set<int64_t> tensors_data_del;
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

// check to be performed in main thread before
// scheduling ShallowCopy to acc thread
void MaybeSyncLaunchBeforeShallowCopy(
    const HbLazyTensor* dest,
    const HbLazyTensor* src);

} // namespace habana_lazy
