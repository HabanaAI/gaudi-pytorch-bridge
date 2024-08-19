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
#include <thread>

#include "backend/habana_device/HPUGraph.h"
#include "backend/helpers/dynamic_shape_info.h"
#include "backend/synapse_helpers/util.h"
#include "habana_helpers/thread_pool/thread_pool.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/ir.h"
#include "habana_lazy/tensor_impl.h"
#include "habana_lazy/view_utils.h"
#include "torch/csrc/jit/ir/ir.h"

enum LazyExecutionMode { kLAZY, kLOWERING };

using Graph = torch::jit::Graph;
using GraphPtr = std::shared_ptr<Graph>;

namespace habana_lazy {

struct HashFn {
  std::size_t operator()(const std::pair<double, at::ScalarType>& pair) const {
    return std::hash<double>()(pair.first) ^
        std::hash<float>()((float)pair.second);
  }
};

class EqualFn {
 public:
  bool operator()(
      const std::pair<double, at::ScalarType>& a,
      const std::pair<double, at::ScalarType>& b) const {
    return a.first == b.first && a.second == b.second;
  }
};

class SingleTonExecThreadPool {
 public:
  static SingleTonExecThreadPool& Get() {
    std::call_once(initialize_once_flag_, CreateInstance);
    return *instance_;
  }

  static habana_helpers::ThreadPoolWithFutures& getInstance() {
    return Get().thread_pool_obj_;
  }

 private:
  SingleTonExecThreadPool() : thread_pool_obj_{1} {}
  SingleTonExecThreadPool(const SingleTonExecThreadPool&) = delete;
  SingleTonExecThreadPool& operator=(const SingleTonExecThreadPool&) = delete;
  static std::unique_ptr<SingleTonExecThreadPool> instance_;
  static std::once_flag initialize_once_flag_;
  static void CreateInstance();
  habana_helpers::ThreadPoolWithFutures thread_pool_obj_;
};

class HbExecutionContext {
 public:
  HbExecutionContext() = default;
  void RegisterTensor(std::shared_ptr<Data> data);
  void UnregisterTensor(Data* data);
  void MarkTensorsExecuted() {
    HbContext* devctx = habana_lazy::HbContextArena::Get()->GetHbContext();
    // ensure that Data is destroyed outside of HbContextArena mutex
    // to avoid deadlock with StridedViewContext mutex that can be
    // acquired during Data d'tors
    std::vector<std::shared_ptr<Data>> data_tensors;
    data_tensors.reserve(devctx->tensors_data.size());
    {
      std::lock_guard<std::recursive_mutex> lock(
          habana_lazy::HbContextArena::Get()->GetMutex());
      std::for_each(
          devctx->tensors_data.begin(),
          devctx->tensors_data.end(),
          [&data_tensors](std::pair<int64_t, std::weak_ptr<Data>> p) {
            std::shared_ptr<Data> data = p.second.lock();
            data_tensors.push_back(data);
            if ((data != nullptr) && (data->execution_status == kEXECUTING)) {
              data->execution_status = kEXECUTION_COMPLETE;
              data->is_executing = false;
            }
          });
    }
  }

  void MarkTensorsExecuted(
      const c10::Device& device,
      const std::vector<int64_t>& indices) {
    // ensure that Data is destroyed outside of HbContextArena mutex
    // to avoid deadlock with StridedViewContext mutex that can be
    // acquired during Data d'tors
    std::vector<std::shared_ptr<Data>> data_tensors;
    data_tensors.reserve(indices.size());
    {
      std::lock_guard<std::recursive_mutex> lock(
          habana_lazy::HbContextArena::Get()->GetMutex());
      HbContext* devctx =
          habana_lazy::HbContextArena::Get()->GetHbContext(device);
      for (const auto& k : indices) {
        if (devctx->tensors_data.find(k) != devctx->tensors_data.end()) {
          std::shared_ptr<Data> data = devctx->tensors_data.at(k).lock();
          data_tensors.push_back(data);
          if (data != nullptr) {
            data->execution_status = kEXECUTION_COMPLETE;
            data->is_executing = false;
          }
        }
      }
    }
  }

  void ClearOpAccmulationFlag(
      const c10::Device& device,
      const std::set<int64_t>& acc_indices) {
    // ensure that Data is destroyed outside of HbContextArena mutex
    // to avoid deadlock with StridedViewContext mutex that can be
    // acquired during Data d'tors
    std::vector<std::shared_ptr<Data>> data_tensors;
    data_tensors.reserve(acc_indices.size());
    {
      std::lock_guard<std::recursive_mutex> lock(
          habana_lazy::HbContextArena::Get()->GetMutex());
      HbContext* devctx =
          habana_lazy::HbContextArena::Get()->GetHbContext(device);

      for (const auto& k : acc_indices) {
        if (devctx->tensors_data.find(k) != devctx->tensors_data.end()) {
          std::shared_ptr<Data> data = devctx->tensors_data.at(k).lock();
          data_tensors.push_back(data);
          if (data != nullptr) {
            if (data->is_op_acc) {
              data->is_op_acc--;
            }
          }
        }
      }
    }
  }

  void MarkAllTensorsExecuted(const c10::Device& device) {
    HbContext* devctx =
        habana_lazy::HbContextArena::Get()->GetHbContext(device);
    // ensure that Data is destroyed outside of HbContextArena mutex
    // to avoid deadlock with StridedViewContext mutex that can be
    // acquired during Data d'tors
    std::vector<std::shared_ptr<Data>> data_tensors;
    data_tensors.reserve(devctx->tensors_data.size());
    {
      std::lock_guard<std::recursive_mutex> lock(
          habana_lazy::HbContextArena::Get()->GetMutex());
      std::for_each(
          devctx->tensors_data.begin(),
          devctx->tensors_data.end(),
          [&data_tensors](std::pair<int64_t, std::weak_ptr<Data>> p) {
            std::shared_ptr<Data> data = p.second.lock();
            data_tensors.push_back(data);
            if (data != nullptr) {
              data->execution_status = kEXECUTION_COMPLETE;
            }
          });
    }
  }
  void MarkTensorExecuting(std::shared_ptr<Data> data);
  void MarkTensorExecuted(std::shared_ptr<Data> data);
  void MarkTensorStatus(
      std::shared_ptr<Data> data,
      LazyTensorExecutionStatus status);
  LazyTensorExecutionStatus getTensorExecutionStatus(
      std::shared_ptr<Data> data);

  void removeRetainedTensor(at::Tensor& tensor) {
    for (auto i = m_retained_tensor_list.begin();
         i != m_retained_tensor_list.end();
         ++i) {
      auto list_tensor_impl = i->unsafeGetTensorImpl();
      if (list_tensor_impl == tensor.unsafeGetTensorImpl()) {
        m_retained_tensor_list.erase(i);
      }
    }
  }

  void saveGraph(GraphPtr p_g) {
    mp_g = p_g;
  }

  GraphPtr getGraph() {
    return mp_g;
  }

  void saveHash(size_t p_h) {
    mp_g_hash = p_h;
  }

  size_t getHash() {
    return mp_g_hash;
  }

  void saveGraphKey(size_t graphKey) {
    mp_g_key = graphKey;
  }

  size_t getGraphKey() {
    return mp_g_key;
  }

  void saveOpStrs(std::string opStrs) {
    mp_g_op_strs = opStrs;
  }

  std::string getOpStrs() {
    return mp_g_op_strs;
  }

  void setCapturing(bool capture) {
    m_capturing_graph = capture;
  }

  bool getCapturing() {
    return m_capturing_graph;
  }

  void setDryRun(bool dry_run) {
    m_dry_run = dry_run;
  }

  bool getDryRun() {
    return m_dry_run;
  }

  void setCaptureGraph(at::hpu::HPUGraph* hpu_graph) {
    HABANA_ASSERT(m_captured_hpu_graph == nullptr || hpu_graph == nullptr)
    m_captured_hpu_graph = hpu_graph;
  }

  at::hpu::HPUGraph* getCaptureGraph() {
    return m_captured_hpu_graph;
  }

  void CaptureGraphMarkStep() {
    m_captured_hpu_graph->mark_step();
  }

  void saveInputsAndOutputs(
      ir::ValueList inputVals,
      ir::ValueList outputVals,
      std::vector<habana_lazy::HbLazyTensor>& tensors,
      const std::vector<int>& indices);

  bool updateInputsRequired(std::vector<size_t>& indices);

  void updateInputs(ir::ValueList inputVals);

  ir::ValueList& getInputs() {
    return m_input_vals;
  }

  ir::ValueList& getOutputs() {
    return m_output_vals;
  }

  std::unordered_map<int64_t, c10::optional<at::Generator>>& getSeedTensorMap() {
    return m_seed_tensor_generator_map;
  }

  std::vector<habana_lazy::HbLazyTensor> getHbLazyTensors() {
    return m_hblazy_tensors;
  }

  std::mutex& GetOpAccTidsMutex() {
    return m_op_acc_tid_mtx;
  }

  std::mutex& GetScalarToTensorMutex() {
    return m_scalar_to_tensor_map_mtx;
  }

  std::unordered_map<size_t, size_t> getUserInputIndices() const noexcept {
    return m_user_input_positions;
  }

  std::unordered_set<size_t> getUserInputMatchIndices() const noexcept {
    return m_user_input_match_index;
  }

  void setMarkedInputs(std::vector<at::Tensor>& marked_user_tensors) {
    m_marked_user_inputs = marked_user_tensors;
  }

  void ClearHPUGraphUserMarkedInputs() {
    m_marked_user_inputs.clear();
  }

  void clear() {
    viewContext.hb_tensors_exclude_out_view.clear();
    m_retained_tensor_list.clear();
    // The scalar_to_tensor_map caches {scalar value, target dtype} -> device
    // tensor This cache avoids repeated H2D DMAs for scalars with target dtype.
    // Hence, the default strategy is to retain the cache across executions and
    // model training iterations. However, if the cache grows too large, due to
    // frequently changing scalar values in any model, the lookup time increases
    // and the host overhead increases on the lazy op accumulation side. To
    // avoid this, PT_HPU_SCALAR_MAP_MAXSIZE sets a max size limit on the cache.
    // Once the cache reaches this size, it gets cleared after an execution. The
    // max value for PT_HPU_SCALAR_MAP_MAXSIZE is heuristically set at 500
    // entries as of now, and can be fine tuned based on performance profiling
    // feedback from model runs.
    {
      std::lock_guard<std::mutex> lock(GetScalarToTensorMutex());
      PT_LAZY_DEBUG(
          "scalar_to_tensor_map size at HbExecutionContext::clear = ",
          scalar_to_tensor_map.size());
      if (GET_ENV_FLAG_NEW(PT_HPU_CLEAR_SCALAR_MAP_ON_MARKSTEP, 1) ||
          scalar_to_tensor_map.size() >
              GET_ENV_FLAG_NEW(PT_HPU_SCALAR_MAP_MAXSIZE)) {
        PT_LAZY_DEBUG("scalar_to_tensor_map cleared");
        scalar_to_tensor_map.clear();
      }
    }

    viewContext.updated_bucket_list.clear();

    {
      std::lock_guard<std::mutex> lock(GetOpAccTidsMutex());
      op_acc_tids.clear();
    }
  }

  void resetGraph() {
    saveGraph(nullptr);
    saveHash(0);
    saveGraphKey(0);
    saveOpStrs("");
    m_input_vals.clear();
    m_output_vals.clear();
    m_hblazy_tensors.clear();
    m_user_input_positions.clear();
    m_user_input_match_index.clear();
  }

  // We want to retain some tensors for special cases where PT releases them
  // but because we are in lazy mode we actually need them for processing
  // later This should only be used in special cases and released on exit
  // cleanly
  std::vector<at::Tensor> m_retained_tensor_list;

  std::unordered_map<
      std::pair<double, at::ScalarType>,
      at::Tensor,
      HashFn,
      EqualFn>
      scalar_to_tensor_map;

  std::vector<std::pair<at::Tensor, at::Tensor>> copy_scalar_to_hpu_tensor_list;

  // Structure to keep the strided view related data
  StridedViewContext viewContext;

  // Handle for the launch thread, only one thread is alive at a time.
  std::future<void> m_launch_thread_handle;
  void JoinPendingLaunchThread(bool wait_only = false);
  void HandleException() {
    if (C10_UNLIKELY(m_launch_thread_exception_handler)) {
      try {
        std::rethrow_exception(m_launch_thread_exception_handler);
      } catch (const std::exception& e) {
        m_launch_thread_exception_handler = nullptr;
        PT_BRIDGE_FATAL(
            "Exception in Launch thread...\nCheck $HABANA_LOGS/ for details",
            e.what());
      } catch (...) {
        m_launch_thread_exception_handler = nullptr;
        PT_BRIDGE_FATAL(
            "Exception in Launch thread...\nCheck $HABANA_LOGS/ for details");
      }
    }
  }
  thread_local static bool m_launch_thread_context;
  std::exception_ptr m_launch_thread_exception_handler = nullptr;
  // Tensorids list which is part of current exec thread
  std::vector<int64_t> executing_tids;

  // Ids of tensors whose is_op_acc field is set
  // These tids are used to reset them post launch
  std::set<int64_t> op_acc_tids;

  thread_local static bool m_async_d2h_context;

  void AddToJobidStreamidMap(
      uint64_t jobId,
      synapse_helpers::hpuStream_t streamId);
  void DelFromJobidStreamidMap(uint64_t jobId);
  bool HaveJobsInStream(synapse_helpers::hpuStream_t);
  std::uint64_t GetUniqueJobId();

 private:
  GraphPtr mp_g;
  size_t mp_g_hash{0};
  size_t mp_g_key{0};
  std::string mp_g_op_strs = "";
  ir::ValueList m_input_vals;
  ir::ValueList m_output_vals;
  std::vector<at::Tensor> m_marked_user_inputs;
  std::vector<habana_lazy::HbLazyTensor> m_hblazy_tensors;
  std::unordered_map<size_t, size_t> m_user_input_positions;
  std::unordered_set<size_t> m_user_input_match_index;
  bool m_capturing_graph{false};
  bool m_dry_run{false};
  at::hpu::HPUGraph* m_captured_hpu_graph{nullptr};
  std::unordered_map<int64_t, c10::optional<at::Generator>>
      m_seed_tensor_generator_map;
  static std::atomic_uint64_t m_unique_jobid_count;

  // unordered map jobid -> synapse_helpers::hpuStream_t
  std::unordered_map<uint64_t, synapse_helpers::hpuStream_t>
      m_jobid_streamid_map;
  std::mutex m_jobid_streamid_map_mtx;
  std::mutex m_op_acc_tid_mtx;
  std::mutex m_scalar_to_tensor_map_mtx;
};

class HbExecutionContextArena {
 public:
  static HbExecutionContextArena& Get() {
    std::call_once(initialize_once_flag_, CreateInstance);
    return *instance_;
  }

  HbExecutionContext* getDeviceExecutionContext(
      int device = 0); // TODO remove device from everywhere
  HbExecutionContextArena() = default;
  const LazyExecutionMode& getExecutionMode();
  void setExecutionMode(LazyExecutionMode m);
  static thread_local LazyExecutionMode execution_mode;
  void resetUniqueGraphCntr() {
    unique_graph_index_counter.clear();
  }
  uint64_t getGraphindexCntr(size_t hash_code) {
    if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_UNIQUE_GRAPH)) {
      if (unique_graph_index_counter.find(hash_code) !=
          unique_graph_index_counter.end()) {
        unique_graph_index_counter[hash_code] += 1;
      } else {
        unique_graph_index_counter[hash_code] = 0;
      }
      return unique_graph_index_counter[hash_code];
    }
    return 0;
  }

 private:
  // Keep a map of all the execution contexts in play
  // Right now we support  a single context per device, map maintains ID to
  // context map
  HbExecutionContext execution_context_;
  std::unordered_map<size_t, uint64_t> unique_graph_index_counter;
  static std::once_flag initialize_once_flag_;
  static std::unique_ptr<HbExecutionContextArena> instance_;
  static void CreateInstance();
};

// The global object for all contexts, we create contexts out of this per
// device as the execution goes on Create a global object for the arena of
// contexts We will keep them aslive as long as program lives and manage
// device contexts across iterations
inline HbExecutionContextArena& get_habana_lazy_executor() {
  return HbExecutionContextArena::Get();
}

inline HbExecutionContext* get_device_lazy_execution_context(int device = 0) {
  return get_habana_lazy_executor().getDeviceExecutionContext(device);
}

/*
 * Helper functions to manage the tensor creation based on the execution
 * state. The execution states are following -
 * 1. PyTorch creates a tensor
 *    - Create a tensor with storage, and one without storage.
 *      From the one without storage, create a lazy tensor and from
 *      the lazy tensor point to the internal one with storage.
 *      Retrun the one one without storage.
 * 2. Accumulate ops in IR graph
 *    - Create only storage less tensors and return.
 * 3. Lowering creates a tensor
 *    - Create a tensor with storage and return.
 */
bool isDeviceInLoweringMode();
}; // namespace habana_lazy
