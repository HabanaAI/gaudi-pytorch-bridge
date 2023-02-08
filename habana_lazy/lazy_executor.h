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
#include <thread>

#include "habana_helpers/dynamic_shape_info.h"
#include "habana_helpers/thread_pool/thread_pool.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/ir.h"
#include "habana_lazy/tensor_impl.h"
#include "habana_lazy/view_utils.h"
#include "pytorch_helpers/habana_device/HPUGraph.h"
#include "synapse_helpers/util.h"
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
  static habana_helpers::ThreadPool& getInstance() {
    static habana_helpers::ThreadPool thread_pool_obj(1);
    return thread_pool_obj;
  }

  static void work() {
    while (getInstance().has_work.load()) {
      if (getInstance().m_stop || !getInstance().has_work.load()) {
        break;
      }
    }
    return;
  }

  static void queueStatus() {
    while (getInstance().has_queued_items.load()) {
      if (getInstance().m_stop || !getInstance().has_queued_items.load()) {
        break;
      }
    }
    return;
  }

 private:
  SingleTonExecThreadPool() = default;
  SingleTonExecThreadPool(const SingleTonExecThreadPool&) = delete;
  SingleTonExecThreadPool& operator=(const SingleTonExecThreadPool&) = delete;
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
    PT_LAZY_DEBUG(
        "scalar_to_tensor_map size at HbExecutionContext::clear = ",
        scalar_to_tensor_map.size());
    if (GET_ENV_FLAG_NEW(PT_HPU_CLEAR_SCALAR_MAP_ON_MARKSTEP, 1) ||
        scalar_to_tensor_map.size() >
            GET_ENV_FLAG_NEW(PT_HPU_SCALAR_MAP_MAXSIZE)) {
      PT_LAZY_DEBUG("scalar_to_tensor_map cleared");
      scalar_to_tensor_map.clear();
    }

    viewContext.updated_bucket_list.clear();
  }

  void resetGraph() {
    saveGraph(nullptr);
    saveHash(0);
    saveGraphKey(0);
    saveOpStrs("");
    m_input_vals.clear();
    m_output_vals.clear();
    m_hblazy_tensors.clear();
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
  void JoinPendingLaunchThread();
  void HandleException() {
    if (C10_UNLIKELY(m_launch_thread_exception_handler)) {
      try {
        std::rethrow_exception(m_launch_thread_exception_handler);
      } catch (const std::exception& e) {
        m_launch_thread_exception_handler = nullptr;
        PT_BRIDGE_FATAL("Exception in Launch thread...\n", e.what());
      } catch (...) {
        m_launch_thread_exception_handler = nullptr;
        PT_BRIDGE_FATAL("Exception in Launch thread...\n");
      }
    }
  }
  thread_local static bool m_launch_thread_context;
  std::exception_ptr m_launch_thread_exception_handler = nullptr;
  // Tensorids list which is part of current exec thread
  std::vector<int64_t> executing_tids;

 private:
  GraphPtr mp_g;
  size_t mp_g_hash{0};
  size_t mp_g_key{0};
  std::string mp_g_op_strs = "";
  ir::ValueList m_input_vals;
  ir::ValueList m_output_vals;
  std::vector<habana_lazy::HbLazyTensor> m_hblazy_tensors;
  bool m_capturing_graph{false};
  at::hpu::HPUGraph* m_captured_hpu_graph{nullptr};
  std::unordered_map<int64_t, c10::optional<at::Generator>>
      m_seed_tensor_generator_map;
};

class HbExecutionContextArena {
 public:
  static HbExecutionContextArena Get();
  HbExecutionContext* getDeviceExecutionContext(
      int device = 0); // TODO remove device from everywhere
  HbExecutionContext* createExecutionContext(int device);
  void removeExecutionContext(int device);
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
  std::unordered_map<int, HbExecutionContext*> m_execution_context_list;
  std::unordered_map<size_t, uint64_t> unique_graph_index_counter;
};

// The global object for all contexts, we create contexts out of this per
// device as the execution goes on Create a global object for the arena of
// contexts We will keep them aslive as long as program lives and manage
// device contexts across iterations
extern HbExecutionContextArena habana_lazy_executor;
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
