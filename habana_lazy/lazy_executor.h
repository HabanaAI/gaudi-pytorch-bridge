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
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/ir.h"
#include "habana_lazy/tensor_impl.h"
#include "synapse_helpers/util.h"
#include "torch/csrc/jit/ir/ir.h"

enum LazyExecutionMode { kLAZY, kLOWERING };

using Graph = torch::jit::Graph;
using GraphPtr = std::shared_ptr<Graph>;

namespace habana_lazy {
enum StrideOPType {
  kStridedOpDefault = 0,
  kStridedOpView,
  kStridedOpSlice,
  kStridedOpTranspose,
  kStridedOpT,
  kStridedOpPermute,
  kStridedOpSqueeze,
  kStridedOpUnsqueeze
};

struct StridedOpSliceParams {
  int64_t dim;
  c10::optional<int64_t> start;
  c10::optional<int64_t> end;
  int64_t step;
};

struct StridedOpTransposeParams {
  int64_t dim0_;
  int64_t dim1_;
};

struct StridedOpSqueezeParams {
  int64_t dim;
};
union OpParams {
  StridedOpSliceParams slice_param;
  StridedOpTransposeParams transpose_param;
  StridedOpSqueezeParams squeeze_param;
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
  StrideOPType optype;
  OpParams params;

  size_t Size() const {
    size_t size = sizeof(*this);
    size += sizes.size() * sizeof(decltype(sizes)::value_type);
    size += strides.size() * sizeof(decltype(strides)::value_type);
    return size;
  }
};

struct HashFn {
  std::size_t operator()(const std::pair<float, at::ScalarType>& pair) const {
    return std::hash<float>()(pair.first) ^
        std::hash<float>()((float)pair.second);
  }
};

class EqualFn {
 public:
  bool operator()(
      const std::pair<float, at::ScalarType>& a,
      const std::pair<float, at::ScalarType>& b) const {
    return a.first == b.first && a.second == b.second;
  }
};

class HbExecutionContext {
 public:
  HbExecutionContext() = default;
  void RegisterTensor(std::shared_ptr<Data> data);
  void UnregisterTensor(Data* data);
  void MarkTensorsExecuted() {
    HbContext* devctx = habana_lazy::HbContextArena::Get()->GetHbContext();
    std::lock_guard<std::recursive_mutex> lock(
        habana_lazy::HbContextArena::Get()->GetMutex());
    std::for_each(
        devctx->tensors_data.begin(),
        devctx->tensors_data.end(),
        [](std::pair<int64_t, std::weak_ptr<Data>> p) {
          std::shared_ptr<Data> data = p.second.lock();
          if ((data != nullptr) && (data->execution_status == kEXECUTING)) {
            data->execution_status = kEXECUTION_COMPLETE;
          }
        });
  }

  void MarkTensorsExecuted(
      const c10::Device& device,
      const std::vector<uint64_t>& indices) {
    std::lock_guard<std::recursive_mutex> lock(
        habana_lazy::HbContextArena::Get()->GetMutex());
    HbContext* devctx =
        habana_lazy::HbContextArena::Get()->GetHbContext(device);
    for (const auto& k : indices) {
      if (devctx->tensors_data.find(k) != devctx->tensors_data.end()) {
        std::shared_ptr<Data> data = devctx->tensors_data.at(k).lock();
        if (data != nullptr) {
          data->execution_status = kEXECUTION_COMPLETE;
        }
      }
    }
  }

  void MarkAllTensorsExecuted(const c10::Device& device) {
    HbContext* devctx =
        habana_lazy::HbContextArena::Get()->GetHbContext(device);
    std::lock_guard<std::recursive_mutex> lock(
        habana_lazy::HbContextArena::Get()->GetMutex());
    std::for_each(
        devctx->tensors_data.begin(),
        devctx->tensors_data.end(),
        [](std::pair<int64_t, std::weak_ptr<Data>> p) {
          std::shared_ptr<Data> data = p.second.lock();
          if (data != nullptr) {
            data->execution_status = kEXECUTION_COMPLETE;
          }
        });
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

  void saveInputsAndOutputs(
      ir::ValueList inputVals,
      ir::ValueList outputVals,
      std::vector<habana_lazy::HbLazyTensor>& tensors,
      const std::vector<int>& indices);

  ir::ValueList& getInputs() {
    return m_input_vals;
  }

  ir::ValueList& getOutputs() {
    return m_output_vals;
  }

  std::vector<habana_lazy::HbLazyTensor> getHbLazyTensors() {
    return m_hblazy_tensors;
  }

  void clear() {
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
    if (GET_ENV_FLAG_NEW(PT_HPU_CLEAR_SCALAR_MAP_ON_MARKSTEP) ||
        scalar_to_tensor_map.size() >
            GET_ENV_FLAG_NEW(PT_HPU_SCALAR_MAP_MAXSIZE)) {
      PT_LAZY_DEBUG("scalar_to_tensor_map cleared");
      scalar_to_tensor_map.clear();
    }
    hb_tensors_out_view.clear();
  }

  size_t viewTableSize() const {
    size_t size = sizeof(view_table);
    size += sizeof(decltype(view_table)::key_type) * view_table.size();

    for (auto const& entry : view_table) {
      size += entry.second.Size();
    }
    return size;
  }

  size_t tensorMapSize() const {
    size_t size = sizeof(orig_tensor_map);
    size += orig_tensor_map.size() *
        (sizeof(decltype(orig_tensor_map)::key_type) +
         sizeof(decltype(orig_tensor_map)::mapped_type));

    return size;
  }

  // We want to retain some tensors for special cases where PT releases them
  // but because we are in lazy mode we actually need them for processing
  // later This should only be used in special cases and released on exit
  // cleanly
  std::vector<at::Tensor> m_retained_tensor_list;

  bool m_is_cached = false;

  std::unordered_map<
      std::pair<float, at::ScalarType>,
      at::Tensor,
      HashFn,
      EqualFn>
      scalar_to_tensor_map;

  std::vector<std::pair<at::Tensor, at::Tensor>> copy_scalar_to_hpu_tensor_list;

  // maps tensor id corresponding to as_strided's o/p with its i/p stride params
  std::unordered_map<int64_t, StrideParams> view_table;
  // maintains most recent version of the original tensor map
  std::unordered_map<int64_t, at::Tensor> orig_tensor_map;

  // view tensors that occurs as graph outputs
  std::vector<habana_lazy::HbLazyTensor> hb_tensors_out_view;

 private:
  GraphPtr mp_g;
  ir::ValueList m_input_vals;
  ir::ValueList m_output_vals;
  std::vector<habana_lazy::HbLazyTensor> m_hblazy_tensors;
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

 private:
  // Keep a map of all the execution contexts in play
  // Right now we support  a single context per device, map maintains ID to
  // context map
  std::unordered_map<int, HbExecutionContext*> m_execution_context_list;
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
