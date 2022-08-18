/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#pragma once

#include <tuple>
#include <utility>

#include "habana_lazy/aten_lazy_bridge.h"
#include "pytorch_helpers/synapse_helpers/env_flags.h"

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
  kStridedOpExpand
};

enum ViewStatus { kViewRead = 0, kViewWrite = 1, kEvaluated };

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

  size_t Size() const {
    size_t size = sizeof(*this);
    size += sizes.size() * sizeof(decltype(sizes)::value_type);
    size += strides.size() * sizeof(decltype(strides)::value_type);
    return size;
  }
};

class StridedViewContext {
 public:
  size_t viewTableSize() const {
    return m_view_table.size();
  }

  size_t viewTableBytes() const {
    size_t size = sizeof(m_view_table);
    size += sizeof(decltype(m_view_table)::key_type) * viewTableSize();

    for (auto const& entry : m_view_table) {
      size += entry.second.Size();
    }
    return size;
  }

  size_t tensorMapSize() const {
    return m_orig_tensor_map.size();
  }

  size_t tensorMapBytes() const {
    size_t size = sizeof(m_orig_tensor_map);
    size += tensorMapSize() *
        (sizeof(decltype(m_orig_tensor_map)::key_type) +
         sizeof(decltype(m_orig_tensor_map)::mapped_type));

    return size;
  }

  std::recursive_mutex& GetViewTableMutex() {
    return m_view_table_mtx;
  }

  void AddViewTableEntry(int64_t tensor_id, StrideParams params);
  void DelViewTableEntry(int64_t tensor_id);
  // std::optional<StrideParams> GetViewTableEntry(int64_t tensor_id);
  StrideParams* GetViewTableEntry(int64_t tensor_id);

  void AddOrigTensorMapEntry(int64_t tensor_id, at::Tensor tensor);
  void DelOrigTensorMapEntry(int64_t tensor_id);
  c10::optional<at::Tensor> GetOrigTensorMapEntry(int64_t tensor_id);

  // contains view tensors that are excluded from graph outputs
  std::vector<habana_lazy::HbLazyTensor> hb_tensors_exclude_out_view;
  bool isLazyViewPresent = false;

  void SetViewStatus(int64_t id, ViewStatus viewStatus);

 private:
  std::recursive_mutex m_view_table_mtx;

  // maps tensor id corresponding to as_strided's o/p with its i/p stride params
  std::unordered_map<int64_t, StrideParams> m_view_table;
  // maintains most recent version of the original tensor map
  std::unordered_map<int64_t, at::Tensor> m_orig_tensor_map;

 public:
  std::vector<HbLazyTensor> updated_bucket_list;
  std::set<int64_t> view_outputs;
};

class HbLazyTensorViews {
  /* Currently stateless, based on need in future can change access of
   * constructor */
 private:
  HbLazyTensorViews() {}

  static at::Tensor add_view_lazy(
      const at::Tensor& self,
      at::IntArrayRef size,
      c10::optional<at::Tensor> out_t);

  static at::Tensor add_slice_lazy(
      const at::Tensor& self,
      const StridedOpSliceParams& params,
      c10::optional<at::Tensor> out_t);

  static at::Tensor add_transpose_lazy(
      const at::Tensor& self,
      const StridedOpTransposeParams& params,
      c10::optional<at::Tensor> out_t);

  static at::Tensor add_t_lazy(
      const at::Tensor& self,
      c10::optional<at::Tensor> out_t);

  static at::Tensor add_permute_lazy(
      const at::Tensor& self,
      std::vector<int64_t> dims_vec,
      c10::optional<at::Tensor> out_t);

  static at::Tensor add_squeeze_unsqueeze_lazy(
      const at::Tensor& self,
      const int64_t dim,
      c10::optional<at::Tensor> out_t,
      std::string node_str);

  static at::Tensor add_expand_lazy(
      const at::Tensor& self,
      std::vector<int64_t> sizes,
      bool implicit,
      c10::optional<at::Tensor> out_t);

 public:
  static bool HandleViews(
      const at::Tensor& t,
      const habana_lazy::HbLazyTensor& hl_t);
  static habana_lazy::HbLazyTensor HandleViewsOrUpdate(
      const at::Tensor& t,
      habana_lazy::HbLazyTensor& hl_t);
  static at::Tensor HandleViewsD2H(const at::Tensor& t);
  static std::vector<at::Tensor> UpdateViewDistributed(
      std::vector<at::Tensor>&);
  static bool HandleViewsD2D(const at::Tensor& src, const at::Tensor& dst);
  static std::vector<at::Tensor> HandleViewsTensorList(const at::TensorList&);
  static at::Tensor add_strided_view_node(
      const at::Tensor& self,
      at::IntArrayRef size_in,
      at::IntArrayRef stride_in,
      int64_t storage_offset,
      bool is_update_view,
      c10::optional<at::Tensor> out,
      bool is_out = false);
  static void updateViewTable(at::Tensor& result, StrideParams& params);
  static StrideParams* getViewTableParams(HbLazyTensor& hl_view_t);
  static at::Tensor get_base_tensor(const at::Tensor& self);
  static const at::Tensor get_recent_base_tensor(const at::Tensor& self);
  static void CustomKernelAddNodeInplace(
      const at::Tensor& self,
      habana_lazy::ir::NodePtr node,
      int64_t& out_index);
  static void HandleViewsLiveTensors(
      HbContext* devctx,
      bool is_allreduce,
      std::set<int64_t>& bucket_recent_id);
  static void StepMarkerAllReduce(const std::vector<at::Tensor>& inputs);
};

at::Tensor add_strided_insert_node(
    const at::Tensor& orig_t,
    const at::Tensor& insert_t,
    at::IntArrayRef strides,
    int64_t offset,
    bool is_flush = true);

at::Tensor add_slice_insert_node(
    const at::Tensor& orig_t,
    const at::Tensor& insert_t,
    const std::vector<StridedOpSliceParams>& params);

bool is_aliased_view(HbLazyTensorImpl& self, HbLazyTensorImpl& other);
} // namespace habana_lazy
