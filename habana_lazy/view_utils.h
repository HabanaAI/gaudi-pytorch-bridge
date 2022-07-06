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

  // maps tensor id corresponding to as_strided's o/p with its i/p stride params
  std::unordered_map<int64_t, StrideParams> view_table;
  // maintains most recent version of the original tensor map
  std::unordered_map<int64_t, at::Tensor> orig_tensor_map;

  // view tensors that occurs as graph outputs
  std::vector<habana_lazy::HbLazyTensor> hb_tensors_out_view;
  bool isLazyViewPresent = false;
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
      c10::optional<at::Tensor> out);
  static void updateViewTable(at::Tensor& result, StrideParams& params);
  static StrideParams& getViewTableParams(HbLazyTensor& hl_view_t);
  static at::Tensor get_base_tensor(const at::Tensor& self);
  static const at::Tensor& get_recent_base_tensor(const at::Tensor& self);
  static void CustomKernelAddNodeInplace(
      const at::Tensor& self,
      habana_lazy::ir::NodePtr node,
      int64_t& out_index);
};

at::Tensor add_strided_insert_node(
    const at::Tensor& orig_t,
    const at::Tensor& insert_t,
    at::IntArrayRef strides,
    int64_t offset,
    bool is_flush = true);

bool is_aliased_view(HbLazyTensorImpl& self, HbLazyTensorImpl& other);
} // namespace habana_lazy
