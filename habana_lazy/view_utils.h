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
#include "habana_lazy/lazy_executor.h"
#include "pytorch_helpers/synapse_helpers/env_flags.h"

namespace habana_lazy {

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
