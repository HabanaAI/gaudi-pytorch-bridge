/**
* Copyright (c) 2021-2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/
#pragma once

#include <tuple>
#include <utility>

#include "backend/synapse_helpers/env_flags.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/hpu_lazy_tensors.h"

namespace habana_lazy {
class StridedViewContext {
 public:
  // contains view tensors that are excluded from graph outputs
  std::vector<habana_lazy::HbLazyTensor> hb_tensors_exclude_out_view;
  bool isLazyViewPresent = false;

  void ReplaceViewBase(int64_t id, at::Tensor& new_base_t);

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

  static at::Tensor add_squeeze_dims_lazy(
      const at::Tensor& self,
      std::vector<int64_t> dims_vec,
      c10::optional<at::Tensor> out_t);

 public:
  static at::Tensor add_expand_lazy(
      const at::Tensor& self,
      std::vector<int64_t> sizes,
      bool implicit,
      c10::optional<at::Tensor> out_t);
  static bool HandleViews(
      const at::Tensor& t,
      const habana_lazy::HbLazyTensor& hl_t);
  static habana_lazy::HbLazyTensor HandleViewsOrUpdate(
      const at::Tensor& t,
      habana_lazy::HbLazyTensor& hl_t);
  static at::Tensor HandleViewsD2H(const at::Tensor& t);
  static std::vector<at::Tensor> UpdateViewDistributed(
      std::vector<at::Tensor>&);
  static void HandleViewsLazyCollective(const at::Tensor& t);
  static bool HandleViewsD2D(const at::Tensor& src, const at::Tensor& dst);
  static std::vector<at::Tensor> HandleViewsTensorList(const at::TensorList);
  static void add_strided_view_node_parallel_impl(
      const at::Tensor& self,
      at::IntArrayRef size_in,
      at::IntArrayRef stride_in,
      int64_t storage_offset,
      bool is_update_view,
      at::Tensor& out,
      bool is_out = false);
  static at::Tensor add_strided_view_node(
      const at::Tensor& self,
      at::IntArrayRef size_in,
      at::IntArrayRef stride_in,
      int64_t storage_offset,
      bool is_update_view,
      c10::optional<at::Tensor> out,
      bool is_out = false);
  static at::Tensor process_strided_view(
      const at::Tensor& self,
      at::IntArrayRef size_in,
      at::IntArrayRef stride_in,
      int64_t storage_offset,
      bool create_storage);
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
  static at::Tensor add_identity_lazy(
      const at::Tensor& self,
      c10::optional<at::Tensor> out_t);
  static std::vector<StridedOpSliceParams> getSliceInsertParams(
      const at::Tensor& recent_orig_t,
      const at::Tensor& recent_src_t,
      const StrideParams& params);
  static size_t updateViewHash(
      const habana_lazy::HbLazyTensor& hl_t,
      size_t hash);
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

} // namespace habana_lazy