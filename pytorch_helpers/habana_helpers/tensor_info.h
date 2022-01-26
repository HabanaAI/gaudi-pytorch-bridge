/******************************************************************************
 * Copyright (C) 2020 Habana Labs, Ltd. an Intel Company
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

#include <iostream>
#include <string>

#include <ATen/Tensor.h>
#include <torch/csrc/jit/ir/ir.h>

#include "habana_helpers/logging.h"
#include "habana_helpers/misc_utils.h"
#include "habana_helpers/tensor_utils.h"

#include "synapse_helpers/device_types.h"
#include "synapse_helpers/graph.h"
#include "synapse_helpers/habana_tensor.h"

class PtTensorInfo;

typedef void (
    *getDMAInputTensorCBType)(const PtTensorInfo& ti, at::Tensor& dma_tensor);

using PtTensorInfoShared = std::shared_ptr<PtTensorInfo>;

class PtTensorInfo {
 public:
  PtTensorInfo(const IValPtrShared& ivpsh);
  PtTensorInfo(const synapse_helpers::tensor& st, const std::string& irn);
  PtTensorInfo(
      const IValPtrShared& ivp,
      const std::string& sn,
      const ValPtr& vp,
      const bool wflag,
      const uint64_t tensor_id,
      const synTensorType stt = DATA_TENSOR,
      const getDMAInputTensorCBType dma_cb = nullptr);
  PtTensorInfo(
      const at::Tensor& pt_tensor,
      const std::string& sn,
      const std::string& irn,
      const bool wflag,
      const uint64_t tensor_id,
      const synTensorType stt = DATA_TENSOR,
      const getDMAInputTensorCBType dma_cb = nullptr);

  PtTensorInfo(std::istream& is);
  // access functions for read write data members
  void* get_buffer() const {
    return buffer_;
  }
  uint64_t get_buffer_syn() const {
    return reinterpret_cast<synapse_helpers::device_ptr>(buffer_);
  }
  void set_buffer(void* bp) {
    buffer_ = bp;
  }

  void* get_buffer_start() const {
    return buffer_start_;
  }
  synapse_helpers::device_ptr get_buffer_start_syn() const {
    return reinterpret_cast<synapse_helpers::device_ptr>(buffer_start_);
  }

  bool is_duplicate() const {
    return is_duplicate_;
  }
  void set_duplicate_flag(bool b) {
    is_duplicate_ = b;
  }

  size_t get_parent_index() const {
    return parent_index_;
  }
  void set_parent_index(size_t i) {
    parent_index_ = i;
  }

  bool is_output() const {
    return (output_index_ != ULONG_MAX);
  }
  size_t get_output_index() const {
    return output_index_;
  }
  void set_output_index(size_t i) {
    output_index_ = i;
  }
  bool is_restrided() const {
    return is_restrided_;
  }
  void set_restrided(bool flag = true) {
    is_restrided_ = flag;
  }

  // The following patch functions need to be used for patching.
  // Note :
  //   For inputs both storage and data ptrs are updated.
  //   For the rest of the tensors offset will be used to calculate the buffer.
  void patch_exact(const at::Tensor& pt_tensor) {
    buffer_ = pt_tensor.data_ptr();
    buffer_start_ = pt_tensor.storage().data_ptr().get();
    auto new_offset = get_buffer_syn() - get_buffer_start_syn();
    TORCH_CHECK(
        offset_ == new_offset,
        "offset_ ",
        offset_,
        "is not matching with the offset of new tensor ",
        new_offset);
  }
  void patch(const PtTensorInfo& t) {
    buffer_start_ = t.buffer_start_;
    buffer_ = (void*)(get_buffer_start_syn() + offset_);
  }
  void patch(const at::Tensor& pt_tensor) {
    buffer_start_ = pt_tensor.storage().data_ptr().get();
    buffer_ = (void*)(get_buffer_start_syn() + offset_);
  }

  friend std::ostream& operator<<(std::ostream& O, const PtTensorInfo& t);

  // access functions for read only data members
  bool is_view_tensor() const {
    return is_view_tensor_;
  }
  const std::string& get_ir_name() const {
    return ir_name_;
  }
  void set_ir_name(std::string& n) {
    ir_name_ = n;
  }
  const std::string& get_syn_name() const {
    return syn_name_;
  }
  const char* get_syn_namec_str() const {
    return syn_name_.c_str();
  }
  unsigned get_numel() const {
    return numel_;
  }
  unsigned get_size() const {
    return size_;
  }
  bool watch_enabled() const {
    return watch_;
  }
  synapse_helpers::device_ptr get_offset() const {
    return offset_;
  }
  void set_offset(synapse_helpers::device_ptr val) {
    offset_ = val;
  }

  void set_shape(const std::vector<int64_t>& shape) {
    shape_ = shape;
    // For IDST the size/numel_ can be zero, avoid division by zero
    auto itemsize = (numel_) ? size_ / numel_ : 0;
    numel_ = 1;
    for (const auto& i : shape) {
      numel_ *= i;
    }
    size_ = numel_ * itemsize;
    update_shape_syn();
  }

  void set_strides(const std::vector<int64_t>& strides) {
    strides_ = strides;
  }

  const std::vector<int64_t>& get_shape() const {
    return shape_;
  };
  const std::vector<int64_t>& get_strides() const {
    return strides_;
  };
  const c10::TensorOptions& get_topts() const {
    return topts_;
  }
  const c10::MemoryFormat& get_mf() const {
    return mf_;
  }
  synTensorType tensor_type() const {
    return tensor_type_;
  }
  const std::array<uint32_t, SYN_GAUDI_MAX_TENSOR_DIM>& syn_shape() {
    return syn_shape_;
  }

  size_t get_dma_tensor_idx() const {
    return dma_tensor_idx_;
  }
  void set_dma_tensor_idx(size_t i) {
    dma_tensor_idx_ = i;
  }
  getDMAInputTensorCBType get_dma_cb() const {
    return dma_cb_;
  }
  habana_lazy::LayoutFormat getHbInternalLayoutFormat() const {
    return hb_internal_lf_;
  }

  bool is_ZST() const {
    return is_ZST_;
  }

  void Serialize(std::ostream& os) const;

  uint64_t get_tensor_id() const {
    return tensor_id_;
  }

  void set_external(bool external) {
    is_external_ = external;
  }

  bool get_external() const {
    return is_external_;
  }

  void set_host_ptr(void* host_ptr) {
    host_ptr_ = host_ptr;
  }

  uint64_t get_host_ptr() const {
    return reinterpret_cast<uint64_t>(host_ptr_);
  }

 private:
  bool is_ZST_{false};
  bool is_view_tensor_{false};
  bool is_restrided_{false};
  bool is_external_{false};

  void* buffer_{nullptr};
  void* buffer_start_{nullptr};
  // offset is used for view tensor only
  synapse_helpers::device_ptr offset_{0};
  std::string ir_name_;
  std::string syn_name_;

  unsigned numel_{0};
  unsigned size_{0};

  // Will hold the index of parent tensor info for aliases
  bool is_duplicate_{false};
  size_t parent_index_{ULONG_MAX};
  size_t output_index_{ULONG_MAX};
  bool watch_ = false;

  std::vector<int64_t> shape_;
  std::vector<int64_t> strides_;

  c10::MemoryFormat mf_;
  c10::TensorOptions topts_;

  habana_lazy::LayoutFormat hb_internal_lf_{habana_lazy::LayoutFormat::kNCHW};

  synTensorType tensor_type_{DATA_TENSOR};
  std::array<uint32_t, SYN_GAUDI_MAX_TENSOR_DIM> syn_shape_{0};
  // uint64_t shape_ndim_{0};

  size_t dma_tensor_idx_{ULONG_MAX};
  getDMAInputTensorCBType dma_cb_{nullptr};
  uint64_t tensor_id_{synapse_helpers::INVALID_SYN_TENSOR_ID};

  void* host_ptr_{nullptr};

  void populate_tinfo(
      const at::Tensor& pt_tensor,
      const std::string& irn,
      const std::string& sn,
      const bool wflag,
      const uint64_t tensor_id,
      const synTensorType stt,
      const getDMAInputTensorCBType dma_cb);
  void update_shape_syn();
};
