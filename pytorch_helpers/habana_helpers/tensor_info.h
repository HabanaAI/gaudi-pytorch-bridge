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

#include <iostream>
#include <string>

#include <ATen/Tensor.h>
#include <torch/csrc/jit/ir/ir.h>

#include "habana_helpers/logging.h"
#include "synapse_helpers/habana_tensor.h"

#include "synapse_helpers/device_types.h"
#include "synapse_helpers/graph.h"

using IVal = torch::jit::IValue;
using IValPtrShared = std::shared_ptr<IVal>;
using ValPtr = torch::jit::Value*;

void PrintATenTensor(const at::Tensor& a);
void PrintATenTensor(const IVal& a);
void PrintATenTensor(const IValPtrShared& a);

class PtTensorInfo;

typedef void (
    *getDMAInputTensorCBType)(const PtTensorInfo& ti, at::Tensor& dma_tensor);

class PtTensorInfo {
 public:
  PtTensorInfo(const IValPtrShared& ivpsh);
  PtTensorInfo(const synapse_helpers::tensor& st, const std::string& irn);
  PtTensorInfo(
      const IValPtrShared& ivp,
      const std::string& sn,
      const ValPtr& vp,
      const bool wflag,
      const synTensorType stt = DATA_TENSOR,
      const getDMAInputTensorCBType dma_cb = nullptr);
  PtTensorInfo(
      const at::Tensor& pt_tensor,
      const std::string& sn,
      const std::string& irn,
      const bool wflag,
      const synTensorType stt = DATA_TENSOR,
      const getDMAInputTensorCBType dma_cb = nullptr);

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
  bool is_tensor() const {
    return is_tensor_;
  }
  bool is_view_tensor() const {
    return is_view_tensor_;
  }
  const IVal& get_ivalue() const {
    return iv_;
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
  const c10::MemoryFormat& get_mf() {
    return mf_;
  }
  synTensorType tensor_type() {
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

 private:
  bool is_tensor_{true};
  bool is_view_tensor_{false};
  bool is_restrided_{false};
  IVal iv_{};

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

  synTensorType tensor_type_{DATA_TENSOR};
  std::array<uint32_t, SYN_GAUDI_MAX_TENSOR_DIM> syn_shape_{0};
  // uint64_t shape_ndim_{0};

  size_t dma_tensor_idx_{ULONG_MAX};
  getDMAInputTensorCBType dma_cb_{nullptr};

  void populate_tinfo(
      const at::Tensor& pt_tensor,
      const std::string& irn,
      const std::string& sn,
      const bool wflag,
      const synTensorType stt,
      const getDMAInputTensorCBType dma_cb);
  void update_shape_syn();
};
