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

#include <ATen/Tensor.h>
#include <torch/csrc/jit/ir/ir.h>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include "backend/helpers/layout.h"
#include "backend/helpers/tensor_utils.h"
#include "backend/synapse_helpers/device_types.h"
#include "backend/synapse_helpers/habana_tensor.h"
#include "habana_helpers/logging.h"

class PtTensorInfo;

typedef void (
    *getDMAInputTensorCBType)(const PtTensorInfo& ti, at::Tensor& dma_tensor);

enum class DMAInputGeneratorType { INVALID, SEEDTENSOR, MAX };

// This structure enables serialization of Recipes for DiskCache.
// A Tensor may have corresponding Generator associated with it.
// This structure holds all the Generator functions.
// While serialization we'll dump the enum associated with its Generator
// function. While deserialization we'll use the enum to get back the original
// Generator function.
struct DMAInputGenerators {
  static getDMAInputTensorCBType getGenerator(DMAInputGeneratorType id) {
    switch (id) {
      case DMAInputGeneratorType::SEEDTENSOR:
        return populateSeedTensor;
        break;
      default:
        return nullptr;
    }
  }

  static void populateSeedTensor(
      const PtTensorInfo& ti,
      at::Tensor& dma_tensor);

  // Add any future generators here and link it with its own entry in
  // DMAInputGeneratorType
};

using PtTensorInfoShared = std::shared_ptr<PtTensorInfo>;

class PtTensorInferenceData {
 public:
  static PtTensorInferenceData& get_instance() {
    static PtTensorInferenceData instance_;
    return instance_;
  }
  using InferenceRangePair = std::pair<float, float>;
  void SetInferenceTensorRange(std::string tensor_name, float min, float max) {
    inference_tensor_map.emplace(tensor_name, std::make_pair(min, max));
  }

  InferenceRangePair GetInferenceTensorRange(
      std::string tensor_name,
      bool& range_found) {
    if (exists(tensor_name)) {
      range_found = true;
      return inference_tensor_map[tensor_name];
    }
    range_found = false;
    return std::make_pair(-1, -1);
  }

  bool exists(std::string tensor_name) {
    bool ret_flag{false};
    if (inference_tensor_map.end() != inference_tensor_map.find(tensor_name)) {
      ret_flag = true;
    }

    return ret_flag;
  }
  void print_map();
  std::string scope_to_key(std::string src);
  void update_map(std::string src, std::string dst);

  void duplicate_key(std::string old, std::string now);
  std::string extract_key_name(
      std::string tensor_name,
      const std::string token);
  void update_entry(
      std::string tensor_name,
      float min,
      float max,
      bool align = false);

 private:
  std::unordered_map<std::string, std::pair<float, float>> inference_tensor_map;
};

class PtTensorInfo {
 public:
  PtTensorInfo(const IValPtrShared& ivpsh);
  PtTensorInfo(const synapse_helpers::tensor& st, const std::string& irn);
  PtTensorInfo(
      const IValPtrShared& ivp,
      const std::string& sn,
      const ValPtr& vp,
      const uint64_t tensor_id,
      const synTensor handle = nullptr,
      const synTensorType stt = DATA_TENSOR,
      DMAInputGeneratorType dma_gen_id = DMAInputGeneratorType::INVALID);
  PtTensorInfo(
      const at::Tensor& pt_tensor,
      const std::string& sn,
      const std::string& irn,
      const uint64_t tensor_id,
      const synTensor handle = nullptr,
      const synTensorType stt = DATA_TENSOR,
      DMAInputGeneratorType dma_gen_id = DMAInputGeneratorType::INVALID);
  PtTensorInfo(
      const std::string& sn,
      const uint64_t tensor_id,
      const synTensor handle = nullptr,
      const synTensorType stt = DATA_TENSOR,
      const std::vector<int64_t> shape = {});

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
  void patch_exact(
      const at::Tensor& pt_tensor,
      const bool shape_agnostic_flag = false) {
    buffer_ = pt_tensor.data_ptr();
    buffer_start_ = pt_tensor.storage().data_ptr().get();
    auto new_offset = get_buffer_syn() - get_buffer_start_syn();
    TORCH_CHECK(
        shape_agnostic_flag || offset_ == new_offset,
        "offset_ ",
        offset_,
        "is not matching with the offset of new tensor ",
        new_offset);
    // For shape agnostic flow storage offset can be different
    // store new offset for patching section offset for duplicates if any
    if (shape_agnostic_flag) {
      offset_ = new_offset;
    }
  }
  void patch(const PtTensorInfo& t, const bool shape_agnostic_flag = false) {
    buffer_start_ = t.buffer_start_;
    if (!shape_agnostic_flag) {
      buffer_ = (void*)(get_buffer_start_syn() + offset_);
    } else {
      buffer_ = t.buffer_;
      offset_ = t.offset_;
    }
  }
  void patch(
      const at::Tensor& pt_tensor,
      const bool shape_agnostic_flag = false) {
    buffer_start_ = pt_tensor.storage().data_ptr().get();
    if (!shape_agnostic_flag) {
      buffer_ = (void*)(get_buffer_start_syn() + offset_);
    } else {
      buffer_ = pt_tensor.data_ptr();
      offset_ = pt_tensor.storage_offset() * pt_tensor.itemsize();
    }
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
  uint64_t get_numel() const {
    return numel_;
  }
  uint64_t get_size() const {
    return size_;
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
  const std::array<uint32_t, SYN_GAUDI_MAX_TENSOR_DIM>& syn_shape() const {
    return syn_shape_;
  }

  size_t get_dma_tensor_idx() const {
    return dma_tensor_idx_;
  }
  void set_dma_tensor_idx(size_t i) {
    dma_tensor_idx_ = i;
  }
  getDMAInputTensorCBType get_dma_cb() const {
    return DMAInputGenerators::getGenerator(dma_gen_id_);
  }
  habana::LayoutFormat getHbInternalLayoutFormat() const {
    return hb_internal_lf_;
  }

  const synapse_helpers::layouts::MemoryPermutation& getHbInternalPermute()
      const {
    return hb_internal_perm_;
  }

  void setHbInternalPermute(
      const synapse_helpers::layouts::MemoryPermutation& permute) {
    hb_internal_perm_ = permute;
  }

  bool getHbDontAllowPermute() const {
    return hb_dont_allow_permute_;
  }

  void setHbDontAllowPermute(bool allow) {
    hb_dont_allow_permute_ = allow;
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

  bool get_allow_permutation() const {
    return is_allow_permutation_;
  }

  bool set_allow_permutation(bool allow_permutation) {
    return is_allow_permutation_ = allow_permutation;
  }

  size_t Size() const {
    size_t size = sizeof(*this);
    size += ir_name_.size() * sizeof(decltype(ir_name_)::value_type);
    size += syn_name_.size() * sizeof(decltype(syn_name_)::value_type);
    size += shape_.size() * sizeof(decltype(shape_)::value_type);
    size += strides_.size() * sizeof(decltype(strides_)::value_type);
    return size;
  }

  synTensor get_orig_syn_handle() const {
    return orig_syn_handle_;
  }

 private:
  bool is_ZST_{false};
  bool is_view_tensor_{false};
  bool is_restrided_{false};
  bool is_external_{false};
  bool is_allow_permutation_{false};

  void* buffer_{nullptr};
  void* buffer_start_{nullptr};
  // offset is used for view tensor only
  synapse_helpers::device_ptr offset_{0};
  std::string ir_name_;
  std::string syn_name_;

  uint64_t numel_{0};
  uint64_t size_{0};

  // Will hold the index of parent tensor info for aliases
  bool is_duplicate_{false};
  size_t parent_index_{ULONG_MAX};
  size_t output_index_{ULONG_MAX};

  std::vector<int64_t> shape_;
  std::vector<int64_t> strides_;

  c10::MemoryFormat mf_;
  c10::TensorOptions topts_;

  habana::LayoutFormat hb_internal_lf_{habana::LayoutFormat::NCHW};
  synapse_helpers::layouts::MemoryPermutation hb_internal_perm_;
  bool hb_dont_allow_permute_;

  synTensorType tensor_type_{DATA_TENSOR};
  std::array<uint32_t, SYN_GAUDI_MAX_TENSOR_DIM> syn_shape_{0};
  // uint64_t shape_ndim_{0};

  size_t dma_tensor_idx_{ULONG_MAX};
  uint64_t tensor_id_{synapse_helpers::INVALID_SYN_TENSOR_ID};

  void* host_ptr_{nullptr};
  DMAInputGeneratorType dma_gen_id_ = DMAInputGeneratorType::INVALID;

  synTensor orig_syn_handle_{nullptr};

  void populate_tinfo(
      const at::Tensor& pt_tensor,
      const std::string& irn,
      const std::string& sn,
      const uint64_t tensor_id,
      const synTensorType stt,
      DMAInputGeneratorType dma_gen_id);
  void update_shape_syn();
};

CREATE_OSTREAM_FORMATTER(PtTensorInfo)
