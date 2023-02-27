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

#include <c10/core/TensorImpl.h>
#include <synapse_common_types.h>
#include <tuple>

#include "backend/helpers/layout.h"
#include "backend/helpers/tensor_utils.h"
#include "backend/synapse_helpers/layout_utils.h"
#include "pytorch_helpers/habana_helpers/pt_version_check.h"

namespace habana {
// you may build with CXXFLAGS=-DHAVE_TORCH_BACKEND_META_SUPPORT=1 to excercise
// the new code path
#ifndef HAVE_TORCH_BACKEND_META_SUPPORT
#define HAVE_TORCH_BACKEND_META_SUPPORT IS_PYTORCH_FORK_AT_LEAST(1, 0)
#endif

#if HAVE_TORCH_BACKEND_META_SUPPORT
using BaseTensorExtraMeta = c10::BackendMeta;
#else
struct BaseTensorExtraMeta : c10::intrusive_ptr_target {
  virtual ~BaseTensorExtraMeta(){};
  virtual c10::intrusive_ptr<BaseTensorExtraMeta> clone(
      const c10::intrusive_ptr<BaseTensorExtraMeta>& ptr) const {
    return ptr;
  }
};
#endif
struct ShapeTensorStruct {
  bool contains_data = false;
  std::vector<int64_t> strides{};
  std::vector<int64_t> stride_ratio{};
  int64_t offset = 0;

  void set_strides_tensor_shape(std::vector<int64_t> input_strides) {
    contains_data = true;
    int len = input_strides.size();
    for (int i = 0; i < len; i++) {
      strides.push_back(input_strides[i]);
    }
  }

  void set_offset_tensor_shape(int64_t offset_value) {
    contains_data = true;
    offset = offset_value;
  }

  void set_stride_ratio(std::vector<int64_t> ratios) {
    contains_data = true;
    int len = ratios.size();
    for (int i = 0; i < len; i++) {
      stride_ratio.push_back(ratios[i]);
    }
  }

  bool has_shape_tensor_data() {
    return contains_data;
  }

  std::vector<int64_t> get_stride_ratios() {
    return stride_ratio;
  }

  int64_t get_offset() {
    return offset;
  }

  std::vector<int64_t> get_stride_shape() {
    return strides;
  }
};

enum class HostDataType {
  INVALID_T = 0,
  INT32_T = 1,
  UINT32_T = 2,
  UINT64_T = 3,
  FLOAT_T = 4
};

inline constexpr std::string_view to_string(const HostDataType& t) {
  switch (t) {
    case HostDataType::INVALID_T:
      return "INVALID";
    case HostDataType::INT32_T:
      return "INT32";
    case HostDataType::UINT32_T:
      return "UINT32";
    case HostDataType::UINT64_T:
      return "UINT64";
    case HostDataType::FLOAT_T:
      return "FLOAT";
  }
  return "<UNKNOWN_HOST_DATA_TYPE>";
}

inline std::ostream& operator<<(std::ostream& O, const HostDataType& t) {
  return O << to_string(t);
}

struct TensorExtraMeta : public BaseTensorExtraMeta {
  c10::intrusive_ptr<BaseTensorExtraMeta> clone(
      const c10::intrusive_ptr<BaseTensorExtraMeta>& ptr) const override {
    return ptr;
  }

  ~TensorExtraMeta() override;

  caffe2::TypeMeta get_type_meta(const at::Tensor& t);

  static void set_const_tensor(
      const at::Tensor& tensor,
      bool is_const_tensor,
      bool relax = false);
  void set_is_const_tensor(bool is_const_tensor) {
    is_const_tensor_ = is_const_tensor;
  }

  void set_tensor_size(c10::IntArrayRef s) {
    sizes_ = s;
  }

  c10::IntArrayRef get_tensor_size() const {
    return sizes_;
  }

  void set_tensor_type(c10::IntArrayRef s) {
    sizes_ = s;
  }

  habana::LayoutFormat get_tensor_layout() const {
    return tensor_layout_;
  }

  void set_tensor_layout(habana::LayoutFormat layout) {
    tensor_layout_ = layout;
  }

  bool is_data_in_host_memory() const {
    return is_data_in_host_memory_;
  }

  void set_data_in_host_memory(bool is_data_in_host_memory) {
    is_data_in_host_memory_ = is_data_in_host_memory;
  }

  bool is_const_tensor() const {
    return is_const_tensor_;
  }

  const synapse_helpers::layouts::MemoryPermutation& get_memory_permutation()
      const {
    return memory_permutation_;
  }

  void set_memory_permutation(
      synapse_helpers::layouts::MemoryPermutation permutation) {
    memory_permutation_ = permutation;
  }

  bool get_dont_allow_permutation() const {
    return dont_allow_permutation_;
  }

  void set_dont_allow_permutation(bool allow) {
    dont_allow_permutation_ = allow;
  }

  void set_tensor_type(synTensorType tensor_type) {
    tensor_type_ = tensor_type;
  }

  bool is_shape_tensor() const {
    return habana_helpers::is_shape_tensor(tensor_type_);
  }

  bool is_H2D_frontend_shape_tensor() const {
    return is_h2d_fe_shape_tensor_;
  }

  void set_H2D_frontend_shape_tensor() {
    is_h2d_fe_shape_tensor_ = true;
  }

  bool peek_H2D_data_for_bucketing() const {
    return is_h2d_bucketing_;
  }

  void set_H2D_data_for_bucketing() {
    is_h2d_bucketing_ = true;
  }

  synTensorType get_tensor_type() const {
    return tensor_type_;
  }

  void increase_permuted_counter() {
    permuted_counter_++;
  }

  unsigned get_permuted_counter() const {
    return permuted_counter_;
  }

  void set_host_data(void* d, int size, int ele_size, HostDataType dt_type);

  void set_redundant() {
    is_redundant_ = true;
  }

  bool is_redundant() const {
    return is_redundant_;
  }

  auto get_host_params() const {
    return std::make_tuple(
        host_ptr_, compile_host_ptr_, size_, el_size_, dt_type_);
  }

  void* get_host_ptr() const {
    return host_ptr_;
  }

  void set_host_ptr(void* host_ptr) {
    host_ptr_ = host_ptr;
  }

  void* get_compile_host_ptr() const {
    return compile_host_ptr_;
  }

  size_t get_host_size() const {
    return size_;
  }

  void set_host_size(size_t size) {
    size_ = size;
  }

  size_t get_host_total_elem() const {
    return total_elem_;
  }

  size_t get_host_el_size() const {
    return el_size_;
  }

  void set_host_el_size(size_t el_size) {
    el_size_ = el_size;
  }

  void set_host_dt_type(HostDataType dt_type) {
    dt_type_ = dt_type;
  }

  void set_compile_host_ptr(void* compile_host_ptr) {
    compile_host_ptr_ = compile_host_ptr;
  }

  HostDataType get_host_dt_type() const {
    return dt_type_;
  }

  ShapeTensorStruct& get_shape_struct() {
    return shape_tensor_struct_;
  }

  template <typename T>
  void get_host_data(std::vector<T>& data) {
    uint64_t host_ptr = reinterpret_cast<uint64_t>(host_ptr_);
    for (size_t i = 0; i < size_; ++i) {
      T* d = reinterpret_cast<T*>(host_ptr);
      data.emplace_back(*d);
      host_ptr += el_size_;
    }
  }

  template <typename T>
  void set_max(const std::vector<T>& d) {
    HABANA_ASSERT(d.size() == size_);
    HABANA_ASSERT(sizeof(T) == el_size_);
    size_t data_size = size_ * el_size_;
    memcpy(compile_host_ptr_, (void*)d.data(), data_size);
  }

  template <typename T>
  void set_min(const std::vector<T>& d) {
    HABANA_ASSERT(d.size() == size_);
    HABANA_ASSERT(sizeof(T) == el_size_);
    size_t data_size = size_ * el_size_;
    char* ptr = static_cast<char*>(compile_host_ptr_) + data_size;
    memcpy(ptr, (void*)d.data(), data_size);
  }

  int get_id() const {
    return id_;
  }

  void set_id(int id) {
    id_ = id;
  }

  void clone_host_buffer_info(const TensorExtraMeta& tmeta) {
    if (tmeta.get_compile_host_ptr()) {
      set_id(tmeta.get_id());
      set_host_size(tmeta.get_host_size());
      set_host_el_size(tmeta.get_host_el_size());
      total_elem_ = 2 * size_ * el_size_;
      set_host_dt_type(tmeta.get_host_dt_type());
      set_compile_host_ptr(tmeta.get_compile_host_ptr());
    }
  }

 private:
  c10::IntArrayRef sizes_{0};
  habana::LayoutFormat tensor_layout_{habana::LayoutFormat::NCHW};
  synTensorType tensor_type_{DATA_TENSOR};
  bool is_const_tensor_{false};
  bool is_data_in_host_memory_{false};

  // Memory permutation represents how tensor layout is set in memory
  synapse_helpers::layouts::MemoryPermutation memory_permutation_{};
  bool dont_allow_permutation_{false};
  unsigned permuted_counter_{0};
  bool is_h2d_fe_shape_tensor_{false};
  bool is_h2d_bucketing_{false};

  void* host_ptr_{nullptr};
  void* compile_host_ptr_{nullptr};
  size_t size_{0};
  size_t el_size_{0};
  HostDataType dt_type_{HostDataType::INVALID_T};
  ShapeTensorStruct shape_tensor_struct_{};
  bool is_redundant_ = false;
  int id_{-1};
  int total_elem_{0};
};

TensorExtraMeta* get_tensor_extra_meta_from_hb_internal_tensor_impl(
    at::TensorImpl& impl,
    [[maybe_unused]] bool relax);

TensorExtraMeta* allocate_tensor_extra_meta(at::TensorImpl& impl);

inline TensorExtraMeta* get_tensor_extra_meta(
    at::TensorImpl& impl,
    [[maybe_unused]] bool relax = false) {
#if HAVE_TORCH_BACKEND_META_SUPPORT
  auto meta{impl.get_backend_meta()};
  if (meta == nullptr)
    return allocate_tensor_extra_meta(impl);
  return reinterpret_cast<TensorExtraMeta*>(meta.get());
#else
  return get_tensor_extra_meta_from_hb_internal_tensor_impl(impl, relax);
#endif
}

inline const TensorExtraMeta* get_ctensor_extra_meta(
    const at::TensorImpl& impl,
    bool relax = false) {
  return const_cast<TensorExtraMeta*>(
      get_tensor_extra_meta(const_cast<at::TensorImpl&>(impl), relax));
}

inline TensorExtraMeta* get_tensor_extra_meta(
    const at::Tensor& tensor,
    bool relax = false) {
  auto impl{tensor.unsafeGetTensorImpl()};
  TORCH_CHECK(impl, "No impl");
  return get_tensor_extra_meta(*impl, relax);
}

} // namespace habana
