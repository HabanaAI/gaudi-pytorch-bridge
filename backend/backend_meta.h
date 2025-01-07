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

#include <c10/core/TensorImpl.h>
#include <synapse_common_types.h>
#include <memory>
#include <string>
#include <tuple>

#include "backend/helpers/layout.h"
#include "backend/helpers/tensor_utils.h"
#include "backend/synapse_helpers/layout_utils.h"
#include "habana_helpers/habana_serialization/include/habana_serialization/const_section.h"
#include "pytorch_helpers/habana_helpers/pt_version_check.h"

namespace habana_lazy {
class HbInternalTensorImpl;
}

namespace habana {

using BaseTensorExtraMeta = c10::BackendMeta;

struct ShapeTensorStruct {
  bool contains_data = false;
  std::vector<int64_t> strides{};
  std::vector<int64_t> stride_ratio{};
  int64_t offset = 0;

  void set_strides_tensor_shape(std::vector<int64_t> input_strides) {
    contains_data = true;
    strides.clear();
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

struct StorageExtraMeta {
  const synapse_helpers::layouts::MemoryPermutation& get_memory_permutation()
      const {
    return memory_permutation_;
  }

  bool get_dont_allow_permutation() const {
    return dont_allow_permutation_;
  }

  void set_dont_allow_permutation(bool allow) {
    dont_allow_permutation_ = allow;
  }

  void set_memory_permutation(
      const synapse_helpers::layouts::MemoryPermutation& permutation) {
    // That strange condition has been added because we have data race between
    // main thread (which read) and launch thread (write). In launch thread we
    // have redundant calls to set permutation which we don't have to change.
    if (permutation != memory_permutation_)
      memory_permutation_ = permutation;
  }

  void set_base_tensor_size(std::vector<int64_t> s) {
    base_sizes_ = s;
  }

  std::vector<int64_t>& get_base_tensor_size() {
    return base_sizes_;
  }

 private:
  // Memory permutation represents how tensor layout is set in memory
  synapse_helpers::layouts::MemoryPermutation memory_permutation_{};
  bool dont_allow_permutation_{false};
  // view meta
  std::vector<int64_t> base_sizes_{};
};

StorageExtraMeta* get_storage_extra_meta(const at::Tensor& tensor);

StorageExtraMeta* get_storage_extra_meta(
    const c10::TensorImpl* tensor_impl,
    at::optional<size_t> nbytes = c10::nullopt,
    bool is_contiguous = true);

StorageExtraMeta* get_storage_base_meta(const at::Tensor& tensor);

/* Decides if the tensor needs to be lowered in the eager jit ir pass
cases in which it is lowered 1. View tensor 2. Not marked as a grad view 3. have
a valid permutation in storage meta or base meta*/
bool is_view_lowering(const at::Tensor& tensor);

/* Base tensor size can be different for the following cases:
1. View output with a non default permutation 2. View tensor whose base is
having non default permutation 3. 1D for other cases*/
std::vector<int64_t> get_base_tensor_size(const at::Tensor& tensor);

enum class ParamType { INVALID = 0, WEIGHT = 1, BIAS = 2, OTHERS = 3 };

inline constexpr std::string_view to_string(const ParamType& t) {
  switch (t) {
    case ParamType::INVALID:
      return "INVALID";
    case ParamType::WEIGHT:
      return "WEIGHT";
    case ParamType::BIAS:
      return "BIAS";
    case ParamType::OTHERS:
      return "OTHERS";
  }
  return "<UNKNOWN_PARAM_TYPE>";
}

struct SendTensorMeta {
 public:
  SendTensorMeta(bool isPermuted, const at::Tensor& tensor)
      : isPermuted_(isPermuted), tensor_(tensor) {}

  const bool& isPermuted() const {
    return isPermuted_;
  }

  const at::Tensor& getTensor() const {
    return tensor_;
  }

 private:
  const bool isPermuted_;
  const at::Tensor tensor_;
};

static constexpr int INVALID_CONST_ID = -1;
static constexpr size_t INVALID_CHECKSUM = 0;
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

  void set_nbytes_inference(size_t _bytes) {
    nbytes_inference_ = _bytes;
  }

  at::optional<size_t> get_nbytes_inference() const {
    return nbytes_inference_;
  }

  bool has_nbytes_inference_valid() const {
    return nbytes_inference_.has_value();
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

  bool has_valid_const_id() const {
    return (const_id_ != INVALID_CONST_ID);
  }

  bool has_valid_checksum() const {
    return (host_checksum_ != INVALID_CHECKSUM);
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
  void update_host_data(void* d, int size, int el_size, bool compile = true);

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

  std::shared_ptr<serialization::ConstSectionDataSerialize>
  get_const_section_data_serializer();

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

  int get_const_id() const {
    return const_id_;
  }

  void set_const_id(int id) {
    const_id_ = id;
  }

  size_t get_host_checksum() const {
    return host_checksum_;
  }

  void set_host_checksum(size_t _checksum) {
    host_checksum_ = _checksum;
  }

  void clone_host_buffer_info(const TensorExtraMeta& tmeta) {
    if (tmeta.get_compile_host_ptr()) {
      set_id(tmeta.get_id());
      set_const_id(tmeta.get_const_id());
      set_host_size(tmeta.get_host_size());
      set_host_el_size(tmeta.get_host_el_size());
      total_elem_ = 2 * size_ * el_size_;
      set_host_dt_type(tmeta.get_host_dt_type());
      set_compile_host_ptr(tmeta.get_compile_host_ptr());
    }
  }

  // view related member functions
  bool is_view_tensor() const {
    return is_view_;
  }

  void set_view_tensor() {
    is_view_ = true;
  }

  bool is_maybe_grad_view() {
    return is_maybe_grad_view_;
  }

  void set_maybe_grad_view() {
    is_maybe_grad_view_ = true;
  }

  bool is_tensor_pipelined() const {
    return is_tensor_pipelined_;
  }

  void set_tensor_pipelined() {
    is_tensor_pipelined_ = true;
  }

  void set_send_org_tensor_meta(bool isPermute, const at::Tensor& src) {
    HABANA_ASSERT(
        send_tensor_meta_ == nullptr, "Send tensor meta is already set");
    send_tensor_meta_ = std::make_shared<SendTensorMeta>(isPermute, src);
  }

  bool is_send_org_tensor_permuted() const {
    if (send_tensor_meta_ == nullptr) {
      return false;
    }
    return send_tensor_meta_->isPermuted();
  }

  std::unique_ptr<at::Tensor> get_send_org_tensor() const {
    if (send_tensor_meta_ == nullptr) {
      return nullptr;
    }
    if (send_tensor_meta_->isPermuted()) {
      return nullptr;
    }
    return std::make_unique<at::Tensor>(send_tensor_meta_->getTensor());
  }

 private:
  c10::IntArrayRef sizes_{0};
  habana::LayoutFormat tensor_layout_{habana::LayoutFormat::NCHW};
  synTensorType tensor_type_{DATA_TENSOR};
  bool is_const_tensor_{false};
  bool is_data_in_host_memory_{false};

  unsigned permuted_counter_{0};
  bool is_h2d_fe_shape_tensor_{false};
  bool is_h2d_bucketing_{false};

  void* host_ptr_{nullptr};
  void* compile_host_ptr_{nullptr};
  std::shared_ptr<serialization::ConstSectionDataSerialize> const_section_data_;
  size_t size_{0};
  size_t el_size_{0};
  at::optional<size_t> nbytes_inference_;
  HostDataType dt_type_{HostDataType::INVALID_T};
  ShapeTensorStruct shape_tensor_struct_{};
  bool is_redundant_ = false;
  size_t host_checksum_{INVALID_CHECKSUM};
  int id_{-1};
  int const_id_{INVALID_CONST_ID};
  int total_elem_{0};
  // view meta
  bool is_view_{false};
  bool is_maybe_grad_view_{false};

  bool is_tensor_pipelined_{false};
  std::shared_ptr<SendTensorMeta> send_tensor_meta_{nullptr};
};

TensorExtraMeta* get_tensor_extra_meta_from_hb_internal_tensor_impl(
    at::TensorImpl& impl,
    [[maybe_unused]] bool relax);

TensorExtraMeta* allocate_tensor_extra_meta(at::TensorImpl& impl);

inline TensorExtraMeta* get_tensor_extra_meta(
    at::TensorImpl& impl,
    [[maybe_unused]] bool relax = false) {
  auto meta{impl.get_backend_meta()};
  if (meta == nullptr)
    return allocate_tensor_extra_meta(impl);
  return reinterpret_cast<TensorExtraMeta*>(meta);
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

inline bool is_tensor_const(const at::TensorImpl& impl, bool relax = false) {
  if (!habana_helpers::IsInferenceMode()) {
    return false;
  }
  return get_ctensor_extra_meta(impl, relax)->is_const_tensor();
}

inline bool is_tensor_const(const at::Tensor& tensor, bool relax = false) {
  if (!habana_helpers::IsInferenceMode()) {
    return false;
  }
  return get_tensor_extra_meta(tensor, relax)->is_const_tensor();
}

inline int get_tensor_const_id(const at::TensorImpl& impl, bool relax = false) {
  if (!habana_helpers::IsInferenceMode()) {
    return INVALID_CONST_ID;
  }
  return get_ctensor_extra_meta(impl, relax)->get_const_id();
}

inline int get_tensor_const_id(const at::Tensor& tensor, bool relax = false) {
  if (!habana_helpers::IsInferenceMode()) {
    return INVALID_CONST_ID;
  }
  return get_tensor_extra_meta(tensor, relax)->get_const_id();
}

inline bool is_tensor_const_with_valid_const_id(
    const at::TensorImpl& impl,
    bool relax = false) {
  if (!habana_helpers::IsInferenceMode()) {
    return false;
  }
  auto tmeta = get_ctensor_extra_meta(impl, relax);
  if (tmeta->is_const_tensor()) {
    HABANA_ASSERT(
        tmeta->has_valid_const_id(),
        "Constant tensor does not have a valid constant id");
    return true;
  }
  return false;
}

inline bool is_tensor_const_with_valid_const_id(
    const at::Tensor& tensor,
    bool relax = false) {
  if (!habana_helpers::IsInferenceMode()) {
    return false;
  }
  auto tmeta = get_tensor_extra_meta(tensor, relax);
  if (tmeta->is_const_tensor()) {
    HABANA_ASSERT(
        tmeta->has_valid_const_id(),
        "Constant tensor does not have a valid constant id");
    return true;
  }
  return false;
}

inline void set_tensor_const(
    const at::Tensor& tensor,
    bool is_const,
    int const_id,
    bool relax = false) {
  if (!habana_helpers::IsInferenceMode()) {
    return;
  }
  auto tmeta = get_tensor_extra_meta(tensor, relax);
  tmeta->set_is_const_tensor(is_const);
  if (is_const) {
    PT_BRIDGE_DEBUG(
        "set_tensor_const: is_const ", is_const, " const_id: ", const_id);
    HABANA_ASSERT(
        const_id != INVALID_CONST_ID,
        "Const id cannot be ",
        INVALID_CONST_ID,
        " for constant tensors")
    HABANA_ASSERT(
        tmeta->get_const_id() == INVALID_CONST_ID ||
            tmeta->get_const_id() == const_id,
        "Constant id already set for the tensor")
    tmeta->set_const_id(const_id);
  }
}

inline void get_and_set_tensor_const(
    const at::Tensor& tensor_src,
    const at::Tensor& tensor,
    bool relax = false) {
  if (!habana_helpers::IsInferenceMode()) {
    return;
  }
  auto tmeta_src = get_tensor_extra_meta(tensor_src, relax);
  auto is_src_const = tmeta_src->is_const_tensor();
  auto src_const_id = tmeta_src->get_const_id();
  habana::set_tensor_const(tensor, is_src_const, src_const_id);
}

inline void get_and_set_tensor_const(
    const at::TensorImpl& impl,
    const at::Tensor& tensor,
    bool relax = false) {
  if (!habana_helpers::IsInferenceMode()) {
    return;
  }
  auto is_src_const = get_ctensor_extra_meta(impl, relax)->is_const_tensor();
  get_tensor_extra_meta(tensor, relax)->set_is_const_tensor(is_src_const);
  auto src_const_id = get_ctensor_extra_meta(impl, relax)->get_const_id();
  get_tensor_extra_meta(tensor, relax)->set_const_id(src_const_id);
}

} // namespace habana
