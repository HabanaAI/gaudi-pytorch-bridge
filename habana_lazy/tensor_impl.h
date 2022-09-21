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

#include <ATen/Tensor.h>
#include <c10/core/DefaultDtype.h>
#include <c10/core/Device.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorImpl.h>
#include <c10/macros/Macros.h>
#include <c10/util/Optional.h>
#include "habana_helpers/tensor_utils.h"
#include "hpu_lazy_tensors.h"
#include "pytorch_helpers/synapse_helpers/layout_utils.h"

namespace habana_lazy {

// Tensor implementation class used to be fed to the at::Tensor.
// Its scope is just to handle an HbLazyTensor.
// While creating PT tensors, we need to connect to HbLazyTensors
// HbLazyTensors are created with this backend which helps in memory and
// lifetime management
class HbLazyTensorImpl : public c10::TensorImpl {
 public:
  HbLazyTensorImpl(HbLazyTensor hb_tensor);
  HbLazyTensorImpl(HbLazyTensor hb_tensor, c10::Storage&& tensor_storage);
  HbLazyTensorImpl(
      HbLazyTensor hb_tensor,
      const c10::Storage& tensor_storage,
      c10::DispatchKeySet key_set);
  HbLazyTensor& tensor() {
    return m_tensor;
  }
  void set_tensor(HbLazyTensor hb_tensor);
  static void AtenInitialize();
  caffe2::TypeMeta GetTypeMeta(const HbLazyTensor& hb_tensor);

  c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach(
      const c10::VariableVersion& version_counter,
      bool allow_tensor_metadata_change) const override;

  c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach(
      c10::VariableVersion&& version_counter,
      bool allow_tensor_metadata_change) const override;

  void shallow_copy_from(const c10::intrusive_ptr<TensorImpl>& impl) override;

  at::IntArrayRef sizes_custom() const override;

  int64_t dim_custom() const override;

  int64_t numel_custom() const override;

  bool is_contiguous_custom(at::MemoryFormat memory_format) const override;

  inline int64_t compute_numel() const;

  const at::Storage& storage() const override;

  bool has_storage() const override;

 private:
  void SetupSizeProperties();
  void SetStorage(at::Storage storage);
  void ComputeArrayStrides(
      std::vector<int64_t>& strides,
      absl::Span<const int64_t> sizes);

  bool m_size_initialized;

  HbLazyTensor m_tensor;
};

enum class HostDataType {
  INVALID_T = 0,
  INT32_T = 1,
  UINT32_T = 2,
  UINT64_T = 3,
  FLOAT_T = 4
};

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

// Habana internal TensorImpl
class HbInternalTensorImpl : public c10::TensorImpl {
 public:
  HbInternalTensorImpl(
      c10::Storage&& tensor_storage,
      const caffe2::TypeMeta& data_type);

  ~HbInternalTensorImpl() {
    if (host_ptr_ || compile_host_ptr_) {
      auto& device = synapse_helpers::HPURegistrar::get_device();
      device.get_host_memory().free(host_ptr_);
      device.get_host_memory().free(compile_host_ptr_);
    }
  }

  static void AtenInitialize();
  caffe2::TypeMeta GetTypeMeta(const at::Tensor& t);

  LayoutFormat GetTensorLayout() const {
    return tensor_layout;
  }
  void SetTensorLayout(LayoutFormat layout) {
    tensor_layout = layout;
  }

  synapse_helpers::layouts::MemoryPermutation GetMemoryPermutation() const {
    return m_memory_permutation;
  }

  void SetMemoryPermutation(
      synapse_helpers::layouts::MemoryPermutation permutation) {
    m_memory_permutation = permutation;
  }

  bool GetDontAllowPermutation() const {
    return m_dont_allow_permutation;
  }

  void SetDontAllowPermutation(bool allow) {
    m_dont_allow_permutation = allow;
  }

  void setTensorType(synTensorType tensor_type) {
    m_tensor_type = tensor_type;
  }

  bool isShapeTensor() {
    return habana_helpers::is_shape_tensor(m_tensor_type);
  }

  bool isH2DFrontEndShapeTensor() {
    return m_is_h2d_fe_shape_tensor;
  }

  void setH2DFrontEndShapeTensor() {
    m_is_h2d_fe_shape_tensor = true;
  }

  synTensorType getTensorType() {
    return m_tensor_type;
  }

  void increasePermutedCounter() {
    m_permuted_counter++;
  }
  unsigned getPermutedCounter() const {
    return m_permuted_counter;
  }

  void set_host_data(void* d, int size, int ele_size, HostDataType dt_type);
  // void set_host_data(std::vector<int32_t> d);
  void* get_host_ptr() const;
  void* get_compile_host_ptr() const;
  size_t get_host_size() const;
  size_t get_host_el_size() const;
  HostDataType get_host_dt_type() const;

  template <typename T>
  void set_min(const std::vector<T>& d);
  template <typename T>
  void set_max(const std::vector<T>& d);
  template <typename T>
  void get_host_data(std::vector<T>& data);
  /*template <typename T>
  void set_min_max(const std::vector<T>& min, const std::vector<T>& max);*/
  ShapeTensorStruct& get_shape_struct();

 private:
  LayoutFormat tensor_layout = LayoutFormat::kNCHW;
  synTensorType m_tensor_type = DATA_TENSOR;

  // Memory permutation represents how tensor layout is set in memory
  synapse_helpers::layouts::MemoryPermutation m_memory_permutation;
  bool m_dont_allow_permutation = false;
  unsigned m_permuted_counter = 0;
  bool m_is_h2d_fe_shape_tensor = false;

  void* host_ptr_ = nullptr;
  void* compile_host_ptr_ = nullptr;
  size_t size_;
  size_t total_elem_;
  size_t el_size_;
  int id_;
  HostDataType dt_type_;
  ShapeTensorStruct shape_tensor_struct_;
};

} // namespace habana_lazy
