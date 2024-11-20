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

#include <ATen/Tensor.h>
#include <c10/core/DefaultDtype.h>
#include <c10/core/Device.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorImpl.h>
#include <c10/macros/Macros.h>
#include <c10/util/Optional.h>
#include "backend/backend_meta.h"
#include "backend/helpers/layout.h"
#include "backend/helpers/tensor_utils.h"
#include "backend/synapse_helpers/layout_utils.h"
#include "hpu_lazy_tensors.h"

namespace habana_lazy {

// Tensor implementation class used to be fed to the at::Tensor.
// Its scope is just to handle an HbLazyTensor.
// While creating PT tensors, we need to connect to HbLazyTensors
// HbLazyTensors are created with this backend which helps in memory and
// lifetime management
class HbLazyTensorImpl : public c10::TensorImpl {
 public:
  HbLazyTensorImpl(HbLazyTensor hb_tensor);
  HbLazyTensorImpl(
      const HbLazyTensor& hb_tensor,
      c10::Storage&& tensor_storage);
  HbLazyTensorImpl(HbLazyTensor&& hb_tensor, c10::Storage&& tensor_storage);
  HbLazyTensorImpl(
      HbLazyTensor&& hb_tensor,
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

#pragma GCC diagnostic push
// Following done as internal fork has different function signature than
// upstream, and clang16 fails on it.
// And GCC fails on unknown option
#pragma GCC diagnostic ignored "-Wpragmas"
#pragma GCC diagnostic ignored "-Winconsistent-missing-override"
  void set_storage_keep_dtype(at::Storage storage);
#pragma GCC diagnostic pop

  void handle_view_cycles(HbLazyTensor& hl_src, HbLazyTensor& hl_dst);

 private:
  void SetupSizeProperties();
  void SetStorage(at::Storage storage);
  void ComputeArrayStrides(
      c10::SmallVectorImpl<int64_t>& strides,
      c10::IntArrayRef sizes);

  bool m_size_initialized;

  HbLazyTensor m_tensor;
};

// Habana internal TensorImpl
class HbInternalTensorImpl : public c10::TensorImpl {
 public:
  HbInternalTensorImpl(
      c10::Storage&& tensor_storage,
      const caffe2::TypeMeta& data_type);

  // Shallow copy of compile_host_ptr_
  void set_compile_host_ptr(const HbInternalTensorImpl* impl) {
    get_tensor_extra_meta().clone_host_buffer_info(
        impl->get_ctensor_extra_meta());
  }

  // TODO [SW-127553] the following functions will be removed as we refactor
  // code further
  // Any new code should no longer assume HbInternalTensorImpl is present
  // use get_extra_tensor_data()->* directly

  bool isRedundant() {
    return get_ctensor_extra_meta().is_redundant();
  }

  void setRedundant() {
    get_tensor_extra_meta().set_redundant();
  }

  void* get_compile_host_ptr() const {
    return get_ctensor_extra_meta().get_compile_host_ptr();
  }

  habana::ShapeTensorStruct& get_shape_struct() {
    return get_tensor_extra_meta().get_shape_struct();
  }

  habana::LayoutFormat GetTensorLayout() const {
    return get_ctensor_extra_meta().get_tensor_layout();
  }

  void SetTensorLayout(habana::LayoutFormat layout) {
    get_tensor_extra_meta().set_tensor_layout(layout);
  }

  void setH2DFrontEndShapeTensor() {
    get_tensor_extra_meta().set_H2D_frontend_shape_tensor();
  }

  void setH2DDataForBucketing() {
    get_tensor_extra_meta().set_H2D_data_for_bucketing();
  }

  synapse_helpers::layouts::MemoryPermutation GetMemoryPermutation() const;

  void SetMemoryPermutation(
      synapse_helpers::layouts::MemoryPermutation permutation);

  bool isShapeTensor() const {
    return get_ctensor_extra_meta().is_shape_tensor();
  }

  void SetTensorSize(size_t size) {
    get_tensor_extra_meta().set_tensor_size(size);
  }

  c10::IntArrayRef GetTensorSize() const {
    return get_ctensor_extra_meta().get_tensor_size();
  }

  static void AtenInitialize();

  void SetIsConstTensor(bool is_const) {
    if (!habana_helpers::IsInferenceMode()) {
      return;
    }
    get_tensor_extra_meta().set_is_const_tensor(is_const);
  }

  bool IsConstTensor() const {
    if (!habana_helpers::IsInferenceMode()) {
      return false;
    }
    return get_ctensor_extra_meta().is_const_tensor();
  }

  const habana::TensorExtraMeta& get_ctensor_extra_meta() const {
    return *habana::get_ctensor_extra_meta(*this);
  }

  habana::TensorExtraMeta& get_tensor_extra_meta() {
    return *habana::get_tensor_extra_meta(*this);
  }
};

} // namespace habana_lazy
