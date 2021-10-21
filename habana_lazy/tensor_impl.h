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
  HbLazyTensorImpl(HbLazyTensor hb_tensor, c10::Storage&& tensor_storage);
  HbLazyTensor& tensor() {
    return m_tensor;
  }
  void set_tensor(HbLazyTensor hb_tensor);
  void set_storage_tensor(at::Tensor internal_tensor);
  static void AtenInitialize();
  caffe2::TypeMeta GetTypeMeta(const HbLazyTensor& hb_tensor);

  c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach(
      const c10::VariableVersion& version_counter,
      bool allow_tensor_metadata_change) const override;

  c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach(
      c10::VariableVersion&& version_counter,
      bool allow_tensor_metadata_change) const override;

  void shallow_copy_from(const c10::intrusive_ptr<TensorImpl>& impl) override;

  at::IntArrayRef sizes() const override;

  int64_t dim() const override;

  int64_t numel() const override;

  bool is_contiguous(at::MemoryFormat memory_format) const override;

  int64_t size(int64_t d) const override;

  const at::Storage& storage() const override;

  bool has_storage() const override;

 private:
  void SetupSizeProperties();
  void SetStorage(at::Storage storage);
  std::vector<int64_t> ComputeArrayStrides(absl::Span<const int64_t> sizes);

  bool m_size_initialized;

  HbLazyTensor m_tensor;

  // The m_storage_tensor shares the storage with the internal tensor, when the
  // internal tensor is created upon the execution of this lazy tensor.
  // The storage object from this m_storage_tensor is then moved to the lazy
  // tensor so that outside callers of storage_ on the lazy tensor can get
  // access to the internal storage.
  at::Tensor m_storage_tensor;
};

// Habana internal TensorImpl
class HbInternalTensorImpl : public c10::TensorImpl {
 public:
  HbInternalTensorImpl(
      c10::Storage&& tensor_storage,
      const caffe2::TypeMeta& data_type);

  static void AtenInitialize();
  caffe2::TypeMeta GetTypeMeta(const at::Tensor& t);

  LayoutFormat GetTensorLayout() const {
    return tensor_layout;
  }
  void SetTensorLayout(LayoutFormat layout) {
    tensor_layout = layout;
  }

  void setShapeTensor(bool valid = true) {
    m_is_shape_tensor = valid;
  }

  bool isShapeTensor() {
    return m_is_shape_tensor;
  }

 private:
  LayoutFormat tensor_layout = LayoutFormat::kNCHW;
  bool m_is_shape_tensor = false;
};
} // namespace habana_lazy
