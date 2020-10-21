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
  static void AtenInitialize();
  caffe2::TypeMeta GetTypeMeta(const HbLazyTensor& hb_tensor);

  c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach(
      const c10::VariableVersion& version_counter,
      bool allow_tensor_metadata_change) const override;

  void shallow_copy_from(const c10::intrusive_ptr<TensorImpl>& impl) override;

 private:
  HbLazyTensor m_tensor;
};

// Habana internal TensorImpl
class HbInternalTensorImpl : public c10::TensorImpl {
 public:
  HbInternalTensorImpl(c10::Storage&& tensor_storage);
  at::Tensor* tensor() {
    return m_tensor;
  }
  void set_tensor(at::Tensor* t);
  static void AtenInitialize();
  caffe2::TypeMeta GetTypeMeta(const at::Tensor& t);

 private:
   at::Tensor* m_tensor;
};
} // namespace habana_lazy
