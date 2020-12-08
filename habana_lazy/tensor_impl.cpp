/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "tensor_impl.h"
#include <c10/core/Device.h>
#include <c10/core/ScalarType.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include "habana_helpers/logging.h"

namespace habana_lazy {

caffe2::TypeMeta HbLazyTensorImpl::GetTypeMeta(const HbLazyTensor& hb_tensor) {
  return c10::scalarTypeToTypeMeta(hb_tensor.dtype());
}

// TODO : PT now needs two keys one for forward and one for autograd
// Need to check what to pass for autograd for Hb
HbLazyTensorImpl::HbLazyTensorImpl(HbLazyTensor hb_tensor)
    : c10::TensorImpl(
          c10::DispatchKeySet{c10::DispatchKey::HABANATensorId},
          c10::scalarTypeToTypeMeta(hb_tensor.dtype()),
          c10::make_optional(hb_tensor.GetDevice())),
      m_size_initialized(false),
      m_tensor(std::move(hb_tensor)) {
  const_cast<HbLazyTensorImpl*>(this)->SetupSizeProperties();
}

HbLazyTensorImpl::HbLazyTensorImpl(
    HbLazyTensor hb_tensor,
    c10::Storage&& tensor_storage)
    : c10::TensorImpl(
          std::move(tensor_storage),
          c10::DispatchKeySet{c10::DispatchKey::HABANATensorId}),
      m_size_initialized(false),
      m_tensor(std::move(hb_tensor)) {
  const_cast<HbLazyTensorImpl*>(this)->SetupSizeProperties();
}

void HbLazyTensorImpl::set_tensor(HbLazyTensor hb_tensor) {
  m_tensor = std::move(hb_tensor);
  const_cast<HbLazyTensorImpl*>(this)->SetupSizeProperties();
}

c10::intrusive_ptr<c10::TensorImpl> HbLazyTensorImpl::shallow_copy_and_detach(
    const c10::VariableVersion& version_counter,
    bool allow_tensor_metadata_change) const {
  auto impl = c10::make_intrusive<HbLazyTensorImpl>(m_tensor);
  copy_tensor_metadata(
      /*src_impl=*/this,
      /*dest_impl=*/impl.get(),
      /*version_counter=*/version_counter,
      /*allow_tensor_metadata_change=*/allow_tensor_metadata_change);
  impl.get()->SetupSizeProperties();
  return impl;
}

void HbLazyTensorImpl::shallow_copy_from(
    const c10::intrusive_ptr<TensorImpl>& impl) {
  // HABANA_ASSERT(0);
  HbLazyTensorImpl* hl_impl = dynamic_cast<HbLazyTensorImpl*>(impl.get());
  copy_tensor_metadata(
      /*src_impl=*/hl_impl,
      /*dest_impl=*/this,
      /*version_counter=*/version_counter(),
      /*allow_tensor_metadata_change=*/allow_tensor_metadata_change());
  hl_impl->m_tensor.ShallowCopyTo(&this->m_tensor);
  const_cast<HbLazyTensorImpl*>(this)->SetupSizeProperties();
}

at::IntArrayRef HbLazyTensorImpl::sizes() const {
  HABANA_ASSERT(m_size_initialized);
  return c10::TensorImpl::sizes();
}

int64_t HbLazyTensorImpl::dim() const {
  HABANA_ASSERT(m_size_initialized);
  return c10::TensorImpl::dim();
}

int64_t HbLazyTensorImpl::numel() const {
  HABANA_ASSERT(m_size_initialized);
  return c10::TensorImpl::numel();
}

bool HbLazyTensorImpl::is_contiguous(at::MemoryFormat memory_format) const {
  // Only check that the storage is already contiguous.
  HABANA_ASSERT(is_contiguous_);
  return true;
}

int64_t HbLazyTensorImpl::size(int64_t d) const {
  HABANA_ASSERT(m_size_initialized);
  return c10::TensorImpl::size(d);
}

std::vector<int64_t> HbLazyTensorImpl::ComputeArrayStrides(
    absl::Span<const int64_t> sizes) {
  std::vector<int64_t> strides(sizes.size(), 1);
  for (auto i = sizes.size(); i > 1; --i) {
    strides[i - 2] = strides[i - 1] * sizes[i - 1];
  }
  return strides;
}

void HbLazyTensorImpl::SetupSizeProperties() {
  if (!m_size_initialized) {
    // Fill up the basic dimension data members which the base class
    // implementation uses in its APIs.
    auto sizes = m_tensor.GetSizes();
    sizes_.clear();
    numel_ = 1;
    for (auto dim : sizes) {
      sizes_.push_back(dim);
      numel_ *= dim;
    }
    strides_.clear();
    for (auto stride : ComputeArrayStrides(sizes)) {
      strides_.push_back(stride);
    }
    m_size_initialized = true;
  }
}

const at::Storage& HbLazyTensorImpl::storage() const {
  // TBD: There maybe cases, specially from the scale out
  // side, where the tensor.storage().data() is used.
  // For this, we can later use the storage from m_tensor.
  std::cerr << "Habana Lazy tensors do not have storage";
  HABANA_ASSERT(0);
}

bool HbLazyTensorImpl::has_storage() const {
  return false;
}

void HbLazyTensorImpl::AtenInitialize() {
  // ATEN specific initialization calls placed below.
}

HbInternalTensorImpl::HbInternalTensorImpl(c10::Storage&& tensor_storage)
    : c10::TensorImpl(
          std::move(tensor_storage),
          c10::DispatchKeySet{c10::DispatchKey::HABANATensorId}) {}

void HbInternalTensorImpl::AtenInitialize() {
  // ATEN specific initialization calls placed below.
}
} // namespace habana_lazy