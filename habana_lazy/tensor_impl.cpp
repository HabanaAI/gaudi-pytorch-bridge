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
#include "synapse_helpers/env_flags.h"

namespace habana_lazy {

caffe2::TypeMeta HbLazyTensorImpl::GetTypeMeta(const HbLazyTensor& hb_tensor) {
  return c10::scalarTypeToTypeMeta(hb_tensor.dtype());
}

// TODO : PT now needs two keys one for forward and one for autograd
// Need to check what to pass for autograd for Hb
HbLazyTensorImpl::HbLazyTensorImpl(HbLazyTensor hb_tensor)
    : c10::TensorImpl(
          c10::DispatchKeySet{
              at::DispatchKey::HPU,
              at::DispatchKey::AutogradHPU},
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
          c10::DispatchKeySet{
              at::DispatchKey::HPU,
              at::DispatchKey::AutogradHPU},
          c10::scalarTypeToTypeMeta(hb_tensor.dtype())),
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

c10::intrusive_ptr<c10::TensorImpl> HbLazyTensorImpl::shallow_copy_and_detach(
    c10::VariableVersion&& version_counter,
    bool allow_tensor_metadata_change) const {
  auto impl = c10::make_intrusive<HbLazyTensorImpl>(m_tensor);
  copy_tensor_metadata(
      /*src_impl=*/this,
      /*dest_impl=*/impl.get(),
      /*version_counter=*/std::move(version_counter),
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
  // HACK
  int64_t n = 1;
  for (const auto& i : sizes()) {
    n *= i;
  }
  return n;
  return c10::TensorImpl::numel();
}

bool HbLazyTensorImpl::is_contiguous(at::MemoryFormat memory_format) const {
  // Only check that the storage is already contiguous.
  // HABANA_ASSERT(is_contiguous_);
  return c10::TensorImpl::is_contiguous(memory_format);
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
    auto sizes_l = m_tensor.GetSizes();
    sizes_and_strides_.set_sizes(sizes_l);
    at::IntArrayRef new_stride = ComputeArrayStrides(sizes_l);
    const auto new_dim = sizes_l.size();
    if (new_dim > 0) {
      for (size_t dim = new_dim - 1;; dim--) {
        if (new_stride[dim] >= 0) {
          sizes_and_strides_.stride_at_unchecked(dim) = new_stride[dim];
        } else {
          // XXX: This behavior is surprising and may need to be removed to
          // support negative strides. Some pytorch functions rely on it:
          // for example, torch.cat (run TestTorch.test_cat_empty).
          if (dim == new_dim - 1) {
            sizes_and_strides_.stride_at_unchecked(dim) = 1;
          } else {
            // Keep stride monotonically increasing to match NumPy.
            sizes_and_strides_.stride_at_unchecked(dim) =
                std::max<int64_t>(
                    sizes_and_strides_.size_at_unchecked(dim + 1), 1) *
                sizes_and_strides_.stride_at_unchecked(dim + 1);
          }
        }
        if (dim == 0)
          break;
      }
    }
    m_size_initialized = true;
  }
}

void HbLazyTensorImpl::SetStorage(at::Storage storage) {
  storage_ = std::move(storage);
  device_opt_ = storage_.device();
}

const at::Storage& HbLazyTensorImpl::storage() const {
  // FIXME Violates const correctness
  c10::TensorImpl* impl = ((HbLazyTensor)m_tensor).getAttachedTensorImpl();
  // return a dummy storage if it isnt allocated yet
  // its a bit dangerous and we need to ensure storage calls are made only after
  // backend memory allocation for output tensors
  if (!impl)
    return storage_;
  const_cast<HbLazyTensorImpl*>(this)->SetStorage(impl->storage());
  return impl->storage();
}

bool HbLazyTensorImpl::has_storage() const {
  return storage_;
}

void HbLazyTensorImpl::AtenInitialize() {
  // ATEN specific initialization calls placed below.
}

HbInternalTensorImpl::HbInternalTensorImpl(
    c10::Storage&& tensor_storage,
    const caffe2::TypeMeta& data_type)
    : c10::TensorImpl(
          std::move(tensor_storage),
          c10::DispatchKeySet{
              at::DispatchKey::HPU,
              at::DispatchKey::AutogradHPU},
          data_type) {}
} // namespace habana_lazy
