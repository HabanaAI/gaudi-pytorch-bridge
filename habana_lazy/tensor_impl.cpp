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
#include <torch/csrc/api/include/torch/version.h>
#include "aten_lazy_bridge.h"
#include "habana_helpers/logging.h"
#include "habana_lazy/lazy_executor.h"
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
    const HbLazyTensor& hb_tensor,
    c10::Storage&& tensor_storage)
    : c10::TensorImpl(
          std::move(tensor_storage),
          c10::DispatchKeySet{
              at::DispatchKey::HPU,
              at::DispatchKey::AutogradHPU},
          c10::scalarTypeToTypeMeta(hb_tensor.dtype())),
      m_size_initialized(false),
      m_tensor(hb_tensor) {
  const_cast<HbLazyTensorImpl*>(this)->SetupSizeProperties();
}

HbLazyTensorImpl::HbLazyTensorImpl(
    HbLazyTensor&& hb_tensor,
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

HbLazyTensorImpl::HbLazyTensorImpl(
    HbLazyTensor&& hb_tensor,
    const c10::Storage& tensor_storage,
    c10::DispatchKeySet key_set)
    : c10::TensorImpl(
          c10::TensorImpl::VIEW,
          c10::Storage(tensor_storage),
          key_set,
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
  auto aten_t = AtenFromHbLazyTensor(
      m_tensor, c10::nullopt, c10::nullopt, c10::nullopt, c10::nullopt);
  auto impl = c10::make_intrusive<HbLazyTensorImpl>(
      HbLazyTensor::Create(aten_t, aten_t.device()));

  copy_tensor_metadata(
      /*src_impl=*/this,
      /*dest_impl=*/impl.get(),
      /*version_counter=*/version_counter,
      /*allow_tensor_metadata_change=*/allow_tensor_metadata_change);
  this->m_tensor.ShallowCopyTo(&impl->m_tensor);
  impl.get()->SetupSizeProperties();
  impl->refresh_numel();
  impl->refresh_contiguous();
  return impl;
}

c10::intrusive_ptr<c10::TensorImpl> HbLazyTensorImpl::shallow_copy_and_detach(
    c10::VariableVersion&& version_counter,
    bool allow_tensor_metadata_change) const {
  auto aten_t = AtenFromHbLazyTensor(
      m_tensor, c10::nullopt, c10::nullopt, c10::nullopt, c10::nullopt);
  auto impl = c10::make_intrusive<HbLazyTensorImpl>(
      HbLazyTensor::Create(aten_t, aten_t.device()));

  copy_tensor_metadata(
      /*src_impl=*/this,
      /*dest_impl=*/impl.get(),
      /*version_counter=*/std::move(version_counter),
      /*allow_tensor_metadata_change=*/allow_tensor_metadata_change);
  this->m_tensor.ShallowCopyTo(&impl->m_tensor);
  impl.get()->SetupSizeProperties();
  impl->refresh_numel();
  impl->refresh_contiguous();
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
  const_cast<HbLazyTensorImpl*>(this)->refresh_numel();
  const_cast<HbLazyTensorImpl*>(this)->refresh_contiguous();
}

at::IntArrayRef HbLazyTensorImpl::sizes_custom() const {
  HABANA_ASSERT(m_size_initialized);
  return sizes_default();
}

int64_t HbLazyTensorImpl::dim_custom() const {
  HABANA_ASSERT(m_size_initialized);
  return dim_default();
}

int64_t HbLazyTensorImpl::numel_custom() const {
  HABANA_ASSERT(m_size_initialized);
  // HACK
  int64_t n = 1;
  for (const auto& i : sizes()) {
    n *= i;
  }
  return n;
  return numel_default();
}

bool HbLazyTensorImpl::is_contiguous_custom(
    at::MemoryFormat memory_format) const {
  // Only check that the storage is already contiguous.
  // HABANA_ASSERT(is_contiguous_);
  return is_contiguous_default(memory_format);
}

inline int64_t HbLazyTensorImpl::compute_numel() const {
  int64_t n = 1;
  for (const auto& i : sizes()) {
    n *= i;
  }
  return n;
}

void HbLazyTensorImpl::ComputeArrayStrides(
    SmallSizeVec& strides,
    absl::Span<const int64_t> sizes) {
  for (auto i = sizes.size(); i > 1; --i) {
    strides[i - 2] = strides[i - 1] * sizes[i - 1];
  }
}

void HbLazyTensorImpl::SetupSizeProperties() {
  if (!m_size_initialized) {
    // Fill up the basic dimension data members which the base class
    // implementation uses in its APIs.
    auto& sizes_l = m_tensor.GetSizes();
    sizes_and_strides_.set_sizes(sizes_l);
    SmallSizeVec new_stride(sizes_l.size(), 1);
    ComputeArrayStrides(new_stride, sizes_l);
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
            int64_t sizes = sizes_and_strides_.size_at_unchecked(dim + 1);

            // Keep stride monotonically increasing to match NumPy.
            sizes_and_strides_.stride_at_unchecked(dim) =
                std::max<int64_t>(sizes, 1) * sizes;
          }
        }
        if (dim == 0)
          break;
      }
    }
    m_size_initialized = true;
    // initialize numel at tensor impl constructor. This enables using numel
    // caching.
    numel_ = compute_numel();
  }
}

void HbLazyTensorImpl::SetStorage(at::Storage storage) {
  storage_ = std::move(storage);
  device_opt_ = storage_.device();
}

void HbLazyTensorImpl::set_storage_keep_dtype(at::Storage storage) {
  if (!GET_ENV_FLAG_NEW(PT_HPU_INFERENCE_STORAGE_OVERRIDE)) {
    TORCH_CHECK(
        allow_tensor_metadata_change(),
        "set_storage ",
        err_msg_tensor_metadata_change_not_allowed);
    storage_ = std::move(storage);
    device_opt_ = storage_.device();
    return;
  }

  TORCH_CHECK(
      allow_tensor_metadata_change(),
      "set_storage ",
      err_msg_tensor_metadata_change_not_allowed);

  device_opt_ = storage_.device();
  // storage is frontend and we are setting in frontend tensor's storage.
  if (storage.data_ptr() == nullptr) {
    storage_ = std::move(storage);
    PT_LAZY_DEBUG("set_storage_keep_dtype called with frontend storage.");
    return;
  } else { // We have backend storage to be set.
    auto aten_t = AtenFromHbLazyTensor(
        m_tensor, c10::nullopt, c10::nullopt, c10::nullopt, c10::nullopt);

    auto hl_t_opt = TryGetHbLazyTensor(aten_t, true, false, false);
    auto hl_t_updated = hl_t_opt.has_value() ? hl_t_opt.value() : m_tensor;
    std::lock_guard<std::recursive_mutex> lock(
        habana_lazy::HbContextArena::Get()->GetMutex());
    if (hl_t_updated.IsExecutionInProgress()) {
      auto context =
          habana_lazy::habana_lazy_executor.getDeviceExecutionContext(0);
      context->JoinPendingLaunchThread();
    }
    c10::TensorImpl* impl =
        ((HbLazyTensor)hl_t_updated).getAttachedTensorImpl();
    // At this point, execution thread is finished.
    if (impl) {
      impl->set_storage_keep_dtype(storage);
      PT_LAZY_DEBUG("set_storage_keep_dtype called with backend storage.");
    } else {
      storage_ = std::move(storage);
      PT_LAZY_DEBUG(
          "set_storage_keep_dtype called with backend storage, but impl in NULL");
    }
  }
}

const at::Storage& HbLazyTensorImpl::storage() const {
  // FIXME Violates const correctness
  // return a dummy storage if it isnt allocated yet
  // its a bit dangerous and we need to ensure storage calls are made only after
  // backend memory allocation for output tensors
  auto aten_t = AtenFromHbLazyTensor(
      m_tensor, c10::nullopt, c10::nullopt, c10::nullopt, c10::nullopt);

  auto hl_t_opt = TryGetHbLazyTensor(aten_t, true, false, false);
  auto hl_t_updated = hl_t_opt.has_value() ? hl_t_opt.value() : m_tensor;

  {
    // m_tensor is_executing is set to True at point where there is no execution
    // thread. is_executing is to False in execution thread, when tensor_data is
    // replaced in data. We acquire this lock so that data state doesn't change
    // in between.
    std::lock_guard<std::recursive_mutex> lock(
        habana_lazy::HbContextArena::Get()->GetMutex());
    if (!hl_t_updated.IsExecutionInProgress()) {
      c10::TensorImpl* impl =
          ((HbLazyTensor)hl_t_updated).getAttachedTensorImpl();
      if (impl && impl->storage() && !storage_.is_alias_of(impl->storage())) {
        const_cast<HbLazyTensorImpl*>(this)->SetStorage(
            c10::Storage(impl->storage()));
      }
    }
  }

  return storage_;
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

void HbInternalTensorImpl::set_host_data(
    void* d,
    int size,
    int el_size,
    HostDataType dt_type) {
  auto& device = synapse_helpers::HPURegistrar::get_device();
  id_ = device.id();
  int total_elem = 2 * size * el_size;
  int data_size = size * el_size;
  auto status = device.get_host_memory().malloc(&host_ptr_, total_elem);
  HABANA_ASSERT(status == synSuccess);
  status = device.get_host_memory().malloc(&compile_host_ptr_, total_elem);
  HABANA_ASSERT(status == synSuccess);
  memcpy(host_ptr_, d, data_size);
  char* ptr = static_cast<char*>(host_ptr_) + data_size;
  memcpy(ptr, d, data_size);
  memcpy(
      static_cast<char*>(compile_host_ptr_),
      static_cast<char*>(host_ptr_),
      total_elem);

  total_elem_ = total_elem;
  size_ = size;
  el_size_ = el_size;
  dt_type_ = dt_type;
}

void* HbInternalTensorImpl::get_host_ptr() const {
  return host_ptr_;
}

void* HbInternalTensorImpl::get_compile_host_ptr() const {
  return compile_host_ptr_;
}

void HbInternalTensorImpl::set_compile_host_ptr(
    const HbInternalTensorImpl* impl) {
  if (impl->compile_host_ptr_) {
    id_ = impl->id_;
    size_ = impl->get_host_size();
    el_size_ = impl->get_host_el_size();
    dt_type_ = impl->get_host_dt_type();
    total_elem_ = 2 * size_ * el_size_;
    compile_host_ptr_ = impl->compile_host_ptr_;
  }
}

size_t HbInternalTensorImpl::get_host_size() const {
  return size_;
}

size_t HbInternalTensorImpl::get_host_el_size() const {
  return el_size_;
}

HostDataType HbInternalTensorImpl::get_host_dt_type() const {
  return dt_type_;
}

ShapeTensorStruct& HbInternalTensorImpl::get_shape_struct() {
  return shape_tensor_struct_;
}

template <typename T>
void HbInternalTensorImpl::get_host_data(std::vector<T>& data) {
  uint64_t host_ptr = reinterpret_cast<uint64_t>(host_ptr_);
  for (size_t i = 0; i < size_; ++i) {
    T* d = reinterpret_cast<T*>(host_ptr);
    data.emplace_back(*d);
    host_ptr += el_size_;
  }
}

template void HbInternalTensorImpl::get_host_data(std::vector<int64_t>& data);
template void HbInternalTensorImpl::get_host_data(std::vector<uint64_t>& data);
template void HbInternalTensorImpl::get_host_data(std::vector<int32_t>& data);
template void HbInternalTensorImpl::get_host_data(std::vector<float>& data);

template <typename T>
void HbInternalTensorImpl::set_max(const std::vector<T>& d) {
  HABANA_ASSERT(d.size() == size_);
  HABANA_ASSERT(sizeof(T) == el_size_);
  size_t data_size = size_ * el_size_;
  memcpy(compile_host_ptr_, (void*)d.data(), data_size);
}

template <typename T>
void HbInternalTensorImpl::set_min(const std::vector<T>& d) {
  HABANA_ASSERT(d.size() == size_);
  HABANA_ASSERT(sizeof(T) == el_size_);
  size_t data_size = size_ * el_size_;
  char* ptr = static_cast<char*>(compile_host_ptr_) + data_size;
  memcpy(ptr, (void*)d.data(), data_size);
}

template void HbInternalTensorImpl::set_min(const std::vector<int64_t>& d);
template void HbInternalTensorImpl::set_min(const std::vector<uint64_t>& d);
template void HbInternalTensorImpl::set_min(const std::vector<int32_t>& d);
template void HbInternalTensorImpl::set_min(const std::vector<uint32_t>& d);
template void HbInternalTensorImpl::set_min(const std::vector<float>& d);

template void HbInternalTensorImpl::set_max(const std::vector<int64_t>& d);
template void HbInternalTensorImpl::set_max(const std::vector<uint64_t>& d);
template void HbInternalTensorImpl::set_max(const std::vector<int32_t>& d);
template void HbInternalTensorImpl::set_max(const std::vector<uint32_t>& d);
template void HbInternalTensorImpl::set_max(const std::vector<float>& d);

} // namespace habana_lazy
