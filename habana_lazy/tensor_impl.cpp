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
#include "tensor_impl.h"
#include <c10/core/Device.h>
#include <c10/core/ScalarType.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <torch/csrc/api/include/torch/version.h>
#include "aten_lazy_bridge.h"
#include "backend/synapse_helpers/env_flags.h"
#include "habana_helpers/logging.h"
#include "habana_lazy/lazy_executor.h"
#include "hpu_ops/hpu_op_helper.h"

namespace habana_lazy {

// helper class to maintain ownership on 'this' in case of
// working on separate thread (acc thread)
struct intrusive_raii_t {
  HbLazyTensorImpl* ptr_;
  intrusive_raii_t(HbLazyTensorImpl* ptr) : ptr_(ptr) {
    if (ptr_)
      c10::raw::weak_intrusive_ptr::incref(ptr_);
  }
  ~intrusive_raii_t() {
    if (ptr_)
      c10::raw::weak_intrusive_ptr::decref(ptr_);
  }
  // not copyable
  intrusive_raii_t(const intrusive_raii_t&) = delete;
  intrusive_raii_t& operator=(const intrusive_raii_t&) = delete;
};

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
  PT_LAZY_TRACE;
  habana_lazy::NoAccThread no_acc_thread;

  auto aten_t = AtenFromHbLazyTensor(
      m_tensor, c10::nullopt, c10::nullopt, c10::nullopt, c10::nullopt);
  auto impl = c10::make_intrusive<HbLazyTensorImpl>(
      HbLazyTensor::Create(aten_t, aten_t.device()));

  copy_tensor_metadata(
      /*src_impl=*/this,
      /*dest_impl=*/impl.get(),
      /*version_counter=*/version_counter,
      /*allow_tensor_metadata_change=*/allow_tensor_metadata_change);
  impl.get()->SetupSizeProperties();
  impl->refresh_numel();
  impl->refresh_contiguous();

  MaybeSyncLaunchBeforeShallowCopy(&this->m_tensor, &impl->m_tensor);

  // increase this refcount to preserve it alive till lambda execution
  auto this_ref =
      std::make_shared<intrusive_raii_t>(const_cast<HbLazyTensorImpl*>(this));
  auto func = [impl, this, this_ref]() mutable {
    this->m_tensor.ShallowCopyTo(&impl->m_tensor);
  };
  RUN_MANUAL_OP_MAYBE_WITH_ACC_THREAD_NO_FLUSH(__FUNCTION__, func, impl);
}

c10::intrusive_ptr<c10::TensorImpl> HbLazyTensorImpl::shallow_copy_and_detach(
    c10::VariableVersion&& version_counter,
    bool allow_tensor_metadata_change) const {
  PT_LAZY_TRACE;
  habana_lazy::NoAccThread no_acc_thread;

  auto aten_t = AtenFromHbLazyTensor(
      m_tensor, c10::nullopt, c10::nullopt, c10::nullopt, c10::nullopt);
  auto impl = c10::make_intrusive<HbLazyTensorImpl>(
      HbLazyTensor::Create(aten_t, aten_t.device()));

  copy_tensor_metadata(
      /*src_impl=*/this,
      /*dest_impl=*/impl.get(),
      /*version_counter=*/std::move(version_counter),
      /*allow_tensor_metadata_change=*/allow_tensor_metadata_change);
  impl.get()->SetupSizeProperties();
  impl->refresh_numel();
  impl->refresh_contiguous();

  MaybeSyncLaunchBeforeShallowCopy(&this->m_tensor, &impl->m_tensor);

  // increase this refcount to preserve it alive till lambda execution
  auto this_ref =
      std::make_shared<intrusive_raii_t>(const_cast<HbLazyTensorImpl*>(this));
  auto func = [impl, this, this_ref]() mutable {
    this->m_tensor.ShallowCopyTo(&impl->m_tensor);
  };
  RUN_MANUAL_OP_MAYBE_WITH_ACC_THREAD_NO_FLUSH(__FUNCTION__, func, impl);
}

/* Handles the below scenario
Example:
b = view(a)
c = view(a)
a.data = b.data

print(a.cpu(), b.cpu(), c.cpu())
*/
void HbLazyTensorImpl::handle_view_cycles(
    HbLazyTensor& hl_src,
    HbLazyTensor& hl_dst) {
  auto src_t = AtenFromHbLazyTensor(
      hl_src, c10::nullopt, c10::nullopt, c10::nullopt, c10::nullopt);

  auto src_updated_t = HbLazyTensorViews::get_recent_base_tensor(src_t);
  auto hl_src_updated = GetHbLazyTensor(src_updated_t);
  auto& params_opt = hl_src_updated.getDataPtr()->stride_params;
  if (params_opt.has_value()) {
    auto& params = params_opt.value();
    auto base = params.base;
    auto recent_base = HbLazyTensorViews::get_recent_base_tensor(base);

    auto base_id = GetHbLazyTensorId(recent_base);

    auto dst_t = AtenFromHbLazyTensor(
        hl_dst, c10::nullopt, c10::nullopt, c10::nullopt, c10::nullopt);
    auto dst_id = GetHbLazyTensorId(dst_t);

    if (dst_id == base_id) {
      // create a different hb lazy tensor for base
      auto base_tensor_data = GetHbLazyTensor(recent_base).EvaluateTensorData();
      auto base_or_parent_impl = c10::make_intrusive<HbLazyTensorImpl>(
          HbLazyTensor::Create(recent_base, recent_base.device()));

      auto new_base_t = AtenFromHbLazyTensor(
          base_or_parent_impl->m_tensor,
          c10::nullopt,
          recent_base.sizes(),
          c10::nullopt,
          c10::nullopt);

      GetHbLazyTensor(new_base_t).SetTensorData(base_tensor_data);

      // for simplicity always use as_strided op for this case. This helps to
      // set just the base and avoid complications in multilevel view
      // scenarios
      params.optype = kStridedOpDefault;
      params.base = new_base_t;

      // Now replace the base tensor for all the other views pointing to the
      // same base
      auto context = get_device_lazy_execution_context();
      context->viewContext.ReplaceViewBase(base_id, new_base_t);
    }
  } // if (params_ptr != nullptr)
}

void HbLazyTensorImpl::shallow_copy_from(
    const c10::intrusive_ptr<TensorImpl>& impl) {
  PT_LAZY_TRACE;
  habana_lazy::NoAccThread no_acc_thread;

  HbLazyTensorImpl* hl_impl = dynamic_cast<HbLazyTensorImpl*>(impl.get());

  handle_view_cycles(hl_impl->m_tensor, this->m_tensor);

  copy_tensor_metadata(
      /*src_impl=*/hl_impl,
      /*dest_impl=*/this,
      /*version_counter=*/version_counter(),
      /*allow_tensor_metadata_change=*/allow_tensor_metadata_change());

  const_cast<HbLazyTensorImpl*>(this)->SetupSizeProperties();
  const_cast<HbLazyTensorImpl*>(this)->refresh_numel();
  const_cast<HbLazyTensorImpl*>(this)->refresh_contiguous();

  MaybeSyncLaunchBeforeShallowCopy(&hl_impl->m_tensor, &this->m_tensor);

  // increase this refcount to preserve it alive till lambda execution
  auto this_ref = std::make_shared<intrusive_raii_t>(this);
  auto func = [impl, this, this_ref]() mutable {
    HbLazyTensorImpl* hl_impl = dynamic_cast<HbLazyTensorImpl*>(impl.get());
    hl_impl->m_tensor.ShallowCopyTo(&this->m_tensor);
  };
  RUN_MANUAL_OP_NO_RETURN_WITH_ACC_THREAD_NO_FLUSH(__FUNCTION__, func);
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
    c10::SmallVectorImpl<int64_t>& strides,
    c10::IntArrayRef sizes) {
  for (auto i = sizes.size(); i > 1; --i) {
    strides[i - 2] = strides[i - 1] * sizes[i - 1];
  }
}

void HbLazyTensorImpl::SetupSizeProperties() {
  if (!m_size_initialized) {
    // Fill up the basic dimension data members which the base class
    // implementation uses in its APIs.
    c10::IntArrayRef sizes_l = m_tensor.GetSizes();
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
  TORCH_CHECK(
      allow_tensor_metadata_change(),
      "set_storage ",
      err_msg_tensor_metadata_change_not_allowed);
  storage_ = storage;
  device_opt_ = storage_.device();

  if (!GET_ENV_FLAG_NEW(PT_HPU_INFERENCE_STORAGE_OVERRIDE)) {
    return;
  }

  // storage is frontend and we may need to set backend tensor's storage also.
  if (storage.data_ptr() != nullptr) {
    auto aten_t = AtenFromHbLazyTensor(
        m_tensor, c10::nullopt, c10::nullopt, c10::nullopt, c10::nullopt);
    auto hl_t_opt = TryGetHbLazyTensor(aten_t, true, false, false);
    auto hl_t_updated = hl_t_opt.has_value() ? hl_t_opt.value() : m_tensor;
    std::lock_guard<std::recursive_mutex> lock(
        habana_lazy::HbContextArena::Get()->GetMutex());
    if (hl_t_updated.IsExecutionInProgress()) {
      auto context = habana_lazy::get_device_lazy_execution_context();
      context->JoinPendingLaunchThread();
    }
    // At this point, execution thread is finished.
    c10::TensorImpl* impl =
        ((HbLazyTensor)hl_t_updated).getAttachedTensorImpl();
    if (impl) {
      impl->set_storage_keep_dtype(storage);
      PT_LAZY_DEBUG("set_storage_keep_dtype called with backend storage.");
    } else {
      PT_LAZY_DEBUG(
          "set_storage_keep_dtype called with backend storage, but impl in NULL!");
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

  // ensure proper order of locking StridedViewContext and HbContextArena
  // mutexes always first mutex is StridedViewContext to be locked inside
  // TryGetHbLazyTensor there is StridedViewContext mutex lock temporarily so it
  // needs to be locked in outer scope, since this function is re-called from
  // the scope below with HbContextArena mutex taken
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

synapse_helpers::layouts::MemoryPermutation HbInternalTensorImpl::
    GetMemoryPermutation() const {
  auto smeta =
      habana::get_storage_extra_meta(static_cast<const c10::TensorImpl*>(this));
  if (!smeta) {
    PT_BRIDGE_DEBUG(
        "Getting permutations from HbInternalTensorImpl ",
        this,
        " without StorageExtraMet-less tensor. Returning defaults..");
    return habana::StorageExtraMeta().get_memory_permutation();
  }
  return smeta->get_memory_permutation();
}

void HbInternalTensorImpl::SetMemoryPermutation(
    synapse_helpers::layouts::MemoryPermutation permutation) {
  auto smeta = habana::get_storage_extra_meta(
      dynamic_cast<const c10::TensorImpl*>(this));
  if (!smeta && !permutation.empty())
    HABANA_ASSERT(
        smeta,
        "Trying to set memory permutations ",
        VecToString(permutation),
        ", but no StorageExtraeta avilable for HbInternalTensorImpl ",
        this);
  smeta->set_memory_permutation(permutation);
}
} // namespace habana_lazy
