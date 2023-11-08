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
#include "aten_lazy_bridge.h"
#include "backend/backend_meta.h"
#include "habana_helpers/misc_utils.h"
#include "habana_kernels/resize.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/lazy_storage.h"
#include "habana_lazy/ops/constant.h"
#include "habana_lazy/ops/hpu_input.h"
#include "pytorch_helpers/habana_helpers/kernels_accumulation.h"
#include "tensor_impl.h"

namespace habana_lazy {

void CreateStorageForAtenTensor(
    size_t tensor_size,
    c10::optional<c10::IntArrayRef> size,
    c10::Storage& lazy_storage) {
  auto storage_size = tensor_size;
  if (size.has_value()) {
    storage_size *= c10::multiply_integers(size.value());
  } else {
    storage_size = 0;
  }
  lazy_storage =
      c10::Storage(c10::make_intrusive<HbLazyStorageImpl>(storage_size));
}

at::Tensor AtenFromHbLazyTensor(
    HbLazyTensor&& HbLazy_tensor,
    c10::optional<synTensorType> tensor_type,
    c10::optional<c10::IntArrayRef> size,
    c10::optional<c10::IntArrayRef> stride,
    c10::optional<c10::MemoryFormat> mem_format) {
  PT_LAZY_TRACE;
  HABANA_ASSERT(HbLazy_tensor.is_null() == false);
  auto is_tensor_const = HbLazy_tensor.IsConstTensor();
  auto const_tensor_id = HbLazy_tensor.GetConstTensorId();
  c10::Storage lazy_storage;
  CreateStorageForAtenTensor(
      scalarTypeToTypeMeta(HbLazy_tensor.dtype()).itemsize(),
      size,
      lazy_storage);
  at::Tensor tensor = at::Tensor(c10::make_intrusive<HbLazyTensorImpl>(
      std::move(HbLazy_tensor), std::move(lazy_storage)));
  InitSizesAndStrides(tensor, tensor_type, size, stride, mem_format);
  habana::set_tensor_const(tensor, is_tensor_const, const_tensor_id);
  return tensor;
}

at::Tensor AtenFromHbLazyTensor(
    const HbLazyTensor& HbLazy_tensor,
    c10::optional<synTensorType> tensor_type,
    c10::optional<c10::IntArrayRef> size,
    c10::optional<c10::IntArrayRef> stride,
    c10::optional<c10::MemoryFormat> mem_format) {
  PT_LAZY_TRACE;
  HABANA_ASSERT(HbLazy_tensor.is_null() == false);
  auto is_tensor_const = HbLazy_tensor.IsConstTensor();
  auto const_tensor_id = HbLazy_tensor.GetConstTensorId();
  c10::Storage lazy_storage;
  CreateStorageForAtenTensor(
      scalarTypeToTypeMeta(HbLazy_tensor.dtype()).itemsize(),
      size,
      lazy_storage);
  at::Tensor tensor = at::Tensor(c10::make_intrusive<HbLazyTensorImpl>(
      HbLazy_tensor, std::move(lazy_storage)));
  InitSizesAndStrides(tensor, tensor_type, size, stride, mem_format);
  habana::set_tensor_const(tensor, is_tensor_const, const_tensor_id);
  return tensor;
}

at::Tensor AtenFromHbLazyTensor(
    HbLazyTensor&& HbLazy_tensor,
    const c10::Storage& storage,
    c10::DispatchKeySet key_set,
    c10::optional<synTensorType> tensor_type,
    c10::optional<c10::IntArrayRef> size,
    c10::optional<c10::IntArrayRef> stride,
    c10::optional<c10::MemoryFormat> mem_format) {
  PT_LAZY_TRACE;
  HABANA_ASSERT(HbLazy_tensor.is_null() == false);
  auto is_tensor_const = HbLazy_tensor.IsConstTensor();
  auto const_tensor_id = HbLazy_tensor.GetConstTensorId();
  at::Tensor tensor = at::Tensor(c10::make_intrusive<HbLazyTensorImpl>(
      std::move(HbLazy_tensor), storage, key_set));
  InitSizesAndStrides(tensor, tensor_type, size, stride, mem_format);
  habana::set_tensor_const(tensor, is_tensor_const, const_tensor_id);
  return tensor;
}

at::Tensor AtenInternalHbTensor(
    c10::Storage&& storage,
    const caffe2::TypeMeta& data_type,
    c10::optional<synTensorType> tensor_type,
    c10::optional<c10::IntArrayRef> size,
    c10::optional<c10::IntArrayRef> stride,
    c10::optional<c10::MemoryFormat> mem_format) {
  at::Tensor tensor = at::Tensor(
      c10::make_intrusive<HbInternalTensorImpl>(std::move(storage), data_type));
  InitSizesAndStrides(tensor, tensor_type, size, stride, mem_format);
  return tensor;
}

HbLazyTensorImpl* GetHbLazyTensorImpl(const at::Tensor& tensor) {
  return dynamic_cast<HbLazyTensorImpl*>(tensor.unsafeGetTensorImpl());
}

namespace {
HbLazyTensor CheckAndUpdateSizeStride(
    HbLazyTensor hl_t,
    const at::Tensor& tensor) {
  PT_LAZY_TRACE;

  auto t = AtenFromHbLazyTensor(
      hl_t, c10::nullopt, c10::nullopt, c10::nullopt, c10::nullopt);
  auto impl = GetHbLazyTensorImpl(t);

  if (impl->storage().data_ptr() != nullptr) {
    auto at_tensor_size_zero = true;
    for (auto i = 0; i < (int)tensor.sizes().size(); i++) {
      if (tensor.sizes().at(i) > 0) {
        at_tensor_size_zero = false;
        break;
      }
    }

    if (at_tensor_size_zero) {
      PT_LAZY_DEBUG("CheckAndUpdateSizeStride: at_internal_tensor is NOT set!");
      return hl_t;
    }

    HbLazyTensor& hl_t_updated = hl_t;
    auto pTensor = hl_t_updated.GetTensorData();
    auto hl_tensor_size_zero = true;
    if (pTensor != c10::nullopt) {
      auto old_tensor_data = pTensor.value();
      if (old_tensor_data.sizes().size() > 0) {
        for (auto i = 0; i < (int)old_tensor_data.sizes().size(); i++) {
          if (old_tensor_data.sizes().at(i) > 0) {
            hl_tensor_size_zero = false;
            break;
          }
        }
      } else {
        hl_tensor_size_zero = false;
      }
    } else {
      hl_tensor_size_zero = false;
    }

    if (!hl_tensor_size_zero) {
      PT_LAZY_DEBUG("CheckAndUpdateSizeStride: at_internal_tensor is NOT set!");
      return hl_t;
    }

    auto type = habana_helpers::getInternalDtype(tensor.scalar_type());
    auto new_dtype = scalarTypeToTypeMeta(type);

    auto at_internal_tensor = AtenInternalHbTensor(
        c10::Storage(impl->storage()),
        new_dtype,
        DATA_TENSOR,
        tensor.sizes(),
        tensor.strides(),
        tensor.options().memory_format_opt());

    // backend tensor should always be contiguous as per view table design
    std::vector<int64_t> contig_strides = at_internal_tensor.strides().vec();
    if (contig_strides.size()) {
      habana_helpers::recalc_strides(
          contig_strides, at_internal_tensor.sizes().vec());
      c10::IntArrayRef new_strides = contig_strides;
      at_internal_tensor.unsafeGetTensorImpl()->set_sizes_and_strides(
          at_internal_tensor.sizes(), new_strides);
    }

    hl_t_updated.SetTensorData(at_internal_tensor);

    PT_LAZY_DEBUG("CheckAndUpdateSizeStride: at_internal_tensor is set!");
    PT_LAZY_DEBUG(
        "CheckAndUpdateSizeStride: at_internal_tensor storage size = ",
        habana_helpers::GetNBytes(at_internal_tensor));

    // Update Size And Stride Info
    c10::IntArrayRef tensor_size = tensor.sizes();
    c10::IntArrayRef tensor_stride = tensor.strides();
    impl->set_sizes_and_strides(tensor_size, tensor_stride);
  }

  return hl_t;
}
} // namespace

c10::optional<HbLazyTensor> TryGetHbLazyTensor(
    const at::Tensor& tensor,
    bool get_updated,
    bool handle_collective,
    bool is_size_strides_update) {
  auto is_tensor_const = habana::is_tensor_const(tensor);
  auto const_id = habana::get_tensor_const_id(tensor);
  HbLazyTensorImpl* impl = GetHbLazyTensorImpl(tensor);
  if (impl == nullptr) {
    return c10::nullopt;
  }

  HbLazyTensor hl_t = impl->tensor();
  // always fetch most recent version of the tensor
  // TODO currently we assert if view handle is missing in any of the kernel.
  // Try bringing it here
  auto& t_shallow_copy_opt = hl_t.getDataPtr()->tensor_shallow_copy;
  if (t_shallow_copy_opt.has_value()) {
    impl = GetHbLazyTensorImpl(t_shallow_copy_opt.value().back());
    hl_t = impl->tensor();
  }

  if (get_updated) {
    auto& base_tensor = hl_t.getDataPtr()->recent_base;
    if (base_tensor.has_value()) {
      impl = GetHbLazyTensorImpl(base_tensor.value());
      hl_t = impl->tensor();
    }
  }

  auto is_impl_const = habana::is_tensor_const(*impl);
  if (is_impl_const || is_tensor_const) {
    if (!habana::is_tensor_const_with_valid_const_id(tensor)) {
      const_id = habana::get_tensor_const_id(*impl);
    }
    hl_t.SetIsConstTensor(true, const_id);
  }

  // It may happen, HPU Lazy Tensor might be created with {0} size
  // during at::empty({0}, ...) call in pytorch-fork
  // As set_sizes_and_strides() call in pytorch-fork doesn't impact
  // the backend tensor properties like size, stride etc,
  // here we try to update these properties using frontend tensor info
  if (is_size_strides_update && hl_t.created_as_zero_size_tensor) {
    hl_t = CheckAndUpdateSizeStride(hl_t, tensor);
  }

  // if producer is collective, mark step
  if (handle_collective && hl_t.IsCollective() &&
      (!habana_lazy::AccThread::IsAccThreadEnabled() ||
       !habana_lazy::AccThread::Get().inAccThreadContext())) {
    PT_LAZY_DEBUG("step marker due to collective op output request");
    PT_IRGRAPH_DEBUG("step marker due to collective op output request");
    habana_lazy::HbLazyTensor::StepMarker({}, nullptr, {}, true);
  }

  return hl_t;
}

HbInternalTensorImpl* GetHbInternalTensorImpl(const at::Tensor& tensor) {
  return dynamic_cast<HbInternalTensorImpl*>(tensor.unsafeGetTensorImpl());
}

HbLazyTensor GetOrCreateHbLazyTensor(
    const at::Tensor& tensor,
    const c10::Device& device) {
  PT_LAZY_TRACE;
  if (!tensor.defined()) {
    return HbLazyTensor(device);
  }
  auto p_hb_tensor = TryGetHbLazyTensor(tensor);
  HbLazyTensor hl_tensor;
  if (p_hb_tensor) {
    hl_tensor = *p_hb_tensor;
  } else {
    hl_tensor = HbLazyTensor::Create(tensor, device);
  }
  return hl_tensor;
}

HbLazyTensor GetHbLazyTensor(
    const at::Tensor& tensor,
    bool get_updated,
    bool handle_collective) {
  HABANA_ASSERT(
      tensor.device().type() == at::kHPU,
      "Got a non-HPU tensor, expecting an HPU tensor");
  auto hb_tensor = TryGetHbLazyTensor(tensor, get_updated, handle_collective);
  HABANA_ASSERT(hb_tensor, "GetHbLazyTensor for a non lazy tensor");
  return *hb_tensor;
}

HbLazyTensor SyncAndGetHbLazyTensor(
    const at::Tensor& tensor,
    bool get_updated,
    bool handle_collective) {
  habana_lazy::AccThread::Get().SyncAccThreadPool();
  return GetHbLazyTensor(tensor, get_updated, handle_collective);
}

int64_t GetHbLazyTensorId(
    const at::Tensor& tensor,
    bool get_updated,
    bool handle_collective) {
  HABANA_ASSERT(
      tensor.device().type() == at::kHPU,
      "Got a non-HPU tensor, expecting an HPU tensor");
  auto hb_tensor = TryGetHbLazyTensor(tensor, get_updated, handle_collective);
  HABANA_ASSERT(hb_tensor, "GetHbLazyTensor for a non lazy tensor");
  return hb_tensor->getTensorUniqueId();
}

HbLazyTensor GetOrCreateHbLazyTensor(
    const c10::optional<at::Tensor>& tensor,
    const c10::Device& device) {
  PT_LAZY_TRACE;
  if (!IsDefined(tensor)) {
    return HbLazyTensor();
  }
  auto hb_tensor = TryGetHbLazyTensor(*tensor);
  return hb_tensor ? *hb_tensor : HbLazyTensor::Create(*tensor, device);
}

void MarkTensorAsOutputFromCollectiveOp(const at::Tensor& tensor) {
  GetHbLazyTensor(tensor).SetCollective();
}

bool IsHbLazyTensor(const at::Tensor& tensor) {
  return GetHbLazyTensorImpl(tensor) != nullptr;
}

ir::Value GetIrValueForNone() {
  return ir::Value(std::make_shared<ir::ScalarConstant>());
}

ir::Value GetIrValueForScalar(const c10::Scalar& scalar) {
  return ir::Value(std::make_shared<ir::ScalarConstant>(scalar));
}

at::Tensor CreateHbLazyTensor(
    at::Tensor tensor,
    const c10::optional<at::Device>& device) {
  PT_LAZY_TRACE;
  if (tensor.defined() && device) {
    bool is_input_lazy = IsHbLazyTensor(tensor);
    HbLazyTensor hblazy_tensor =
        HbLazyTensor::Create(std::move(tensor), *device);
    if (!is_input_lazy) {
      tensor = AtenFromHbLazyTensor(
          hblazy_tensor,
          c10::nullopt,
          c10::nullopt,
          c10::nullopt,
          c10::nullopt);
    } else {
      return tensor;
    }
  }
  return tensor;
}

ir::Value GetIrValueForListConstruct(
    const ir::ValueList& values,
    bool optional) {
  return {std::make_shared<ir::ListConstruct>(values, optional)};
}

void* GetLazyTensorDataPtr(const at::Tensor& t) {
  auto lazy_t = GetHbLazyTensor(t);
  auto internal_tensor = lazy_t.GetHbLazyTensorDataForMedia();
  TORCH_CHECK(
      internal_tensor,
      "Internal error: GetLazyTensorDataPtr doesn't have "
      "tensor with HBM storage");

  return internal_tensor->data_ptr();
}

} // namespace habana_lazy
