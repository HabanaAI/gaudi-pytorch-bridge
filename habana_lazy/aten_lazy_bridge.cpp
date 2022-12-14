/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "aten_lazy_bridge.h"
#include "habana_helpers/misc_utils.h"
#include "habana_kernels/resize.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/lazy_storage.h"
#include "habana_lazy/ops/constant.h"
#include "habana_lazy/ops/hpu_input.h"
#include "pytorch_helpers/habana_helpers/kernels_accumulation.h"
#include "tensor_impl.h"

namespace habana_lazy {

at::Tensor HbLazyToAtenTensor(
    HbLazyTensor HbLazy_tensor,
    const at::TensorOptions& tensor_options) {
  at::Tensor tensor = HbLazy_tensor.ToTensor(/*detached=*/false);
  // We need to copy the tensor since it is cached within the HbLazyTensor, and
  // returning it directly might expose it to in place changes. Which there was
  // COW option :)
  return tensor.to(tensor_options, /*non_blocking=*/false, /*copy=*/true);
}

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
  c10::Storage lazy_storage;
  CreateStorageForAtenTensor(
      scalarTypeToTypeMeta(HbLazy_tensor.dtype()).itemsize(),
      size,
      lazy_storage);
  at::Tensor tensor = at::Tensor(c10::make_intrusive<HbLazyTensorImpl>(
      std::move(HbLazy_tensor), std::move(lazy_storage)));
  InitSizesAndStrides(tensor, tensor_type, size, stride, mem_format);
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
  c10::Storage lazy_storage;
  CreateStorageForAtenTensor(
      scalarTypeToTypeMeta(HbLazy_tensor.dtype()).itemsize(),
      size,
      lazy_storage);
  at::Tensor tensor = at::Tensor(c10::make_intrusive<HbLazyTensorImpl>(
      HbLazy_tensor, std::move(lazy_storage)));
  InitSizesAndStrides(tensor, tensor_type, size, stride, mem_format);
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
  at::Tensor tensor = at::Tensor(c10::make_intrusive<HbLazyTensorImpl>(
      std::move(HbLazy_tensor), storage, key_set));
  InitSizesAndStrides(tensor, tensor_type, size, stride, mem_format);
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
    }

    if (!hl_tensor_size_zero) {
      PT_LAZY_DEBUG("CheckAndUpdateSizeStride: at_internal_tensor is NOT set!");
      return hl_t;
    }

    auto type = c10::typeMetaToScalarType(tensor.dtype());
    type = type == c10::ScalarType::Long ? c10::ScalarType::Int : type;
    type = type == c10::ScalarType::Double ? c10::ScalarType::Float : type;
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
        habana_lazy::GetNBytes(at_internal_tensor));

    // Update Size And Stride Info
    c10::IntArrayRef tensor_size = tensor.sizes();
    c10::IntArrayRef tensor_stride = tensor.strides();
    impl->set_sizes_and_strides(tensor_size, tensor_stride);
  }

  return hl_t;
}

c10::optional<HbLazyTensor> TryGetHbLazyTensor(
    const at::Tensor& tensor,
    bool get_updated,
    bool handle_collective,
    bool is_size_strides_update) {
  HbLazyTensorImpl* impl = GetHbLazyTensorImpl(tensor);
  if (impl == nullptr) {
    return c10::nullopt;
  }

  auto context = habana_lazy_executor.getDeviceExecutionContext(0);
  HbLazyTensor hl_t = impl->tensor();
  // always fetch most recent version of the tensor
  // TODO currently we assert if view handle is missing in any of the kernel.
  // Try bringing it here
  auto id = hl_t.getTensorUniqueId();

  auto t_shallow_copy_opt = hl_t.getDataPtr()->tensor_shallow_copy;
  if (t_shallow_copy_opt.has_value()) {
    impl = GetHbLazyTensorImpl(t_shallow_copy_opt.value());
    hl_t = impl->tensor();
    id = hl_t.getTensorUniqueId();
  }

  if (get_updated) {
    LOCK_VIEW_TABLE_MUTEX(context->viewContext);
    c10::optional<at::Tensor> base_tensor =
        context->viewContext.GetOrigTensorMapEntry(id);
    if (base_tensor != c10::nullopt) {
      impl = GetHbLazyTensorImpl(base_tensor.value());
      hl_t = impl->tensor();
    }
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
       !habana_lazy::AccThread::Get().inThreadPool())) {
    PT_LAZY_DEBUG("step marker due to collective op output request");
    HbLazyTensor::StepMarker({});
  }

  return hl_t;
}

HbInternalTensorImpl* GetHbInternalTensorImpl(const at::Tensor& tensor) {
  return dynamic_cast<HbInternalTensorImpl*>(tensor.unsafeGetTensorImpl());
}

static inline size_t calculate_nbytes(
    size_t num_bytes,
    const caffe2::TypeMeta d_type) {
  auto s_type = c10::typeMetaToScalarType(d_type);
  if (s_type == c10::ScalarType::Long) {
    PT_LAZY_DEBUG("GetNBytes() called for tensor with dtype 'Long'!");
    // As our allocation is for 'Int'
    num_bytes /= 2;
  }
  if (s_type == c10::ScalarType::Double) {
    PT_LAZY_DEBUG("GetNBytes() called for tensor with dtype 'Double'!");
    // As our allocation is for 'Float'
    num_bytes /= 2;
  }
  return num_bytes;
}

// Use GetNBytes instead of nbytes
size_t GetNBytes(c10::StorageImpl* impl, const caffe2::TypeMeta d_type) {
  size_t num_bytes = calculate_nbytes(impl->nbytes(), d_type);
  return num_bytes;
}

size_t GetNBytes(at::TensorImpl* impl) {
  size_t num_bytes = calculate_nbytes(impl->storage().nbytes(), impl->dtype());
  return num_bytes;
}

size_t GetNBytes(const at::Tensor& tensor) {
  size_t num_bytes = calculate_nbytes(tensor.nbytes(), tensor.dtype());
  return num_bytes;
}

size_t GetNBytes(at::Tensor& tensor) {
  size_t num_bytes = calculate_nbytes(tensor.nbytes(), tensor.dtype());
  return num_bytes;
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

c10::optional<at::Device> GetHblazyDevice(const at::Tensor& tensor) {
  auto hb_tensor = TryGetHbLazyTensor(tensor);
  if (!hb_tensor) {
    return c10::nullopt;
  }
  return hb_tensor->GetDevice();
}

ir::Value GetIrValueForListConstruct(
    const ir::ValueList& values,
    bool optional) {
  return ir::Value(std::make_shared<ir::ListConstruct>(values, optional));
}

std::vector<at::Tensor> HpuGetFallbackTensorList(
    const std::vector<at::Tensor>& tensors) {
  std::vector<at::Tensor> fbtensors;
  fbtensors.reserve(tensors.size());
  for (const auto& tensor : tensors) {
    fbtensors.push_back(tensor.to(c10::kCPU));
  }
  return fbtensors;
}

void HpuGatherLazyFallbackTensorList(
    const std::vector<at::Tensor>& tensors,
    std::vector<HbLazyTensor>& tensors_to_execute) {
  for (const auto& tensor : tensors) {
    tensors_to_execute.push_back(GetOrCreateHbLazyTensor(tensor));
  }
}
void HpuGatherLazyFallbackOptTensorList(
    const std::vector<c10::optional<at::Tensor>>& tensors,
    std::vector<HbLazyTensor>& tensors_to_execute) {
  for (const auto& tensor : tensors) {
    if (tensor.has_value() && tensor.value().defined()) {
      tensors_to_execute.push_back(GetOrCreateHbLazyTensor(tensor.value()));
    }
  }
}

const std::vector<c10::optional<at::Tensor>> HpuGetFallbackOptTensorList(
    const std::vector<c10::optional<at::Tensor>>& tensors) {
  std::vector<c10::optional<at::Tensor>> fbtensors;
  fbtensors.reserve(tensors.size());
  for (const auto& tensor : tensors) {
    if (tensor.has_value() && tensor.value().defined()) {
      fbtensors.emplace_back(tensor.value().to(c10::kCPU));
    } else {
      fbtensors.emplace_back(tensor);
    }
  }
  return fbtensors;
}

c10::List<c10::optional<at::Tensor>> HpuGetFallbackOptTensorList(
    const c10::List<c10::optional<at::Tensor>>& tensors) {
  c10::List<c10::optional<at::Tensor>> fbtensors;
  fbtensors.reserve(tensors.size());
  for (c10::optional<at::Tensor> tensor : tensors) {
    if (tensor.has_value() && tensor.value().defined()) {
      fbtensors.emplace_back(tensor.value().to(c10::kCPU));
    } else {
      fbtensors.emplace_back(tensor);
    }
  }
  return fbtensors;
}

at::Tensor CreateHpuTensor(
    const at::Tensor& tensor,
    const c10::optional<c10::Device>& device) {
  if (tensor.defined() && device) {
    return tensor.contiguous().to(device.value());
  }
  return tensor;
}

std::vector<at::Tensor> CreateHpuTensors(
    const std::vector<at::Tensor>& tensors,
    const c10::optional<c10::Device>& device) {
  std::vector<at::Tensor> htensors;
  htensors.reserve(tensors.size());
  for (auto& tensor : tensors) {
    htensors.push_back(CreateHpuTensor(tensor, device));
  }
  return htensors;
}

void HpuUpdateTensors(
    std::vector<at::Tensor>& dst_tensors,
    std::vector<at::Tensor>& src_tensors,
    const std::vector<size_t>& indices) {
  for (auto index : indices) {
    auto dst = dst_tensors.at(index);
    auto src = src_tensors.at(index);
    // https://github.com/pytorch/pytorch/wiki/Developer-FAQ#how-does-out-work-in-pytorch
    // says:
    // When a user passes one or more tensors to out= the contract is as
    // follows:
    // * if an out tensor has no elements it may be resized
    // * passing out= tensors is numerically equivalent to performing the
    //   operation and "safe" copying its results to the (possibly resized if
    //   empty) out tensors

    // if (dst.numel() == 0) {
    if (dst.sizes() != src.sizes()) {
      auto shape = at::DimVector(src.sizes());
      THHTensor_resizeNd(
          dst.unsafeGetTensorImpl(), shape.size(), shape.data(), nullptr);
    }

    dst.copy_(src, /*non_blocking*/ true);
  }
}
c10::optional<c10::Device> GetHpuDevice(const at::Tensor& tensor) {
  return tensor.device();
}

c10::optional<c10::Device> GetHpuDevice(
    const c10::optional<at::Tensor>& tensor) {
  if (!tensor.has_value()) {
    return c10::nullopt;
  }
  return GetHpuDevice(*tensor);
}

c10::optional<c10::Device> GetHpuDevice(const at::TensorList& tensors) {
  for (const auto& tensor : tensors) {
    auto device = GetHpuDevice(tensor);
    if (device) {
      return device;
    }
  }
  return c10::nullopt;
}

c10::optional<c10::Device> GetHpuDevice(
    const at::TensorOptions& tensor_options) {
  if (!tensor_options.has_device()) {
    return c10::nullopt;
  }
  return GetHpuDevice(tensor_options.device());
}

c10::optional<c10::Device> GetHpuDevice(const c10::Device& device) {
  if (device.type() != at::kHPU) {
    return c10::nullopt;
  }
  return device;
}

c10::optional<c10::Device> GetHpuDevice(
    const c10::optional<c10::Device>& device) {
  if (!device) {
    return c10::nullopt;
  }
  return GetHpuDevice(*device);
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
