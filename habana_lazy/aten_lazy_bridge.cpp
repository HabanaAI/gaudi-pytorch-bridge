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
#include "habana_lazy/ops/constant.h"
#include "habana_lazy/ops/hpu_input.h"
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

at::Tensor AtenFromHbLazyTensor(
    HbLazyTensor HbLazy_tensor,
    c10::optional<synTensorType> tensor_type,
    c10::optional<c10::IntArrayRef> size,
    c10::optional<c10::IntArrayRef> stride,
    c10::optional<c10::MemoryFormat> mem_format) {
  HABANA_ASSERT(HbLazy_tensor.is_null() == false);
  at::Tensor tensor = at::Tensor(
      c10::make_intrusive<HbLazyTensorImpl>(std::move(HbLazy_tensor)));
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

c10::optional<HbLazyTensor> TryGetHbLazyTensor(const at::Tensor& tensor) {
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
  if (context->orig_tensor_map.find(id) != context->orig_tensor_map.end()) {
    impl = GetHbLazyTensorImpl(context->orig_tensor_map[id]);
  }

  auto hl_t_updated = impl->tensor();
  // if producer is collective, mark step
  const auto& ir_value = hl_t_updated.GetIrValue();
  const auto& ir_node = ir_value.mp_node;
  const auto& ir_op = ir_node->op();

  PT_LAZY_DEBUG(
      "op: ", ir_op.toQualString(), " ir value: ", ir_value.ToString());
  if (IsCollective(ir_op)) {
    PT_LAZY_DEBUG("Collective op output requested, triggering a mark_step");
    HbLazyTensor::StepMarker({});
  }

  return hl_t_updated;
}

HbInternalTensorImpl* GetHbInternalTensorImpl(const at::Tensor& tensor) {
  return dynamic_cast<HbInternalTensorImpl*>(tensor.unsafeGetTensorImpl());
}

void setTensorAsInputNode(HbLazyTensor hl_tensor) {
  if (!hl_tensor.CurrentIrValue()) {
    ir::Value val = hl_tensor.createIrValueFromData();
    ir::NodePtr node = std::make_shared<ir::Input>(hl_tensor);
    val.SetNode(
        node,
        hl_tensor.GetDevice(),
        hl_tensor.GetSizes(),
        hl_tensor.dtype_optional());
    hl_tensor.AssignIrValue(val);
  } else {
    // TORCH_CHECK(
    //    false,
    //    " Habana Lazy Trying to set a tensor as leaf input node but IR value
    //    is set already");
  }
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

HbLazyTensor GetHbLazyTensor(const at::Tensor& tensor) {
  HABANA_ASSERT(
      tensor.device().type() == at::kHPU,
      "Got a non-HPU tensor, expecting an HPU tensor");
  auto hb_tensor = TryGetHbLazyTensor(tensor);
  HABANA_ASSERT(hb_tensor, "GetHbLazyTensor for a non lazy tensor");
  return *hb_tensor;
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
    fbtensors.push_back(std::move(tensor.to(c10::kCPU)));
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
  auto interal_tensor = lazy_t.GetHbLazyTensorData();
  TORCH_CHECK(
      interal_tensor,
      "Intenal error: GetLazyTensorDataPtr doesn't have "
      "tensor with HBM storage");
  return interal_tensor->data_ptr();
}

} // namespace habana_lazy
