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
#include "habana_lazy/ops/constant.h"
#include "habana_lazy/ops/hpu_input.h"

namespace habana_lazy {

////////////////////////////Util functions : Move to seperate file if
/// needed/////////////////////////////////////////////
// Checks whether a c10::optional<Tensor> is defined.
inline bool IsDefined(const c10::optional<at::Tensor>& tensor) {
  return tensor.has_value() && tensor.value().defined();
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

at::Tensor HbLazyToAtenTensor(
    HbLazyTensor HbLazy_tensor,
    const at::TensorOptions& tensor_options) {
  at::Tensor tensor = HbLazy_tensor.ToTensor(/*detached=*/false);
  // We need to copy the tensor since it is cached within the HbLazyTensor, and
  // returning it directly might expose it to in place changes. Which there was
  // COW option :)
  return tensor.to(tensor_options, /*non_blocking=*/false, /*copy=*/true);
}

at::Tensor AtenFromHbLazyTensor(HbLazyTensor HbLazy_tensor) {
  HABANA_ASSERT(HbLazy_tensor.is_null() == false);
  at::Tensor tensor = at::Tensor(
      c10::make_intrusive<HbLazyTensorImpl>(std::move(HbLazy_tensor)));

  return tensor;
}

at::Tensor AtenInternalHbTensor(
    c10::Storage&& storage,
    const caffe2::TypeMeta& data_type) {
  at::Tensor tensor = at::Tensor(
      c10::make_intrusive<HbInternalTensorImpl>(std::move(storage), data_type));
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
  return impl->tensor();
}

HbInternalTensorImpl* GetHbInternalTensorImpl(const at::Tensor& tensor) {
  return dynamic_cast<HbInternalTensorImpl*>(tensor.unsafeGetTensorImpl());
}

void setTensorAsInputNode(HbLazyTensor hl_tensor) {
  if (!hl_tensor.CurrentIrValue()) {
    ir::Value val = hl_tensor.createIrValueFromData();
    ir::NodePtr node = std::make_shared<ir::Input>(hl_tensor);
    val.SetNode(node);
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
  auto hb_tensor = TryGetHbLazyTensor(tensor);
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
      tensor = AtenFromHbLazyTensor(hblazy_tensor);
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

ir::Value GetIrValueForListConstruct(const ir::ValueList values) {
  return ir::Value(std::make_shared<ir::ListConstruct>(values));
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

const std::vector<c10::optional<at::Tensor>> HpuGetFallbackOptTensorList(
    const std::vector<c10::optional<at::Tensor>>& tensors) {
  std::vector<c10::optional<at::Tensor>> fbtensors;
  fbtensors.reserve(tensors.size());
  for (const auto& tensor : tensors) {
    if (tensor.has_value()) {
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
    return tensor.to(device.value());
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
  if (device.type() != at::kHABANA) {
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
} // namespace habana_lazy
