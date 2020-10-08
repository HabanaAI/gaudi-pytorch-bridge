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
#include "habana_lazy/constant.h"

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

at::Tensor AtenFromHbLazyTensor(
    HbLazyTensor HbLazy_tensor,
    c10::Storage&& storage) {
  at::Tensor tensor = HbLazy_tensor.is_null()
      ? at::Tensor()
      : at::Tensor(c10::make_intrusive<HbLazyTensorImpl>(
            std::move(HbLazy_tensor), std::move(storage)));
  return tensor;
}

at::Tensor AtenFromHbLazyTensor(HbLazyTensor HbLazy_tensor) {
  at::Tensor tensor = HbLazy_tensor.is_null()
      ? at::Tensor()
      : at::Tensor(
            c10::make_intrusive<HbLazyTensorImpl>(std::move(HbLazy_tensor)));

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

HbLazyTensor GetOrCreateHbLazyTensor(
    const at::Tensor& tensor,
    const c10::Device& device) {
  if (!tensor.defined()) {
    return HbLazyTensor();
  }
  auto hb_tensor = TryGetHbLazyTensor(tensor);
  return hb_tensor ? *hb_tensor : HbLazyTensor::Create(tensor, device);
}

HbLazyTensor GetHbLazyTensor(const at::Tensor& tensor) {
  auto hb_tensor = TryGetHbLazyTensor(tensor);
  return *hb_tensor;
}

HbLazyTensor GetOrCreateHbLazyTensor(
    const c10::optional<at::Tensor>& tensor,
    const c10::Device& device) {
  if (!IsDefined(tensor)) {
    return HbLazyTensor();
  }
  auto hb_tensor = TryGetHbLazyTensor(*tensor);
  return hb_tensor ? *hb_tensor : HbLazyTensor::Create(*tensor, device);
}

bool IsHbLazyTensor(const at::Tensor& tensor) {
  return GetHbLazyTensorImpl(tensor) != nullptr;
}

Value GetIrValueForScalar(const c10::Scalar& scalar) {
  return Constant<c10::Scalar>(scalar).IrValue();
}

} // namespace habana_lazy