/******************************************************************************
 * Copyright (C) 2021-2023 Habana Labs, Ltd. an Intel Company
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

#include "generated/lazy/clamp.h"
#include "generated/lazy/clamp_max.h"
#include "habana_helpers/dtype_helpers.h"

namespace habana {
// Use min/max of self tensor's dtype for clamping instead of
// blanket float limits. Use min/max of self's dtype as seen
// at backend(lowering).
// Clamp uses max/min guids which support only bf16, fp32 and int32.
// So the following table is limited to those dtypes.
// Long and Double at FE are seen as int and float at BE.
// Hence include these.
float self_type_max_for_be(c10::ScalarType type) {
  float max = std::numeric_limits<float>::max();
  switch (type) {
    case c10::ScalarType::Long:
    case c10::ScalarType::Int:
      // We should ideally use int max = 2147483647, But since
      // get_tensor_for_scalar() takes float value as argument,
      // we need to cast 2147483647 to float which becomes 2147483648.
      // This exceeds the int max limit. This causes issue down the line in
      // validateDownCast(). Hence use the largest integer value that
      // when converted to float becomes 2147483647. This is a tradeoff.
      // This int value is 2147483583, which is less than int max by 64.
      // Hence the clamping of these highest 64 integer values may not be
      // proper.
      // TODO:  Check if scalar caching can take types other than float in
      // the cache map. If yes, try using what ever is self's scalartype
      // instead of blanket float.
      max = (float)2147483583;
      break;

    case c10::ScalarType::Double:
    case c10::ScalarType::Float:
      max = (float)std::numeric_limits<float>::max();
      break;
    case c10::ScalarType::BFloat16:
      max = 3.38953139E38;
      break;
    default:
      // TODO: handle other dtypes
      PT_KERNEL_WARN("Using float max for unsupported type", type)
  }
  return max;
}

float self_type_min_for_be(c10::ScalarType type) {
  float min = std::numeric_limits<float>::lowest();
  switch (type) {
    case c10::ScalarType::Long:
    case c10::ScalarType::Int:
      min = (float)std::numeric_limits<int>::lowest();
      break;

    case c10::ScalarType::Double:
    case c10::ScalarType::Float:
      min = std::numeric_limits<float>::lowest();
      break;
    case c10::ScalarType::BFloat16:
      min = -3.38953139E38;
      break;
    default:
      // TODO: handle other dtypes
      PT_KERNEL_WARN("Using float min for unsupported type", type)
  }
  return min;
}

static void convert_params_to_tensors(
    at::Stack& inputs,
    at::ScalarType compute_dtype) {
  // For lazy eager Skip scalar handling at FE
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 2) {
    return;
  }
  const auto& self = inputs[0].toTensor();
  float min = inputs[1].isScalar() ? inputs[1].toScalar().to<float>()
                                   : self_type_min_for_be(compute_dtype);
  float max = inputs[2].isScalar() ? inputs[2].toScalar().to<float>()
                                   : self_type_max_for_be(compute_dtype);
  inputs[1] = habana_lazy::get_tensor_for_scalar(
      min, self.options().dtype(compute_dtype));
  inputs[2] = habana_lazy::get_tensor_for_scalar(
      max, self.options().dtype(compute_dtype));
}

HPU_OP_FRONTEND_CUSTOM_CTOR(habana_lazy::LazyOp, ClampFE, -1, at::Tensor) {}
HPU_OP_FRONTEND_CUSTOM_CTOR_ONLY(habana_lazy::LazyOp, ClampFE, at::Tensor&) {
  convert_params_to_tensors(
      get_inputs(), inputs.at(0).toTensor().scalar_type());
}

HPU_OP_FRONTEND_CREATE_RESULT_ONLY(habana_lazy::LazyOp, ClampFE, at::Tensor) {
  auto& inputs = get_inputs();
  const auto& dtype = get_scalar_types()[0];
  convert_params_to_tensors(inputs, dtype);
  const auto& t = inputs.at(0).toTensor();
  return habana_lazy::empty_hpu_lazy(
      t.sizes(), t.options().dtype(dtype), t.suggest_memory_format(), false);
}

} // namespace habana
