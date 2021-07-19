/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "hpu_op_helper.h"
namespace habana {

template <typename T>
T& get(fint_t&);

template <>
int& get<int>(fint_t& u) {
  return u.i;
}
template <>
float& get<float>(fint_t& u) {
  return u.f;
}

template <typename ScalarType>
static std::shared_ptr<void> ClampParams(
    ScalarType min,
    ScalarType max,
    size_t& size) {
  using P = ns_ClampKernel::Params;
  size = sizeof(P);
  auto params = std::make_shared<P>();

  get<ScalarType>(params->lowerBound) = min;
  get<ScalarType>(params->upperBound) = max;

  return params;
}

std::shared_ptr<void> HabanaOperatorHelper::FillClampParams(
    const at::Stack& stack,
    size_t& size) {
  if (c10::isFloatingType(stack[0].toTensor().scalar_type())) {
    float min = stack[1].isScalar() ? stack[1].toScalar().to<float>()
                                    : -std::numeric_limits<float>::max();
    float max = stack[2].isScalar() ? stack[2].toScalar().to<float>()
                                    : std::numeric_limits<float>::max();
    return ClampParams(min, max, size);
  } else {
    int min = stack[1].isScalar() ? stack[1].toScalar().to<int>()
                                  : -std::numeric_limits<int>::max();
    int max = stack[2].isScalar() ? stack[2].toScalar().to<int>()
                                  : std::numeric_limits<int>::max();
    return ClampParams(min, max, size);
  }
} // namespace habana

std::shared_ptr<void> HabanaOperatorHelper::FillClampMinParams(
    const at::Stack& stack,
    size_t& size) {
  return ClampParams(
      stack[1].toScalar().toFloat(), std::numeric_limits<float>::max(), size);
}

std::shared_ptr<void> HabanaOperatorHelper::FillClampMaxParams(
    const at::Stack& stack,
    size_t& size) {
  return ClampParams(
      -std::numeric_limits<float>::max(), stack[1].toScalar().toFloat(), size);
}

std::vector<at::Tensor> GetMetaTensorList(
    const std::vector<at::Tensor>& tensors) {
  std::vector<at::Tensor> metatensors;
  metatensors.reserve(tensors.size());
  for (const auto& tensor : tensors) {
    metatensors.emplace_back(at::empty_meta(
        tensor.sizes(), tensor.options(), tensor.suggest_memory_format()));
  }
  return metatensors;
}

std::vector<c10::optional<at::Tensor>> GetMetaOptTensorList(
    const std::vector<c10::optional<at::Tensor>>& tensors) {
  std::vector<c10::optional<at::Tensor>> metatensors;
  metatensors.reserve(tensors.size());
  for (const auto& tensor : tensors) {
    if (tensor.has_value()) {
      const auto& tv = tensor.value();
      metatensors.emplace_back(
          at::empty_meta(tv.sizes(), tv.options(), tv.suggest_memory_format()));
    } else {
      metatensors.emplace_back(tensor);
    }
  }
  return metatensors;
}
} // namespace habana
