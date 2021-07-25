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

#define PARAMS_STUB(structname) \
  size = sizeof(structname);    \
  auto params = std::make_shared<structname>()

template <typename ScalarType>
static std::shared_ptr<void> ClampParams(
    ScalarType min,
    ScalarType max,
    size_t& size) {
  PARAMS_STUB(ns_ClampKernel::Params);

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
}

std::shared_ptr<void> HabanaOperatorHelper::FillClampMinParams(
    const at::Stack& stack,
    size_t& size) {
  if (c10::isFloatingType(stack[0].toTensor().scalar_type())) {
    return ClampParams(
        stack[1].toScalar().toFloat(), std::numeric_limits<float>::max(), size);
  }
  return ClampParams(
      stack[1].toScalar().toInt(), std::numeric_limits<int>::max(), size);
}

std::shared_ptr<void> HabanaOperatorHelper::FillClampMaxParams(
    const at::Stack& stack,
    size_t& size) {
  if (c10::isFloatingType(stack[0].toTensor().scalar_type())) {
    return ClampParams(
        -std::numeric_limits<float>::max(),
        stack[1].toScalar().toFloat(),
        size);
  }
  return ClampParams(
      -std::numeric_limits<int>::max(), stack[1].toScalar().toInt(), size);
}

std::shared_ptr<void> HabanaOperatorHelper::FillHardSigmoidParams(
    const at::Stack&,
    size_t& size) {
  PARAMS_STUB(ns_HardSigmoidKernel::Params);
  constexpr float alpha = 1 / 6.0f;
  constexpr float beta = 1 / 2.0f;

  params->alpha = alpha;
  params->beta = beta;

  return params;
}

std::shared_ptr<void> HabanaOperatorHelper::FillMseLossParams(
    const at::Stack& stack,
    size_t& size) {
  PARAMS_STUB(ns_MSELossKernel::Params);

  auto mode = stack.at(stack.at(2).isInt() ? 2 : 3).toInt();
  switch (mode) {
    case at::Reduction::Reduction::None:
      params->mode = MSELossMode_t::MSE_LOSS_REDUCTION_MODE_NONE;
      break;
    case at::Reduction::Reduction::Mean:
      params->mode = MSELossMode_t::MSE_LOSS_REDUCTION_MODE_MEAN;
      break;
    case at::Reduction::Reduction::Sum:
      params->mode = MSELossMode_t::MSE_LOSS_REDUCTION_MODE_SUM;
      break;
    default:
      TORCH_CHECK(false, "Unsupported reduction mode in mseloss: ", mode);
  }
  return params;
}

sizes_vec HabanaOperatorHelper::MseLossOutputShape(
    const torch::Tensor& self,
    int64_t reduction) {
  if (reduction == at::Reduction::Reduction::None) {
    return {self.sizes().vec()};
  }
  return {{}};
}

sizes_vec HabanaOperatorHelper::PowOutputShape(const torch::Tensor& self) {
  return {self.sizes().vec()};
}

std::shared_ptr<void> HabanaOperatorHelper::FillCumsumParams(
    const at::Stack& stack,
    size_t& size) {
  PARAMS_STUB(ns_CumSumKernel::Params);
  auto self = stack.at(0).toTensor();
  auto dim = at::maybe_wrap_dim(stack.at(1).toInt(), self.dim(), true);
  params->axis = static_cast<int>(self.sizes().vec().size() - dim - 1);

  return params;
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
