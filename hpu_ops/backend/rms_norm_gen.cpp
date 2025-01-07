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
#include <cstdint>
#include <vector>

#include "generated/backend/rms_norm.h"

namespace habana {

OutputMetaDataVector RMSNormMeta(const at::Stack& stack) {
  auto data_in = stack.at(0).toTensor();
  auto gamma = stack.at(1).toTensor();

  std::vector<std::int64_t> inverse_root_mean_square_sizes{
      data_in.sizes().vec()};
  inverse_root_mean_square_sizes.back() = 1;

  const auto data_in_dtype =
      (data_in.scalar_type() != gamma.scalar_type())
      ? c10::ScalarType::Float
      : data_in.scalar_type();

  OutputMetaData first_output;
  first_output.shape = data_in.sizes().vec();
  first_output.dtype = data_in_dtype;
  OutputMetaData second_output;
  second_output.shape = inverse_root_mean_square_sizes;
  second_output.dtype = data_in_dtype;
  return {first_output, second_output};
}

std::shared_ptr<void> RMSNormParams(
    const at::Stack& stack,
    std::size_t& size) {
  const auto epsilon = stack.at(2).toScalar().toFloat();

  PARAMS_STUB(ns_LayerNormKernel::ParamsRmsNorm);
  params->epsValid = true;
  params->eps = epsilon;
  params->fastMath = false;
  return params;
}

std::shared_ptr<void> RMSNormFastParams(
    const at::Stack& stack,
    std::size_t& size) {
  const auto epsilon = stack.at(2).toScalar().toFloat();

  PARAMS_STUB(ns_LayerNormKernel::Params);
  params->epsValid = true;
  params->eps = epsilon;
  return params;
}

std::shared_ptr<void> RMSNormBwdParams(
    const at::Stack& stack,
    std::size_t& size) {
  auto use_stages = stack.at(4).toScalar().to<bool>();
  auto bwd_mode = stack.at(5).toScalar().to<int>();

  PARAMS_STUB(ns_RmsNorm::ParamsV3);
  params->useStages = use_stages;
  params->bwdMode = static_cast<RmsNormBwdMode_t>(bwd_mode);
  return params;
}

OutputMetaDataVector RMSNormBwdMeta(const at::Stack& stack) {
  const auto& data_in = stack.at(1).toTensor();
  const auto& gamma = stack.at(2).toTensor();

  auto type = data_in.scalar_type();
  bool types_match = true;
  for (std::size_t i{0}; i < 4; ++i) {
    if (type != stack.at(i).toTensor().scalar_type()) {
      types_match = false;
      break;
    }
  }

  OutputMetaData first_output;
  first_output.shape = data_in.sizes().vec();
  first_output.dtype = types_match ? type : c10::ScalarType::Float;
  OutputMetaData second_output;
  second_output.shape = gamma.sizes().vec();
  second_output.dtype = types_match ? type : c10::ScalarType::Float;
  return {first_output, second_output};
}
} // namespace habana
