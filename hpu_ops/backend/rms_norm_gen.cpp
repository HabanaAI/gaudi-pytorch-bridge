/*******************************************************************************
 * Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
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
  second_output.dtype = c10::ScalarType::Float;
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
  auto data_in = stack.at(1).toTensor();
  auto gamma = stack.at(2).toTensor();

  OutputMetaData first_output;
  first_output.shape = data_in.sizes().vec();
  first_output.dtype = data_in.scalar_type();
  OutputMetaData second_output;
  second_output.shape = gamma.sizes().vec();
  second_output.dtype = gamma.scalar_type();
  return {first_output, second_output};
}
} // namespace habana
