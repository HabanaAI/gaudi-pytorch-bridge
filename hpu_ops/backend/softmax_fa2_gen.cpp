/**
 * Copyright (c) 2025 Intel Corporation
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
#include "backend/habana_device/HPUGuardImpl.h"
#include "generated/backend/softmax_fa2.h"
#include "hpu_ops/fp8_ops.h"

namespace habana {

FillParamsT FillSoftmaxFa2Params(const at::Stack& stack) {
  const int64_t inputRank = stack.at(0).toTensor().dim();
  const at::ScalarType inputDtype = stack.at(0).toTensor().scalar_type();
  PARAMS_STUB(ns_SoftmaxFA2::ParamsV3);

  const int64_t dim = stack.at(4).toInt();
  params->dim = get_dim_in_tpc_order(dim, inputRank);
  TORCH_CHECK(
      params->dim == 0,
      "softmax_fa2: only last input dimension is supported, got: ",
      dim,
      ", when input rank is: ",
      inputRank);

  params->mode = SoftmaxMode_t::DEFAULT_SOFTMAX;
  params->invAttnHeads = 0;
  params->modeV2 = inputDtype == at::ScalarType::Float8_e4m3fn
      ? SoftmaxFA2Mode_t::FA_SOFTMAX_HF8_2B
      : SoftmaxFA2Mode_t::FA_DEFAULT_SOFTMAX;
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  const bool isGaudi3 =
      habana::HPUDeviceContext::get_device().type() == synDeviceGaudi3;
  if (isGaudi3) {
    // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
    params->modeV2 = static_cast<SoftmaxFA2Mode_t>(
        params->modeV2 | SoftmaxFA2Mode_t::FA_CL_ALIGNED_RETAINED);
  }
  params->experimentalMode = stack.at(5).toBool();
  TORCH_CHECK(
      isGaudi3 || params->experimentalMode == false,
      "softmax_fa2: experimental mode is supported only on Gaudi3.")

  params->boxGlobalDepthSize = 0;
  params->boxGlobalDepthStart = 0;
  params->boxGlobalWidthStart = 0;
  return paramsT;
}

OutputMetaDataVector SoftmaxFa2Meta(const at::Stack& stack) {
  OutputMetaDataVector metaVec(4);

  const auto input = stack.at(0).toTensor();
  metaVec[0] = {input.scalar_type(), input.sizes().vec()};

  const auto inputM = stack.at(2).toTensor();
  metaVec[1] = {inputM.scalar_type(), inputM.sizes().vec()};

  const auto inputL = stack.at(3).toTensor();
  metaVec[2] = {inputL.scalar_type(), inputL.sizes().vec()};
  metaVec[3] = {inputL.scalar_type(), inputL.sizes().vec()};

  return metaVec;
}

SharedMetaDataVector SoftmaxFa2SharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode /*unused*/) {
  const auto input = stack.at(0).toTensor();
  const auto descale = stack.at(1).toOptional<at::Tensor>();
  const auto inputM = stack.at(2).toTensor();
  const auto inputL = stack.at(3).toTensor();

  SharedMetaData sharedMeta("softmax_fa2_fwd");
  sharedMeta.inputs_data.emplace_back(getSharedMetaFromTensor(input));
  sharedMeta.inputs_data.emplace_back(getSharedMetaFromOptionalTensor(descale));
  sharedMeta.inputs_data.emplace_back(
      createOptionalNotPresentSharedMetaTensor());
  sharedMeta.inputs_data.emplace_back(
      createOptionalNotPresentSharedMetaTensor());
  sharedMeta.inputs_data.emplace_back(getSharedMetaFromTensor(inputM));
  sharedMeta.inputs_data.emplace_back(getSharedMetaFromTensor(inputL));

  sharedMeta.outputs_data.emplace_back(getSharedMetaFromTensor(input));
  sharedMeta.outputs_data.emplace_back(getSharedMetaFromTensor(inputM));
  sharedMeta.outputs_data.emplace_back(getSharedMetaFromTensor(inputL));
  sharedMeta.outputs_data.emplace_back(getSharedMetaFromTensor(inputL));

  return {sharedMeta};
}

} // namespace habana
