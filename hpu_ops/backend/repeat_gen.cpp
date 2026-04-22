/**
 * Copyright (c) 2023-2025 Intel Corporation
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
#include "generated/backend/repeat.h"
#include "habana_helpers/conversion.h"
#include "habana_kernels/repeat.h"

namespace habana {

OutputMetaDataVector RepeatMeta(const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  auto repeats = stack.at(1).isTensor() ? stack.at(1).toTensor().sizes().vec()
                                        : stack.at(1).toIntList().vec();

  OutputMetaDataVector metaVec(1);
  auto& meta = metaVec.front();
  meta.dtype = self.scalar_type();
  meta.shape = RepeatOperator::compute_output_shape(self, repeats);

  return metaVec;
}

SharedMetaDataVector RepeatSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode /*unused*/) {
  const auto& self = stack_tensor(stack, 0);
  auto dtype = self.scalar_type();
  auto inputRank = self.dim();
  auto outputRank = inputRank;

  if (!stack.at(1).isTensor()) {
    auto repeats = static_cast<int64_t>(stack.at(1).toIntList().size());
    outputRank = std::max(repeats, outputRank);
  }

  SharedMetaDataVector meta;
  meta.reserve(1);
  auto& repeatSharedMeta = meta.emplace_back("repeat_pt_fwd");
  repeatSharedMeta.inputs_data.emplace_back(inputRank, dtype);
  repeatSharedMeta.outputs_data.emplace_back(outputRank, dtype);

  return meta;
}

FillParamsT FillRepeatFwdParams(const at::Stack& stack) {
  PARAMS_STUB(ns_RepeatPt::Params);
  auto repeats = stack.at(1).toIntVector();

  for (size_t i = 0; i < repeats.size(); i++) {
    params->repeat[i] = safe_convert<int>(repeats[i]);
  }
  params->size = safe_convert<unsigned int>(repeats.size());

  return paramsT;
}

} // namespace habana
