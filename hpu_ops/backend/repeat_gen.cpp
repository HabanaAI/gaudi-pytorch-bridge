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
#include "habana_kernels/repeat.h"

namespace habana {

OutputMetaDataVector RepeatMeta(const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  auto repeats = stack.at(1).isTensor() ? stack.at(1).toTensor().sizes().vec()
                                        : stack.at(1).toIntList().vec();

  OutputMetaData meta{};
  meta.dtype = self.scalar_type();
  meta.shape = RepeatOperator::compute_output_shape(self, repeats);

  return {meta};
}

SharedMetaDataVector RepeatSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  const auto& self = stack_tensor(stack, 0);
  auto dtype = self.scalar_type();
  auto inputRank = self.dim();
  auto outputRank = inputRank;

  if (!stack.at(1).isTensor()) {
    auto repeats = static_cast<int64_t>(stack.at(1).toIntList().size());
    outputRank = std::max(repeats, outputRank);
  }

  SharedMetaData repeatSharedMeta{"repeat_pt_fwd"};
  repeatSharedMeta.inputs_data.emplace_back(inputRank, dtype);
  repeatSharedMeta.outputs_data.emplace_back(outputRank, dtype);

  return {repeatSharedMeta};
}

FillParamsT FillRepeatFwdParams(const at::Stack& stack) {
  PARAMS_STUB(ns_RepeatPt::Params);
  auto repeats = stack.at(1).toIntVector();

  for (unsigned int i = 0; i < repeats.size(); i++) {
    params->repeat[i] = repeats[i];
  }
  params->size = repeats.size();

  return paramsT;
}

} // namespace habana
