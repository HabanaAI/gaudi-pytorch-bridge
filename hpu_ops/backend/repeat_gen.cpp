/*******************************************************************************
 * Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
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

SharedMetaDataVector RepeatSharedMeta(const at::Stack& stack) {
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

std::shared_ptr<void> FillRepeatFwdParams(
    const at::Stack& stack,
    size_t& size) {
  PARAMS_STUB(ns_RepeatPt::Params);
  auto repeats = stack.at(1).toIntVector();

  for (unsigned int i = 0; i < repeats.size(); i++) {
    params->repeat[i] = repeats[i];
  }
  params->size = repeats.size();

  return params;
}

} // namespace habana
