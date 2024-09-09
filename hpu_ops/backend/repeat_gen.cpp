/******************************************************************************
 * Copyright (C) 2023-2024 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
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
