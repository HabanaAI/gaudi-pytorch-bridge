/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/backend/dot.h"

namespace habana {
sizes_vec DotOutputShape(const at::Stack& stack) {
  const at::Tensor self = stack_tensor(stack, 0);
  const at::Tensor other = stack_tensor(stack, 1);
  TORCH_CHECK(
      self.dim() == 1 && other.dim() == 1,
      "Dot Op: 1D tensors expected, but got ",
      self.dim(),
      "D and ",
      other.dim(),
      "D tensors");
  TORCH_CHECK(
      self.sizes() == other.sizes(),
      "Dot Op: Tensor must have same size, but got ",
      self.sizes(),
      "and ",
      other.sizes(),
      "size tensors");
  return {{}};
}

OutputMetaDataVector DotMeta(const at::Stack& stack) {
  OutputMetaData meta;

  meta.shape = DotOutputShape(stack)[0];
  meta.dtype = habana_helpers::DTypeHelper::get_compute_dtype(
      stack,
      c10::nullopt,
      habana_helpers::DTypeHelper::DtypePromoteVariant::kPromoteToCommon,
      false,
      c10::nullopt,
      false,
      false);

  return {meta};
}
} // namespace habana
