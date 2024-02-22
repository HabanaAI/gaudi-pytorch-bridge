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
#include "habana_eager/ops/masked_select.h"
#include "hpu_ops/op_logger.h"

namespace habana {
namespace eager {

at::Tensor masked_select_eager(const at::Tensor& self, const at::Tensor& mask) {
  auto new_size = at::infer_size(self.sizes(), mask.sizes());
  auto new_mask = at::broadcast_to(mask, new_size);
  auto new_self = at::broadcast_to(self, new_size);

  return at::index(new_self, {new_mask});
}

} // namespace eager
} // namespace habana