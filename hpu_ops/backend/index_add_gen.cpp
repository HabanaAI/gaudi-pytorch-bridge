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

#include "generated/backend/index_add.h"

namespace habana {

std::shared_ptr<void> FillIndexAddParams(const at::Stack& stack, size_t& size) {
  PARAMS_STUB(ns_IndexAdd::Params);

  auto self = stack.at(0).toTensor();
  auto dim = stack.at(1).toScalar().to<int>();

  params->axis = get_dim_in_tpc_order(dim, self.dim());
  params->alpha = stack.at(4).toScalar().to<double>();
  return params;
}
} // namespace habana
