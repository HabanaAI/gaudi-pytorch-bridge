/******************************************************************************
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
#include "generated/backend/scatter_add.h"
namespace habana {

const unsigned SELF_INDEX = 0;
const unsigned DIM_INDEX = 1;

std::shared_ptr<void> ScatterAddParams(const at::Stack& stack, size_t& size) {
  PARAMS_STUB(ns_ScatterKernel::ParamsV2);
  const auto dim = stack.at(DIM_INDEX).toInt();

  const auto rank = stack.at(SELF_INDEX).toTensor().dim();
  params->dim = dim;
  params->axis = get_dim_in_tpc_order(dim, rank);

  return params;
}

OutputMetaDataVector ScatterAddMeta(const at::Stack& stack) {
  const auto selfTensor = stack.at(SELF_INDEX).toTensor();

  OutputMetaData meta;
  meta.dtype = selfTensor.scalar_type();
  meta.shape = selfTensor.sizes().vec();

  return {meta};
}
} // namespace habana
