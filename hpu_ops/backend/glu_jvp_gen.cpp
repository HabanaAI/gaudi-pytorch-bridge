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


#include "generated/backend/glu_jvp.h"

namespace habana {

OutputMetaDataVector GluJvpMeta(const at::Stack& stack) {
  OutputMetaData output;
  const auto input_tensor = stack.at(0).toTensor();
  output.shape = input_tensor.sizes().vec();
  output.dtype = input_tensor.scalar_type();
  return {output};
}

std::shared_ptr<void> FillGluJvpParams(
    const at::Stack& stack,
    size_t& size) {
  const auto dim = stack.at(3).toScalar().toInt();

  PARAMS_STUB(ns_GatherKernel::Params);
  params->axis = dim;
  return params;
}
}
