/******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
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
#include "common_functions_helpers.h"

std::vector<torch::Tensor> TensorAndViewVecToViewVec(
    const std::vector<TensorAndView>& src) {
  std::vector<torch::Tensor> result;
  for (auto&& v : src) {
    result.push_back(v.view);
  }
  return result;
}
