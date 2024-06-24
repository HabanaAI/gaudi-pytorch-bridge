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
#include "generated/backend/isin.h"

namespace habana {

std::shared_ptr<void> FillIsinParams(const at::Stack& stack, size_t& size) {
  PARAMS_STUB(ns_Isin::Params);
  params->invert = stack.at(3).toBool();

  return params;
}

OutputMetaDataVector IsinMeta(const at::Stack& stack) {
  const auto& shape = stack.at(0).toTensor().sizes().vec();
  return {OutputMetaData{c10::ScalarType::Bool, shape}};
}

OutputMetaDataVector ScalarIsinMeta(const at::Stack&) {
  return {OutputMetaData{c10::ScalarType::Bool, {}}};
}

} // namespace habana