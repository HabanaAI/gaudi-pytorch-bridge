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
#include "hpu_ops/op_logger.h"

namespace habana {
template <>
std::string to_string(const at::Tensor& t) {
  std::ostringstream oss;
  if (t.defined()) {
    oss << t.toString() << t.sizes();
  } else {
    oss << "[UndefinedTensor]";
  }
  return oss.str();
}

template <>
std::string to_string(const at::Generator&) {
  return "generator";
}
} // namespace habana
