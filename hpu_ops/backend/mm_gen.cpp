/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/backend/mm.h"
namespace habana {
sizes_vec MmOutputShape(const at::Stack& stack) {
  TORCH_CHECK(
      (stack.at(0).isTensor() && stack.at(1).isTensor()),
      " Matmul Input type expected to be tensors");
  auto mat1 = stack.at(0).toTensor();
  auto mat2 = stack.at(1).toTensor();
  return {{mat1.size(0), mat2.size(1)}};
}
} // namespace habana
