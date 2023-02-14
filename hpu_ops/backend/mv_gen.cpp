/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/backend/mv.h"

namespace habana {

sizes_vec MvOpsOutputShape(const at::Stack& stack) {
  const at::Tensor mat1 = stack_tensor(stack, 0);
  sizes_vec shape = std::vector<std::vector<int64_t>>{{mat1.sizes()[0]}};
  return shape;
}
} // namespace habana
