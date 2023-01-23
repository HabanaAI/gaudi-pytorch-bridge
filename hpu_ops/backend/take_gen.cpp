/******************************************************************************
 * Copyright (C) 2022 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/backend/take.h"
#include "hpu_ops/hpu_op_helper.h"

namespace habana {

sizes_vec TakeOutputShape(const at::Stack& stack) {
  return {stack_tensor(stack, 1).sizes().vec()};
}
} // namespace habana