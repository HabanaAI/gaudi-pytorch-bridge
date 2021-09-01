/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/hpu_op.h"

namespace habana {
sizes_vec HabanaOperatorHelper::PowOutputShape(const at::Stack& stack) {
  return {stack_tensor(stack, 1).sizes().vec()};
}

} // namespace habana
