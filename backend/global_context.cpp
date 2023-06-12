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

#include "backend/global_context.h"

namespace habana {

namespace backend {

ScalarCache& GlobalContext::GetScalarCache() {
  return scalar_cache_;
}

void GlobalContext::Clear() {
  scalar_cache_.ClearCache();
}

} // namespace backend
} // namespace habana