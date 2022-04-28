/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/_unique.h"
#include "generated/unique_dim.h"

namespace habana {

FALLBACK_CHECK(UniqueFallbackCheck, bool sorted) {
  return !sorted; // sorted=True currently not supported in HPU.
};
} // namespace habana