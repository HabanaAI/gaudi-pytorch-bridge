/******************************************************************************
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

#pragma once

#include "hpu_ops/hpu_op_helper.h"

#define SHARED_META(name)                \
  SharedMetaDataVector name##SharedMeta( \
      const at::Stack& stack, const std::string& guid);

namespace habana {

SHARED_META(Input0)
SHARED_META(Input0ToOut0And1)
SHARED_META(AdaptiveBwd)
SHARED_META(AvgPoolBwd)
SHARED_META(FillCumSumProd)
SHARED_META(IsFiniteInfNan)
SHARED_META(Rounding)
SHARED_META(Compare)

SHARED_META(UnaryForeach)
} // namespace habana
