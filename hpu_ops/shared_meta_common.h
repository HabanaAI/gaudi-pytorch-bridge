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

#define SHARED_META(name) \
  SharedMetaDataVector name##SharedMeta(const at::Stack& stack);

#define SHARED_META_GUID(name)           \
  SharedMetaDataVector name##SharedMeta( \
      const at::Stack& stack, const std::string& guid);

namespace habana {

SHARED_META_GUID(Input0)
SHARED_META_GUID(Input0ToOut0And1)
SHARED_META_GUID(AdaptiveBwd)
SHARED_META_GUID(AvgPoolBwd)
SHARED_META_GUID(FillCumSumProd)
SHARED_META_GUID(IsFiniteInfNan)
SHARED_META_GUID(Rounding)
SHARED_META_GUID(Compare)
SHARED_META_GUID(ForeachCompound)
SHARED_META(BoolCast)
SHARED_META_GUID(LogicalBinary)
SHARED_META_GUID(UnaryForeach)
} // namespace habana
