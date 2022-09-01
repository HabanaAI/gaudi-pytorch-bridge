/******************************************************************************
 * Copyright (C) 2021 Habana Labs, Ltd. an Intel Company
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

// This file will act as shim layer for various torch param conversions in diff
// torch version, which was impacted due to version upgrades.
#include <torch/csrc/api/include/torch/version.h>
#include <torch/library.h>

#if ((TORCH_VERSION_MAJOR == 1) && (TORCH_VERSION_MINOR < 13))
#define INTARRAY_PARAM(intArray) (intArray)
#else
#define INTARRAY_PARAM(intArray) \
  (c10::SymIntArrayRef(          \
      reinterpret_cast<const c10::SymInt*>(intArray.data()), intArray.size()))
#endif
