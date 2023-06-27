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
#include <torch/torch.h>

bool isGaudi();
bool isGaudi2();
bool isGaudi3();

#define GTEST_SKIPPED_ON_PLATFORM_CAUSE(platform, cause)              \
  do {                                                                \
    if (is##platform()) {                                             \
      GTEST_SKIP() << "Test skipped on " #platform ", cause: " cause; \
    }                                                                 \
  } while (0)

#define GTEST_SKIPPED_ON_GAUDI3_CAUSE_DYNAMIC_SHAPES_NOT_SUPPORTED() \
  GTEST_SKIPPED_ON_PLATFORM_CAUSE(                                   \
      Gaudi3, "Dynamic shapes are not supported on Gaudi3")
