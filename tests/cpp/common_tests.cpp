/*******************************************************************************
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
#include <gtest/gtest.h>

#include "common/utils.h"

TEST(Common, LoadedLibraryType) {
#ifdef EAGER_TESTS
  EXPECT_EQ(common::getLoadedLibraryType(), common::LibraryType::EAGER);
#else
  EXPECT_EQ(common::getLoadedLibraryType(), common::LibraryType::LAZY);
#endif
}
