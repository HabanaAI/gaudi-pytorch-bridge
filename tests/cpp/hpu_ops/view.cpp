/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "util.h"
#define SIZE(...) __VA_ARGS__

#define HPU_VIEW_SIZE_TEST(name, in_size, DTYPE, out_size) \
  TEST_F(HpuOpTest, name) {                                \
    GenerateInputs(1, {in_size}, {DTYPE});                 \
    auto expected = GetCpuInput(0).view(out_size);         \
    auto result = GetHpuInput(0).view(out_size);           \
    Compare(expected, result);                             \
  }

class HpuOpTest : public HpuOpTestUtil {};

/**
 * Test cases fail for view op
 * Issue Raised: https://jira.habana-labs.com/browse/SW-95463
 */

HPU_VIEW_SIZE_TEST(view_size, SIZE({1024}), torch::kInt64, SIZE({2, 512}))
