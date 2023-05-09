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
#pragma once

void runResourceApplyMomentumOptTest(
    int num_params,
    int M,
    int N,
    double momentum);

#define RESOURCE_APPLY_MOMENTUM_OPT_TEST(BASE)     \
  TEST_F(BASE, ResourceApplyMomentumOptTest) {     \
    runResourceApplyMomentumOptTest(2, 4, 4, 0.9); \
  }
