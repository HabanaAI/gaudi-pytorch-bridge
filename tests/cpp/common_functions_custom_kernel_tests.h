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

#include <cstdint>
#include <vector>

void runResourceApplyMomentumOptTest(
    int num_params,
    int M,
    int N,
    double momentum);

#define RESOURCE_APPLY_MOMENTUM_OPT_TEST(BASE)     \
  TEST_F(BASE, ResourceApplyMomentumOptTest) {     \
    runResourceApplyMomentumOptTest(2, 4, 4, 0.9); \
  }

void runLarsOptTest(
    int num_params,
    int M,
    int N,
    const std::vector<int64_t>& skip_masks,
    double eeta,
    double weight_decay,
    double eps,
    double lr,
    bool params_zero,
    bool grads_zero);

#define LARS_OPT_TEST(BASE)                                               \
  TEST_F(BASE, LarsOptTest) {                                             \
    runLarsOptTest(3, 4, 4, {1, 0, 1}, 0.9, 0.8, 0.1, 0.7, false, false); \
  }                                                                       \
  TEST_F(BASE, LarsOptTestParamsZero) {                                   \
    runLarsOptTest(3, 4, 4, {1, 0, 1}, 0.9, 0.8, 0.1, 0.7, true, false);  \
  }                                                                       \
  TEST_F(BASE, LarsOptTestGradsZero) {                                    \
    runLarsOptTest(3, 4, 4, {1, 0, 1}, 0.9, 0.8, 0.1, 0.7, false, true);  \
  }                                                                       \
  TEST_F(BASE, LarsOptTest1D) {                                           \
    runLarsOptTest(3, 8, 1, {1, 0, 1}, 0.9, 0.8, 0.1, 0.7, false, false); \
  }
