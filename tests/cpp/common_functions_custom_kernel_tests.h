/**
* Copyright (c) 2021-2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/
#pragma once

#include <cstdint>
#include <vector>

void runResourceApplyMomentumOptTest(
    int num_params,
    int M,
    int N,
    double momentum,
    bool enable_views);

#define RESOURCE_APPLY_MOMENTUM_OPT_TEST(BASE, ENA_VW)     \
  TEST_F(BASE, ResourceApplyMomentumOptTest) {             \
    runResourceApplyMomentumOptTest(2, 4, 4, 0.9, ENA_VW); \
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
    bool grads_zero,
    bool enable_views);

#define LARS_OPT_TEST(BASE, ENA_VW)                                    \
  TEST_F(BASE, LarsOptTest) {                                          \
    runLarsOptTest(                                                    \
        3, 4, 4, {1, 1, 0}, 0.9, 0.8, 0.1, 0.7, false, false, ENA_VW); \
  }                                                                    \
  TEST_F(BASE, LarsOptTestViewOnMasked) {                              \
    runLarsOptTest(                                                    \
        3, 4, 4, {1, 0, 1}, 0.9, 0.8, 0.1, 0.7, false, false, ENA_VW); \
  }                                                                    \
  TEST_F(BASE, LarsOptTestParamsZero) {                                \
    runLarsOptTest(                                                    \
        3, 4, 4, {1, 1, 0}, 0.9, 0.8, 0.1, 0.7, true, false, ENA_VW);  \
  }                                                                    \
  TEST_F(BASE, LarsOptTestGradsZero) {                                 \
    runLarsOptTest(                                                    \
        3, 4, 4, {1, 1, 0}, 0.9, 0.8, 0.1, 0.7, false, true, ENA_VW);  \
  }                                                                    \
  TEST_F(BASE, LarsOptTest1D) {                                        \
    runLarsOptTest(                                                    \
        3, 8, 1, {1, 1, 0}, 0.9, 0.8, 0.1, 0.7, false, false, ENA_VW); \
  }

void runAdamwOptTest(
    int num_params,
    int M,
    int N,
    double modified_weight_decay,
    bool enable_views);

#define ADAMW_OPT_TEST(BASE, ENA_VW)       \
  TEST_F(BASE, AdamwOptTestWd) {           \
    runAdamwOptTest(2, 4, 4, 0.9, ENA_VW); \
  }                                        \
  TEST_F(BASE, AdamwOptTestNoWd) {         \
    runAdamwOptTest(2, 4, 4, 1.0, ENA_VW); \
  }

void runLambPhase2OptimizerTest(
    int num_params,
    int M,
    int N,
    const double weight_decay,
    const bool use_lamb,
    const bool with_view);

#define LAMB_PHASE2_OPT_TEST(BASE)                          \
  TEST_F(BASE, LambPhase2Test) {                            \
    runLambPhase2OptimizerTest(2, 4, 3, 0.9, true, false);  \
  }                                                         \
  TEST_F(BASE, LambPhase2TestNoWd) {                        \
    runLambPhase2OptimizerTest(2, 4, 3, 0.0, true, false);  \
  }                                                         \
  TEST_F(BASE, LambPhase2TestNoLamb) {                      \
    runLambPhase2OptimizerTest(2, 4, 3, 0.9, false, false); \
  }                                                         \
  TEST_F(BASE, LambPhase2TestWithView) {                    \
    runLambPhase2OptimizerTest(2, 4, 3, 0.9, true, true);   \
  }

void runEmaOptTest(
    int num_params,
    int M,
    int N,
    double decay_val,
    bool enable_views);

#define EMA_OPT_TEST(BASE, ENA_VW)       \
  TEST_F(BASE, EmaOptTest) {             \
    runEmaOptTest(2, 4, 4, 0.9, ENA_VW); \
  }

void runLambPhase1OptimizerTest(
    int num_params,
    int M,
    int N,
    double weight_decay,
    int bias_correction,
    int step,
    int grad_averaging,
    bool with_view);

#define LAMB_PHASE1_OPT_TEST(BASE, ENA_VW)                     \
  TEST_F(BASE, LambPhase1Test) {                               \
    runLambPhase1OptimizerTest(1, 3, 4, 0.1, 1, 1, 1, false);  \
  }                                                            \
  TEST_F(BASE, LambPhase1TestNoBiasCorrection) {               \
    runLambPhase1OptimizerTest(2, 3, 4, 0.1, 0, 1, 1, false);  \
  }                                                            \
  TEST_F(BASE, LambPhase1TestLargerStep) {                     \
    runLambPhase1OptimizerTest(2, 3, 4, 0.1, 1, 3, 1, false);  \
  }                                                            \
  TEST_F(BASE, LambPhase1TestNoGradAveraging) {                \
    runLambPhase1OptimizerTest(2, 3, 4, 0.1, 1, 1, 0, false);  \
  }                                                            \
  TEST_F(BASE, LambPhase1TestNoWeightDecay) {                  \
    runLambPhase1OptimizerTest(2, 3, 4, 0.0, 1, 1, 1, false);  \
  }                                                            \
  TEST_F(BASE, LambPhase1TestWithViews) {                      \
    runLambPhase1OptimizerTest(1, 3, 4, 0.1, 1, 1, 1, ENA_VW); \
  }
