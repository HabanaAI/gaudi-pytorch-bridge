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

struct AtTensorPair {
  at::Tensor cpu;
  at::Tensor hpu;
};

enum class NativeLayerNormTestWeight {
  Defined,
  Undefined,
};

enum class NativeLayerNormTestBias {
  Defined,
  Undefined,
};

enum class NativeLayerNormTestMode {
  Forward,
  Backward,
  BackwardGal,
  FwdBwdAffine,
};

std::vector<AtTensorPair> native_layer_norm_test(
    NativeLayerNormTestMode,
    NativeLayerNormTestWeight,
    NativeLayerNormTestBias,
    bool verbose = false);

#define LAYER_NORM_TEST_2(BASE, MODE, WEIGHT, BIAS)                    \
  TEST_F(BASE, LayerNorm##MODE##Weight##WEIGHT##Bias##BIAS##Execute) { \
    auto results = native_layer_norm_test(                             \
        NativeLayerNormTestMode::MODE,                                 \
        NativeLayerNormTestWeight::WEIGHT##ined,                       \
        NativeLayerNormTestBias::BIAS##ined);                          \
    for (int i = 0; i < results.size(); ++i) {                         \
      auto& result = results[i];                                       \
      EXPECT_EQ(result.hpu.is_same_size(result.cpu), true)             \
          << "HPU: " << result.hpu.sizes()                             \
          << " vs CPU: " << result.cpu.sizes();                        \
      EXPECT_EQ(allclose(result.hpu, result.cpu, 0.01, 0.01), true)    \
          << "Maximum abs diff = "                                     \
          << (result.hpu - result.cpu).abs().max().item<float>();      \
    }                                                                  \
  }

#define LAYER_NORM_TEST_1(...)        \
  LAYER_NORM_TEST_2(__VA_ARGS__, Def) \
  LAYER_NORM_TEST_2(__VA_ARGS__, Undef)

#define LAYER_NORM_TEST(...)          \
  LAYER_NORM_TEST_1(__VA_ARGS__, Def) \
  LAYER_NORM_TEST_1(__VA_ARGS__, Undef)
