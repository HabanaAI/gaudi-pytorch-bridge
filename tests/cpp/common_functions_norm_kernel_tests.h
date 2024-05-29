/******************************************************************************
 * Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
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

#include "utils/dtype_supported_on_device.h"
#include "utils/hint_tolerance_values.h"

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
    c10::ScalarType dtype,
    int dsIterNo,
    int dsItersCount,
    bool verbose = false);

#define LAYER_NORM_TEST_3(                                                    \
    BASE, MODE, WEIGHT, BIAS, DT, DTYPE, PREC, DSVAL, DSLAB)                  \
  TEST_F(                                                                     \
      BASE, LayerNorm##MODE##Weight##WEIGHT##Bias##BIAS##DT##DSLAB##xecute) { \
    if (!IsDtypeSupportedOnCurrentDevice(torch::DTYPE)) {                     \
      GTEST_SKIP();                                                           \
    }                                                                         \
    for (int dsi = 0; dsi < DSVAL; ++dsi) {                                   \
      auto results = native_layer_norm_test(                                  \
          NativeLayerNormTestMode::MODE,                                      \
          NativeLayerNormTestWeight::WEIGHT##ined,                            \
          NativeLayerNormTestBias::BIAS##ined,                                \
          torch::DTYPE,                                                       \
          dsi,                                                                \
          DSVAL);                                                             \
      for (int i = 0; i < results.size(); ++i) {                              \
        auto& result = results[i];                                            \
        if ((torch::DTYPE != torch::kFloat32) &&                              \
            (result.hpu.scalar_type() != result.cpu.scalar_type())) {         \
          result.cpu = result.cpu.to(result.hpu.scalar_type());               \
        }                                                                     \
        EXPECT_EQ(result.hpu.is_same_size(result.cpu), true)                  \
            << "HPU: " << result.hpu.sizes()                                  \
            << " vs CPU: " << result.cpu.sizes();                             \
        EXPECT_EQ(allclose(result.hpu, result.cpu, PREC, PREC), true)         \
            << HintToleranceValues(result.hpu, result.cpu, PREC, PREC);       \
      }                                                                       \
    }                                                                         \
  }

#define LAYER_NORM_TEST_2(...)                                 \
  LAYER_NORM_TEST_3(__VA_ARGS__, F32, kFloat32, 0.605, 1, E)   \
  LAYER_NORM_TEST_3(__VA_ARGS__, BF16, kBFloat16, 0.605, 1, E) \
  LAYER_NORM_TEST_3(__VA_ARGS__, F16, kFloat16, 0.605, 1, E)

#define LAYER_NORM_TEST_1(...)        \
  LAYER_NORM_TEST_2(__VA_ARGS__, Def) \
  LAYER_NORM_TEST_2(__VA_ARGS__, Undef)

#define LAYER_NORM_TEST(...)          \
  LAYER_NORM_TEST_1(__VA_ARGS__, Def) \
  LAYER_NORM_TEST_1(__VA_ARGS__, Undef)

#define LAYER_NORM_TEST_DS(...)                                           \
  LAYER_NORM_TEST_3(__VA_ARGS__, Def, Def, F32, kFloat32, 0.01, 3, DsE)   \
  LAYER_NORM_TEST_3(__VA_ARGS__, Def, Def, BF16, kBFloat16, 0.01, 3, DsE) \
  LAYER_NORM_TEST_3(__VA_ARGS__, Def, Def, F16, kFloat16, 0.01, 3, DsE)
