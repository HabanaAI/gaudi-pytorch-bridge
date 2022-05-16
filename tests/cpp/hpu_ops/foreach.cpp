/******************************************************************************
 * Copyright (C) 2022 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "util.h"

class HpuOpTest : public HpuOpTestUtil {};

#define FOREACH(outplace_op)                                                 \
  TEST_F(HpuOpTest, outplace_op) {                                           \
    static constexpr int n = 5;                                              \
    std::vector<at::Tensor> cpu_in;                                          \
    std::vector<at::Tensor> hpu_in;                                          \
    GenerateInputs(n, {{4, 2, 3}, {4, 0, 5}, {128}, {64, 1}, {2, 3, 4, 5}}); \
    for (int i = 0; i < n; ++i) {                                            \
      cpu_in.push_back(GetCpuInput(i));                                      \
      hpu_in.push_back(GetHpuInput(i));                                      \
    }                                                                        \
                                                                             \
    auto exp = outplace_op(cpu_in);                                          \
    auto res = outplace_op(hpu_in);                                          \
                                                                             \
    for (int i = 0; i < n; ++i) {                                            \
      Compare(exp[i], res[i]);                                               \
    }                                                                        \
  }

#define FOREACH_(inplace_op)                                                 \
  TEST_F(HpuOpTest, inplace_op##_) {                                         \
    static constexpr int n = 5;                                              \
    std::vector<at::Tensor> cpu_in;                                          \
    std::vector<at::Tensor> hpu_in;                                          \
    GenerateInputs(n, {{4, 2, 3}, {4, 0, 5}, {128}, {64, 1}, {2, 3, 4, 5}}); \
    for (int i = 0; i < n; ++i) {                                            \
      cpu_in.push_back(GetCpuInput(i));                                      \
      hpu_in.push_back(GetHpuInput(i));                                      \
    }                                                                        \
                                                                             \
    inplace_op##_(cpu_in);                                                   \
    inplace_op##_(hpu_in);                                                   \
                                                                             \
    for (int i = 0; i < n; ++i) {                                            \
      Compare(cpu_in[i], hpu_in[i]);                                         \
    }                                                                        \
  }

#define FOREACH_TESTS(op) FOREACH(_foreach_##op) FOREACH_(_foreach_##op)

/*FOREACH_TESTS(zero)*/
FOREACH_TESTS(exp)
FOREACH_TESTS(sqrt)
FOREACH_TESTS(abs)
FOREACH_TESTS(acos)
FOREACH_TESTS(asin)
FOREACH_TESTS(atan)
FOREACH_TESTS(ceil)
FOREACH_TESTS(cos)
FOREACH_TESTS(cosh)
FOREACH_TESTS(erf) /*FOREACH_TESTS(erfc)*/
FOREACH_TESTS(expm1)
FOREACH_TESTS(floor)
FOREACH_TESTS(log) /*FOREACH_TESTS(log10)*/
FOREACH_TESTS(log1p)
FOREACH_TESTS(log2)
FOREACH_TESTS(neg)
FOREACH_TESTS(tan)
FOREACH_TESTS(tanh)
FOREACH_TESTS(sin)
FOREACH_TESTS(sinh)
FOREACH_TESTS(round)
/*FOREACH_TESTS(lgamma)*/
/*FOREACH_TESTS(frac)*/
FOREACH_TESTS(reciprocal)
FOREACH_TESTS(sigmoid)
FOREACH_TESTS(trunc)
