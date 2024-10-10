/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "compare.h"
#include <math.h>
#include <iostream>
template <typename test_t> struct test_condition {};

template <> struct test_condition<float> {
    static constexpr float atol=1.e-6;
    static constexpr float rtol=1.e-6;
};

template <> struct test_condition<bfloat16> {
    static constexpr float atol=1.e-6;
    static constexpr float rtol=1.e-2;
};

template <typename test_t>
bool compare(float* ref, test_t* test, int n) {
  float atol = test_condition<test_t>::atol;
  float rtol = test_condition<test_t>::rtol;
  for (int i = 0; i < n; ++i) {
    float test_v = (float)test[i];
    // abs(A-B) < rtol*abs(B)+atol
    float abs_err = fabs(ref[i] - test_v);
    float prec_margin = rtol * fabs(ref[i]) + atol - abs_err;
    if (prec_margin < 0) {
      std::cerr << "compare failed at index " << i << ", reference value " << ref[i] << ", actual value " << test_v
                << ", absolute error " << abs_err << " " << (abs_err < atol ? "ok " : "FAIL")
                << ", precision prec_margin " << prec_margin << "\n";
      std::cerr << ", sizeof test_t" << sizeof(test_t) << "\n";
      return false;
    }
  }
  return true;
}
template bool compare<float>(float* ref, float* test, int n);
template bool compare<bfloat16>(float* ref, bfloat16* test, int n);
