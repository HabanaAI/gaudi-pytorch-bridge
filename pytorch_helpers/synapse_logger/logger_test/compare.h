/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#pragma once

#include <stdint.h>

typedef union {
  float f;
  struct {
    unsigned int mantisa : 23;
    unsigned int exponent : 8;
    unsigned int sign : 1;
  } parts;
} float_cast;

union bfloat16 {
  uint16_t f;
  struct {
    uint16_t mantisa : 7;
    uint16_t exponent : 8;
    uint16_t sign : 1;
  } parts;
  operator float() const {
    float_cast x;
    x.parts.mantisa = parts.mantisa << (23 - 7);
    x.parts.exponent = parts.exponent;
    x.parts.sign = parts.sign;
    return x.f;
  }
};

template <typename test_t>
bool compare(float* ref, test_t* test, int n);
