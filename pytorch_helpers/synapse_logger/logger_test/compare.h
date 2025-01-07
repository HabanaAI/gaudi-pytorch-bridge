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
