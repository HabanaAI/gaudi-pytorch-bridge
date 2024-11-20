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
/************************************************************************************************
float16 class definition - code origin -
/trees/npu-stack/func-sim/agents/tpc/includes/fma_bfp16.h
*************************************************************************************************
*/

#include <cmath>
#include <cstdint>

class float16 {
 public:
  float16();

  float16(float v);

  static float16 max();

  static float16 min(); /* smallest representable positive */

  static float16 maxNegative(); /* largest (in abs) representable negative */

  void operator=(float16 a) {
    this->val = a.val;
  }

  void operator=(uint16_t a) {
    this->val = a;
  }

  friend bool operator<(float16 a, float16 b);

  friend bool operator>(float16 a, float16 b);

  friend bool operator<=(float16 a, float16 b);

  friend bool operator>=(float16 a, float16 b);

  friend bool operator==(float16 a, float16 b);

  friend bool operator!=(float16 a, float16 b);

  friend float operator+(float a, float16 b);
  friend float operator-(float a, float16 b);
  friend float operator*(float a, float16 b);
  friend float operator/(float a, float16 b);

  friend float float16ToFP32(float16 v);

  friend float16 fp32ToFloat16(float v);

  operator double() const;

  operator float() const;

  uint16_t get_val() {
    return val;
  }

  void set_val(uint16_t a) {
    this->val = a;
  }

  uint16_t val;
};

inline float abs(float16 val) {
  return std::abs(float16ToFP32(val));
}

inline float sqrt(float16 val) {
  return std::sqrt(float16ToFP32(val));
}

static_assert(
    sizeof(float16) == sizeof(uint16_t),
    "reinterpret casting to float16 won't work");

/*
 * Converts float buffer to bf16 buffer.
 * Allocates a new fp16 buffer, the caller is responsible to delete it.
 */
float16* floatBufferToFloat16Buffer(float* floatBuffer, unsigned numElements);
/*
 * Converts float buffer to fp16 buffer that was allocated by the caller.
 */
void floatBufferToFloat16Buffer(
    float* floatBuffer,
    unsigned numElements,
    float16* float16Buffer);
/*
 * Converts fp16 buffer to float buffer.
 * Allocates a new float buffer, the caller is responsible to delete it.
 */
float* float16BufferToFloatBuffer(float16* float16Buffer, uint64_t numElements);
