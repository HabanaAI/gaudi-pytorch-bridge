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

#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define FLOAT_BF16_MIN_VAL (0x0080)
#define FLOAT_BF16_MAX_VAL (0x7f7f)
#define EXPONENT_OFFSET_FP32 (23)

inline float bf16ToFloat(uint16_t a) {
  uint32_t val_32b = ((uint32_t)a) << 16;
  float* val_fp32 = reinterpret_cast<float*>(&val_32b);
  return *val_fp32;
}

inline uint16_t floatToBf16(float input) {
  uint32_t* val_32b = reinterpret_cast<uint32_t*>(&input);
  uint32_t inputUint = *val_32b;
  uint16_t res;

  if (std::isnan(input) || std::isinf(input)) {
    return *val_32b >> 16;
  } else {
    uint32_t inputSign = (inputUint & (1UL << 31)) >> 31;
    bool roundedMSB = ((inputUint & (1 << 15)) != 0);

    int32_t inputExponent = (inputUint >> EXPONENT_OFFSET_FP32) & 0xFF;

    int32_t outputExponent = inputExponent;

    uint32_t inputMantissa =
        inputUint & ((1 << (EXPONENT_OFFSET_FP32 + 1)) - 1);
    inputMantissa |= (1 << EXPONENT_OFFSET_FP32);

    int32_t outputMantissa = inputMantissa >> 16;

    if (roundedMSB) {
      outputMantissa++;
    }
    if (outputMantissa & (1 << 8)) {
      outputExponent++;
    }
    res = (inputSign << 15) | (outputExponent << 7) | (outputMantissa & 0x7F);
  }
  return res;
}

inline float floatTobf16ToFloat(float a) {
  return bf16ToFloat(floatToBf16(a));
}

/*
Multiply two Bfloat parameters
*/
inline uint16_t bf16Mult(uint16_t a, uint16_t b) {
  float dst = bf16ToFloat(a);
  float src = bf16ToFloat(b);
  float res = dst * src;
  uint16_t res_bf16 = floatToBf16(res);
  return res_bf16;
}

#define OVERLOAD_OPERATOR(op, ret)                         \
  ret operator op(const bfloat16& lhs, const float& rhs) { \
    return (bf16ToFloat(lhs.val) op rhs);                  \
  }

class bfloat16 {
 public:
  bfloat16(float v = 0) {
    val = floatToBf16(v);
  }

  float operator-(float rhs) {
    return bf16ToFloat(val) - rhs;
  }
  bool operator<(float rhs) const {
    return bf16ToFloat(val) < rhs;
  }
  bool operator>(float rhs) const {
    return bf16ToFloat(val) > rhs;
  }

  operator float() const {
    return bf16ToFloat(val);
  }

  friend bool operator==(const bfloat16& lhs, const bfloat16& rhs) {
    return lhs.val == rhs.val;
  }

  friend bool operator!=(const bfloat16& lhs, const bfloat16& rhs) {
    return !(lhs.val == rhs.val);
  }

  uint16_t val;
};

inline OVERLOAD_OPERATOR(-, float);
inline OVERLOAD_OPERATOR(*, float);

inline bfloat16 sqrt(bfloat16 val) {
  return std::sqrt(float(val));
}
inline bfloat16 abs(bfloat16 val) {
  return std::abs(float(val));
}
inline bfloat16 abs2(bfloat16 val) {
  return float(val) * float(val);
}

/*
 * Converts float buffer to bf16 buffer
 * Caller is responsible to delete the allocated buffer
 */
inline bfloat16* floatBufferToBf16Buffer(
    float* floatBuffer,
    unsigned numElements) {
  uint16_t* bf16Buffer = new uint16_t[numElements];
  // foreach element in float buffer - convert to bf16 representation
  for (unsigned i = 0; i < numElements; i++) {
    bf16Buffer[i] = floatToBf16(floatBuffer[i]);
  }
  return reinterpret_cast<bfloat16*>(bf16Buffer);
}

/*
 * Converts bf16 buffer to float buffer
 * Caller is responsible to delete the allocated buffer
 */
inline float* bf16BufferTofloatBuffer(
    bfloat16* bf16Buffer,
    unsigned numElements) {
  float* floatBuffer = new float[numElements];
  // foreach element in bf16 buffer - convert to float
  for (unsigned i = 0; i < numElements; i++) {
    floatBuffer[i] = float(bf16Buffer[i]);
  }
  return floatBuffer;
}

static_assert(
    sizeof(bfloat16) == sizeof(uint16_t),
    "reinterpret casting to bfloat16 won't work");
