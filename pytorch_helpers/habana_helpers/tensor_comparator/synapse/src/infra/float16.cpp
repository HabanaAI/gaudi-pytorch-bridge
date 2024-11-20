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

#include "float16.h"
#include "sim_fp16.h"

/*****************************************************************************
Code origin - trees/npu_stack/tpcsim/src/fma_bfp16.cpp
******************************************************************************
*/

float16::float16() {
  this->val = 0;
}

bool operator<(float16 a, float16 b) {
  float a_fp32, b_fp32;
  fp16_to_fp32(a.get_val(), a_fp32);
  fp16_to_fp32(b.get_val(), b_fp32);

  return a_fp32 < b_fp32;
}

bool operator>(float16 a, float16 b) {
  float a_fp32, b_fp32;
  fp16_to_fp32(a.get_val(), a_fp32);
  fp16_to_fp32(b.get_val(), b_fp32);

  return a_fp32 > b_fp32;
}

bool operator<=(float16 a, float16 b) {
  float a_fp32, b_fp32;
  fp16_to_fp32(a.get_val(), a_fp32);
  fp16_to_fp32(b.get_val(), b_fp32);

  return a_fp32 <= b_fp32;
}

bool operator>=(float16 a, float16 b) {
  float a_fp32, b_fp32;
  fp16_to_fp32(a.get_val(), a_fp32);
  fp16_to_fp32(b.get_val(), b_fp32);

  return a_fp32 >= b_fp32;
}

bool operator==(float16 a, float16 b) {
  float a_fp32, b_fp32;
  fp16_to_fp32(a.get_val(), a_fp32);
  fp16_to_fp32(b.get_val(), b_fp32);

  return a_fp32 == b_fp32;
}

bool operator!=(float16 a, float16 b) {
  float a_fp32, b_fp32;
  fp16_to_fp32(a.get_val(), a_fp32);
  fp16_to_fp32(b.get_val(), b_fp32);

  return a_fp32 != b_fp32;
}

float operator+(float a, float16 b) {
  float b_fp32;
  fp16_to_fp32(b.get_val(), b_fp32);

  return a + b_fp32;
}

float operator-(float a, float16 b) {
  float b_fp32;
  fp16_to_fp32(b.get_val(), b_fp32);

  return a - b_fp32;
}

float operator*(float a, float16 b) {
  float b_fp32;
  fp16_to_fp32(b.get_val(), b_fp32);

  return a * b_fp32;
}

float operator/(float a, float16 b) {
  float b_fp32;
  fp16_to_fp32(b.get_val(), b_fp32);

  return a / b_fp32;
}

/*****************************************************************************
 * Synapse additions
 *****************************************************************************/

float16::operator float() const {
  return float16ToFP32(*this);
}

float16::operator double() const {
  return double(float16ToFP32(*this));
}

float16::float16(float v) {
  uint16_t value;
  fp32_to_fp16(v, value, RND_TO_NE);
  this->val = value;
}

float16 float16::max() {
  return float16(uint16_t(FLT_MAX_FP16));
}

float16 float16::min() {
  return float16(uint16_t(FLT_MIN_FP16));
}

float16 float16::maxNegative() {
  return float16(uint16_t(FLT_MINUS_MAX_FP16));
}

float float16ToFP32(float16 v) {
  float retVal = 0.0;
  fp16_to_fp32(v.get_val(), retVal);
  return retVal;
}

float16 fp32ToFloat16(float v) {
  uint16_t val;
  fp32_to_fp16(v, val, RND_TO_NE);
  return float16(val);
}

float16* floatBufferToFloat16Buffer(float* floatBuffer, unsigned numElements) {
  // caller is responsible to delete the allocated buffer
  uint16_t* fp16Buffer = new uint16_t[numElements];
  // foreach element in float buffer - convert to fp16 representation
  for (unsigned i = 0; i < numElements; i++) {
    fp16Buffer[i] = fp32ToFloat16(floatBuffer[i]).val;
  }
  return reinterpret_cast<float16*>(fp16Buffer);
}

void floatBufferToFloat16Buffer(
    float* floatBuffer,
    unsigned numElements,
    float16* fp16Buffer) {
  // caller is responsible to allocate fp16 buffer
  // foreach element in float buffer - convert to fp16 representation
  for (unsigned i = 0; i < numElements; i++) {
    fp16Buffer[i] = fp32ToFloat16(floatBuffer[i]);
  }
}

float* float16BufferToFloatBuffer(
    float16* float16Buffer,
    uint64_t numElements) {
  // caller is responsible to allocate fp16 buffer
  float* floatBuffer = new float[numElements];
  // foreach element in bf16 buffer - convert to float
  for (uint64_t i = 0; i < numElements; i++) {
    floatBuffer[i] = float(float16Buffer[i]);
  }
  return floatBuffer;
}