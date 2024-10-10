#pragma once
/*****************************************************************************
float16 definitions collected from tpcsim project.
 Do not modify this codes without consulting with Hilla Ben-Yaacov
******************************************************************************
*/

/*****************************************************************************
Code origin - trees/npu_stack/tpcsim/includes/fma_bfp16.h
******************************************************************************
*/

#pragma once

#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define RND_TO_NE 0
#define RND_TO_0 1
#define RND_TO_PINF 2
#define RND_TO_NINF 3
#define RND_SR 4
#define RND_CSR 5
#define RND_HALF_AZ 6

#define DEFAULT_NAN_BFP16 0x7FFF
#define DEFAULT_NAN_FP16 0x7FFF
#define DEFAULT_NAN_FP32 0x7FFFFFFF
#define DEFAULT_NAN_FP8 0x7F
#define QNAN_BIT 0x0040
#define QNAN_BIT_FP32 0x00400000
#define DNORM_FTZ 1

// Gaudi2
#define VPE_RM_STOCHASTIC_W_RNE_DNORM 7
#define VPE_RM_NEAREST_EVEN 0
#define VPE_RM_STOCHASTIC_GEN3 4
#define VPE_RM_TO_0_GEN3 1
#define VPE_RM_NINF_GEN3 3
#define VPE_RM_INF 2
#define VPE_RM_RHAZ 6

#define EXP_WIDTH_FP8 5
#define MAN_WIDTH_FP8 2
#define EXP_BIAS_FP8 15
uint8_t getRoundMode();

// lead zero detect - find leading 1
inline uint8_t lzd(uint32_t x) {
  if (x == 0) {
    return 32;
  }
  int i = 31;
  for (; i >= 0; --i) {
    if ((x & (1 << i)) != 0) {
      break;
    }
  }
  return i;
}

// sbs implements select bits x[high:low]
inline uint32_t sbs(uint32_t x, uint8_t high, uint8_t low) {
  return (high == 31) ? (x >> low) : ((x & ((1U << (high + 1)) - 1)) >> low);
}

// cbs implements concatenate bits {x[31-pos:0],y[pos-1,0]}
inline uint32_t cbs(uint32_t x, uint32_t y, uint8_t pos) {
  return ((x << pos) | (y & ((1 << pos) - 1)));
}

// ibs implements insert bits x[high:low] = y[high-low-1:0]
inline uint32_t ibs(uint32_t x, uint32_t high, uint32_t low, uint32_t y) {
  return (high == 31) ? ((x & ((1U << low) - 1)) | (y << low))
                      : ((x & (~((1U << (high + 1)) - (1U << low)))) |
                         ((y << low) & (((1U << (high + 1)) - 1))));
}

inline bool is_nan_fp32(uint32_t x) {
  return ((sbs(x, 30, 23) == 0xFF) && (sbs(x, 22, 0) != 0));
}

inline bool is_inf_fp32(uint32_t x) {
  return ((sbs(x, 30, 23) == 0xFF) && (sbs(x, 22, 0) == 0));
}

inline bool is_denorm_fp32(uint32_t x) {
  return ((sbs(x, 30, 23) == 0x00) && (sbs(x, 22, 0) != 0));
}

inline bool is_zero_fp32(uint32_t x) {
  return ((sbs(x, 30, 23) == 0x00) && (sbs(x, 22, 0) == 0));
}

inline bool is_nan_fp16(uint16_t x) {
  return ((sbs(x, 14, 10) == 0x1F) && (sbs(x, 9, 0) != 0));
}
inline bool is_inf_fp16(uint16_t x) {
  return ((sbs(x, 14, 10) == 0x1F) && (sbs(x, 9, 0) == 0));
}
inline bool is_denorm_fp16(uint16_t x) {
  return ((sbs(x, 14, 10) == 0x00) && (sbs(x, 9, 0) != 0));
}
inline bool is_zero_fp16(uint16_t x) {
  return ((sbs(x, 14, 10) == 0x00) && (sbs(x, 9, 0) == 0));
}

#define SIGN_MASK_FP8 0x80

inline bool fp8_is_zero(uint8_t val) {
  return (val & (~SIGN_MASK_FP8)) ? 0 : 1;
}

inline bool fp8_is_infinity(uint8_t val, uint8_t exponent_offset_fp8) {
  bool isAllExponentBitsSet =
      sbs(val, 6, exponent_offset_fp8) == sbs(0xff, 6, exponent_offset_fp8);
  bool isAllMantissaBitsZero = (sbs(val, exponent_offset_fp8 - 1, 0) == 0);
  return (isAllExponentBitsSet & isAllMantissaBitsZero);
}

inline bool fp8_is_nan(uint8_t val, uint8_t exponent_offset_fp8) {
  bool isAllExponentBitsSet =
      sbs(val, 6, exponent_offset_fp8) == sbs(0xff, 6, exponent_offset_fp8);
  bool isAnyMantissaBitSet = (sbs(val, exponent_offset_fp8 - 1, 0) != 0);
  return (isAllExponentBitsSet & isAnyMantissaBitSet);
}

inline bool fp8_is_negative(uint8_t val) {
  return ((val & SIGN_MASK_FP8) == SIGN_MASK_FP8);
}

inline bool fp8_is_denormal(
    uint8_t val,
    uint8_t exponent_offset_fp8) { // Do not consider zero as denormal
  bool isAllExponentBitsZero = sbs(val, 6, exponent_offset_fp8) == 0;
  bool isAnyMantissaBitSet = (sbs(val, exponent_offset_fp8 - 1, 0) != 0);
  return (isAllExponentBitsZero & isAnyMantissaBitSet);
}
/*****************************************************************************
Code origin - trees/npu-stack/tpcsim/conversions/ConvUtils.h
******************************************************************************
*/
float fp8_to_fp32(
    uint8_t input,
    uint8_t exp_width = EXP_WIDTH_FP8,
    uint8_t man_width = MAN_WIDTH_FP8,
    uint8_t exp_bias = EXP_BIAS_FP8,
    bool clip_fp = false);
uint8_t fp32_to_fp8(
    float input,
    uint8_t exp_width = EXP_WIDTH_FP8,
    uint8_t man_width = MAN_WIDTH_FP8,
    uint8_t exp_bias = EXP_BIAS_FP8,
    int roundingMode = RND_TO_NE,
    int32_t lfsrVal = 0,
    bool ftz_fp8 = 0,
    bool clip_fp = false);
void fp16_to_fp32(uint16_t input, float& output);
void fp32_to_fp16(
    float input,
    uint16_t& output,
    int roundingMode = getRoundMode(),
    int32_t lfsrVal = 0,
    bool clip_fp = false);
int fp_accommodate_rounding(
    uint32_t intValuePreRounding,
    bool roundedMSB,
    bool roundedLSBs,
    unsigned int sign,
    int roundingMode,
    uint32_t lfsrVal = 0,
    uint32_t discardedAlignedLeft = 0);

inline int lzcnt(uint32_t bits, uint32_t int_num) {
  int msb = bits - 1;
  int lsb = 0;
  int i = msb;
  for (; i >= lsb; --i) {
    if ((int_num & (1 << i)) != 0) {
      break;
    }
  }
  return bits - i - 1;
}

/*****************************************************************************
Code origin - trees/npu-stack/tpcsim/src/VPE_ISA_GEN.h
******************************************************************************
*/
#define SIGN_OFFSET_FP32 31
#define SIGN_MASK_FP32 0x80000000
#define EXPONENT_OFFSET_FP32 23
#define EXPONENT_MASK_FP32 0x7F800000
#define EXPONENT_BIAS_FP32 127
#define SIGNIFICAND_MASK_FP32 0x007FFFFF
#define NAN_FP32 0x7fffffff
#define MINUS_INF_FP32 0xff800000
#define PLUS_INF_FP32 0x7f800000

#define UNIT_VAL_BF16 0x3F80
#define SIGN_OFFSET_BF16 15
#define SIGN_MASK_BF16 0x8000
#define EXPONENT_OFFSET_BF16 7
#define EXPONENT_MASK_BF16 0x7F80
#define EXPONENT_BIAS_BF16 127
#define SIGNIFICAND_MASK_BF16 0x007F
#define MINUS_NAN_BF16 0xFFFF
#define PLUS_NAN_BF16 0x7FFF
#define MINUS_INF_BF16 0xFF80
#define PLUS_INF_BF16 0x7F80
#define FLT_MIN_BF16 0x0080
#define FLT_MAX_BF16 0x7f7f
#define FLT_MINUS_MAX_BF16 0xff7f

#define UNIT_VAL_FP16 0x3c00
#define SIGN_OFFSET_FP16 15
#define SIGN_MASK_FP16 0x8000
#define EXPONENT_OFFSET_FP16 10
#define EXPONENT_MASK_FP16 0x7C00
#define EXPONENT_BIAS_FP16 15
#define SIGNIFICAND_MASK_FP16 0x03FF
#define MINUS_NAN_FP16 0xffff
#define PLUS_NAN_FP16 0x7fff
#define MINUS_INF_FP16 0xfc00
#define PLUS_INF_FP16 0x7c00
#define FLT_MIN_FP16 0x0400
#define FLT_MAX_FP16 0x7bff
#define FLT_MINUS_MAX_FP16 0xfbff

#define UNIT_VAL_FP8_152 0x3c
#define SIGN_OFFSET_FP8_152 7
#define EXPONENT_OFFSET_FP8_152 2
#define EXPONENT_MASK_FP8_152 0x7C
#define EXPONENT_BIAS_FP8_152 15
#define SIGNIFICAND_MASK_FP8_152 0x03
#define MINUS_NAN_FP8_152 0xff
#define PLUS_NAN_FP8_152 0x7f
#define MINUS_INF_FP8_152 0xfc
#define PLUS_INF_FP8_152 0x7c
#define FLT_MIN_FP8_152 0x04
#define FLT_MAX_FP8_152 0x7b
#define FLT_MINUS_MAX_FP8_152 0xfb
