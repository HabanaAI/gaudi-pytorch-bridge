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

#include "sim_fp16.h"
/*****************************************************************************
float16 implementation collected from tpcsim project.
Do not modify this codes without consulting with Hilla Ben-Yaacov
******************************************************************************
*/

/*****************************************************************************
Code origin - trees/npu_stack/tpcsim/src/fma_bfp16.cpp
******************************************************************************
*/
uint8_t getRoundMode() {
  int roundingMode = fegetround();

  uint8_t roundingModeToSet = RND_TO_NE;
  switch (roundingMode) {
    case FE_TONEAREST:
      roundingModeToSet = RND_TO_NE;
      break;
    case FE_UPWARD:
      roundingModeToSet = RND_TO_PINF;
      break;
    case FE_DOWNWARD:
      roundingModeToSet = RND_TO_NINF;
      break;
    case FE_TOWARDZERO:
      roundingModeToSet = RND_TO_0;
      break;
  }
  return roundingModeToSet;
}

/*****************************************************************************
Code origin - trees/npu-stack/tpcsim/conversions/ConvUtils.cpp
******************************************************************************
*/
int fp_accommodate_rounding(
    uint32_t intValuePreRounding,
    bool roundedMSB,
    bool roundedLSBs,
    unsigned int sign,
    int roundingMode,
    uint32_t lfsrVal,
    uint32_t discardedAlignedLeft) {
  uint32_t result = 0;
  result = intValuePreRounding;
  switch (roundingMode) {
    case RND_TO_0:
      result = intValuePreRounding;
      break;
    case RND_TO_PINF:
      if ((sign == 0) && ((roundedMSB == 1) || (roundedLSBs == 1))) {
        result = intValuePreRounding + 1;
      }
      break;
    case RND_TO_NINF:
      if ((sign == 1) && ((roundedMSB == 1) || (roundedLSBs == 1))) {
        result = intValuePreRounding + 1;
      }
      break;
    case RND_HALF_AZ:
      if (roundedMSB == 1) // half or above half will be rounded away from zero
      {
        result = intValuePreRounding + 1;
      }
      break;
    case RND_SR:
      if (discardedAlignedLeft >= lfsrVal) {
        result = intValuePreRounding + 1;
      }
      break;
    case RND_TO_NE:
    default:
      if ((((intValuePreRounding & 0x1) == 1) && (roundedMSB == 1)) ||
          (((intValuePreRounding & 0x1) == 0) && (roundedMSB == 1) &&
           (roundedLSBs == 1))) {
        result = intValuePreRounding + 1;
      }
      break;
  }
  return result;
}

/*****************************************************************************
Code origin - trees/npu-stack/tpcsim/conversions/FP2FP.cpp
******************************************************************************
*/
void fp32_to_fp16(
    float input,
    uint16_t& output,
    int roundingMode,
    int32_t lfsrVal,
    bool clip_fp) {
  int inputExponent, inputSign, unbiasedExp = 0;
  uint32_t inputMantissa;
  bool roundedMSB = 0, roundedLSBs = 0;
  int minExp = -25;
  int minNormExp = -14;
  int maxExp = 15;

  /*
   * convert the float input to bits
   * note - changed from original code to comply with strict aliasing
   * compilation rules
   */
  const uint32_t* inputUintPtr = reinterpret_cast<const uint32_t*>(&input);
  const uint32_t& inputUint = *inputUintPtr;

  inputMantissa = (inputUint & SIGNIFICAND_MASK_FP32);
  inputExponent = (inputUint & EXPONENT_MASK_FP32) >> EXPONENT_OFFSET_FP32;
  inputSign = (inputUint & SIGN_MASK_FP32) >> SIGN_OFFSET_FP32;

  int32_t outputExponent;
  int32_t outputMantissa;
  int32_t outputSign = inputSign;
  if (is_nan_fp32(inputUint)) {
    // return the same NAN always (0x7FFF), as NVDA does
    outputSign = 0x0;
    outputExponent = 0x1F;
    outputMantissa = 0x3FF;
  } else if (is_zero_fp32(inputUint)) {
    // return +-0
    outputExponent = 0x0;
    outputMantissa = 0x0;
  } else if (is_inf_fp32(inputUint)) {
    // return +-infinity
    outputExponent = 0x1F;
    outputMantissa = 0x0;
  } else {
    // Valid number
    unbiasedExp = inputExponent - EXPONENT_BIAS_FP32;
    inputMantissa |= (1 << EXPONENT_OFFSET_FP32);

    if (unbiasedExp > maxExp) {
      if ((roundingMode == RND_TO_0) ||
          (inputSign && (roundingMode == RND_TO_PINF)) ||
          (!inputSign && (roundingMode == RND_TO_NINF)))

      { // +- 65504.0 - that's what NVDA does
        outputMantissa = 0x3FF;
        outputExponent = maxExp + EXPONENT_BIAS_FP16;
      } else { // +-infinity
        outputExponent = 0x1F;
        outputMantissa = 0x0;
      }
    } else if (unbiasedExp < minExp) {
      // The result will be either 0 or 0x1
      roundedMSB = 0;
      roundedLSBs = 1;
      outputMantissa = fp_accommodate_rounding(
          0, roundedMSB, roundedLSBs, inputSign, roundingMode, lfsrVal);
      outputExponent = 0x0;
    } else { // minExp <= unbiasedExp <= maxExp
      outputExponent = unbiasedExp;
      int rc_bit_idx = (unbiasedExp < minNormExp)
          ? -(unbiasedExp + 2)
          : (EXPONENT_OFFSET_FP32 - EXPONENT_OFFSET_FP16 - 1);
      roundedMSB = (inputMantissa >> rc_bit_idx) & 0x1;
      roundedLSBs = inputMantissa & ((1 << rc_bit_idx) - 1);
      uint32_t discardedAlignedLeft = inputMantissa << (31 - rc_bit_idx);
      outputMantissa = inputMantissa >> (rc_bit_idx + 1);
      outputMantissa = fp_accommodate_rounding(
          outputMantissa,
          roundedMSB,
          roundedLSBs,
          inputSign,
          roundingMode,
          lfsrVal,
          discardedAlignedLeft);
      if (((unbiasedExp < minNormExp) &&
           (outputMantissa & (1 << EXPONENT_OFFSET_FP16))) ||
          (outputMantissa &
           (1 << (EXPONENT_OFFSET_FP16 + 1)))) { // Should handle two cases:
        // 1. The number was denormal, and after rounding became normal
        // 2. The number was rounded to the 1.0 * 2^(next exponent)
        outputExponent = outputExponent + 1;
      }
      if (outputExponent > maxExp) {
        // return infinity
        outputExponent = 0x1F;
        outputMantissa = 0x0;
      } else {
        if (outputExponent < minNormExp) {
          outputExponent = 0x0;
        } else {
          outputExponent += EXPONENT_BIAS_FP16;
        }
        // normalize - leave 10 bits
        outputMantissa &= SIGNIFICAND_MASK_FP16;
      }
    }
  }
  output = outputMantissa | (outputExponent << EXPONENT_OFFSET_FP16) |
      (outputSign << SIGN_OFFSET_FP16);

  if (clip_fp & is_inf_fp16(output)) {
    outputMantissa = 0x3FF;
    outputExponent = 0x1E;
    output = outputMantissa | (outputExponent << EXPONENT_OFFSET_FP16) |
        (inputSign << SIGN_OFFSET_FP16);
  }
}

void fp16_to_fp32(uint16_t input, float& output) {
  const uint16_t& inputUint = input;
  uint32_t& outputUint = *(uint32_t*)&output;

  int32_t inputMantissa = (inputUint & SIGNIFICAND_MASK_FP16);
  int32_t inputExponent =
      (inputUint & EXPONENT_MASK_FP16) >> EXPONENT_OFFSET_FP16;
  int32_t inputSign = (inputUint & SIGN_MASK_FP16) >> SIGN_OFFSET_FP16;

  int32_t outputExponent;
  int32_t outputMantissa;
  int32_t outputSign = inputSign;

  if (is_zero_fp16(inputUint)) {
    outputExponent = 0x0;
    outputMantissa = 0x0;
  } else if (is_nan_fp16(inputUint)) {
    outputExponent = 0xFF;
    outputMantissa = 0x007FFFFF;
    outputSign = 0;
  } else if (is_inf_fp16(inputUint)) {
    outputExponent = 0xFF;
    outputMantissa = 0x0;
  } else {
    outputExponent = inputExponent - EXPONENT_BIAS_FP16 + EXPONENT_BIAS_FP32;
    int32_t mantissaForAdjustment = inputMantissa;
    if (is_denorm_fp16(inputUint)) {
      int shift = lzcnt(EXPONENT_OFFSET_FP16, inputMantissa);
      // Shift leading 1 to bit 10 (normalize) and fixup the exponent
      // accordingly
      mantissaForAdjustment =
          (inputMantissa << (shift + 1)) & SIGNIFICAND_MASK_FP16;
      outputExponent -= shift;
    }
    // Normal case
    outputMantissa = mantissaForAdjustment
        << (EXPONENT_OFFSET_FP32 - EXPONENT_OFFSET_FP16);
  }

  outputUint = outputMantissa | outputExponent << EXPONENT_OFFSET_FP32 |
      outputSign << SIGN_OFFSET_FP32;
}

uint8_t fp32_to_fp8(
    float input,
    uint8_t exp_width,
    uint8_t man_width,
    uint8_t exp_bias,
    int roundingMode,
    int32_t lfsrVal,
    bool ftz_fp8,
    bool clip_fp) {
  uint8_t output;
  int inputExponent, inputSign, unbiasedExp = 0;
  uint32_t inputMantissa;
  bool roundedMSB = 0, roundedLSBs = 0;
  // int minExp = -25;
  int minNormExp = 1 - exp_bias; //-14
  int maxExp = ((1 << exp_width) - 1) - exp_bias - 1; // 15
  int minExp = minNormExp - man_width -
      1; //-25 //min denormal value can come from rounding of 0.5
  int32_t exponent_offset_fp8 = man_width;
  int32_t sign_offset_fp8 = 7;
  int roundingModeDnorm = (roundingMode == VPE_RM_STOCHASTIC_W_RNE_DNORM)
      ? VPE_RM_NEAREST_EVEN
      : roundingMode;
  int roundingModeNorm = (roundingMode == VPE_RM_STOCHASTIC_W_RNE_DNORM)
      ? VPE_RM_STOCHASTIC_GEN3
      : roundingMode;

  /*
   * convert the float input to bits
   * note - changed from original code to comply with strict aliasing
   * compilation rules
   */
  const uint32_t* inputUintPtr = reinterpret_cast<const uint32_t*>(&input);
  const uint32_t& inputUint = *inputUintPtr;

  inputMantissa = (inputUint & SIGNIFICAND_MASK_FP32);
  inputExponent = (inputUint & EXPONENT_MASK_FP32) >> EXPONENT_OFFSET_FP32;
  inputSign = (inputUint & SIGN_MASK_FP32) >> SIGN_OFFSET_FP32;

  int32_t outputExponent;
  int32_t outputMantissa;
  int32_t outputSign = inputSign;
  if (is_nan_fp32(inputUint)) {
    // return the same NAN always (0x7F)
    outputSign = 0x0;
    outputExponent = sbs(0xff, exp_width - 1, 0); // 0x1F;
    outputMantissa = sbs(0xff, man_width - 1, 0); // 0x3;
  } else if (is_zero_fp32(inputUint)) {
    // return +-0
    outputExponent = 0x0;
    outputMantissa = 0x0;
  } else if (is_inf_fp32(inputUint)) {
    // return +-infinity
    outputExponent = sbs(0xff, exp_width - 1, 0); // 0x1F;
    outputMantissa = 0x0;
  } else {
    // Valid number
    unbiasedExp = inputExponent - EXPONENT_BIAS_FP32;
    inputMantissa |= (1 << EXPONENT_OFFSET_FP32);

    if (unbiasedExp > maxExp) {
      if ((roundingMode == (VPE_RM_TO_0_GEN3)) ||
          (inputSign && (roundingMode == (VPE_RM_INF))) ||
          (!inputSign && (roundingMode == (VPE_RM_NINF_GEN3))))

      { // +- max_normal
        outputMantissa = sbs(0xff, man_width - 1, 0); // 0x3;
        outputExponent = maxExp + exp_bias;
      } else { // +-infinity
        outputExponent = sbs(0xff, exp_width - 1, 0); // 0x1F;
        outputMantissa = 0x0;
      }
    } else if (unbiasedExp < minExp) {
      // The result will be either 0 or 0x1
      roundedMSB = 0;
      roundedLSBs = 1;
      if (ftz_fp8 || (roundingModeDnorm == VPE_RM_STOCHASTIC_GEN3))
        outputMantissa = 0;
      else
        outputMantissa = fp_accommodate_rounding(
            0,
            roundedMSB,
            roundedLSBs,
            inputSign,
            roundingModeDnorm,
            lfsrVal,
            0);
      outputExponent = 0x0;
    } else { // minExp <= unbiasedExp <= maxExp
      outputExponent = unbiasedExp;
      int rc_bit_idx = (unbiasedExp < minNormExp)
          ? ((EXPONENT_OFFSET_FP32 - exponent_offset_fp8 - 1) +
             (minNormExp - unbiasedExp))
          : (EXPONENT_OFFSET_FP32 - exponent_offset_fp8 - 1);
      roundedMSB = (((inputMantissa >> rc_bit_idx)) & 0x1) != 0;
      roundedLSBs = (inputMantissa & ((1 << rc_bit_idx) - 1)) != 0;
      uint32_t discardedAlignedLeft = inputMantissa << (31 - rc_bit_idx);
      outputMantissa = inputMantissa >> (rc_bit_idx + 1);
      if (unbiasedExp < minNormExp) {
        if (ftz_fp8)
          outputMantissa = 0;
        else
          outputMantissa = fp_accommodate_rounding(
              outputMantissa,
              roundedMSB,
              roundedLSBs,
              inputSign,
              roundingModeDnorm,
              lfsrVal,
              discardedAlignedLeft);
      } else
        outputMantissa = fp_accommodate_rounding(
            outputMantissa,
            roundedMSB,
            roundedLSBs,
            inputSign,
            roundingModeNorm,
            lfsrVal,
            discardedAlignedLeft);
      if (((unbiasedExp < minNormExp) &&
           (outputMantissa & (1 << exponent_offset_fp8))) ||
          (outputMantissa &
           (1
            << (exponent_offset_fp8 + 1)))) { // Should handle two cases:
                                              // 1. The number was denormal, and
                                              // after rounding became normal
                                              // 2. The number was rounded to
                                              // the 1.0 * 2^(next exponent)
        outputExponent = outputExponent + 1;
      }
      if (outputExponent > maxExp) {
        // return infinity
        outputExponent = sbs(0xff, exp_width - 1, 0); // 0x1F;
        outputMantissa = 0x0;
      } else {
        if (outputExponent < minNormExp) {
          outputExponent = 0x0;
          if (ftz_fp8)
            outputMantissa = 0;
        } else {
          outputExponent += exp_bias;
        }
        // normalize - leave man_width bits
        outputMantissa = sbs(outputMantissa, man_width - 1, 0);
      }
    }
  }
  output = outputMantissa | (outputExponent << exponent_offset_fp8) |
      (outputSign << sign_offset_fp8);

  if (clip_fp & fp8_is_infinity(output, exponent_offset_fp8)) {
    // outputExponent = sbs(0xff, exp_width - 1, 0) - 1;//exponent is max_exp-1
    // (all ones minus 1) outputMantissa = sbs(0xff, man_width - 1, 0);
    // //mantissa is all ones output = outputMantissa | (outputExponent <<
    // exponent_offset_fp8) | (outputSign << sign_offset_fp8);
    output = output - 1; // will return +/-max_norm_value
  }

  return output;
}

float fp8_to_fp32(
    uint8_t input,
    uint8_t exp_width,
    uint8_t man_width,
    uint8_t exp_bias,
    bool clip_fp) {
  const uint8_t inputUint = input;
  union {
    uint32_t u;
    float f;
  } retVal;
  uint32_t& outputUint = retVal.u;
  int32_t exponent_offset_fp8 = man_width;
  int32_t sign_offset_fp8 = 7;

  int32_t inputMantissa = sbs(inputUint, man_width - 1, 0);
  int32_t inputExponent = sbs(inputUint, 6, exponent_offset_fp8);
  int32_t inputSign = sbs(inputUint, sign_offset_fp8, sign_offset_fp8);

  int32_t outputExponent;
  int32_t outputMantissa;
  int32_t outputSign = inputSign;

  if (fp8_is_zero(inputUint)) {
    outputExponent = 0x0;
    outputMantissa = 0x0;
  } else if (fp8_is_nan(inputUint, exponent_offset_fp8)) {
    outputExponent = 0xFF;
    outputMantissa = 0x007FFFFF;
    outputSign = 0;
  } else if (fp8_is_infinity(inputUint, exponent_offset_fp8)) {
    outputExponent = 0xFF;
    outputMantissa = 0x0;
  } else {
    outputExponent = inputExponent - exp_bias + EXPONENT_BIAS_FP32;
    int32_t mantissaForAdjustment = inputMantissa;
    if (fp8_is_denormal(inputUint, exponent_offset_fp8)) {
      int shift = lzcnt(exponent_offset_fp8, inputMantissa);
      // Shift leading 1 (normalize) and fixup the exponent accordingly
      mantissaForAdjustment =
          sbs((inputMantissa << (shift + 1)), man_width - 1, 0);
      outputExponent -= shift;
    }
    outputMantissa = mantissaForAdjustment
        << (EXPONENT_OFFSET_FP32 - exponent_offset_fp8);
  }
  outputUint = outputMantissa | outputExponent << EXPONENT_OFFSET_FP32 |
      outputSign << SIGN_OFFSET_FP32;

  if (clip_fp) {
    if (is_inf_fp32(outputUint)) {
      //	outputMantissa = 0x7FFFFF; //all ones
      //	outputExponent = 0xFE; //max_exp-1
      //	outputUint = outputMantissa | outputExponent <<
      // EXPONENT_OFFSET_FP32 | outputSign << SIGN_OFFSET_FP32;

      outputUint = outputUint - 1; // will return +/-max_norm_value
    }
  }
  return retVal.f;
}
