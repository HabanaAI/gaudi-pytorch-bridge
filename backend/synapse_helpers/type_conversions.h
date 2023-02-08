/*******************************************************************************
 * Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
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
#pragma once

#include <synapse_common_types.h>

namespace synapse_helpers {

inline uint32_t size_of_syn_data_type(synDataType dataType) {
  switch (dataType) {
    case syn_type_int8: // alias to syn_type_fixed
    case syn_type_uint8: // 8-bit unsigned integer
    case syn_type_fp8_143: // 8-bit floating point
    case syn_type_fp8_152: // 8-bit floating point
      return 1;
    case syn_type_bf16: // 16-bit float- 8 bits exponent, 7 bits mantissa, 1 bit
                        // sign
    case syn_type_int16: // 16-bit integer
    case syn_type_fp16: // 16-bit floating point
    case syn_type_uint16: // 16-bit unsigned integer
      return 2;
    case syn_type_float: // alias to syn_type_single, IEEE compliant
    case syn_type_int32: // 32-bit integer
    case syn_type_uint32: // 32-bit unsigned integer
    case syn_type_hb_float: // 32-bit floating point, not compliant with IEEE
      return 4;
    case syn_type_int64: // 64-bit integer
      return 8;
    default:
      return -1; // invalid
  }
}

} // namespace synapse_helpers
