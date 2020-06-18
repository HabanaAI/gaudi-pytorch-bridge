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

#include <synapse_common_types.h>

namespace synapse_helpers {

inline uint32_t size_of_syn_data_type(synDataType dataType) {
  switch (dataType) {
    case syn_type_int8: // alias to syn_type_fixed
    case syn_type_uint8: // 8-bit unsigned integer
      return 1;
    case syn_type_bf16: // 16-bit float- 8 bits exponent, 7 bits mantissa, 1 bit
                        // sign
    case syn_type_int16: // 16-bit integer
      return 2;
    case syn_type_float: // alias to syn_type_single
    case syn_type_int32: // 32-bit integer
      return 4;
    default:
      return -1; // invalid
  }
}

} // namespace synapse_helpers
