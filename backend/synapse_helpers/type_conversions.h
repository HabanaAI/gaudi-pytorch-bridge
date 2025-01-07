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
