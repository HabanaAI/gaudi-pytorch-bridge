###############################################################################
#
#  Copyright (c) 2021-2024 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
###############################################################################

import pytest
import torch

fp8_dtypes = [torch.float8_e5m2, torch.float8_e4m3fn]

MASK_FLOAT32_152 = torch.tensor(2145386496, dtype=torch.int)  # 0 11111111 11000000000000000000000b
MASK_FLOAT32_143 = torch.tensor(2146435072, dtype=torch.int)  # 0 11111111 11100000000000000000000b
MASK_FLOAT32 = {torch.float8_e5m2: MASK_FLOAT32_152, torch.float8_e4m3fn: MASK_FLOAT32_143}

MASK_ROUND_FLOAT32_152 = torch.tensor(1048575, dtype=torch.int)  # 0 00000000 00011111111111111111111b
MASK_ROUND_FLOAT32_143 = torch.tensor(524287, dtype=torch.int)  # 0 00000000 00001111111111111111111b
MASK_ROUND_FLOAT32 = {torch.float8_e5m2: MASK_ROUND_FLOAT32_152, torch.float8_e4m3fn: MASK_ROUND_FLOAT32_143}

EXCESSIVE_BITS_FLOAT32_152 = torch.tensor(21, dtype=torch.int)
EXCESSIVE_BITS_FLOAT32_143 = torch.tensor(20, dtype=torch.int)
EXCESSIVE_BITS_FLOAT32 = {
    torch.float8_e5m2: EXCESSIVE_BITS_FLOAT32_152,
    torch.float8_e4m3fn: EXCESSIVE_BITS_FLOAT32_143,
}

MASK_BFLOAT16_152 = torch.tensor(32736, dtype=torch.short)  # 0 11111111 1100000b
MASK_BFLOAT16_143 = torch.tensor(32752, dtype=torch.short)  # 0 11111111 1110000b
MASK_BFLOAT16 = {torch.float8_e5m2: MASK_BFLOAT16_152, torch.float8_e4m3fn: MASK_BFLOAT16_143}

MASK_ROUND_BFLOAT16_152 = torch.tensor(15, dtype=torch.short)  # 0 00000000 0001111b
MASK_ROUND_BFLOAT16_143 = torch.tensor(7, dtype=torch.short)  # 0 00000000 0000111b
MASK_ROUND_BFLOAT16 = {torch.float8_e5m2: MASK_ROUND_BFLOAT16_152, torch.float8_e4m3fn: MASK_ROUND_BFLOAT16_143}

EXCESSIVE_BITS_BFLOAT16_152 = torch.tensor(5, dtype=torch.short)
EXCESSIVE_BITS_BFLOAT16_143 = torch.tensor(4, dtype=torch.short)
EXCESSIVE_BITS_BFLOAT16 = {
    torch.float8_e5m2: EXCESSIVE_BITS_BFLOAT16_152,
    torch.float8_e4m3fn: EXCESSIVE_BITS_BFLOAT16_143,
}

FP8_MAX_152 = torch.tensor(57344 * 0.9, dtype=torch.float)
FP8_MAX_143 = torch.tensor(240 * 0.9, dtype=torch.float)
FP8_MAX = {torch.float8_e5m2: FP8_MAX_152, torch.float8_e4m3fn: FP8_MAX_143}


def simulateFp8Precision(input, out_dtype):
    dtype = input.dtype
    if dtype == torch.float:
        int_type = torch.int
        mask = MASK_FLOAT32[out_dtype]
        mask_round = MASK_ROUND_FLOAT32[out_dtype]
        excessive_bits = EXCESSIVE_BITS_FLOAT32[out_dtype]
    else:
        int_type = torch.short
        mask = MASK_BFLOAT16[out_dtype]
        mask_round = MASK_ROUND_BFLOAT16[out_dtype]
        excessive_bits = EXCESSIVE_BITS_BFLOAT16[out_dtype]
    signs = torch.where(input < 0.0, -1.0, 1.0).to(dtype)
    asInt = input.view(int_type)
    mant_odd = torch.bitwise_and(
        torch.bitwise_right_shift(asInt, excessive_bits),
        torch.tensor(1, dtype=int_type),
    )
    asInt_masked = asInt + mask_round
    asInt_odded = asInt_masked + mant_odd
    masked = torch.bitwise_and(asInt_odded, mask)
    return masked.view(dtype) * signs
