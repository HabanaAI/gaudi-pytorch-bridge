###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################

import pytest
import torch

IS_NATIVE_FP8 = hasattr(torch, "float8_e5m2")
FP8_NAMES = ["143", "152"]
FP8_NAMES_LEGACY = FP8_NAMES + [None]  # None falls back to simulating fp8 with torch.int8

MASK_FLOAT32_152 = torch.tensor(2145386496, dtype=torch.int)  # 0 11111111 11000000000000000000000b
MASK_FLOAT32_143 = torch.tensor(2146435072, dtype=torch.int)  # 0 11111111 11100000000000000000000b
MASK_FLOAT32 = {"152": MASK_FLOAT32_152, "143": MASK_FLOAT32_143}

MASK_ROUND_FLOAT32_152 = torch.tensor(1048575, dtype=torch.int)  # 0 00000000 00011111111111111111111b
MASK_ROUND_FLOAT32_143 = torch.tensor(524287, dtype=torch.int)  # 0 00000000 00001111111111111111111b
MASK_ROUND_FLOAT32 = {"152": MASK_ROUND_FLOAT32_152, "143": MASK_ROUND_FLOAT32_143}

EXCESSIVE_BITS_FLOAT32_152 = torch.tensor(21, dtype=torch.int)
EXCESSIVE_BITS_FLOAT32_143 = torch.tensor(20, dtype=torch.int)
EXCESSIVE_BITS_FLOAT32 = {
    "152": EXCESSIVE_BITS_FLOAT32_152,
    "143": EXCESSIVE_BITS_FLOAT32_143,
}

MASK_BFLOAT16_152 = torch.tensor(32736, dtype=torch.short)  # 0 11111111 1100000b
MASK_BFLOAT16_143 = torch.tensor(32752, dtype=torch.short)  # 0 11111111 1110000b
MASK_BFLOAT16 = {"152": MASK_BFLOAT16_152, "143": MASK_BFLOAT16_143}

MASK_ROUND_BFLOAT16_152 = torch.tensor(15, dtype=torch.short)  # 0 00000000 0001111b
MASK_ROUND_BFLOAT16_143 = torch.tensor(7, dtype=torch.short)  # 0 00000000 0000111b
MASK_ROUND_BFLOAT16 = {"152": MASK_ROUND_BFLOAT16_152, "143": MASK_ROUND_BFLOAT16_143}

EXCESSIVE_BITS_BFLOAT16_152 = torch.tensor(5, dtype=torch.short)
EXCESSIVE_BITS_BFLOAT16_143 = torch.tensor(4, dtype=torch.short)
EXCESSIVE_BITS_BFLOAT16 = {
    "152": EXCESSIVE_BITS_BFLOAT16_152,
    "143": EXCESSIVE_BITS_BFLOAT16_143,
}

FP8_MAX_152 = torch.tensor(57344 * 0.9, dtype=torch.float)
FP8_MAX_143 = torch.tensor(240 * 0.9, dtype=torch.float)
FP8_MAX = {"152": FP8_MAX_152, "143": FP8_MAX_143}


def variant_from_dtype(dtype):
    return "152" if (dtype is None or dtype is torch.float8_e5m2) else "143"


def simulateFp8Precision(input, out_dtype=None):
    variant = variant_from_dtype(out_dtype)
    dtype = input.dtype
    if dtype == torch.float:
        int_type = torch.int
        mask = MASK_FLOAT32[variant]
        mask_round = MASK_ROUND_FLOAT32[variant]
        excessive_bits = EXCESSIVE_BITS_FLOAT32[variant]
    else:
        int_type = torch.short
        mask = MASK_BFLOAT16[variant]
        mask_round = MASK_ROUND_BFLOAT16[variant]
        excessive_bits = EXCESSIVE_BITS_BFLOAT16[variant]
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


def dtype_from_string(dtype):
    if dtype == "143":
        return torch.float8_e4m3fn
    if dtype == "152":
        return torch.float8_e5m2
    return dtype


def check_native_fp8(dtype):
    if not IS_NATIVE_FP8 and dtype in FP8_NAMES:
        pytest.skip("Native fp8 types are not supported in pytorch package.")
