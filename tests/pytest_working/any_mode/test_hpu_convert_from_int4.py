###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################

import numpy as np
import pytest
import torch
from test_utils import (
    check_ops_executed_in_jit_ir,
    clear_t_compile_logs,
    compare_tensors,
    is_gaudi1,
    is_pytest_mode_compile,
    is_pytest_mode_eager,
)

pytestmark = [pytest.mark.skipif(is_gaudi1(), reason="Gaudi doesn't support fp8")]

dtypes = [torch.float32, torch.bfloat16, torch.float8_e5m2, torch.float8_e4m3fn]


# packs tensor of int4/uint4 numbers into int32 elements
def pack_1d(input):
    packed_size = int(input.shape[-1] / 8)
    packed = np.empty((packed_size,), dtype=int)
    for b in range(packed_size):
        base2 = ""
        for i in range(8):
            base2 = np.binary_repr(input[b * 8 + i], 4) + base2
        packed[b] = int(base2, 2)
    return packed


def pack_int4_into_int32(input, out_shape):
    input_flat = input.flatten()
    return pack_1d(input_flat).reshape(out_shape)


@pytest.mark.skipif(is_pytest_mode_eager(), reason="convert_from_int4 is not supported in eager mode")
@pytest.mark.parametrize("packed_shape", [(10,), (4, 6, 4)])
@pytest.mark.parametrize("variant", ["int4", "uint4"])
@pytest.mark.parametrize("is_zero_point, packed_zero_point", [(True, True), (True, False), (False, False)])
@pytest.mark.parametrize("is_scale", [True, False])
@pytest.mark.parametrize("out_dtype", dtypes)
def test_convert_from_int4(packed_shape, variant, is_zero_point, packed_zero_point, is_scale, out_dtype):
    if out_dtype in [torch.float8_e5m2, torch.float8_e4m3fn]:
        pytest.skip("https://jira.habana-labs.com/browse/SW-182397")

    fn = getattr(torch.ops.hpu, "convert_from_" + variant)

    # dequantize_4_bits cguid executes subtraction in 8bits dtype if zero_point is 4bits
    sub_dtype = out_dtype
    if packed_zero_point:
        sub_dtype = torch.int8 if variant == "int4" else torch.uint8

    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        fn = torch.compile(fn, backend="hpu_backend")

    # packed_shape relates to input tensor of int4 numbers packed into int32 elements
    real_shape = list(packed_shape)
    real_shape[-1] = real_shape[-1] * 8

    # Generates tensor of range 0-15 to simulate uint4 values.
    # For int4 variant rolls values > 7 to be negative.
    input = torch.randint(0, 16, real_shape, dtype=torch.int)
    input_hpu = torch.tensor(pack_int4_into_int32(input, packed_shape), dtype=torch.int).to("hpu")
    if variant == "int4":
        input = torch.where(input > 7, input - 16, input)
    input = input.to(sub_dtype)

    scale = (torch.randn(real_shape) * 50.0).to(out_dtype) if is_scale else torch.ones(real_shape).to(out_dtype)
    scale_hpu = scale.to("hpu")

    zero_point = torch.tensor(0.0)
    zero_point_hpu = None
    if is_zero_point:
        if packed_zero_point:
            zero_point = torch.randint(1, 5, real_shape, dtype=torch.int)
            # Prevent underflow by assuring zero_point values are no bigger than weights (only for unsigned variant).
            if variant == "uint4":
                zero_point = torch.where(zero_point > input, 0, zero_point)
            zero_point_hpu = torch.tensor(pack_int4_into_int32(zero_point, packed_shape), dtype=torch.int).to("hpu")
            zero_point = zero_point.to(sub_dtype)
        else:
            zero_point = (torch.randn(real_shape) * 5.0).to(out_dtype)
            zero_point_hpu = zero_point.to("hpu")

    result_hpu = fn(input_hpu, scale_hpu, zero_point_hpu, out_dtype)

    # sub i8/u8 is currently not supported by the bridge
    if packed_zero_point:
        subtraction = (input - zero_point).to(out_dtype).to("hpu")
    else:
        subtraction = input.to("hpu") - zero_point.to("hpu")
    result = (subtraction * scale.to("hpu")).cpu()

    compare_tensors(result_hpu, result, atol=0.001, rtol=0.001)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("convert_from_" + variant)
