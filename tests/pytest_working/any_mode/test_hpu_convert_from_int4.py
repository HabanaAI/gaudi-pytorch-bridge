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


import math

import habana_frameworks.torch.core as htcore
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

pytestmark = [pytest.mark.skipif(is_gaudi1(), reason="Gaudi doesn't support int4")]

dtypes = [torch.float32, torch.bfloat16, torch.float8_e5m2, torch.float8_e4m3fn]
dtypes_str = ["fp32", "bf16", "f8_e5m2", "f8_e4m3fn"]


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


def pack_tensor(input, bits=4):
    normal = input.to(torch.int32)
    q = torch.zeros((normal.shape[0], normal.shape[1] // 32 * bits), dtype=torch.int32)
    i = 0
    col = 0
    while col < q.shape[1]:
        if bits in [2, 4, 8]:
            for j in range(i, i + (32 // bits)):
                q[:, col] |= normal[:, j] << (bits * (j - i))
            i += 32 // bits
            col += 1
        else:
            raise NotImplementedError("Only 2,4,8 bits are supported.")
    q = q.to(torch.int32)
    return q


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


def pack_cuda_old(linear_weight, scales, zeros, g_idx, infeatures, outfeatures, group_size, bits):
    W = linear_weight.data.clone()

    scales = scales.t().contiguous()
    zeros = zeros.t().contiguous()
    scale_zeros = zeros * scales
    cuda_scales = scales.clone().to(dtype=linear_weight.dtype)

    intweight = []
    for idx in range(infeatures):
        g_idx = idx // group_size
        intweight.append(torch.round((W[:, idx] + scale_zeros[g_idx]) / cuda_scales[g_idx]).to(torch.int)[:, None])
    intweight = torch.cat(intweight, dim=1)
    intweight = intweight.t().contiguous()
    intweight = intweight.numpy().astype(np.uint32)

    i = 0
    row = 0
    qweight = np.zeros((intweight.shape[0] // 32 * bits, intweight.shape[1]), dtype=np.uint32)
    while row < qweight.shape[0]:
        if bits in [2, 4, 8]:
            for j in range(i, i + (32 // bits)):
                qweight[row] |= intweight[j] << (bits * (j - i))
            i += 32 // bits
            row += 1
        else:
            raise NotImplementedError("Only 2,4,8 bits are supported.")

    qweight = qweight.astype(np.int32)
    cuda_qweight = torch.from_numpy(qweight)

    zeros -= 1
    zeros = zeros.numpy().astype(np.uint32)
    qzeros = np.zeros((zeros.shape[0], zeros.shape[1] // 32 * bits), dtype=np.uint32)
    i = 0
    col = 0
    while col < qzeros.shape[1]:
        if bits in [2, 4, 8]:
            for j in range(i, i + (32 // bits)):
                qzeros[:, col] |= zeros[:, j] << (bits * (j - i))
            i += 32 // bits
            col += 1
        else:
            raise NotImplementedError("Only 2,3,4,8 bits are supported.")

    qzeros = qzeros.astype(np.int32)
    cuda_qzeros = torch.from_numpy(qzeros)
    return cuda_qweight, cuda_qzeros, cuda_scales


def unpack_weight_cuda(qweight, bits, wf, group_size, cuda):
    if bits in [2, 4, 8]:
        weight = torch.bitwise_right_shift(
            torch.unsqueeze(qweight, 1).expand(-1, 32 // bits, -1),
            wf.unsqueeze(-1),
        ).to(torch.int16 if bits == 8 else torch.int8)
        weight = torch.bitwise_and(weight, (2**bits) - 1)
        if cuda:
            weight = weight.reshape(-1, group_size, weight.shape[2])
        else:
            weight = weight.reshape(-1, weight.shape[2])
    else:
        raise NotImplementedError("Only 2,4,8 bits are supported.")
    return weight


def unpack_zeros_cuda_old(in_qzeros, in_scales, bits, wf, cuda):
    zeros = torch.bitwise_right_shift(
        torch.unsqueeze(in_qzeros, 2).expand(-1, -1, 32 // bits),
        wf.unsqueeze(0),
    ).to(torch.int16 if bits == 8 else torch.int8)

    zeros = zeros + 1
    zeros = torch.bitwise_and(zeros, (2**bits) - 1).to(
        in_scales.dtype
    )  # NOTE: It appears that casting here after the `zeros = zeros + 1` is important.

    if cuda:
        zeros = zeros.reshape(-1, 1, zeros.shape[1] * zeros.shape[2])
    else:
        zeros = zeros.reshape(-1, zeros.shape[1] * zeros.shape[2])
    return zeros


def get_weight_cuda_old(use_zeros, bits, in_qweight, in_qzeros, in_scales, in_group_size):
    if bits in [2, 4, 8]:
        wf = torch.tensor(list(range(0, 32, bits)), dtype=torch.int32).unsqueeze(0)
    if wf.device != in_scales.device:
        wf = wf.to(in_scales.device)

    if bits in [2, 4, 8]:
        zeros = unpack_zeros_cuda_old(in_qzeros, in_scales, bits, wf, True)

        scales = in_scales
        scales = scales.reshape(-1, 1, scales.shape[-1])

        weight = torch.bitwise_right_shift(
            torch.unsqueeze(in_qweight, 1).expand(-1, 32 // bits, -1),
            wf.unsqueeze(-1),
        ).to(torch.int16 if bits == 8 else torch.int8)
        weight = torch.bitwise_and(weight, (2**bits) - 1)
        weight = weight.reshape(-1, in_group_size, weight.shape[2])
    else:
        raise NotImplementedError("Only 2,3,4,8 bits are supported.")

    weight = scales * (weight - zeros) if use_zeros else scales * weight
    weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])
    return weight


# taken from AutoGPTQ/tests/test_repacking.py
def gen_quant4(k, n, groupsize=-1):
    maxq = 2**4 - 1
    w = torch.randn((k, n), dtype=torch.bfloat16, device="cpu")

    original_w = w.clone()

    if groupsize != -1:
        w = w.reshape((-1, groupsize, n))
        w = w.permute(1, 0, 2)
        w = w.reshape((groupsize, -1))

    s = torch.max(torch.abs(w), 0, keepdim=True)[0]
    s *= 2 / maxq

    # Quantize.
    w = torch.round(w / s).int()

    # Unsigned storage.
    w += (maxq + 1) // 2
    w = torch.clamp(w, 0, maxq)

    # Dequantize.
    ref = (w - (maxq + 1) // 2).bfloat16() * s

    if groupsize != -1:

        def reshape(w):
            w = w.reshape((groupsize, -1, n))
            w = w.permute(1, 0, 2)
            w = w.reshape((k, n)).contiguous()
            return w

        ref = reshape(ref)
        w = reshape(w)

    s = s.reshape((-1, n)).contiguous()
    linear = torch.nn.Linear(k, n, bias=False)
    linear.weight.data = ref.t()

    return original_w, linear, s


import os


def prepare_data(
    infeatures,
    outfeatures,
    group_size,
    bits,
    variant,
    out_dtype,
    is_zero_point,
    packed_zero_point,
    input_values,
    scale_values,
    zeros_values,
):
    _, linear, s = gen_quant4(infeatures, outfeatures, group_size)

    input = linear.weight.to(out_dtype)
    scale = s.to(out_dtype)
    if scale_values == "1":
        scale = torch.ones_like(scale)
    if scale_values == "range" or scale_values == "range_int":
        range_t = torch.tensor(list(range(1, infeatures + 1)), dtype=torch.int32)
        shape_s = s.shape
        scale = (
            (torch.ones(s.numel(), dtype=torch.int32).reshape(-1, infeatures) * range_t).reshape(shape_s).contiguous()
        )
        if scale_values == "range":
            scale = scale / 10.0
    htcore.mark_step()

    packed_shape_zero = (math.ceil(infeatures / group_size), outfeatures)
    real_shape_zero = list(packed_shape_zero)
    real_shape_zero[0] = real_shape_zero[0] * 8
    zeros = None
    if is_zero_point:
        zeros = torch.ones((infeatures // group_size, outfeatures), dtype=torch.int32)
        if zeros_values == "range":
            if variant == "int4":
                range_t_zeros = torch.tensor(list(range(1, 5)), dtype=torch.int32)
                seq_size = 4
            else:
                range_t_zeros = torch.tensor(list(range(1, 9)), dtype=torch.int32)
                seq_size = 8
            shape_z = zeros.shape
            zeros = (
                (torch.ones(zeros.numel(), dtype=torch.int32).reshape(-1, seq_size) * range_t_zeros)
                .reshape(shape_z)
                .contiguous()
            )
        elif zeros_values.isnumeric():
            zeros = torch.full(zeros.shape, int(zeros_values), dtype=torch.int32)
        else:  # "0"
            zeros = torch.full(zeros.shape, 0, dtype=torch.int32)
    else:
        zeros = torch.full((infeatures // group_size, outfeatures), 0, dtype=torch.int32)

    input_unpacked = None
    if input_values == "normal":
        input_unpacked = input
    elif input_values == "range":
        if variant == "int4":
            range_t_input = torch.tensor(list(range(0, 4)), dtype=torch.int32)
            seq_size = 4
        else:
            range_t_input = torch.tensor(list(range(0, 8)), dtype=torch.int32)
            seq_size = 8
        shape_w = input.shape
        input_unpacked = (
            (torch.ones(input.numel(), dtype=torch.int32).reshape(-1, seq_size) * range_t_input)
            .reshape(shape_w)
            .contiguous()
        )
    elif input_values.isnumeric():
        input_unpacked = torch.full(input.shape, int(input_values), dtype=input.dtype)
    else:
        input_unpacked = torch.full(input.shape, 7, dtype=input.dtype)
    override_weight = input_unpacked is not None
    if override_weight:
        input_to_cuda = input_unpacked
    else:
        input_to_cuda = input

    return input_to_cuda, scale, zeros


def hpu_preprocessing(wf, qweight, qzeros, scales, bits, group_size):
    qweight = qweight.cpu()
    weight = unpack_weight_cuda(qweight, bits, wf, group_size, False)
    new_qweight = pack_tensor(weight)
    qweight = new_qweight.to("hpu")

    zeros = unpack_zeros_cuda_old(qzeros, scales, bits, wf, False).cpu()
    new_qzeros = pack_tensor(zeros)
    qzeros = new_qzeros.to("hpu")
    return qzeros, qweight


def prepare_data_for_hpu(bits, group_size, cuda_qweight, cuda_qzeros, cuda_scales):
    if bits in [2, 4, 8]:
        wf = torch.tensor(list(range(0, 32, bits)), dtype=torch.int32).unsqueeze(0)
    return hpu_preprocessing(wf, cuda_qweight, cuda_qzeros, cuda_scales, bits, group_size)


@pytest.mark.skipif(is_pytest_mode_eager(), reason="convert_from_int4 is not supported in eager mode")
@pytest.mark.parametrize("infeatures, outfeatures", [(64, 64)])
@pytest.mark.parametrize("variant", ["uint4"])
@pytest.mark.parametrize(
    "is_zero_point, packed_zero_point",
    [(False, True), (True, False), (True, True)],
    ids=["no_zeros", "zeros_unpacked", "zeros_packed"],
)
@pytest.mark.parametrize(
    "zeros_values",
    ["range", "8", "-8", "0", "1"],
    ids=["zeros_as_range", "zeros_as_8", "zeros_as_-8", "zeros_as_0", "zeros_as_1"],
)
@pytest.mark.parametrize(
    "scale_values", ["rand", "1", "range", "range_int"], ids=["scale", "no_scale", "scale_range", "scale_range_int"]
)
@pytest.mark.parametrize(
    "input_values",
    ["0", "normal", "1", "7", "range"],
    ids=["input_zeros", "input_normal", "input_1", "input_7", "input_range"],
)
@pytest.mark.parametrize("out_dtype", dtypes, ids=dtypes_str)
def test_convert_from_int4_AutoGPTQ(
    infeatures,
    outfeatures,
    variant,
    is_zero_point,
    packed_zero_point,
    zeros_values,
    scale_values,
    input_values,
    out_dtype,
):
    if out_dtype in [torch.float8_e5m2, torch.float8_e4m3fn]:
        pytest.skip("https://jira.habana-labs.com/browse/SW-182397")

    group_size = 16
    bits = 4

    input_to_cuda, scale, zeros = prepare_data(
        infeatures,
        outfeatures,
        group_size,
        bits,
        variant,
        out_dtype,
        is_zero_point,
        packed_zero_point,
        input_values,
        scale_values,
        zeros_values,
    )
    cuda_qweight, cuda_qzeros, cuda_scales = pack_cuda_old(
        input_to_cuda, scale.T, zeros.T, -1, infeatures, outfeatures, group_size, bits
    )
    scale_hpu = cuda_scales.to(out_dtype).to("hpu")
    htcore.mark_step()

    if input_values == "0":
        cuda_qweight = torch.zeros_like(cuda_qweight)
    zeros_hpu, qweight_hpu = prepare_data_for_hpu(bits, group_size, cuda_qweight, cuda_qzeros, cuda_scales)

    fn = getattr(torch.ops.hpu, "convert_from_" + variant)
    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        fn = torch.compile(fn, backend="hpu_backend")
    if not is_zero_point:
        result_hpu = fn(qweight_hpu, scale_hpu, None, out_dtype)
    else:
        result_hpu = fn(qweight_hpu, scale_hpu, zeros_hpu, out_dtype)
    htcore.mark_step()
    result_pt = get_weight_cuda_old(is_zero_point, bits, cuda_qweight, cuda_qzeros, cuda_scales, group_size)

    torch.set_printoptions(edgeitems=64)
    compare_tensors(result_hpu.cpu(), result_pt.cpu(), atol=0.001, rtol=0.001)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("convert_from_" + variant)
