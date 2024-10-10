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
from enum import Enum

import habana_frameworks.torch.core as htcore
import pytest
import torch
from test_utils import compare_tensors, format_tc, is_gaudi1


class DeviceMode(Enum):
    CPU_TO_HPU = 0
    HPU_TO_CPU = 1


device_modes = [DeviceMode.HPU_TO_CPU]

dst_formats = [torch.channels_last, torch.contiguous_format]

shapes_strides = [((1, 3, 32, 32), None), ((1, 3, 32, 32), (3072, 1, 96, 3)), ((1, 3, 32, 32), (3072, 1024, 1, 32))]

dtypes = [torch.bfloat16, torch.float, torch.int]
if not is_gaudi1():
    dtypes.append(torch.float16)
    dtypes.append(torch.short)


# following test validates "to" operator when two input arguments i.e.
# memory_format and device are changed in the same "to" op call.
@pytest.mark.parametrize("shape_strides", shapes_strides, ids=format_tc)
@pytest.mark.parametrize("device_mode", device_modes)
@pytest.mark.parametrize("dst_format", dst_formats, ids=format_tc)
@pytest.mark.parametrize("dtype", dtypes, ids=format_tc)
def test_hpu_to(shape_strides, device_mode, dst_format, dtype):
    shape = shape_strides[0]
    strides = shape_strides[1]

    src_device = torch.device("cpu" if device_mode == DeviceMode.CPU_TO_HPU else "hpu")
    dst_device = torch.device("hpu" if device_mode == DeviceMode.CPU_TO_HPU else "cpu")

    src_format = torch.contiguous_format if dst_formats == torch.channels_last else torch.channels_last

    if dtype in (torch.int, torch.short):
        ref_input = torch.randint(size=shape, low=0, high=2, dtype=dtype, device="cpu")
    else:
        ref_input = torch.rand(shape, dtype=dtype, device="cpu")

    # if strides are set then view input will be created with regards to strides
    if strides != None:
        ref_input = ref_input.as_strided(shape, strides)
    else:
        ref_input = ref_input.to(memory_format=src_format)

    input = ref_input.to(device=src_device)

    # The main part of the test. "to" operator is invoked with two
    # arguments i.e. memory_format and device in one call. Conversion is
    # performed in accordance with the input test mode i.e. DeviceMode and
    # dst_format e.g for CPU_TO_HPU mode and torch.channels_last dst_format
    # following conversion will be applied:
    # torch.device("cpu")->torch.device("hpu")
    # and
    # torch.contiguous_format->torch.channels_last
    result = input.to(non_blocking=False, copy=True, memory_format=dst_format, device=dst_device)

    # result has to be copied to "cpu" in order to compare it correctly
    result_cpu = result.to(device=torch.device("cpu"))

    compare_tensors(ref_input, result_cpu, atol=0, rtol=0)

    assert result_cpu.is_contiguous(memory_format=dst_format)


# following test validates "to" operator memory format w.r.t synapse permutation
@pytest.mark.parametrize("dst_format", dst_formats, ids=format_tc)
def test_conv_hpu_to(dst_format):
    input = torch.arange(27, dtype=torch.float32, requires_grad=False).reshape(1, 3, 3, 3)
    weight = torch.arange(27, dtype=torch.float32, requires_grad=False).reshape(3, 3, 3, 1)

    # cpu
    out = torch.nn.functional.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)

    # hpu
    input_hpu = input.to("hpu")
    weight_hpu = weight.to("hpu")
    out_hpu = torch.nn.functional.conv2d(
        input_hpu,
        weight_hpu,
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
    )

    if dst_format == torch.channels_last:
        out = out.to(memory_format=dst_format)
    out_cpu = out_hpu.to(non_blocking=False, copy=True, memory_format=dst_format, device="cpu")
    assert torch.allclose(out, out_cpu, atol=0.001, rtol=0.001)
