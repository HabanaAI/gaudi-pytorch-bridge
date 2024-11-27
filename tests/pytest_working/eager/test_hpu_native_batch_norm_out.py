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
from test_utils import compare_tensors, format_tc, hpu, is_gaudi1

dtypes = [torch.bfloat16, torch.float]
if not is_gaudi1():
    dtypes.append(torch.half)

shapes = [(2, 3, 4, 5), (4, 3, 8)]


def fn(input, weight, bias, mean, var):
    return torch.native_batch_norm(
        input, weight, bias, running_mean=mean, running_var=var, training=True, momentum=0.1, eps=1e-5
    )


def fn_out(input, weight, bias, mean, var, out1, out2, out3):
    torch.native_batch_norm(
        input,
        weight,
        bias,
        running_mean=mean,
        running_var=var,
        training=True,
        momentum=0.1,
        eps=1e-5,
        out=(out1, out2, out3),
    )
    return (out1, out2, out3)


@pytest.mark.parametrize("shape", shapes, ids=format_tc)
@pytest.mark.parametrize("dtype", dtypes, ids=format_tc)
def test_native_batch_norm(dtype, shape):
    num_channels = shape[1]
    input = torch.rand(shape, dtype=dtype)
    weight = torch.rand(num_channels, dtype=torch.float)
    bias = torch.rand(num_channels, dtype=torch.float)
    mean = torch.rand(num_channels, dtype=torch.float)
    var = torch.rand(num_channels, dtype=torch.float)

    hpu_input = input.to(hpu)
    hpu_weight = weight.to(hpu)
    hpu_bias = bias.to(hpu)
    hpu_mean = mean.to(hpu)
    hpu_var = var.to(hpu)

    cpu_result = fn(input, weight, bias, mean, var)
    hpu_result = fn(hpu_input, hpu_weight, hpu_bias, hpu_mean, hpu_var)

    compare_tensors(hpu_result, cpu_result, rtol=1e-3, atol=1e-6)


@pytest.mark.parametrize("shape", shapes, ids=format_tc)
@pytest.mark.parametrize("dtype", dtypes, ids=format_tc)
def test_native_batch_norm_out(dtype, shape):
    if len(shape) == 3:
        pytest.skip("SW-186931")

    num_channels = shape[1]
    input = torch.rand(shape, dtype=dtype)
    weight = torch.rand(num_channels, dtype=torch.float)
    bias = torch.rand(num_channels, dtype=torch.float)
    mean = torch.rand(num_channels, dtype=torch.float)
    var = torch.rand(num_channels, dtype=torch.float)

    hpu_input = input.to(hpu)
    hpu_weight = weight.to(hpu)
    hpu_bias = bias.to(hpu)
    hpu_mean = mean.to(hpu)
    hpu_var = var.to(hpu)

    out1 = torch.zeros(shape).to(dtype)
    out2 = torch.zeros(num_channels).to(torch.float)
    out3 = torch.zeros(num_channels).to(torch.float)

    hpu_out1 = out1.to(hpu)
    hpu_out2 = out2.to(hpu)
    hpu_out3 = out3.to(hpu)

    cpu_result = fn_out(input, weight, bias, mean, var, out1, out2, out3)
    hpu_result = fn_out(hpu_input, hpu_weight, hpu_bias, hpu_mean, hpu_var, hpu_out1, hpu_out2, hpu_out3)

    compare_tensors(hpu_result, cpu_result, rtol=1e-3, atol=1e-6)
