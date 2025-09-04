###############################################################################
#
#  Copyright (c) 2025 Intel Corporation
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
from test_utils import (
    compare_tensors,
    compile_function_if_compile_mode,
    format_tc,
    hpu,
    is_gaudi1,
)


@pytest.mark.skipif(is_gaudi1(), reason="im2col is not supported on Gaudi1")
@pytest.mark.parametrize("kernel_size", [14], ids=format_tc)
@pytest.mark.parametrize("dilation", [1], ids=format_tc)
@pytest.mark.parametrize("padding", [0], ids=format_tc)
@pytest.mark.parametrize("stride", [14], ids=format_tc)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16], ids=format_tc)
@pytest.mark.parametrize("batch", [1, 8, 32], ids=format_tc)
def test_im2col(batch, dtype, stride, padding, dilation, kernel_size):
    input_shape = [batch, 3, 560, 560]
    input_cpu = torch.randn(input_shape, dtype=dtype)
    fn = torch.ops.aten.im2col

    result_cpu = fn(input_cpu, kernel_size, dilation=dilation, padding=padding, stride=stride)

    fn = compile_function_if_compile_mode(fn)
    input_hpu = input_cpu.to(hpu)
    result_hpu = fn(input_hpu, kernel_size, dilation=dilation, padding=padding, stride=stride)

    compare_tensors(result_hpu, result_cpu, rtol=0, atol=0)
