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
from test_utils import cpu, hpu


@pytest.mark.parametrize("src_dtype", [torch.int8, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("dst_dtype", [torch.int16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("devices", [(cpu, hpu), (hpu, cpu), (hpu, hpu)])
def test_memcpy_with_cast(src_dtype, dst_dtype, devices):
    src_device, dst_device = devices

    def memcpy_with_cast(src, dst_dtype):
        dst = torch.tensor(1, device=dst_device, dtype=dst_dtype)
        src = src.to(dst)
        return src

    if pytest.mode == "lazy":
        func = memcpy_with_cast
    elif pytest.mode == "compile":
        func = torch.compile(memcpy_with_cast, backend="hpu_backend")
    elif pytest.mode == "eager":
        func = memcpy_with_cast

    input_tensor = torch.tensor(range(0, 10), device=src_device, dtype=src_dtype)
    output_tensor = func(input_tensor, dst_dtype)

    assert torch.allclose(
        input_tensor.to("cpu").to(torch.float32),
        output_tensor.to("cpu").to(torch.float32),
        atol=1e-3,
        rtol=1e-3,
    )


@pytest.mark.parametrize("dtype", [torch.double, torch.int64])
def test_dma_unsupported_type(dtype):
    input_cpu_ = torch.randint(0, 1024, (2, 2)).to(dtype)
    assert torch.all(torch.eq(input_cpu_[1], input_cpu_.to("hpu")[1].to("cpu")))
