###############################################################################
# Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
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
