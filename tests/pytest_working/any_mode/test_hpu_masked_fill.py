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
import torch
import pytest
import habana_frameworks.torch.dynamo.compile_backend

#This test checks if the masked_fill op will accept a value tensor on the CPU while the input tensor is on the HPU
@pytest.mark.parametrize("shape", [(2,7)])
@pytest.mark.parametrize("value", [2])
@pytest.mark.parametrize("scalar_value", [True, False])
@pytest.mark.parametrize("dtype", [torch.float])
def test_hpu_masked_mixed_devices(shape, value, scalar_value, dtype):
    if (pytest.mode == "eager" and scalar_value == False):
        pytest.xfail("[SW-161344] RuntimeError: Expected all tensors to be on the HPU device")
    def fn(input, mask, value):
        input.masked_fill_(mask, value)
    mask = torch.randint(low=0, high=2, size=shape, dtype=torch.bool, device="hpu")
    input = torch.rand(shape, dtype=dtype, device="hpu")
    value = value if scalar_value else torch.tensor(value, dtype=dtype, device="cpu")

    wrapped_fn = torch.compile(fn, backend="aot_hpu_training_backend") if pytest.mode == "compile" else fn
    wrapped_fn(input, mask, value)
    assert (input.device.type == "hpu")