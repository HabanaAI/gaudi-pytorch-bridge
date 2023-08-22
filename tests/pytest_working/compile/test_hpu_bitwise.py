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
import habana_frameworks.torch.core as htcore

@pytest.mark.parametrize("dtype", [torch.int8, torch.int, torch.uint8, torch.int16])
@pytest.mark.parametrize("op_code",[torch.bitwise_and, torch.bitwise_or, torch.bitwise_xor])
def test_bitwise_tensor(dtype, op_code):

        def fn(input1, input2):
            return op_code(input1, input2)

        # CPU
        x = torch.tensor([-1, -2, 3], dtype=dtype)
        y = torch.tensor([1, 0, 3], dtype=dtype)
        hx = x.to('hpu')
        hy = y.to('hpu')

        result = fn(x, y)

        # HPU
        compiled_fn = torch.compile(fn, backend="aot_hpu_training_backend")

        hresult = compiled_fn(hx, hy)

        assert torch.allclose(result, hresult.cpu(), atol = 0.001, rtol = 0.001)

@pytest.mark.parametrize("dtype", [torch.int8, torch.int, torch.uint8, torch.int16])
@pytest.mark.parametrize("op_code",[torch.bitwise_and])
#@pytest.mark.parametrize("op_code",[torch.bitwise_or, torch.bitwise_xor])
#These will be supported once PT2.1 migration is done
#https://jira.habana-labs.com/browse/SW-158110
def test_bitwise_scalar(dtype, op_code):

        def fn(input1, input2):
            return op_code(input1, input2)

        # CPU
        x = torch.tensor([-1, -2, 3], dtype=dtype)
        y = 5
        hx = x.to('hpu')

        result = fn(x, y)

        # HPU
        compiled_fn = torch.compile(fn, backend="aot_hpu_training_backend")

        hresult = compiled_fn(hx, y)
        print(result)
        print(hresult)

        assert torch.allclose(result, hresult.cpu(), atol = 0.001, rtol = 0.001)

@pytest.mark.parametrize("dtype", [torch.int8, torch.bool, torch.int, torch.uint8, torch.int16])
def test_bitwise_not(dtype):

        def fn(input):
            return torch.bitwise_not(input)

        # CPU
        x = torch.tensor([-1, -2, 3], dtype=dtype)
        hx = x.to('hpu')

        result = fn(x)

        # HPU
        compiled_fn = torch.compile(fn, backend="aot_hpu_training_backend")

        hresult = compiled_fn(hx)

        assert torch.allclose(result, hresult.cpu(), atol = 0.001, rtol = 0.001)

