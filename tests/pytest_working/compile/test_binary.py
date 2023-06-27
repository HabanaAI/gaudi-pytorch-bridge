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
import random
import habana_frameworks.torch.core
import habana_frameworks.torch.dynamo.compile_backend

@pytest.mark.xfail(reason="Attempting to broadcast a dimension of length 4 at -1! [SW-148715]")
@pytest.mark.parametrize("func", [torch.add, torch.sub, torch.rsub])
@pytest.mark.parametrize("shapes", [[tuple(), tuple()], [(1,),(2,)], [(4,4), (2,2)], [(1000,1000), (1000, 1000)]])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16, torch.int32])
def test_binary(func, shapes, dtype):
        def fn(input, other, alpha):
            return func(input, other, alpha=alpha)

        compiled_cpu_fn = torch.compile(fn)
        compiled_hpu_fn = torch.compile(fn, backend="aot_hpu_training_backend")

        if (dtype.is_floating_point):
            alpha = random.random()
        else:
            alpha = random.randint(-128, 127)
        if (shapes[0] == tuple()):
            input = torch.tensor(alpha, dtype=dtype)
        else:
            input = (torch.randn(size=shapes[0]) * 3).to(dtype)
        if (shapes[1] == tuple()):
            other = torch.tensor(alpha, dtype=dtype)
        else:
            other = (torch.randn(size=shapes[1]) * 3).to(dtype)

        input_hpu = input.to("hpu")
        other_hpu = other.to("hpu")
        expected = compiled_cpu_fn(input, other, alpha)
        result = compiled_hpu_fn(input_hpu, other_hpu, alpha).cpu()

        rtol = 1e-02 if dtype == torch.bfloat16 else 1e-07
        atol = 1e-02 if dtype == torch.bfloat16 else 1e-08
        assert torch.allclose(result, expected, rtol=rtol, atol=atol)