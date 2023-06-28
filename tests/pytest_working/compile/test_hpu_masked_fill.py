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
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.dynamo.compile_backend
from test_utils import env_var_in_scope

@pytest.mark.xfail(reason="CI problem: undefined symbol: _ZN6habana5graph12GraphStorage3getEv [SW-150162]")
@pytest.mark.parametrize("shape", [(1,1), (2,2), (3,4,5,6,7)])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16, torch.int8, torch.int32])
def test_masked_fill(shape, dtype):
    with env_var_in_scope(PT_HPU_LAZY_MODE="0", PT_HPU_DETERMINISTIC_ENABLE="0", PT_HPU_COMPILE_USE_RECIPES=True):
        def fn(input, mask, fill_value):
            return torch.ops.aten.masked_fill(input, mask, fill_value)
        compiled_fn = torch.compile(fn, backend="aot_hpu_training_backend")

        if (dtype.is_floating_point):
            val = random.random()
            cpuInput = torch.randn(shape, dtype=dtype)
        else:
            val = random.randint(-128, 127)
            cpuInput = torch.randint(low=-128, high=127, size=shape, dtype=dtype)
        hpuInput = cpuInput.to("hpu")

        cpuMask = torch.randint(low=0, high=2, size=shape, dtype=torch.bool)
        hpuMask = cpuMask.to("hpu")
        expected = fn(cpuInput, cpuMask, val)
        result = compiled_fn(hpuInput, hpuMask, val).cpu()
        assert torch.equal(result, expected)
