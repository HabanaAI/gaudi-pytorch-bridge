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

@pytest.mark.parametrize("shape", [[2, 7], [2, 3, 4]])
@pytest.mark.parametrize("op", [torch.minimum, torch.maximum])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16, torch.int])
def test_hpu_minimum_maximum(shape, op, dtype):
    def fn(input, other):
        return op(input, other)

    if dtype == torch.int:
        cpu_input = torch.randint(low=-100, high=100, size=shape, dtype=dtype)
        cpu_other = torch.randint(low=-100, high=100, size=shape, dtype=dtype)
    else:
        cpu_input = torch.rand(shape, dtype=dtype)
        cpu_other = torch.rand(shape, dtype=dtype)

    hpu_input = cpu_input.to("hpu")
    hpu_other = cpu_other.to("hpu")
    cpu_compiled_fn = torch.compile(fn)
    hpu_compiled_fn = torch.compile(fn, backend="aot_hpu_training_backend")

    cpu_output = cpu_compiled_fn(cpu_input, cpu_other)
    hpu_output = hpu_compiled_fn(hpu_input, hpu_other).cpu()
    assert torch.equal(cpu_output, hpu_output)