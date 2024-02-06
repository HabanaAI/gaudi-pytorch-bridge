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
import os
import torch
import pytest
import habana_frameworks.torch.dynamo.compile_backend
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.common_methods_invocations import ReductionOpInfo
from torch.testing._internal.common_device_type import ops
from test_utils import hpu


@pytest.mark.parametrize("op", [torch.amin, torch.amax])
@pytest.mark.parametrize("shapes", [(3, 4, 5, 6), (2, 3, 5, 4)])
@pytest.mark.parametrize("dim", [-4, -3, -2, -1, 0, 1, 2, 3])
@pytest.mark.parametrize("dtype", ["float"])
def test_hpu_amax_amin(op, shapes, dim, dtype):
    def fn(input, dim):
        return op(input, dim)

    cpu_input = torch.randn(shapes, dtype=getattr(torch, dtype))
    hpu_input = cpu_input.to(hpu)
    torch._dynamo.reset()

    cpu_wrapped_fn = torch.compile(fn) if pytest.mode == "compile" else fn
    hpu_wrapped_fn = torch.compile(fn, backend="hpu_backend") if pytest.mode == "compile" else fn
    cpu_output = cpu_wrapped_fn(cpu_input, dim)
    hpu_output = hpu_wrapped_fn(hpu_input, dim).cpu()
    assert torch.allclose(cpu_output, hpu_output)


@pytest.mark.parametrize("dtype", [torch.bool])
def test_hpu_amax_amin_bool(dtype):
    os.environ["PT_HPU_PLACE_ON_CPU"] = ""

    def convert_boolean_tensors(x):
        if not isinstance(x, torch.Tensor) or x.dtype != dtype:
            return x

        # Map False -> 0 and True -> Random value in [2, 255]
        true_vals = torch.randint(2, 255, x.shape).to(torch.uint8)
        false_vals = torch.zeros(()).to(torch.uint8)
        x_int = torch.where(x, true_vals, false_vals)

        ret = x_int.view(torch.bool)
        return ret

    for op in [x for x in op_db if (x.name == "amax" or x.name == "amin")]:
        for sample in op.sample_inputs(hpu, dtype):
            expect = op(sample.input, *sample.args, **sample.kwargs)
            transformed = sample.transform(convert_boolean_tensors)
            actual = op(transformed.input, *transformed.args, **transformed.kwargs)

            assert torch.allclose(expect, actual)
