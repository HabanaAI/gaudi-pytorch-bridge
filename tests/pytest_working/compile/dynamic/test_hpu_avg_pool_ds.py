###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################
import copy

import habana_frameworks.torch.dynamo.compile_backend
import pytest
import torch
from test_utils import format_tc, is_gaudi3, setup_teardown_env_fixture


@pytest.mark.parametrize("shape", [[1, 16, 3, 2], [1, 1, 16, 3, 2]], ids=format_tc)
@pytest.mark.parametrize("output_size", [[4, 3, 2]], ids=format_tc)
@pytest.mark.parametrize("dtype", [torch.float], ids=format_tc)
@pytest.mark.parametrize(
    "setup_teardown_env_fixture",
    [{"PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES": 1}],
    indirect=True,
)
def test_hpu_adaptive_avg_pool3d_bwd_dynamic(shape, output_size, dtype, setup_teardown_env_fixture):
    if is_gaudi3():
        pytest.skip("DSD not supported on G3")
    shapes = [copy.copy(shape), copy.copy(shape), copy.copy(shape)]
    shapes[1][-2] = shape[-2] * 2
    shapes[2][-2] = shape[-2] * 3

    def fn(input):
        fwd = torch.ops.aten.adaptive_avg_pool3d(input, output_size)
        grad = torch.ones_like(fwd)
        fwd.backward(grad)
        return input.grad

    torch._dynamo.reset()
    hpu_compiled_fn = torch.compile(fn, backend="hpu_backend")
    inputs_cpu = [torch.rand(inputShape, dtype=dtype) for inputShape in shapes]
    inputs_hpu = [input_cpu.to("hpu") for input_cpu in inputs_cpu]
    for i in range(len(inputs_cpu)):
        inputs_cpu[i].requires_grad = True
        inputs_hpu[i].requires_grad = True
    for i in range(len(inputs_cpu)):
        cpu_output = fn(inputs_cpu[i])
        hpu_output = hpu_compiled_fn(inputs_hpu[i])
        assert torch.allclose(cpu_output, hpu_output.cpu())
