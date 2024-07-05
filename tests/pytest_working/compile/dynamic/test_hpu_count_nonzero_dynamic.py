# ******************************************************************************
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
# ******************************************************************************
import copy

import habana_frameworks.torch.dynamo.compile_backend
import pytest
import torch
from test_utils import format_tc, is_gaudi1, setup_teardown_env_fixture

params = [
    ([8, 2, 3], [0, 2]),
    ([4, 4, 4, 2, 2], []),
    ([1, 9, 5], 2),
    ([2, 4], [-1]),
]


dtypes = [
    torch.bfloat16,
    torch.float,
    torch.int,
    torch.short,
    torch.bool,
]

if not is_gaudi1():
    dtypes.append(torch.float16)


@pytest.mark.parametrize("shape, dim", params, ids=format_tc)
@pytest.mark.parametrize("dtype", dtypes, ids=format_tc)
@pytest.mark.parametrize(
    "setup_teardown_env_fixture",
    [{"PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES": 1}],
    indirect=True,
)
def test_hpu_count_nonzero_dynamic(shape, dim, dtype, setup_teardown_env_fixture):
    torch._dynamo.reset()
    shapes = [copy.copy(shape), copy.copy(shape), copy.copy(shape)]
    for i in range(len(shape)):
        shapes[1][i] = shape[i] * 2
        shapes[2][i] = shape[i] * 3

    def fn(input, dim):
        return torch.count_nonzero(input, dim)

    inputs_cpu = [torch.randint(low=0, high=2, size=inputShape, dtype=dtype) for inputShape in shapes]
    if dtype in (torch.bfloat16, torch.float16, torch.float):
        inputs_cpu = [cpu_input * torch.rand(shapes[idx], dtype=dtype) for idx, cpu_input in enumerate(inputs_cpu)]

    inputs_hpu = [input_cpu.to("hpu") for input_cpu in inputs_cpu]

    hpu_compiled_fn = torch.compile(fn, backend="hpu_backend")

    outputs_cpu = []
    outputs_hpu = []
    for i in range(len(inputs_cpu)):
        outputs_cpu.append(fn(inputs_cpu[i], dim))
        outputs_hpu.append(hpu_compiled_fn(inputs_hpu[i], dim))
        assert torch.allclose(outputs_cpu[i], outputs_hpu[i].cpu())
