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
import habana_frameworks.torch.dynamo.compile_backend
import habana_frameworks.torch.internal.bridge_config as bc
import pytest
import torch
from test_utils import format_tc, is_gaudi1

params = [
    ([8, 2, 3], [0, 2]),
    ((4, 4, 4, 2, 2), []),
    ((1, 9, 5), 2),
    ((2, 4), [-1]),
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
@pytest.mark.skipif(
    bc.get_pt_hpu_gpu_migration(),
    reason="Test not suitable for GPU Migration functionality. Default 'inductor' backend is also mapped to 'hpu_backend'.",
)
def test_hpu_count_nonzero(shape, dim, dtype):
    torch._dynamo.reset()

    def fn(input, dim):
        return torch.count_nonzero(input, dim)

    cpu_input = torch.randint(low=0, high=2, size=shape, dtype=dtype)
    if dtype in (torch.bfloat16, torch.float16, torch.float):
        cpu_input *= torch.rand(shape, dtype=dtype)

    hpu_input = cpu_input.to("hpu")

    cpu_compiled_fn = torch.compile(fn)
    hpu_compiled_fn = torch.compile(fn, backend="hpu_backend")

    cpu_output = cpu_compiled_fn(cpu_input, dim)
    hpu_output = hpu_compiled_fn(hpu_input, dim).to("cpu")

    assert torch.equal(cpu_output, hpu_output)
