###############################################################################
#
#  Copyright (c) 2021-2025 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
###############################################################################

from copy import deepcopy

import pytest
import torch
from test_utils import (
    compile_function_if_compile_mode,
    format_tc,
)

dtypes = [torch.half, torch.bfloat16, torch.float, torch.int]


@pytest.mark.parametrize("alpha", [1, 2])
@pytest.mark.parametrize("dim", [0, 1, 2])
@pytest.mark.parametrize("shape", [[5, 4, 7], [12, 5, 3, 5]], ids=format_tc)
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64], ids=format_tc)
@pytest.mark.parametrize("dtype", dtypes, ids=format_tc)
@pytest.mark.parametrize("out_variant", [True, False])
def test_hpu_index_add(dtype, index_dtype, shape, dim, alpha, out_variant):
    def fn(input, indices, source):
        if out_variant:
            out = torch.ones_like(input)
            torch.index_add(input, dim, indices, source, alpha=alpha, out=out)
            return out
        else:
            return torch.index_add(input, dim, indices, source, alpha=alpha)

    compiled_fn_hpu = compile_function_if_compile_mode(fn)

    idx_cpu = torch.randint(size=[shape[dim] - 2], low=0, high=shape[dim], dtype=index_dtype)
    idx_cpu = torch.unique(idx_cpu)

    source_shape = deepcopy(shape)
    source_shape[dim] = idx_cpu.numel()

    if dtype == torch.int:
        input_cpu = torch.randint(size=shape, low=-100, high=100, dtype=dtype)
        source_cpu = torch.randint(size=source_shape, low=-100, high=100, dtype=dtype)
        alpha = int(alpha)
    else:
        input_cpu = torch.rand(shape, dtype=dtype)
        source_cpu = torch.rand(source_shape, dtype=dtype)

    input_hpu = input_cpu.to("hpu")
    idx_hpu = idx_cpu.to("hpu")
    source_hpu = source_cpu.to("hpu")

    result_cpu = fn(input_cpu, idx_cpu, source_cpu)
    result_hpu = compiled_fn_hpu(input_hpu, idx_hpu, source_hpu)

    assert torch.allclose(result_cpu, result_hpu.cpu())
