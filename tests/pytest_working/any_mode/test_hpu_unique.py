###############################################################################
# Copyright (c) 2021-2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################

import pytest
import torch
from test_utils import (
    compile_function_if_compile_mode,
    hpu,
)

param_return_counts = [True, False]
param_return_inverse = [True, False]
param_dim = [None, 0, 1]


@pytest.mark.parametrize("return_counts", param_return_counts)
@pytest.mark.parametrize("return_inverse", param_return_inverse)
@pytest.mark.parametrize("dim", param_dim)
def test_hpu_unique(return_counts, return_inverse, dim):
    shape = (3, 3, 3)
    sorted = dim is None

    def fn(input, sorted, return_inverse, return_counts, dim):
        return torch.unique(input, sorted=sorted, return_inverse=return_inverse, return_counts=return_counts, dim=dim)

    fn_cpu = fn
    fn_hpu = compile_function_if_compile_mode(fn)

    input_cpu = torch.randint(0, 3, shape)
    if dim is not None:
        input_cpu, _ = torch.sort(input_cpu, dim=dim)
    input_hpu = input_cpu.to(hpu)

    def postprocess(output, inverse, counts):
        if not inverse and not counts:
            return [output]
        else:
            return [*output]

    output_cpu = fn_cpu(input_cpu, sorted, return_inverse, return_counts, dim)
    res_cpu = postprocess(output_cpu, return_inverse, return_counts)

    output_hpu = fn_hpu(input_hpu, sorted, return_inverse, return_counts, dim)
    res_hpu = postprocess(output_hpu, return_inverse, return_counts)

    for c, h in zip(res_cpu, res_hpu, strict=False):
        assert torch.equal(c, h.cpu())
