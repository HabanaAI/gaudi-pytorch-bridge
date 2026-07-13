###############################################################################
# Copyright (c) 2026 Intel Corporation
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
    check_ops_executed_in_jit_ir,
    compile_function_if_compile_mode,
    format_tc,
    is_pytest_mode_compile,
)


def _l2norm_last_dim(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / torch.sqrt(torch.sum(x * x, dim=-1, keepdim=True) + eps)


@pytest.mark.parametrize("epsilon", [1e-6, 1e-4])
@pytest.mark.parametrize("shape", [[17], [17, 5, 3], [2, 3, 4, 5, 6]], ids=format_tc)
def test_l2_norm(epsilon, shape):
    def fn(x, eps):
        return torch.ops.hpu.l2_norm(x, epsilon=eps)

    input_cpu = torch.rand(shape, dtype=torch.float32).to(torch.bfloat16)
    input_hpu = input_cpu.to("hpu")

    compiled_fn = compile_function_if_compile_mode(fn)

    result_cpu = _l2norm_last_dim(input_cpu.to(torch.float32), epsilon)
    result_hpu = compiled_fn(input_hpu, epsilon)

    assert result_hpu.dtype == torch.float32
    assert tuple(result_hpu.shape) == tuple(shape)
    assert torch.allclose(result_cpu, result_hpu.cpu(), rtol=1e-5, atol=1e-6)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("l2_norm")
