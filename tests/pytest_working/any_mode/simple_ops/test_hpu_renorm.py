###############################################################################
#
#  Copyright (c) 2025 Intel Corporation
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


import random

import pytest
import torch
from test_utils import (
    check_ops_executed_in_jit_ir,
    compile_function_if_compile_mode,
    format_tc,
    is_pytest_mode_compile,
)


@pytest.mark.parametrize("p", [1.0, 5.0, float("inf")])
@pytest.mark.parametrize("max_norm", [1.0, 4.0, 200.0])
@pytest.mark.parametrize("shape", [[3, 3]], ids=format_tc)
def test_renorm_out(p, max_norm, shape):
    def fn(input, p, dim, maxnorm):
        return input.renorm(p, dim, maxnorm)

    dtype = torch.float

    input_cpu = torch.randn(shape, dtype=dtype)
    input_hpu = input_cpu.to("hpu")
    # This is considered safe because it is not used for security or cryptographic operations.
    dim = random.randint(0, len(shape) - 1)  # nosec B311

    compiled_fn = compile_function_if_compile_mode(fn)

    result_cpu = fn(input_cpu, p, dim, max_norm)
    result_hpu = compiled_fn(input_hpu, p, dim, max_norm)

    assert torch.allclose(result_cpu, result_hpu.cpu())
    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("renorm")
