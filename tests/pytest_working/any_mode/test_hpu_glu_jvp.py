###############################################################################
#
#  Copyright (c) 2021-2024 Intel Corporation
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


import pytest
import torch
from test_utils import (
    check_ops_executed_in_jit_ir,
    compare_tensors,
    compile_function_if_compile_mode,
    format_tc,
    is_pytest_mode_compile,
)


@pytest.mark.parametrize("shape", [[2, 2, 4], [4, 6, 4, 2, 6]], ids=format_tc)
@pytest.mark.parametrize("dim", [-2, -1, 0, 1, 2])
def test_hpu_glu(shape, dim):
    def fn(glu, x, dx, dim):
        return torch.ops.aten.glu_jvp(glu, x, dx, dim)

    in_shape = shape.copy()
    in_shape[dim] = in_shape[dim] // 2

    cpu_input_glu = torch.rand(in_shape, dtype=torch.float)
    cpu_input_x = torch.rand(shape, dtype=torch.float)
    cpu_input_dx = torch.rand(shape, dtype=torch.float)
    hpu_input_glu = cpu_input_glu.to("hpu")
    hpu_input_x = cpu_input_x.to("hpu")
    hpu_input_dx = cpu_input_dx.to("hpu")

    cpu_output = fn(cpu_input_glu, cpu_input_x, cpu_input_dx, dim)
    fn_hpu = compile_function_if_compile_mode(fn)

    hpu_output = fn_hpu(hpu_input_glu, hpu_input_x, hpu_input_dx, dim)

    compare_tensors(hpu_output, cpu_output, atol=0.001, rtol=1.0e-3)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir({"glu_jvp"})
