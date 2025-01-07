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
from test_utils import check_ops_executed_in_jit_ir, compile_function_if_compile_mode, format_tc, is_pytest_mode_compile


@pytest.mark.parametrize("shape", [(20,), (5, 4, 3)], ids=format_tc)
@pytest.mark.parametrize("op_name", ["exp", "sqrt", "rsqrt"])
def test_hpu_exp_fast_math(shape, op_name):
    self_cpu = torch.rand(shape, dtype=torch.bfloat16) * 10
    self_hpu = self_cpu.to("hpu")

    fn_cpu = getattr(torch, op_name)
    hpu_op_name = op_name + "_fast_math"
    fn_hpu = getattr(torch.ops.hpu, hpu_op_name)
    fn_hpu = compile_function_if_compile_mode(fn_hpu)

    result_hpu = fn_hpu(self_hpu)
    result_cpu = fn_cpu(self_cpu)

    torch.testing.assert_close(result_hpu.cpu(), result_cpu, atol=0.01, rtol=0.11)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir(hpu_op_name)
