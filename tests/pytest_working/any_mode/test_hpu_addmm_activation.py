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
    is_gaudi1,
    is_pytest_mode_compile,
)

alpha_beta_pairs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0), (2.0, 2.0)]


@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16, torch.float16], ids=format_tc)
@pytest.mark.parametrize("n, m, p", [(1, 1, 1), (3, 4, 5)])
@pytest.mark.parametrize("use_gelu", [False, True])
@pytest.mark.parametrize("alpha, beta", alpha_beta_pairs)
def test_addmm_activation_out(dtype, n, m, p, use_gelu, alpha, beta):
    if is_gaudi1() and dtype == torch.float16:
        pytest.skip("Half is not supported on Gaudi.")

    def fn(input, mat1, mat2, beta, alpha, use_gelu, output):
        torch._addmm_activation(
            input=input, mat1=mat1, mat2=mat2, beta=beta, alpha=alpha, use_gelu=use_gelu, out=output
        )

    fn_hpu = compile_function_if_compile_mode(fn)

    input_shape = (n, p)
    mat1_shape = (n, m)
    mat2_shape = (m, p)

    input = torch.randn(input_shape, dtype=dtype)
    mat1 = torch.randn(mat1_shape, dtype=dtype)
    mat2 = torch.randn(mat2_shape, dtype=dtype)
    input_hpu = input.to("hpu")
    mat1_hpu = mat1.to("hpu")
    mat2_hpu = mat2.to("hpu")
    beta_hpu = beta
    alpha_hpu = alpha

    output_cpu = torch.empty_like(input)
    output_hpu = torch.empty_like(input).to("hpu")

    fn(input, mat1, mat2, beta, alpha, use_gelu, output_cpu)

    fn_hpu(input_hpu, mat1_hpu, mat2_hpu, beta_hpu, alpha_hpu, use_gelu, output_hpu)

    atol = rtol = 1e-3 if dtype == torch.float else 1e-2
    compare_tensors(output_hpu, output_cpu, atol=atol, rtol=rtol)

    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("_addmm_activation")
