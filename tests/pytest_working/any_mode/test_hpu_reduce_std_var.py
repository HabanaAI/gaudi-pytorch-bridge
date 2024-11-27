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
import habana_frameworks.torch.dynamo.compile_backend  # noqa # pylint: disable=unused-import
import pytest
import torch
from test_utils import format_tc, is_pytest_mode_compile


def std_var_common_test(shape, dim, op, dtype, correction):
    def fn(input):
        return op(input, dim=dim, correction=correction)

    cpu_input = torch.rand(shape, dtype=dtype)
    hpu_input = cpu_input.to("hpu")

    torch._dynamo.reset()

    hpu_wrapped_fn = torch.compile(fn, backend="hpu_backend") if pytest.mode == "compile" else fn

    cpu_output = fn(cpu_input)
    hpu_output = hpu_wrapped_fn(hpu_input)

    return (cpu_output, hpu_output)


@pytest.mark.parametrize("shape, dim", [([], 0), ((2, 3), 0), ((2, 3), None), ((2, 3, 4), 2)], ids=format_tc)
@pytest.mark.parametrize("op", [torch.var_mean, torch.std_mean])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16], ids=format_tc)
@pytest.mark.parametrize("correction", [0, 1, 1.1, 0.0], ids=format_tc)
def test_hpu_std_var_mean(shape, dim, op, dtype, correction):
    if is_pytest_mode_compile() and shape == []:
        pytest.skip(reason="LoweringException in computation on CPU")
    if shape == [] and correction in (1.1, 1, None):
        pytest.skip(
            reason="std_mean/var_mean: degrees of freedom is <= 0."
            "Correction should be strictly less than the reduction factor."
        )
    cpu_output, hpu_output = std_var_common_test(shape, dim, op, dtype, correction)

    tol = 1e-2 if dtype == torch.bfloat16 else 1e-5
    assert torch.allclose(cpu_output[0], hpu_output[0].cpu(), rtol=tol, atol=tol, equal_nan=True)
    assert torch.allclose(cpu_output[1], hpu_output[1].cpu(), rtol=tol, atol=tol)


@pytest.mark.parametrize("shape, dim", [((2, 3), 0), ((2, 3), None), ((2, 3, 4), 2)], ids=format_tc)
@pytest.mark.parametrize("op", [torch.var, torch.std])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16], ids=format_tc)
@pytest.mark.parametrize("correction", [0, 1, 1.1, 0.0], ids=format_tc)
def test_hpu_std_var(shape, dim, op, dtype, correction):
    cpu_output, hpu_output = std_var_common_test(shape, dim, op, dtype, correction)

    tol = 1e-2 if dtype == torch.bfloat16 else 1e-5
    assert torch.allclose(cpu_output, hpu_output.cpu(), rtol=tol, atol=tol)
