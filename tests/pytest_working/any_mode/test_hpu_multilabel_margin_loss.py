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
from test_utils import compile_function_if_compile_mode, format_tc, is_gaudi1, is_pytest_mode_compile

dtypes = [torch.float32, torch.bfloat16]
if not is_gaudi1():
    dtypes.append(torch.float16)


@pytest.mark.parametrize(
    "reduction",
    [
        "mean",
        "sum",
        "none",
    ],
    ids=format_tc,
)
@pytest.mark.parametrize("C, N", [(6, 2), (6, None)], ids=format_tc)
@pytest.mark.parametrize("dtype", dtypes, ids=format_tc)
@pytest.mark.skipif(
    is_pytest_mode_compile(),
    reason="AssertionError: Eager fallback in nodes: ['squeeze_1:torch.ops.aten.squeeze.dims'] due to eagerizing view leaf tensors.",
)
def test_multilabel_margin_loss(C, N, dtype, reduction):
    def func(x, y, reduction):
        result = torch.nn.functional.multilabel_margin_loss(x, y, reduction=reduction)
        grad = torch.ones_like(result)
        result.backward(grad)
        return result, x.grad

    cpu_input = torch.rand((N, C) if N is not None else C, requires_grad=True)
    hpu_input = cpu_input.to(dtype=dtype).to("hpu").detach()
    hpu_input.requires_grad = True

    cpu_target = torch.rand(cpu_input.shape)
    cpu_target = torch.multinomial(cpu_target, C, replacement=False)

    indexes = torch.randint(1, C, (N,) if N is not None else (1,))
    indexes_mask = torch.nn.functional.one_hot(indexes, C).to(torch.bool)

    cpu_target = torch.where(indexes_mask, -1, cpu_target)
    cpu_target = cpu_target.reshape(cpu_input.shape)
    hpu_target = cpu_target.to("hpu")

    # CPU eager version and CPU compile versions gives different results and different shapes
    # so perform reference testing with compile also on CPU
    fn_cpu = torch.compile(func) if is_pytest_mode_compile() else func
    cpu_output, cpu_grad = fn_cpu(cpu_input, cpu_target, reduction)
    hpu_func = compile_function_if_compile_mode(func)
    hpu_output, hpu_grad = hpu_func(hpu_input, hpu_target, reduction)

    rtol = 0.001 if dtype == torch.float16 else None
    atol = 3e-4 if dtype == torch.float16 else None
    torch.testing.assert_close(hpu_output.cpu(), cpu_output.to(dtype), rtol=rtol, atol=atol)
    torch.testing.assert_close(hpu_grad.cpu(), cpu_grad.to(dtype), rtol=rtol, atol=atol)
