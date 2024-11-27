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
import habana_frameworks.torch.dynamo.compile_backend
import pytest
import torch
from test_utils import format_tc


@pytest.mark.parametrize("shape", [(2, 3), (1, 2, 3, 4)], ids=format_tc)
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"], ids=format_tc)
@pytest.mark.parametrize("delta", [1.0, 0.5])
@pytest.mark.parametrize("backward", [False, True])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16], ids=format_tc)
def test_hpu_huber_loss(shape, reduction, delta, backward, dtype):
    def fn_fwd(model, input, target):
        return model(input, target)

    def fn_bwd(model, input, target):
        output = model(input, target)
        grad = torch.ones_like(output)
        output.backward(grad)
        return input.grad

    model = torch.nn.HuberLoss(reduction=reduction, delta=delta)
    cpu_input = torch.rand(shape, dtype=dtype)
    hpu_input = cpu_input.to("hpu")
    if backward:
        cpu_input.requires_grad = True
        hpu_input.requires_grad = True
        fn = fn_bwd
    else:
        fn = fn_fwd

    cpu_target = torch.rand(shape, dtype=dtype)
    hpu_target = cpu_target.to("hpu")
    torch._dynamo.reset()

    hpu_wrapped_fn = torch.compile(fn, backend="hpu_backend") if pytest.mode == "compile" else fn

    cpu_output = fn(model, cpu_input, cpu_target)
    hpu_output = hpu_wrapped_fn(model, hpu_input, hpu_target).cpu()

    tol = 1e-3 if dtype == torch.bfloat16 else 1e-5
    assert torch.allclose(cpu_output, hpu_output, rtol=tol, atol=tol)
