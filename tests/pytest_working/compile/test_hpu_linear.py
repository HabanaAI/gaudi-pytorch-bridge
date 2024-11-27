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
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.dynamo.compile_backend
import pytest
import torch

out_features = 10
in_features = 7


@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16, torch.half])
def test_linear(dtype):
    def fn(model, input):
        return model(input)

    # CPU
    input = torch.randn((2, 3, 4, in_features))
    h_input = input.to("hpu").detach().requires_grad_()
    model = torch.nn.Linear(in_features, out_features, True)
    result = fn(model, input)
    model = model.to("hpu")

    # HPU
    compiled_fn = torch.compile(fn, backend="hpu_backend")
    hresult = compiled_fn(model, h_input)

    assert torch.allclose(result, hresult.cpu(), atol=0.001, rtol=0.001)
