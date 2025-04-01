###############################################################################
#
#  Copyright (c) 2021-2025 Intel Corporation
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
from compile.test_dynamo_utils import use_eager_fallback
from test_utils import is_pytest_mode_compile

# For the below test case for einsum will generate a fx graph which has
# bmm, view and permute op, in some fx passes we are handling wrongly
# for bmm and view, this tests will verify not to match unnecessary pattern
# which includes bmm and view
# fx graph :
# class <lambda>(torch.nn.Module):
#    def forward(self, arg0_1: "bf16[1, 108, 512]", arg1_1: "bf16[1, 128, 108, 108]"):
#         # File: /home/pradas/qnpu/pt_/src/pytorch-integration/tests/pytest_working/compile/test_hpu_bmm_view.py:42 in fn, code: hpu_input2 = cpu_input2.to("hpu")
#        unsqueeze: "bf16[1, 128, 108, 108, 1]" = torch.ops.aten.unsqueeze.default(arg1_1, 4);  arg1_1 = None
#        permute: "bf16[1, 128, 108, 1, 108]" = torch.ops.aten.permute.default(unsqueeze, [0, 1, 2, 4, 3]);  unsqueeze = None
#        unsqueeze_1: "bf16[1, 108, 512, 1]" = torch.ops.aten.unsqueeze.default(arg0_1, 3);  arg0_1 = None
#        unsqueeze_2: "bf16[1, 108, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 4);  unsqueeze_1 = None
#        permute_1: "bf16[1, 1, 1, 512, 108]" = torch.ops.aten.permute.default(unsqueeze_2, [0, 3, 4, 2, 1]);  unsqueeze_2 = None
#        permute_2: "bf16[128, 108, 108, 1, 1]" = torch.ops.aten.permute.default(permute, [1, 2, 4, 0, 3]);  permute = None
#        view: "bf16[1, 13824, 108]" = torch.ops.aten.view.default(permute_2, [1, 13824, 108]);  permute_2 = None
#        permute_3: "bf16[108, 1, 512, 1, 1]" = torch.ops.aten.permute.default(permute_1, [4, 0, 3, 1, 2]);  permute_1 = None
#        view_1: "bf16[1, 108, 512]" = torch.ops.aten.view.default(permute_3, [1, 108, 512]);  permute_3 = None
#        bmm: "bf16[1, 13824, 512]" = torch.ops.aten.bmm.default(view, view_1);  view = view_1 = None
#        view_2: "bf16[128, 108, 1, 1, 512]" = torch.ops.aten.view.default(bmm, [128, 108, 1, 1, 512]);  bmm = None
#        permute_4: "bf16[1, 128, 108, 512, 1]" = torch.ops.aten.permute.default(view_2, [3, 0, 1, 4, 2]);  view_2 = None
#        view_3: "bf16[1, 128, 108, 512]" = torch.ops.aten.view.default(permute_4, [1, 128, 108, 512]);  permute_4 = None
#        return (view_3,)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_einsum(dtype):
    torch.manual_seed(1234)

    def fn(a, b):
        out = torch.einsum("bhql,blc->bhqc", a, b)
        return out

    # CPU
    cpu_input1 = torch.rand((1, 128, 108, 108), dtype=dtype)
    cpu_input2 = torch.rand((1, 108, 512), dtype=dtype)
    cpu_result = fn(cpu_input1, cpu_input2)

    if is_pytest_mode_compile():
        fn = torch.compile(fn, backend="hpu_backend")

    # HPU
    hpu_input1 = cpu_input1.to("hpu")
    hpu_input2 = cpu_input2.to("hpu")

    with use_eager_fallback():
        hpu_result = fn(hpu_input1, hpu_input2)

    tolerance = 5e-2 if dtype == torch.bfloat16 else 1e-4
    assert torch.allclose(cpu_result, hpu_result.cpu(), atol=tolerance, rtol=tolerance)
