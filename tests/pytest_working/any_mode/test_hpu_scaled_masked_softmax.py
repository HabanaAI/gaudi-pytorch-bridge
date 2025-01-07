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
from test_utils import check_ops_executed_in_jit_ir, clear_t_compile_logs, compare_tensors, is_pytest_mode_compile


@pytest.mark.parametrize("scale", [0.75])
@pytest.mark.parametrize("shape", [(32, 48)])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_scaled_masked_softmax(scale, shape, dtype):
    torch.manual_seed(12345)

    input = (torch.rand(shape, dtype=dtype) - 0.5) * 5
    mask = torch.randint(0, 2, shape, dtype=torch.int).to(torch.bfloat16)
    scale_softmax = 0.17

    # Ref copied from model_garden original implementation of this softmax variant
    attention_scores = torch.mul(input, scale_softmax)
    # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
    attention_scores = attention_scores + mask
    # Normalize the attention scores to probabilities.
    result_ref = torch.nn.functional.softmax(attention_scores, dim=-1)

    hpu_op = torch.ops.hpu.scaled_masked_softmax
    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        hpu_op = torch.compile(
            torch.ops.hpu.scaled_masked_softmax,
            backend="hpu_backend",
        )
    result = hpu_op(input.to("hpu"), mask.to("hpu"), scale_softmax)

    atol = 1e-3 if dtype == torch.float else 1e-1
    rtol = atol
    compare_tensors(result, result_ref, atol=atol, rtol=rtol)
    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("scaled_masked_softmax")
