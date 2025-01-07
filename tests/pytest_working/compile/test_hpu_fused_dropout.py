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

# Fused dropout op on HPU and CPU devices will always give different results.
# This test checks if:
# 1. Fused dropout op run on HPU without any runtime errors
# 2. Fused dropout op will return the same values for the same seed
# 3. Fused dropout op will return different values for different seed


@pytest.mark.parametrize("shape", [(2, 3, 4), (2, 3, 4, 5)])
@pytest.mark.parametrize("ratio", [0.5, 0.75])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_exponential(shape, ratio, dtype):
    def fn(input):
        return torch._fused_dropout(input, ratio)

    compiled_fn = torch.compile(fn, backend="hpu_backend")

    cpu_input = torch.rand(shape, dtype=dtype)
    hpu_input_1 = cpu_input.to("hpu")
    hpu_input_2 = cpu_input.to("hpu")
    hpu_input_3 = cpu_input.to("hpu")

    torch.manual_seed(2)
    result_1 = compiled_fn(hpu_input_1)

    torch.manual_seed(2)
    result_2 = compiled_fn(hpu_input_2)

    assert torch.equal(result_1[0], result_2[0])
    assert torch.equal(result_1[1], result_2[1])

    torch.manual_seed(3)
    result_3 = compiled_fn(hpu_input_3)

    assert torch.any(torch.ne(result_1[0], result_3[0]))
    assert torch.any(torch.ne(result_1[1], result_3[1]))
