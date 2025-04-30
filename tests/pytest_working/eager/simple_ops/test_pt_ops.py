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

import numpy as np
import pytest
import torch


def test_alias():
    def raw_function(x):
        y = x[...]
        y = y + 2
        return y

    x = torch.randn(3, 4)
    hx = x.to("hpu")

    result_cpu = raw_function(x)

    result_hpu = raw_function(hx).to("cpu")
    assert torch.allclose(result_cpu, result_hpu, rtol=1e-3, atol=1e-3)


def test_to_copy_dtype():
    def raw_function(x, dtype):
        return torch.ops.aten._to_copy(x, dtype=dtype)

    input_tensor = torch.Tensor(np.random.randint(-1, 1, (20, 20)))
    dtype = input_tensor.dtype
    cpu_tensor = input_tensor.ge(0)
    hpu_tensor = cpu_tensor.to("hpu")

    result_cpu = raw_function(cpu_tensor, dtype)
    result_hpu = raw_function(hpu_tensor, dtype).to("cpu")
    assert torch.equal(result_cpu, result_hpu)


@pytest.mark.parametrize("dim", [0, 1, 2, [0, 1], [0, 2], [1, 2], [0, 1, 2]])
@pytest.mark.parametrize("unbiased", [True, False])
@pytest.mark.parametrize("keepdim", [False, True])
def test_var_dim(dim, unbiased, keepdim):
    def raw_function(x):
        return torch.var(x, dim=dim, unbiased=unbiased, keepdim=keepdim)

    cpu_tensor = torch.randn(2, 3, 4)
    hpu_tensor = cpu_tensor.to("hpu")

    result_cpu = raw_function(cpu_tensor)
    result_hpu = raw_function(hpu_tensor).to("cpu")
    assert torch.allclose(result_cpu, result_hpu, rtol=1e-3, atol=1e-3)


def test_index_put_bool():
    tensor1 = torch.zeros(size=[2, 3, 7], dtype=torch.bfloat16)
    tensor2 = torch.ones(size=[2, 3], dtype=torch.bool)
    tensor1 = tensor1.to("hpu")
    tensor2 = tensor2.to("hpu")
    tensor1[tensor2, :] = 7.0
    assert torch.all(torch.eq(tensor1, 7.0))
