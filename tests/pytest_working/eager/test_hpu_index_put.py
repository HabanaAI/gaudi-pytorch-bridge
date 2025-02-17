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

import torch
from test_utils import compare_tensors

cpu = torch.device("cpu")
hpu = torch.device("hpu")


def test_cpu_tensor_hpu_index():
    x = torch.tensor([1, 2, 3.0])
    x_cpy = x.to(hpu)
    index = torch.tensor([2, 1])
    x[(index)] = 0
    x_cpy[(index.to(hpu))] = 0
    compare_tensors(x, x_cpy.to(cpu), atol=0.001, rtol=1.0e-3)


def test_hpu_tensor_cpu_index():
    x = torch.tensor([1, 2, 3.0])
    x_hpu = x.to(hpu)
    index = torch.tensor([2, 1])
    x[(index)] = 0
    x_hpu[(index.to(hpu))] = 0
    compare_tensors(x, x_hpu.to(cpu), atol=0.001, rtol=1.0e-3)


def test_hpu_tensor_cpu_index_cpu_value():
    x = torch.tensor([1, 2, 3.0])
    x_hpu = x.to(hpu)
    index = torch.tensor([2, 1])
    value = torch.tensor([4, 5.0])
    x[(index)] = value
    x_hpu[(index.to(hpu))] = value.to(hpu)
    compare_tensors(x, x_hpu.to(cpu), atol=0.001, rtol=1.0e-3)


def test_cpu_tensor_hpu_indices():
    x = torch.tensor([1, 2, 3.0]).reshape(3, 1)
    x_cpy = x.to(hpu)
    index1 = torch.tensor([2, 1])
    index1_hpu = index1.to(hpu)
    index2 = torch.tensor([0])
    index2_hpu = index2.to(hpu)
    x[(index1, index2)] = 0
    x_cpy[(index1_hpu, index2_hpu)] = 0

    compare_tensors(x, x_cpy.to(cpu), atol=0.001, rtol=1.0e-3)


def test_cpuTensor_cpuValue_hpu_indices():
    x = torch.tensor([1, 2, 3.0]).reshape(3, 1)
    x_cpy = x.to("hpu")
    index1 = torch.tensor([2, 1])
    index1_hpu = index1.to(hpu)
    index2 = torch.tensor([0])
    index2_hpu = index2.to(hpu)
    new_vals = torch.tensor([4, 5.0])
    x[(index1, index2)] = new_vals
    x_cpy[(index1_hpu, index2_hpu)] = new_vals.to(hpu)

    compare_tensors(x, x_cpy.to(cpu), atol=0.001, rtol=1.0e-3)


def test_cpuTensor_hpuValue_hpu_indices():
    x = torch.tensor([1, 2, 3.0]).reshape(3, 1)
    x_cpy = x.to("hpu")
    index1 = torch.tensor([2, 1])
    index1_hpu = index1.to(hpu)
    index2 = torch.tensor([0])
    index2_hpu = index2.to(hpu)
    new_vals = torch.tensor([4, 5.0])
    x[(index1, index2)] = new_vals
    x_cpy[(index1_hpu, index2_hpu)] = new_vals.to(hpu)

    compare_tensors(x, x_cpy.to(cpu), atol=0.001, rtol=1.0e-3)


def test_index_put_():
    tensor_self = torch.randn(10, 10)
    tensor_self_hpu = tensor_self.to(hpu)
    value = torch.tensor(10.0)
    indices = (torch.randint(10, (10,)), torch.randint(10, (10,)))

    tensor_self.index_put_(indices, value)
    tensor_self_hpu.index_put_((indices[0].to(hpu), indices[1].to(hpu)), value.to(hpu))

    compare_tensors(tensor_self, tensor_self_hpu.to(cpu), atol=0.001, rtol=1.0e-3)


def test_index_put_bool():
    tensor1 = torch.zeros(size=[2, 3, 7], dtype=torch.bfloat16)
    tensor2 = torch.ones(size=[2, 3], dtype=torch.bool)
    tensor1 = tensor1.to(hpu)
    tensor2 = tensor2.to(hpu)
    tensor1[tensor2, :] = 7.0
    assert torch.all(torch.eq(tensor1, 7.0))
