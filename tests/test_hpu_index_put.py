# ******************************************************************************
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
# ******************************************************************************

import torch
from test_utils import compare_tensors

cpu = torch.device("cpu")
hpu = torch.device("hpu")

def test_cpu_tensor_hpu_index():
    x = torch.tensor([1, 2, 3.])
    x_cpy = x
    index = torch.tensor([2, 1])
    index_hpu = index.to(hpu)
    x[(index)] = 0
    x_cpy[(index)] = 0
    compare_tensors(x, x_cpy, atol=0.001, rtol=1.e-3)

def test_hpu_tensor_cpu_index():
    x = torch.tensor([1, 2, 3.])
    x_hpu = x.to(hpu)
    index = torch.tensor([2, 1])
    x[(index)] = 0
    x_hpu[(index)] = 0
    compare_tensors(x, x_hpu.to(cpu), atol=0.001, rtol=1.e-3)

def test_hpu_tensor_cpu_index_cpu_value():
    x = torch.tensor([1, 2, 3.])
    x_hpu = x.to(hpu)
    index = torch.tensor([2, 1])
    value = torch.tensor([4, 5.])
    x[(index)] = value
    x_hpu[(index)] = value
    compare_tensors(x, x_hpu.to(cpu), atol=0.001, rtol=1.e-3)

def test_cpu_tensor_hpu_indices():
    x = torch.tensor([1, 2, 3.]).reshape(3, 1)
    x_cpy=x
    index1 = torch.tensor([2, 1])
    index1_hpu = index1.to(hpu)
    index2 = torch.tensor([0])
    index2_hpu = index2.to(hpu)
    x[(index1, index2)] = 0
    x_cpy[(index1_hpu, index2_hpu)] = 0

    compare_tensors(x, x_cpy, atol=0.001, rtol=1.e-3)

def test_cpuTensor_cpuValue_hpu_indices():
    x = torch.tensor([1, 2, 3.]).reshape(3, 1)
    x_cpy=x
    index1 = torch.tensor([2, 1])
    index1_hpu = index1.to(hpu)
    index2 = torch.tensor([0])
    index2_hpu = index2.to(hpu)
    new_vals = torch.tensor([4, 5.])
    x[(index1, index2)] = new_vals
    x_cpy[(index1_hpu, index2_hpu)] = new_vals

    compare_tensors(x, x_cpy, atol=0.001, rtol=1.e-3)

def test_cpuTensor_hpuValue_hpu_indices():
    x = torch.tensor([1, 2, 3.]).reshape(3,1)
    x_cpy=x
    index1 = torch.tensor([2, 1])
    index1_hpu = index1.to(hpu)
    index2 = torch.tensor([0])
    index2_hpu = index2.to(hpu)
    new_vals = torch.tensor([4, 5.])
    new_vals_hpu= new_vals.to(hpu)
    x[(index1, index2)] = new_vals
    x_cpy[(index1_hpu, index2_hpu)] = new_vals

    compare_tensors(x, x_cpy, atol=0.001, rtol=1.e-3)

if __name__ == "__main__":
   test_cpu_tensor_hpu_index()
   test_hpu_tensor_cpu_index()
   test_hpu_tensor_cpu_index_cpu_value()
   test_cpu_tensor_hpu_indices()
   test_cpuTensor_cpuValue_hpu_indices()
   test_cpuTensor_hpuValue_hpu_indices()
   
