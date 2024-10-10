###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################

import numpy as np
import pytest
import torch
import torch.nn.functional as F


def test_parallel_graphs():
    torch.manual_seed(2562825)

    def raw_fnc(tensor_one, tensor_two):
        maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=1)
        tensor_one = 2 * tensor_one
        tensor_one = torch.relu(tensor_one)
        tensor_one = tensor_one.to(device="cpu")
        tensor_one = maxpool(tensor_one)
        tensor_one = tensor_one.to(device="hpu")

        tensor_two = 2 * tensor_two
        tensor_two = torch.relu(tensor_two)
        tensor_two = tensor_two.to(device="cpu")
        tensor_two = maxpool(tensor_two)
        tensor_two = tensor_two.to(device="hpu")

        return tensor_one + tensor_two

    compiled_fnc = torch.compile(raw_fnc, backend="hpu_backend")

    tensor_one = torch.randn(2, 2, 2, 2).to(device="hpu")
    tensor_two = torch.randn(2, 2, 2, 2).to(device="hpu")

    out = compiled_fnc(tensor_one, tensor_two)
    out_raw = raw_fnc(tensor_one, tensor_two)

    torch.allclose(out, out_raw)


def test_device_partition_cpuinput():
    def raw_function(x):
        tmp1 = x * 2 + 1
        tmp2 = tmp1.to("hpu")
        tmp3 = tmp2 + 11
        tmp4 = tmp3.to("cpu")
        tmp5 = tmp4 / 3

        tmp11 = x * 3 + 2
        tmp12 = tmp11.to("hpu")
        tmp13 = tmp12 + 12
        tmp14 = tmp13.to("cpu")
        tmp15 = tmp14 / 4

        return tmp5 + tmp15

    compiled_function_training = torch.compile(raw_function, backend="hpu_backend")
    compiled_function_inference = torch.compile(raw_function, backend="hpu_backend")

    tensor = torch.Tensor(np.arange(-10.0, 10.0, 0.1))

    result_nocompile = raw_function(tensor)
    result_compile_train = compiled_function_training(tensor)
    result_compile_infer = compiled_function_inference(tensor)

    assert torch.allclose(result_nocompile, result_compile_train)
    assert torch.allclose(result_compile_infer, result_compile_train)


def test_device_partition_hpuinput():
    def raw_function(x):
        tmp1 = x * 2 + 1
        tmp2 = tmp1.to("cpu")
        tmp3 = tmp2 + 11
        tmp4 = tmp3.to("hpu")
        tmp5 = tmp4 / 3

        tmp11 = x * 3 + 2
        tmp12 = tmp11.to("cpu")
        tmp13 = tmp12 + 12
        tmp14 = tmp13.to("hpu")
        tmp15 = tmp14 / 4

        return tmp5 + tmp15

    compiled_function_training = torch.compile(raw_function, backend="hpu_backend")
    compiled_function_inference = torch.compile(raw_function, backend="hpu_backend")

    tensor = torch.Tensor(np.arange(-10.0, 10.0, 0.1)).to("hpu")

    result_nocompile = raw_function(tensor)
    result_compile_train = compiled_function_training(tensor)
    result_compile_infer = compiled_function_inference(tensor)

    assert torch.allclose(result_nocompile, result_compile_train, rtol=1e-06)
    assert torch.allclose(result_compile_infer, result_compile_train, rtol=1e-06)


def test_leaf_views_1():
    def raw_function(x, y, z):
        tmp0 = x + y

        tmp1 = F.relu(tmp0)

        tmp2 = z * 2

        tmp3 = tmp1 * tmp2.to("hpu")

        tmp4 = x / tmp3

        tmp5 = F.tanh(tmp4)

        return torch.transpose(tmp5, 0, 1), tmp3.to("cpu")

    compiled_function = torch.compile(raw_function, backend="hpu_backend")

    input_tensor1 = torch.rand(8, 1, 32, 32).to("hpu")
    input_tensor2 = torch.rand(8, 1, 32, 32).to("hpu")
    input_tensor3 = torch.rand(8, 1, 32, 32).to("cpu")

    result1, result2 = raw_function(input_tensor1, input_tensor2, input_tensor3)
    result1_compiled, result2_compiled = compiled_function(input_tensor1, input_tensor2, input_tensor3)

    assert torch.allclose(result1.cpu(), result1_compiled.cpu())
    assert torch.allclose(result2.cpu(), result2_compiled.cpu())


def test_leaf_views_1_dynamic():
    def raw_function(x, y, z):
        tmp0 = x + y

        tmp1 = F.relu(tmp0)

        tmp2 = z * 2

        tmp3 = tmp1 * tmp2.to("hpu")

        tmp4 = x / tmp3

        tmp5 = F.tanh(tmp4)

        return torch.transpose(tmp5, 0, 1), tmp3.to("cpu")

    compiled_function = torch.compile(raw_function, backend="hpu_backend", dynamic=True)

    input_tensor1 = torch.rand(8, 1, 32, 32).to("hpu")
    input_tensor2 = torch.rand(8, 1, 32, 32).to("hpu")
    input_tensor3 = torch.rand(8, 1, 32, 32).to("cpu")

    result1, result2 = raw_function(input_tensor1, input_tensor2, input_tensor3)
    result1_compiled, result2_compiled = compiled_function(input_tensor1, input_tensor2, input_tensor3)

    assert torch.allclose(result1.cpu(), result1_compiled.cpu())
    assert torch.allclose(result2.cpu(), result2_compiled.cpu())


def test_leaf_views_1_really_dynamic():
    def raw_function(x, y, z):
        tmp0 = x + y

        tmp1 = F.relu(tmp0)

        tmp2 = z * 2

        tmp3 = tmp1 * tmp2.to("hpu")

        tmp4 = x / tmp3

        tmp5 = F.tanh(tmp4)

        return torch.transpose(tmp5, 0, 1), tmp3.to("cpu")

    compiled_function = torch.compile(raw_function, backend="hpu_backend", dynamic=True)

    input_tensor1 = torch.rand(8, 1, 16, 16).to("hpu")
    input_tensor2 = torch.rand(8, 1, 16, 16).to("hpu")
    input_tensor3 = torch.rand(8, 1, 16, 16).to("cpu")

    result1, result2 = raw_function(input_tensor1, input_tensor2, input_tensor3)
    result1_compiled, result2_compiled = compiled_function(input_tensor1, input_tensor2, input_tensor3)

    assert torch.allclose(result1.cpu(), result1_compiled.cpu())
    assert torch.allclose(result2.cpu(), result2_compiled.cpu())

    input_tensor1 = torch.rand(16, 2, 32, 32).to("hpu")
    input_tensor2 = torch.rand(16, 2, 32, 32).to("hpu")
    input_tensor3 = torch.rand(16, 2, 32, 32).to("cpu")

    result1, result2 = raw_function(input_tensor1, input_tensor2, input_tensor3)
    result1_compiled, result2_compiled = compiled_function(input_tensor1, input_tensor2, input_tensor3)

    assert torch.allclose(result1.cpu(), result1_compiled.cpu())
    assert torch.allclose(result2.cpu(), result2_compiled.cpu())


def test_leaf_views_2():
    def raw_function(x):
        x = x.add(1.0)
        x = x[:]
        x = x.add(2.0)
        x = x[:, :]
        return x.t()

    compiled_function = torch.compile(raw_function, backend="hpu_backend")

    a1 = torch.ones([2, 4], requires_grad=False).to("hpu")

    result_nocompile = raw_function(a1)
    result_compile = compiled_function(a1)

    assert torch.allclose(result_nocompile.cpu(), result_compile.cpu())


def test_leaf_views_3():
    def raw_function(x):
        b = x[::2]
        c = torch.relu(b)
        d = b[:]
        return c, d

    compiled_function = torch.compile(raw_function, backend="hpu_backend")

    a1 = torch.ones([2, 4], requires_grad=False).to("hpu")

    result1_nocompile, result2_nocompile = raw_function(a1)
    result1_compile, result2_compile = compiled_function(a1)

    assert torch.allclose(result1_nocompile.cpu(), result1_compile.cpu())
    assert torch.allclose(result2_nocompile.cpu(), result2_compile.cpu())
