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


def test_relu_cpuinput():
    def raw_function(x):
        return torch.relu(x)

    compiled_function = torch.compile(raw_function, backend="hpu_backend")

    tensor = torch.Tensor(np.arange(-10.0, 10.0, 0.1))

    result_nocompile = raw_function(tensor)
    result_compile = compiled_function(tensor)

    assert torch.allclose(result_nocompile, result_compile)


def test_relu_hpuinput():
    def raw_function(x):
        return torch.relu(x)

    compiled_function = torch.compile(raw_function, backend="hpu_backend")

    tensor = torch.Tensor(np.arange(-10.0, 10.0, 0.1)).to("hpu")

    result_nocompile = raw_function(tensor)
    result_compile = compiled_function(tensor)

    assert torch.allclose(result_nocompile, result_compile)


def test_relu_hpuinput_mixed():
    def raw_function(x):
        tmp1 = x * 2 - 1
        return torch.relu(tmp1)

    compiled_function = torch.compile(raw_function, backend="hpu_backend")

    tensor = torch.Tensor(np.arange(-10.0, 10.0, 0.1)).to("hpu")

    result_nocompile = 2 * raw_function(tensor) + 3

    result_compile = compiled_function(tensor)
    result_mixed = 2 * result_compile + 3

    assert torch.allclose(result_nocompile, result_mixed)


def test_output_middle_node():
    def raw_function(x):
        tmp1 = x * 2 - 1
        tmp2 = 1 / tmp1 + 1
        tmp3 = torch.relu(tmp2)
        return tmp2

    compiled_function = torch.compile(raw_function, backend="hpu_backend")

    tensor = torch.Tensor(np.arange(-10.0, 10.0, 0.1)).to("hpu")

    result_nocompile = raw_function(tensor)

    result_compile = compiled_function(tensor)

    assert torch.allclose(result_nocompile, result_compile)


def test_output_first_and_last_node():
    def raw_function(x):
        tmp1 = x * 2 - 1
        tmp2 = 1 / tmp1 + 1
        tmp3 = torch.relu(tmp2)
        return tmp1, tmp3

    compiled_function = torch.compile(raw_function, backend="hpu_backend")

    tensor = torch.Tensor(np.arange(-10.0, 10.0, 0.1)).to("hpu")

    result_nocompile_1, result_nocompile_2 = raw_function(tensor)

    result_compile_1, result_compile_2 = compiled_function(tensor)

    assert torch.allclose(result_nocompile_1, result_compile_1)
    assert torch.allclose(result_nocompile_2, result_compile_2)


def test_output_reverse_order_node():
    def raw_function(x):
        tmp1 = x * 2 - 1
        tmp2 = 1 / tmp1 + 1
        tmp3 = torch.relu(tmp2)
        return tmp3, tmp2, tmp1

    compiled_function = torch.compile(raw_function, backend="hpu_backend")

    tensor = torch.Tensor(np.arange(-10.0, 10.0, 0.1)).to("hpu")

    result_nocompile_1, result_nocompile_2, result_nocompile_3 = raw_function(tensor)

    result_compile_1, result_compile_2, result_compile_3 = compiled_function(tensor)

    assert torch.allclose(result_nocompile_1, result_compile_1)
    assert torch.allclose(result_nocompile_2, result_compile_2)
    assert torch.allclose(result_nocompile_3, result_compile_3)


def test_multiple_runs():
    import torch.nn.functional as F

    def raw_function(x):
        x = x * 2 + 1
        x = x + 11
        x = x / 3
        return F.relu(x)

    compiled_function = torch.compile(raw_function, backend="hpu_backend")

    for _ in range(10):
        input_tensor = torch.rand(8, 8).to("hpu")
        tensor_raw = raw_function(input_tensor)
        tensor_compiled = compiled_function(input_tensor)
        assert torch.allclose(tensor_raw, tensor_compiled, rtol=1e-06)


def test_create_tensor():
    def raw_function():
        out = torch.ones([2, 4], requires_grad=False, device=torch.device("hpu"))
        return out

    compiled_function = torch.compile(raw_function, backend="hpu_backend")

    result_nocompile = raw_function()
    result_compile = compiled_function()

    assert torch.allclose(result_nocompile.cpu(), result_compile.cpu())


def test_relu_than_maxpool():
    class TestModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.MaxPool2d(kernel_size=2, stride=2))

        def forward(self, x):
            x = self.layer1(x)
            return x

    model = TestModel().to("hpu")

    def raw_function(x):
        return model(x)

    compiled_fnc = torch.compile(raw_function, backend="hpu_backend")

    tensor = torch.rand(8, 16, 10, 10, device="cpu").to("hpu")

    res_ver = raw_function(tensor)
    res = compiled_fnc(tensor)

    assert torch.allclose(res, res_ver, rtol=1e-06)


def test_relu_then_maxpool_ignore_first_output():
    torch.manual_seed(9361478)

    class TestModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = torch.nn.Sequential(
                torch.nn.ReLU(), torch.nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
            )

        def forward(self, x):
            x = self.layer1(x)
            return x[1]

    model = TestModel().to("hpu")

    def raw_function(x):
        return model(x)

    compiled_fnc = torch.compile(raw_function, backend="hpu_backend")

    tensor = torch.rand(8, 16, 10, 10, device="cpu").to("hpu")

    res_ver = raw_function(tensor)
    res = compiled_fnc(tensor)

    assert torch.all(torch.eq(res, res_ver))


def test_remove_detach():
    import torch.nn.functional as F

    def raw_function(x):
        x = x * 2 + 1
        x = x.detach()
        x = x / 3
        return F.relu(x)

    compiled_function = torch.compile(raw_function, backend="hpu_backend")
    input_tensor = torch.rand(2, 2).to("hpu")
    tensor_raw = raw_function(input_tensor)
    tensor_compiled = compiled_function(input_tensor)
    assert torch.allclose(tensor_raw, tensor_compiled, rtol=1e-06)


# This test is only to check if pass replacing getitem with ListUnpack
# will not crash.
# Comparing compiled output with raw_function might be problematic
# due to known issues with "aten::split_with_sizes (SW-140890)
def test_split_with_sizes():
    def raw_function(x):
        x = torch.split(x, [1, 4])
        return x

    compiled_function = torch.compile(raw_function, backend="hpu_backend")

    input_tensor = torch.arange(10).reshape(5, 2).to(device="hpu")

    out1, _ = compiled_function(input_tensor)
    assert torch.equal(out1.to("cpu"), torch.tensor([[0, 1]]))
