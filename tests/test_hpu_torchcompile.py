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

import os
import torch
import torch.nn.functional as F
import numpy as np
import pytest

from contextlib import contextmanager


def set_flag_in_env(name: str, value):
    if value is None:
        # Nothing to do here
        return
    elif isinstance(value, str):
        os.environ[name] = value
    elif isinstance(value, bool):
        os.environ[name] = str(int(value))
    elif isinstance(value, int):
        os.environ[name] = str(value)
    else:
        assert False, f"Value '{value}' invalid or not supported"


@contextmanager
def env_var_in_scope(vars={}):
    orig_vars = {}
    for key in vars.keys():
        orig_vars[key] = os.environ.get(key, None)
        set_flag_in_env(key, vars[key])
    try:
        yield
    finally:
        for key in orig_vars.keys():
            # restore environment variable
            if orig_vars[key] is not None:
                os.environ[key] = orig_vars[key]
            else:
                if key in os.environ:
                    del os.environ[key]


def test_relu_cpuinput():
    with env_var_in_scope({"PT_HPU_LAZY_MODE": "0", "PT_HPU_DETERMINISTIC_ENABLE": "0"}):
        import habana_frameworks.torch.core as htcore

        def raw_function(x):
            return torch.relu(x)

        compiled_function_training = torch.compile(raw_function, backend="aot_hpu_training_backend")
        compiled_function_inference = torch.compile(raw_function, backend="aot_hpu_inference_backend")

        tensor = torch.Tensor(np.arange(-10.0, 10.0, 0.1))

        result_nocompile = raw_function(tensor)
        result_compile_train = compiled_function_training(tensor)
        result_compile_infer = compiled_function_inference(tensor)

        assert torch.allclose(result_nocompile, result_compile_train)
        assert torch.allclose(result_compile_infer, result_compile_train)


def test_relu_hpuinput():
    with env_var_in_scope({"PT_HPU_LAZY_MODE": "0", "PT_HPU_DETERMINISTIC_ENABLE": "0"}):
        import habana_frameworks.torch.core as htcore

        def raw_function(x):
            return torch.relu(x)

        compiled_function_training = torch.compile(raw_function, backend="aot_hpu_training_backend")
        compiled_function_inference = torch.compile(raw_function, backend="aot_hpu_inference_backend")

        tensor = torch.Tensor(np.arange(-10.0, 10.0, 0.1)).to("hpu")

        result_nocompile = raw_function(tensor)
        result_compile_train = compiled_function_training(tensor)
        result_compile_infer = compiled_function_inference(tensor)

        assert torch.allclose(result_nocompile, result_compile_train)
        assert torch.allclose(result_compile_infer, result_compile_train)


def test_device_partition_cpuinput():
    with env_var_in_scope({"PT_HPU_LAZY_MODE": "0", "PT_HPU_DETERMINISTIC_ENABLE": "0"}):
        import habana_frameworks.torch.core as htcore

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

        compiled_function_training = torch.compile(raw_function, backend="aot_hpu_training_backend")
        compiled_function_inference = torch.compile(raw_function, backend="aot_hpu_inference_backend")

        tensor = torch.Tensor(np.arange(-10.0, 10.0, 0.1))

        result_nocompile = raw_function(tensor)
        result_compile_train = compiled_function_training(tensor)
        result_compile_infer = compiled_function_inference(tensor)

        assert torch.allclose(result_nocompile, result_compile_train)
        assert torch.allclose(result_compile_infer, result_compile_train)


def test_device_partition_hpuinput():
    with env_var_in_scope({"PT_HPU_LAZY_MODE": "0", "PT_HPU_DETERMINISTIC_ENABLE": "0"}):
        import habana_frameworks.torch.core as htcore

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

        compiled_function_training = torch.compile(raw_function, backend="aot_hpu_training_backend")
        compiled_function_inference = torch.compile(raw_function, backend="aot_hpu_inference_backend")

        tensor = torch.Tensor(np.arange(-10.0, 10.0, 0.1)).to("hpu")

        result_nocompile = raw_function(tensor)
        result_compile_train = compiled_function_training(tensor)
        result_compile_infer = compiled_function_inference(tensor)

        assert torch.allclose(result_nocompile, result_compile_train, rtol=1e-06)
        assert torch.allclose(result_compile_infer, result_compile_train, rtol=1e-06)


def test_leaf_views_1():
    with env_var_in_scope({"PT_HPU_LAZY_MODE": "0", "PT_HPU_DETERMINISTIC_ENABLE": "0"}):
        import habana_frameworks.torch.core as htcore

        def raw_function(x, y, z):
            tmp0 = x + y

            tmp1 = F.relu(tmp0)

            tmp2 = z * 2

            tmp3 = tmp1 * tmp2.to("hpu")

            tmp4 = x / tmp3

            tmp5 = F.tanh(tmp4)

            return torch.transpose(tmp5, 0, 1), tmp3.to("cpu")

        compiled_function = torch.compile(raw_function, backend="aot_hpu_inference_backend")

        input_tensor1 = torch.rand(8, 1, 32, 32).to("hpu")
        input_tensor2 = torch.rand(8, 1, 32, 32).to("hpu")
        input_tensor3 = torch.rand(8, 1, 32, 32).to("cpu")

        result1, result2 = raw_function(input_tensor1, input_tensor2, input_tensor3)
        result1_compiled, result2_compiled = compiled_function(input_tensor1, input_tensor2, input_tensor3)

        assert torch.allclose(result1.cpu(), result1_compiled.cpu())
        assert torch.allclose(result2.cpu(), result2_compiled.cpu())

def test_leaf_views_1_dynamic():
    with env_var_in_scope({"PT_HPU_LAZY_MODE": "0", "PT_HPU_DETERMINISTIC_ENABLE": "0"}):
        import habana_frameworks.torch.core as htcore

        def raw_function(x, y, z):
            tmp0 = x + y

            tmp1 = F.relu(tmp0)

            tmp2 = z * 2

            tmp3 = tmp1 * tmp2.to("hpu")

            tmp4 = x / tmp3

            tmp5 = F.tanh(tmp4)

            return torch.transpose(tmp5, 0, 1), tmp3.to("cpu")

        compiled_function = torch.compile(raw_function, backend="aot_hpu_inference_backend", dynamic=True)

        input_tensor1 = torch.rand(8, 1, 32, 32).to("hpu")
        input_tensor2 = torch.rand(8, 1, 32, 32).to("hpu")
        input_tensor3 = torch.rand(8, 1, 32, 32).to("cpu")

        result1, result2 = raw_function(input_tensor1, input_tensor2, input_tensor3)
        result1_compiled, result2_compiled = compiled_function(input_tensor1, input_tensor2, input_tensor3)

        assert torch.allclose(result1.cpu(), result1_compiled.cpu())
        assert torch.allclose(result2.cpu(), result2_compiled.cpu())

def test_leaf_views_1_really_dynamic():
    with env_var_in_scope({"PT_HPU_LAZY_MODE": "0", "PT_HPU_DETERMINISTIC_ENABLE": "0"}):
        import habana_frameworks.torch.core as htcore

        def raw_function(x, y, z):
            tmp0 = x + y

            tmp1 = F.relu(tmp0)

            tmp2 = z * 2

            tmp3 = tmp1 * tmp2.to("hpu")

            tmp4 = x / tmp3

            tmp5 = F.tanh(tmp4)

            return torch.transpose(tmp5, 0, 1), tmp3.to("cpu")

        compiled_function = torch.compile(raw_function, backend="aot_hpu_inference_backend", dynamic=True)

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
    with env_var_in_scope({"PT_HPU_LAZY_MODE": "0", "PT_HPU_DETERMINISTIC_ENABLE": "0"}):
        import habana_frameworks.torch.core as htcore

        def raw_function(x):
            x = x.add(1.0)
            x = x[:]
            x = x.add(2.0)
            x = x[:, :]
            return x.t()

        compiled_function = torch.compile(raw_function, backend="aot_hpu_inference_backend")

        a1 = torch.ones([2, 4], requires_grad=False).to("hpu")

        result_nocompile = raw_function(a1)
        result_compile = compiled_function(a1)

        assert torch.allclose(result_nocompile.cpu(), result_compile.cpu())


def test_leaf_views_3():
    with env_var_in_scope({"PT_HPU_LAZY_MODE": "0", "PT_HPU_DETERMINISTIC_ENABLE": "0"}):
        import habana_frameworks.torch.core as htcore

        def raw_function(x):
            b = x[::2]
            c = torch.relu(b)
            d = b[:]
            return c, d

        compiled_function = torch.compile(raw_function, backend="aot_hpu_inference_backend")

        a1 = torch.ones([2, 4], requires_grad=False).to("hpu")

        result1_nocompile, result2_nocompile = raw_function(a1)
        result1_compile, result2_compile = compiled_function(a1)

        assert torch.allclose(result1_nocompile.cpu(), result1_compile.cpu())
        assert torch.allclose(result2_nocompile.cpu(), result2_compile.cpu())


def test_create_tensor():
    with env_var_in_scope({"PT_HPU_LAZY_MODE": "0", "PT_HPU_DETERMINISTIC_ENABLE": "0"}):
        import habana_frameworks.torch.core as htcore

        def raw_function():
            out = torch.ones([2, 4], requires_grad=False, device=torch.device("hpu"))
            return out

        compiled_function = torch.compile(raw_function, backend="aot_hpu_inference_backend")

        result_nocompile = raw_function()
        result_compile = compiled_function()

        assert torch.allclose(result_nocompile.cpu(), result_compile.cpu())


def test_multiple_runs():
    def raw_function(x):
        x = x * 2 + 1
        x = x + 11
        x = x / 3
        return F.relu(x)

    compiled_function = torch.compile(raw_function, backend="aot_hpu_inference_backend")

    for _ in range(10):
        input_tensor = torch.rand(8, 8).to("hpu")
        tensor_raw = raw_function(input_tensor)
        tensor_compiled = compiled_function(input_tensor)
        assert torch.allclose(tensor_raw, tensor_compiled, rtol=1e-06)


def test_use_random():
    with env_var_in_scope({"PT_HPU_LAZY_MODE": "0", "PT_HPU_DETERMINISTIC_ENABLE": "0"}):
        import habana_frameworks.torch.core as htcore

        def raw_function():
            out = torch.rand(8, 1, 32, 32, device=torch.device("hpu"))
            return out

        compiled_function = torch.compile(raw_function, backend="aot_hpu_inference_backend")

        torch.manual_seed(0xBADC0FEE)
        result_nocompile = raw_function()

        torch.manual_seed(0xBADC0FEE)
        result_compile = compiled_function()

        assert torch.allclose(result_nocompile.cpu(), result_compile.cpu())


def test_simple_sgd_convnet():
    with env_var_in_scope({"PT_HPU_LAZY_MODE": "0", "PT_HPU_DETERMINISTIC_ENABLE": "0"}):
        import habana_frameworks.torch.core as htcore

        torch.manual_seed(2562825)

        class LeNet5(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = torch.nn.Sequential(
                    torch.nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
                    torch.nn.BatchNorm2d(6),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(kernel_size=2, stride=2),
                )
                self.layer2 = torch.nn.Sequential(
                    torch.nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
                    torch.nn.BatchNorm2d(16),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(kernel_size=2, stride=2),
                )

                self.fc = torch.nn.Linear(400, 120)
                self.relu = torch.nn.ReLU()
                self.fc1 = torch.nn.Linear(120, 84)
                self.relu1 = torch.nn.ReLU()
                self.fc2 = torch.nn.Linear(84, 10)

            def forward(self, x):
                out = self.layer1(x)
                out = self.layer2(out)
                out = out.reshape(out.size(0), -1)
                out = self.fc(out)
                out = self.relu(out)
                out = self.fc1(out)
                out = self.relu1(out)
                out = self.fc2(out)

                return out

        model = LeNet5().to("hpu")
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()

        def raw_function_test(x, y):
            result = model(x)
            loss = criterion(result, y)

            return loss, result

        def raw_function_train(x, y):
            loss, _ = raw_function_test(x, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            return loss

        compiled_function_test = torch.compile(raw_function_test, backend="aot_hpu_inference_backend")
        compiled_function_train = torch.compile(raw_function_train, backend="aot_hpu_training_backend")

        input_tensor1 = torch.rand(8, 1, 32, 32).to("hpu")
        input_tensor2 = torch.randint(0, 9, (8,)).to("hpu")

        loss_nocompile0, result_nocompile0 = raw_function_test(input_tensor1, input_tensor2)
        loss_compile0, result_compile0 = compiled_function_test(input_tensor1, input_tensor2)

        assert torch.allclose(loss_nocompile0, loss_compile0, rtol=1e-03)
        assert torch.allclose(result_nocompile0, result_compile0, rtol=1e-03)

        loss_compile1 = compiled_function_train(input_tensor1, input_tensor2)
        loss_compile2 = compiled_function_train(input_tensor1, input_tensor2)
        loss_compile3 = compiled_function_train(input_tensor1, input_tensor2)
        loss_compile4 = compiled_function_train(input_tensor1, input_tensor2)

        assert loss_compile4 < loss_compile3
        assert loss_compile3 < loss_compile2
        assert loss_compile2 < loss_compile1


def test_simple_sgd_convnet_with_device_pingpong():
    with env_var_in_scope({"PT_HPU_LAZY_MODE": "0", "PT_HPU_DETERMINISTIC_ENABLE": "0"}):
        import habana_frameworks.torch.core as htcore

        torch.manual_seed(2562825)

        class LeNet5(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = torch.nn.Sequential(
                    torch.nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
                    torch.nn.BatchNorm2d(6),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(kernel_size=2, stride=2),
                )
                self.layer2 = torch.nn.Sequential(
                    torch.nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
                    torch.nn.BatchNorm2d(16),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(kernel_size=2, stride=2),
                )

                self.fc = torch.nn.Linear(400, 120)
                self.relu = torch.nn.ReLU()
                self.fc1 = torch.nn.Linear(120, 84)
                self.relu1 = torch.nn.ReLU()
                self.fc2 = torch.nn.Linear(84, 10)

            def forward(self, x):
                out = self.layer1(x)
                out = self.layer2(out)
                out = out.reshape(out.size(0), -1)
                out = self.fc(out)

                out = out.to("cpu")

                out = self.relu(out)

                out = out.to("hpu")

                out = self.fc1(out)
                out = self.relu1(out)
                out = self.fc2(out)

                return out

        model = LeNet5().to("hpu")
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()

        def raw_function_test(x, y):
            result = model(x)
            loss = criterion(result, y)

            return loss, result

        def raw_function_train(x, y):
            loss, _ = raw_function_test(x, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            return loss

        compiled_function_test = torch.compile(raw_function_test, backend="aot_hpu_inference_backend")
        compiled_function_train = torch.compile(raw_function_train, backend="aot_hpu_training_backend")

        input_tensor1 = torch.rand(8, 1, 32, 32).to("hpu")
        input_tensor2 = torch.randint(0, 9, (8,)).to("hpu")

        loss_nocompile0, result_nocompile0 = raw_function_test(input_tensor1, input_tensor2)
        loss_compile0, result_compile0 = compiled_function_test(input_tensor1, input_tensor2)

        assert torch.allclose(loss_nocompile0, loss_compile0, rtol=1e-03)
        assert torch.allclose(result_nocompile0, result_compile0, rtol=1e-03)

        loss_compile1 = compiled_function_train(input_tensor1, input_tensor2)
        loss_compile2 = compiled_function_train(input_tensor1, input_tensor2)
        loss_compile3 = compiled_function_train(input_tensor1, input_tensor2)
        loss_compile4 = compiled_function_train(input_tensor1, input_tensor2)

        assert loss_compile4 < loss_compile3
        assert loss_compile3 < loss_compile2
        assert loss_compile2 < loss_compile1

@pytest.mark.xfail # Adam have issues when deepcopying FX graph in the backend: https://github.com/pytorch/pytorch/issues/96949
def test_simple_adam_convnet():
    with env_var_in_scope({"PT_HPU_LAZY_MODE": "0", "PT_HPU_DETERMINISTIC_ENABLE": "0"}):
        import habana_frameworks.torch.core as htcore

        torch.manual_seed(2562825)

        class LeNet5(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = torch.nn.Sequential(
                    torch.nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
                    torch.nn.BatchNorm2d(6),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(kernel_size=2, stride=2),
                )
                self.layer2 = torch.nn.Sequential(
                    torch.nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
                    torch.nn.BatchNorm2d(16),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(kernel_size=2, stride=2),
                )

                self.fc = torch.nn.Linear(400, 120)
                self.relu = torch.nn.ReLU()
                self.fc1 = torch.nn.Linear(120, 84)
                self.relu1 = torch.nn.ReLU()
                self.fc2 = torch.nn.Linear(84, 10)

            def forward(self, x):
                out = self.layer1(x)
                out = self.layer2(out)
                out = out.reshape(out.size(0), -1)
                out = self.fc(out)
                out = self.relu(out)
                out = self.fc1(out)
                out = self.relu1(out)
                out = self.fc2(out)

                return out

        model = LeNet5().to("hpu")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()

        def raw_function_test(x, y):
            result = model(x)
            loss = criterion(result, y)

            return loss, result

        def raw_function_train(x, y):
            loss, _ = raw_function_test(x, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            return loss

        compiled_function_test = torch.compile(raw_function_test, backend="aot_hpu_inference_backend")
        compiled_function_train = torch.compile(raw_function_train, backend="aot_hpu_training_backend")

        input_tensor1 = torch.rand(8, 1, 32, 32).to("hpu")
        input_tensor2 = torch.randint(0, 9, (8,)).to("hpu")

        loss_nocompile0, result_nocompile0 = raw_function_test(input_tensor1, input_tensor2)
        loss_compile0, result_compile0 = compiled_function_test(input_tensor1, input_tensor2)

        assert torch.allclose(loss_nocompile0, loss_compile0, rtol=1e-03)
        assert torch.allclose(result_nocompile0, result_compile0, rtol=1e-03)

        loss_compile1 = compiled_function_train(input_tensor1, input_tensor2)
        loss_compile2 = compiled_function_train(input_tensor1, input_tensor2)
        loss_compile3 = compiled_function_train(input_tensor1, input_tensor2)
        loss_compile4 = compiled_function_train(input_tensor1, input_tensor2)

        assert loss_compile4 < loss_compile3
        assert loss_compile3 < loss_compile2
        assert loss_compile2 < loss_compile1

@pytest.mark.xfail # Adam have issues when deepcopying FX graph in the backend: https://github.com/pytorch/pytorch/issues/96949
def test_simple_adam_convnet_with_device_pingpong():
    with env_var_in_scope({"PT_HPU_LAZY_MODE": "0", "PT_HPU_DETERMINISTIC_ENABLE": "0"}):
        import habana_frameworks.torch.core as htcore

        torch.manual_seed(2562825)

        class LeNet5(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = torch.nn.Sequential(
                    torch.nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
                    torch.nn.BatchNorm2d(6),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(kernel_size=2, stride=2),
                )
                self.layer2 = torch.nn.Sequential(
                    torch.nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
                    torch.nn.BatchNorm2d(16),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(kernel_size=2, stride=2),
                )

                self.fc = torch.nn.Linear(400, 120)
                self.relu = torch.nn.ReLU()
                self.fc1 = torch.nn.Linear(120, 84)
                self.relu1 = torch.nn.ReLU()
                self.fc2 = torch.nn.Linear(84, 10)

            def forward(self, x):
                out = self.layer1(x)
                out = self.layer2(out)
                out = out.reshape(out.size(0), -1)
                out = self.fc(out)

                out = out.to("cpu")

                out = self.relu(out)

                out = out.to("hpu")

                out = self.fc1(out)
                out = self.relu1(out)
                out = self.fc2(out)

                return out

        model = LeNet5().to("hpu")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()

        def raw_function_test(x, y):
            result = model(x)
            loss = criterion(result, y)

            return loss, result

        def raw_function_train(x, y):
            loss, _ = raw_function_test(x, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            return loss

        compiled_function_test = torch.compile(raw_function_test, backend="aot_hpu_inference_backend")
        compiled_function_train = torch.compile(raw_function_train, backend="aot_hpu_training_backend")

        input_tensor1 = torch.rand(8, 1, 32, 32).to("hpu")
        input_tensor2 = torch.randint(0, 9, (8,)).to("hpu")

        loss_nocompile0, result_nocompile0 = raw_function_test(input_tensor1, input_tensor2)
        loss_compile0, result_compile0 = compiled_function_test(input_tensor1, input_tensor2)

        assert torch.allclose(loss_nocompile0, loss_compile0, rtol=1e-03)
        assert torch.allclose(result_nocompile0, result_compile0, rtol=1e-03)

        loss_compile1 = compiled_function_train(input_tensor1, input_tensor2)
        loss_compile2 = compiled_function_train(input_tensor1, input_tensor2)
        loss_compile3 = compiled_function_train(input_tensor1, input_tensor2)
        loss_compile4 = compiled_function_train(input_tensor1, input_tensor2)

        assert loss_compile4 < loss_compile3
        assert loss_compile3 < loss_compile2
        assert loss_compile2 < loss_compile1


def test_relu_than_maxpool():
    with env_var_in_scope({"PT_HPU_LAZY_MODE": "0", "PT_HPU_DETERMINISTIC_ENABLE": "0"}):
        import habana_frameworks.torch.core as htcore
        torch.manual_seed(2562825)

        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = torch.nn.Sequential(
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(kernel_size=2, stride=2)
                )

            def forward(self, x):
                x = self.layer1(x)
                return x

        model = TestModel().to("hpu")

        def raw_function(x):
            return model(x)

        compiled_fnc = torch.compile(raw_function, backend="aot_hpu_inference_backend")

        tensor = torch.rand(8, 16, 10, 10, device="cpu").to("hpu")

        res_ver = raw_function(tensor)
        res = compiled_fnc(tensor)

        assert torch.allclose(res, res_ver, rtol=1e-06)