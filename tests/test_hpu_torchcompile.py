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


@pytest.mark.xfail
def test_simple_convolution():
    with env_var_in_scope({"PT_HPU_LAZY_MODE": "0", "PT_HPU_DETERMINISTIC_ENABLE": "0"}):
        import habana_frameworks.torch.core as htcore

        torch.manual_seed(2562825)

        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Sequential(
                    torch.nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
                )

            def forward(self, x):
                out = self.layer(x)
                return out

        torch.manual_seed(2562825)
        model = Net().to("hpu")
        torch.manual_seed(2562825)
        raw_model = Net().to("hpu")

        compiled_model = torch.compile(model, backend="aot_hpu_inference_backend")

        tensor = torch.rand(8, 1, 32, 32).to("hpu")

        res_eager = raw_model(tensor)
        res_graph = compiled_model(tensor)
        assert torch.allclose(res_eager, res_graph, rtol=1e-03)


def test_simple_convolution_mixed():
    with env_var_in_scope({"PT_HPU_LAZY_MODE": "0", "PT_HPU_DETERMINISTIC_ENABLE": "0"}):
        import habana_frameworks.torch.core as htcore

        class Net_1(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Sequential(
                    torch.nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
                    torch.nn.BatchNorm2d(6),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(kernel_size=2, stride=2),
                )

            def forward(self, x):
                out = self.layer(x)
                return out

        class Net_2(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Sequential(
                    torch.nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
                    torch.nn.BatchNorm2d(16),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(kernel_size=2, stride=2),
                )

            def forward(self, x):
                out = self.layer(x)
                return out

        model_1 = Net_1().to("hpu")

        def raw_function_1(x):
            return model_1(x)

        model_2 = Net_2().to("hpu")

        def raw_function_2(x):
            return model_2(x)

        tensor = torch.rand(8, 1, 32, 32).to("hpu")

        compiled_function_1 = torch.compile(raw_function_1, backend="aot_hpu_inference_backend")
        compiled_function_2 = torch.compile(raw_function_2, backend="aot_hpu_inference_backend")

        res_eager = raw_function_2(raw_function_1(tensor))
        res_graph_to_eager = raw_function_2(compiled_function_1(tensor))
        res_eager_to_graph = compiled_function_2(raw_function_1(tensor))

        assert torch.allclose(res_eager, res_graph_to_eager, rtol=1e-03)
        assert torch.allclose(res_eager, res_eager_to_graph, rtol=1e-03)


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


@pytest.mark.xfail
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


@pytest.mark.xfail  # Adam have issues when deepcopying FX graph in the backend: https://github.com/pytorch/pytorch/issues/96949
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


@pytest.mark.xfail  # Adam have issues when deepcopying FX graph in the backend: https://github.com/pytorch/pytorch/issues/96949
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


def test_simple_view():
    with env_var_in_scope({"PT_HPU_LAZY_MODE": "0", "PT_HPU_DETERMINISTIC_ENABLE": "0"}):
        import habana_frameworks.torch.core as htcore

        def raw_function(x):
            return torch.relu(x)

        compiled_function_inference = torch.compile(raw_function, backend="aot_hpu_inference_backend")

        input_tensor = torch.rand(3, 3, device="cpu").to("hpu")

        res = compiled_function_inference(input_tensor)

        print(compiled_function_inference.__class__)

        tensor_view = input_tensor.as_strided((2, 2), (1, 2))

        res_view = compiled_function_inference(tensor_view)

        print(input_tensor)
        print(tensor_view)
        print(res)
        print(res_view)
