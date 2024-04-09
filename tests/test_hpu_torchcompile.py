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

import pytest
import torch
from test_utils import env_var_in_scope


@pytest.mark.xfail
def test_simple_convolution():
    with env_var_in_scope({"PT_HPU_LAZY_MODE": "0"}):
        pass

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

        compiled_model = torch.compile(model, backend="hpu_backend")

        tensor = torch.rand(8, 1, 32, 32).to("hpu")

        res_eager = raw_model(tensor)
        res_graph = compiled_model(tensor)
        assert torch.allclose(res_eager, res_graph, rtol=1e-03)


@pytest.mark.xfail(reason="KeyError: 'torch_dynamo_backends")
def test_simple_convolution_mixed():
    with env_var_in_scope({"PT_HPU_LAZY_MODE": "0"}):
        pass

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

        compiled_function_1 = torch.compile(raw_function_1, backend="hpu_backend")
        compiled_function_2 = torch.compile(raw_function_2, backend="hpu_backend")

        res_eager = raw_function_2(raw_function_1(tensor))
        res_graph_to_eager = raw_function_2(compiled_function_1(tensor))
        res_eager_to_graph = compiled_function_2(raw_function_1(tensor))

        assert torch.allclose(res_eager, res_graph_to_eager, rtol=1e-03)
        assert torch.allclose(res_eager, res_eager_to_graph, rtol=1e-03)


@pytest.mark.xfail(reason="KeyError: 'torch_dynamo_backends")
def test_simple_sgd_convnet():
    with env_var_in_scope({"PT_HPU_LAZY_MODE": "0"}):
        pass

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

        compiled_function_test = torch.compile(raw_function_test, backend="hpu_backend")
        compiled_function_train = torch.compile(raw_function_train, backend="hpu_backend")

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


@pytest.mark.skip
def test_simple_sgd_convnet_with_device_pingpong():
    with env_var_in_scope({"PT_HPU_LAZY_MODE": "0"}):
        pass

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

        compiled_function_test = torch.compile(raw_function_test, backend="hpu_backend")
        compiled_function_train = torch.compile(raw_function_train, backend="hpu_backend")

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
    with env_var_in_scope({"PT_HPU_LAZY_MODE": "0"}):

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

        compiled_function_test = torch.compile(raw_function_test, backend="hpu_backend")
        compiled_function_train = torch.compile(raw_function_train, backend="hpu_backend")

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
    with env_var_in_scope({"PT_HPU_LAZY_MODE": "0"}):
        pass

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

        compiled_function_test = torch.compile(raw_function_test, backend="hpu_backend")
        compiled_function_train = torch.compile(raw_function_train, backend="hpu_backend")

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


@pytest.mark.xfail(reason="KeyError: 'torch_dynamo_backends")
def test_simple_view():
    with env_var_in_scope({"PT_HPU_LAZY_MODE": "0"}):
        pass

        def raw_function(x):
            return torch.relu(x)

        compiled_function_inference = torch.compile(raw_function, backend="hpu_backend")

        input_tensor = torch.rand(3, 3, device="cpu").to("hpu")

        res = compiled_function_inference(input_tensor)

        print(compiled_function_inference.__class__)

        tensor_view = input_tensor.as_strided((2, 2), (1, 2))

        res_view = compiled_function_inference(tensor_view)

        print(input_tensor)
        print(tensor_view)
        print(res)
        print(res_view)


@pytest.mark.xfail(reason="AttributeError: 'NoneType' object has no attribute 'reset'")
def test_cache_metrics_enabled_and_graph_compilaton():
    with env_var_in_scope({"PT_HPU_LAZY_MODE": "0", "PT_HPU_ENABLE_CACHE_METRICS": "1"}):
        from habana_frameworks.torch.hpu.metrics import metric_global

        gc_metric = metric_global("graph_compilation")
        gc_metric.reset()
        rc_metric = metric_global("recipe_cache")
        rc_metric.reset()

        import habana_frameworks.torch.core as htcore

        def raw_function(x):
            return torch.relu(x)

        compiled_function_inference = torch.compile(raw_function, backend="hpu_backend")
        input_tensor = torch.rand(3, 3, device="cpu").to("hpu")
        last_total_time = 0
        for curr_iter in range(5):
            res = compiled_function_inference(input_tensor)
            gc_metric_dict = dict(gc_metric.stats())
            rc_metric_dict = dict(rc_metric.stats())
            assert gc_metric_dict["TotalNumber"] == 1
            assert rc_metric_dict["TotalMiss"] == 1
            assert gc_metric_dict["TotalTime"] >= last_total_time
            last_total_time = gc_metric_dict["TotalTime"]


@pytest.mark.xfail(reason="AttributeError: 'NoneType' object has no attribute 'reset'")
def test_metrics_eager_mode():
    with env_var_in_scope({"PT_HPU_LAZY_MODE": "0", "PT_HPU_ENABLE_CACHE_METRICS": "1"}):
        from habana_frameworks.torch.hpu.metrics import metric_global

        rc_metric = metric_global("recipe_cache")
        rc_metric.reset()
        gc_metric = metric_global("graph_compilation")
        gc_metric.reset()

        import habana_frameworks.torch.core as htcore
        import habana_frameworks.torch.utils.experimental as htexp

        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Sequential(
                    torch.nn.Conv2d(1, 6, kernel_size=9, stride=2, padding=1),
                    torch.nn.Conv2d(6, 3, kernel_size=1, stride=1, padding=0),
                )

            def forward(self, x):
                out = self.layer(x)
                return out

        tensor = torch.rand(1, 1, 16, 16).to("hpu")
        for curr_iter in range(2):
            model = Net().to("hpu")
            # print for the sake of consuming the output, so that it's not pruned
            print(model(tensor))
            gc_metric_dict = dict(gc_metric.stats())
            rc_metric_dict = dict(rc_metric.stats())
            # eager compilation not supported on Gaudi1
            assert gc_metric_dict["TotalNumber"] == 0 or htexp._get_device_type() == htexp.synDeviceType.synDeviceGaudi
            assert rc_metric_dict["TotalMiss"] == 0
            assert rc_metric_dict["TotalHit"] == 0
