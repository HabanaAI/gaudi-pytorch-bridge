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

import copy

import habana_frameworks.torch as ht
import habana_frameworks.torch.hpex.experimental.transformer_engine as te
import numpy as np
import pytest
import torch
from habana_frameworks.torch.hpex.experimental.transformer_engine.recipe import DelayedScaling, Format
from test_utils import _kernel_copy_to_device, compare_tensors, is_gaudi1

g = ht.hpu.HPUGraph()
s = ht.hpu.Stream()


def warp_func(first):
    if first:
        with ht.hpu.stream(s):
            g.capture_begin()
            a = torch.full((1000,), 1, device="hpu")
            b = a
            b = b + 1
            g.capture_end()
    else:
        a = torch.full((1000,), 1, device="hpu")
        ht.core.mark_step()
        ht.hpu.default_stream().synchronize()
        g.replay()


def test_graph_capture_simple():
    for i in range(10):
        if i == 0:
            warp_func(True)
        else:
            warp_func(False)
    ht.hpu.synchronize()


@pytest.mark.skip(reason="Results mismatch")
def test_graph_training():
    # N, D_in, H, D_out = 640, 4096, 2048, 1024
    N, D_in, H, D_out = 2, 2, 2, 2
    module1_cpu = torch.nn.Linear(D_in, H).to("cpu")
    module1_hpu = _kernel_copy_to_device(module1_cpu, "hpu")
    loss_fn = torch.nn.MSELoss()
    optimizer_cpu = torch.optim.SGD(module1_cpu.parameters(), lr=0.1)
    optimizer_hpu = torch.optim.SGD(module1_hpu.parameters(), lr=0.1)
    x_cpu = torch.randn(N, D_in, device="cpu")
    x_hpu = x_cpu.to("hpu")
    module1_hpu = ht.hpu.make_graphed_callables(module1_hpu, (x_hpu,))
    real_inputs_cpu = [torch.rand_like(x_cpu) for _ in range(100)]
    real_inputs_hpu = [input.to("hpu") for input in real_inputs_cpu]
    real_targets_cpu = [torch.randn(N, D_out, device="cpu") for _ in range(100)]
    real_targets_hpu = [target.to("hpu") for target in real_targets_cpu]

    for data, target in zip(real_inputs_hpu, real_targets_hpu):
        optimizer_hpu.zero_grad(set_to_none=True)
        tmp = module1_hpu(data)
        loss_hpu = loss_fn(tmp, target)
        loss_hpu.backward()
        optimizer_hpu.step()

    for data, target in zip(real_inputs_cpu, real_targets_cpu):
        optimizer_cpu.zero_grad(set_to_none=True)
        tmp = module1_cpu(data)
        loss_cpu = loss_fn(tmp, target)
        loss_cpu.backward()
        optimizer_cpu.step()
    for j, (p, q) in enumerate(zip(module1_hpu.parameters(), module1_cpu.parameters())):
        if p.requires_grad and q.requires_grad:
            compare_tensors(p, q, atol=0.001, rtol=1.0e-3)
    compare_tensors(loss_hpu, loss_cpu, atol=0.001, rtol=1.0e-3)


def wrapped_func(data, target, module1, loss_fn):
    tmp = module1(data)
    loss = loss_fn(tmp[:].add_(1.0), target)
    return loss


class Model(torch.nn.Module):
    def __init__(self, inp_size, out_size, inner_size):
        super(Model, self).__init__()
        self.Linear1 = torch.nn.Linear(inp_size, inner_size)
        self.Linear2 = torch.nn.Linear(inner_size, out_size)
        self.h = torch.nn.ModuleList([torch.nn.Linear(inp_size, inp_size) for i in range(20)])

    def forward(self, inp):
        for i, (block) in enumerate(self.h):
            if i % 5 == 0:
                ht.core.mark_step()
            inp = block(inp)
        res = self.Linear1(inp)
        ht.core.mark_step()
        return self.Linear2(res)


@pytest.mark.skip(reason="Results mismatch")
def test_multiple_graph_capture():
    # N, D_in, H, D_out = 640, 4096, 2048, 1024
    N, D_in, H, D_out, inner = 2, 2, 2, 2, 4
    module1_cpu = Model(D_in, H, inner).to("cpu")
    module1_hpu = _kernel_copy_to_device(module1_cpu, "hpu")
    loss_fn = torch.nn.MSELoss()
    module1_hpu = ht.hpu.wrap_in_hpu_graph(module1_hpu)
    x_cpu = torch.randn(N, D_in, device="cpu")
    real_inputs_cpu = [torch.rand_like(x_cpu) for _ in range(100)]
    real_inputs_hpu = [input.to("hpu") for input in real_inputs_cpu]
    real_targets_cpu = [torch.randn(N, D_out, device="cpu") for _ in range(100)]
    real_targets_hpu = [target.to("hpu") for target in real_targets_cpu]
    loss_hpu_vec = []
    loss_cpu_vec = []

    for data, target in zip(real_inputs_hpu, real_targets_hpu):
        loss_hpu = wrapped_func(data, target, module1_hpu, loss_fn)
        loss_hpu_vec.append(loss_hpu)
        ht.core.mark_step()

    for data, target in zip(real_inputs_cpu, real_targets_cpu):
        loss_cpu = wrapped_func(data, target, module1_cpu, loss_fn)
        loss_cpu_vec.append(loss_cpu)
    compare_tensors(loss_hpu_vec, loss_cpu_vec, atol=0.001, rtol=1.0e-3)


def test_multiple_graph_capture_memoptimization(asynchronous=False, dry_run=False, release_memory_test=False):
    # N, D_in, H, D_out = 640, 4096, 2048, 1024
    N, D_in, H, D_out, inner = 20, 20, 20, 20, 40
    module1_cpu = Model(D_in, H, inner).to("cpu")
    module1_hpu = _kernel_copy_to_device(module1_cpu, "hpu")
    loss_fn = torch.nn.MSELoss()
    module1_hpu = ht.hpu.wrap_in_hpu_graph(
        module1_hpu, asynchronous=asynchronous, disable_tensor_cache=True, dry_run=dry_run
    )
    x_cpu = torch.randn(N, D_in, device="cpu")
    ITERATION = 10
    real_inputs_cpu = [torch.rand_like(x_cpu) for _ in range(ITERATION)]
    real_inputs_hpu = [input.to("hpu") for input in real_inputs_cpu]
    real_targets_cpu = [torch.randn(N, D_out, device="cpu") for _ in range(ITERATION)]
    real_targets_hpu = [target.to("hpu") for target in real_targets_cpu]
    loss_hpu_vec = []
    loss_cpu_vec = []

    count = 0
    for data, target in zip(real_inputs_hpu, real_targets_hpu):
        loss_hpu = wrapped_func(data, target, module1_hpu, loss_fn)
        loss_hpu_vec.append(loss_hpu)
        ht.core.mark_step()
        if release_memory_test:
            module1_hpu.clear_cache()
        count = count + 1

    for data, target in zip(real_inputs_cpu, real_targets_cpu):
        loss_cpu = wrapped_func(data, target, module1_cpu, loss_fn)
        loss_cpu_vec.append(loss_cpu)
    compare_tensors(loss_hpu_vec, loss_cpu_vec, atol=0.001, rtol=1.0e-3)


def test_tensor_packer():
    x = torch.randn(3, 4).to("hpu")
    y = torch.randn(3, 4).to("hpu")
    z = torch.randn(3, 4).to("hpu")

    output = {"x": x, "y": y}, z

    tensor_packer = ht.hpu.TensorPacker()
    tensors, metadata = tensor_packer.pack(output)
    output_unpacked = tensor_packer.unpack(tensors, metadata)

    metadata_expected = "({'x': #0, 'y': #1}, #2)"
    assert str(metadata) == metadata_expected, "Incorrect metadata:\nExpected {0},  but got {1}".format(
        metadata_expected, metadata
    )

    assert output == output_unpacked


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(4, 4)
        self.fc2 = torch.nn.Linear(4, 4)
        self.fc3 = torch.nn.Linear(4, 4)
        self.fc4 = torch.nn.Linear(4, 4)

    def forward(self, x, y, boolean_var=False):
        x = self.fc1(x)
        y = self.fc2(y)
        z = self.fc3(x + y)
        if boolean_var:
            x = self.fc4(z)
        else:
            y = self.fc4(z)
        return {"x": x, "y": y}, z


@pytest.mark.parametrize("disable_tensor_cache", [True, False])
@pytest.mark.parametrize("dry_run", [True, False])
def test_cached_module_training(disable_tensor_cache, dry_run, save_model=False):
    torch.manual_seed(12345)
    model = Net().to("hpu")
    state_dict = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    meta_args = [((3, 4), True), ((5, 4), False), ((11, 4), True)]

    net_input = []
    net_output = []
    for i in range(2):
        for item in meta_args:
            x = torch.randn(item[0]).to("hpu")
            x.requires_grad_()
            y = torch.randn(item[0]).to("hpu")
            net_input.append({"x": x, "y": y, "boolean_var": item[1]})
            net_output.append(torch.randn(item[0][0]).to("hpu"))

    def train_model():
        m = 0
        for inp, y in zip(net_input, net_output):
            output = model(**inp)
            y_pred = torch.mean(output[1], 1)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(y_pred, y)
            loss.backward()
            optimizer.step()
            ht.core.mark_step()
            inp["x"].grad.add_(1.0)
            m = m + inp["x"].grad.sum()
            inp["x"].grad.zero_()
        return loss.cpu(), m.cpu()

    if save_model:
        torch.save(model, "model.pb")
        model = torch.load("model.pb")
    loss_original, m_original = train_model()
    model.load_state_dict(state_dict)

    ht.hpu.ModuleCacher()(model=model, inplace=True, disable_tensor_cache=disable_tensor_cache)

    if save_model:
        torch.save(model, "model_cached.pb")
        model = torch.load("model_cached.pb")

    loss_cached, m_cached = train_model()
    assert loss_original == loss_cached
    assert m_original == m_cached


class ModelHpu(torch.nn.Module):
    def __init__(self, inp_size, out_size):
        super(ModelHpu, self).__init__()
        self.Linear1 = torch.nn.Linear(inp_size, out_size)

    def forward(self, inp, m):
        res = self.Linear1(inp)
        return res - m


def test_graph_capture_scalar(asynchronous=False, disable_tensor_cache=False):
    N, D_in, H, D_out = 2, 2, 2, 2
    module1_cpu = ModelHpu(D_in, H).to("cpu")
    module1_hpu = _kernel_copy_to_device(module1_cpu, "hpu")
    loss_fn = torch.nn.MSELoss()
    module1_hpu = ht.hpu.wrap_in_hpu_graph(
        module1_hpu, asynchronous=asynchronous, disable_tensor_cache=disable_tensor_cache
    )
    x_cpu = torch.randn(N, D_in, device="cpu")
    ITERATION = 5
    real_inputs_cpu = [torch.rand_like(x_cpu) for _ in range(ITERATION)]
    real_inputs_cpu_scalar = [torch.tensor(i) for i in range(ITERATION)]

    real_inputs_hpu = [input.to("hpu") for input in real_inputs_cpu]
    real_inputs_hpu_scalar = [input.to("hpu") for input in real_inputs_cpu_scalar]

    real_targets_cpu = [torch.randn(N, D_out, device="cpu") for _ in range(ITERATION)]
    real_targets_hpu = [target.to("hpu") for target in real_targets_cpu]
    loss_hpu_vec = []
    loss_cpu_vec = []

    for data, data2, target in zip(real_inputs_hpu, real_inputs_hpu_scalar, real_targets_hpu):
        loss_hpu = wrapped_func_scalar(data, data2, target, module1_hpu, loss_fn)
        loss_hpu_vec.append(loss_hpu)
        ht.core.mark_step()

    for data, data2, target in zip(real_inputs_cpu, real_inputs_cpu_scalar, real_targets_cpu):
        loss_cpu = wrapped_func_scalar(data, data2, target, module1_cpu, loss_fn)
        loss_cpu_vec.append(loss_cpu)

    compare_tensors(loss_hpu_vec, loss_cpu_vec, atol=0.001, rtol=1.0e-3)


def wrapped_func_scalar(data, data2, target, module1, loss_fn):
    tmp = module1(data, data2)
    loss = loss_fn(tmp[:], target)
    return loss


@pytest.mark.skip(reason="Results mismatch")
def test_multiple_graph_capture_with_views():
    N, D_in, H, D_out, inner = 2, 2, 2, 2, 2
    module1_cpu = Model(D_in, H, inner).to("cpu")
    module1_hpu = _kernel_copy_to_device(module1_cpu, "hpu")
    loss_fn = torch.nn.MSELoss()
    module1_hpu = ht.hpu.wrap_in_hpu_graph(module1_hpu, disable_tensor_cache=True)
    x_cpu = torch.randn(N, D_in, device="cpu")
    real_inputs_cpu = [torch.rand_like(x_cpu) for _ in range(2)]
    real_inputs_hpu = [input.to("hpu") for input in real_inputs_cpu]
    real_inputs_cpu1 = [torch.rand_like(x_cpu) for _ in range(2)]
    real_inputs_hpu1 = [input.to("hpu") for input in real_inputs_cpu1]
    real_targets_cpu = [torch.ones(N, D_out, device="cpu") for _ in range(2)]
    real_targets_hpu = [target.to("hpu") for target in real_targets_cpu]
    loss_hpu_vec = []
    loss_cpu_vec = []

    # Input as view first time while capture, second time not view
    for data, data1, target in zip(real_inputs_hpu, real_inputs_hpu1, real_targets_hpu):
        data = torch.transpose(data, 0, 1)
        loss_hpu = wrapped_func(data, target, module1_hpu, loss_fn)
        loss_hpu_vec.append(loss_hpu)
        loss_hpu = wrapped_func(data1, target, module1_hpu, loss_fn)
        loss_hpu_vec.append(loss_hpu)
        ht.core.mark_step()

    for data, data1, target in zip(real_inputs_cpu, real_inputs_cpu1, real_targets_cpu):
        data = torch.transpose(data, 0, 1)
        loss_cpu = wrapped_func(data, target, module1_cpu, loss_fn)
        loss_cpu_vec.append(loss_cpu)
        loss_cpu = wrapped_func(data, target, module1_cpu, loss_fn)
        loss_cpu_vec.append(loss_cpu)
        ht.core.mark_step()

    compare_tensors(loss_hpu_vec, loss_cpu_vec, atol=0.001, rtol=1.0e-3)
    loss_hpu_vec = []
    loss_cpu_vec = []

    # Input as not view first time while capture, view on second turn
    for data, data1, target in zip(real_inputs_hpu, real_inputs_hpu1, real_targets_hpu):
        loss_hpu = wrapped_func(data, target, module1_hpu, loss_fn)
        loss_hpu_vec.append(loss_hpu)
        data = torch.transpose(data, 0, 1)
        loss_hpu = wrapped_func(data, target, module1_hpu, loss_fn)
        loss_hpu_vec.append(loss_hpu)
        ht.core.mark_step()

    for data, data1, target in zip(real_inputs_cpu, real_inputs_cpu1, real_targets_cpu):
        data = torch.transpose(data, 0, 1)
        loss_cpu = wrapped_func(data, target, module1_cpu, loss_fn)
        loss_cpu_vec.append(loss_cpu)
        loss_cpu = wrapped_func(data, target, module1_cpu, loss_fn)
        loss_cpu_vec.append(loss_cpu)
        ht.core.mark_step()

    compare_tensors(loss_hpu_vec, loss_cpu_vec, atol=0.001, rtol=1.0e-3)
    loss_hpu_vec = []
    loss_cpu_vec = []

    # Input as not view first time while capture, view on second turn
    for data, data1, target in zip(real_inputs_hpu, real_inputs_hpu1, real_targets_hpu):
        data = torch.transpose(data, 0, 1)
        loss_hpu = wrapped_func(data, target, module1_hpu, loss_fn)
        loss_hpu_vec.append(loss_hpu)
        data[:] = data1[:]
        loss_hpu = wrapped_func(data, target, module1_hpu, loss_fn)
        loss_hpu_vec.append(loss_hpu)
        data[0:1] = data1.sum()
        loss_hpu = wrapped_func(data, target, module1_hpu, loss_fn)
        loss_hpu_vec.append(loss_hpu)
        data = torch.t(data)
        loss_hpu = wrapped_func(data, target, module1_hpu, loss_fn)
        loss_hpu_vec.append(loss_hpu)
        ht.core.mark_step()

    for data, data1, target in zip(real_inputs_cpu, real_inputs_cpu1, real_targets_cpu):
        data = torch.transpose(data, 0, 1)
        loss_cpu = wrapped_func(data, target, module1_cpu, loss_fn)
        loss_cpu_vec.append(loss_cpu)
        data[:] = data1[:]
        loss_cpu = wrapped_func(data, target, module1_cpu, loss_fn)
        loss_cpu_vec.append(loss_cpu)
        data[0:1] = data1.sum()
        loss_cpu = wrapped_func(data, target, module1_cpu, loss_fn)
        loss_cpu_vec.append(loss_cpu)
        data = torch.t(data)
        loss_cpu = wrapped_func(data, target, module1_cpu, loss_fn)
        loss_cpu_vec.append(loss_cpu)
        ht.core.mark_step()

    compare_tensors(loss_hpu_vec, loss_cpu_vec, atol=0.001, rtol=1.0e-3)
    loss_hpu_vec = []
    loss_cpu_vec = []


def test_wrap_hpugraphs_max_graphs(max_graphs=10):
    D_in, H, D_out, inner = 2, 2, 2, 4
    N = [1, 2, 3, 4, 5]
    module1_cpu = Model(D_in, H, inner).to("cpu")
    module1_hpu = _kernel_copy_to_device(module1_cpu, "hpu")
    loss_fn = torch.nn.MSELoss()
    module1_hpu = ht.hpu.wrap_in_hpu_graph(module1_hpu, max_graphs=max_graphs)
    ITER = 20
    real_inputs_cpu = []
    real_targets_cpu = []
    for n in N:
        x_cpu = torch.randn(n, D_in, device="cpu")
        inp_samples = [torch.rand_like(x_cpu) for _ in range(ITER)]
        real_inputs_cpu.extend(inp_samples)
        tgt_samples = [torch.randn(n, D_out, device="cpu") for _ in range(ITER)]
        real_targets_cpu.extend(tgt_samples)
    real_inputs_hpu = [input.to("hpu") for input in real_inputs_cpu]
    real_targets_hpu = [target.to("hpu") for target in real_targets_cpu]
    loss_hpu_vec = []
    loss_cpu_vec = []

    for data, target in zip(real_inputs_hpu, real_targets_hpu):
        loss_hpu = wrapped_func(data, target, module1_hpu, loss_fn)
        loss_hpu_vec.append(loss_hpu)
        ht.core.mark_step()

    for data, target in zip(real_inputs_cpu, real_targets_cpu):
        loss_cpu = wrapped_func(data, target, module1_cpu, loss_fn)
        loss_cpu_vec.append(loss_cpu)
    compare_tensors(loss_hpu_vec, loss_cpu_vec, atol=0.001, rtol=1.0e-3)


@pytest.mark.skipif(is_gaudi1(), reason="G1 unsupported dtype")
@pytest.mark.parametrize("disable_tensor_cache", [True])
def test_cached_module_training_fp8(disable_tensor_cache):
    torch.manual_seed(12345)
    input0 = torch.tensor([0.1, 0.2, 0.3, 0.4]).to("hpu")
    input1 = torch.tensor([1, 2, 3, 4]).to("hpu")
    input2 = torch.tensor([10, 20, 30, 40]).to("hpu")
    input3 = torch.tensor([100, 200, 300, 400]).to("hpu")

    fp8_format = Format.E5M2
    fp8_recipe = DelayedScaling(
        fp8_format=fp8_format,
        amax_history_len=1,
        amax_compute_algo="max",
        margin=0,
        reduce_amax=False,
    )

    torch.manual_seed(12345)
    my_linear_ref = te.Linear(4, 3, bias=True)
    torch.manual_seed(12345)
    my_linear_test = te.Linear(4, 3, bias=True)

    inputs = [input1, input2, input3, input2, input1, input2, input3, input3, input1, input1, input3]
    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        # Run one iteration before capturing, because scales are not computed during first iteration (it's a different graph)
        out_ref = my_linear_ref(input0)
        loss_ref = out_ref.sum()
        loss_ref.backward()
        out_test = my_linear_test(input0)
        loss_test = out_test.sum()
        loss_test.backward()

        grad_w_ref = my_linear_ref.weight.grad.clone().to(torch.float).cpu().detach()
        grad_b_ref = my_linear_ref.bias.grad.clone().to(torch.float).cpu().detach()
        grad_w_test = my_linear_test.weight.grad.clone().to(torch.float).cpu().detach()
        grad_b_test = my_linear_test.bias.grad.clone().to(torch.float).cpu().detach()
        assert np.array_equal(
            out_test.cpu().to(torch.float).detach().numpy(),
            out_ref.cpu().to(torch.float).detach().numpy(),
            equal_nan=True,
        ), f"Out data mismatch at init run"
        assert np.array_equal(
            grad_w_test.numpy(), grad_w_ref.numpy(), equal_nan=True
        ), f"Grad weight data mismatch at init run"
        assert np.array_equal(
            grad_b_test.numpy(), grad_b_ref.numpy(), equal_nan=True
        ), f"Grad bias data mismatch at init run"
        my_linear_ref.zero_grad(set_to_none=False)
        my_linear_test.zero_grad(set_to_none=False)

        # Wrap the modules in hpu_graph wrapper
        fp8_meta = my_linear_test.save_fp8_meta()
        x = torch.zeros_like(input1)
        my_linear_test = ht.hpu.ModuleCacher(max_graphs=10)(
            have_grad_accumulation=True, model=my_linear_test, inplace=True, disable_tensor_cache=disable_tensor_cache
        )
        out_x = my_linear_test(x).cpu()
        my_linear_test.load_fp8_meta(fp8_meta)
        my_linear_test.zero_grad()

        # Run recorded graph n times
        for i in range(0, len(inputs)):
            my_linear_test.set_iteration_count(i)
            out_test = my_linear_test(inputs[i])
            loss_test = out_test.sum()
            loss_test.backward()
            grad_w_test = my_linear_test.weight.grad.clone().to(torch.float).cpu().detach()
            grad_b_test = my_linear_test.bias.grad.clone().to(torch.float).cpu().detach()

            out_ref = my_linear_ref(inputs[i])
            loss_ref = out_ref.sum()
            loss_ref.backward()
            grad_w_ref = my_linear_ref.weight.grad.clone().to(torch.float).cpu().detach()
            grad_b_ref = my_linear_ref.bias.grad.clone().to(torch.float).cpu().detach()

            assert np.array_equal(
                out_test.cpu().to(torch.float).detach().numpy(),
                out_ref.cpu().to(torch.float).detach().numpy(),
                equal_nan=True,
            ), f"Out data mismatch at {i}"
            assert np.array_equal(
                grad_w_test.numpy(), grad_w_ref.numpy(), equal_nan=True
            ), f"Grad weight data mismatch at {i}"
            assert np.array_equal(
                grad_b_test.numpy(), grad_b_ref.numpy(), equal_nan=True
            ), f"Grad bias data mismatch at {i}"


def test_module_cacher_no_requires_grad():
    class Network(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden = torch.nn.Linear(4, 4)

        def forward(self, x, z):
            x = self.hidden(x)
            x = torch.relu_(x)
            with torch.no_grad():
                y = x.new_ones(*x.shape, requires_grad=False)[-1]
                y = (y + 2) * x
                z = z + y * x
                k = x
            return {"T1": x, "T2": y, "T3": z, "T4": k, "flag": 2}

    torch.manual_seed(12345)
    model = Network().to("hpu")
    state_dict = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    meta_args = [((4, 4)), ((4, 4))]

    net_input = []
    net_output = []
    for i in range(2):
        for item in meta_args:
            x = torch.randn(item[0]).to("hpu")
            x.requires_grad_()
            y = torch.randn(item[0]).to("hpu")
            net_input.append({"x": x, "z": y})
            net_output.append(torch.randn(item[0]).to("hpu"))

    def train_model():
        m = 0
        for inp, y in zip(net_input, net_output):
            output = model(**inp)
            y_pred = output["T1"] + output["T2"] + output["T3"] + output["T4"]
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(y_pred, y)
            loss.backward()
            optimizer.step()
            ht.core.mark_step()
            inp["x"].grad.add_(1.0)
            m = m + inp["x"].grad.sum()
            inp["x"].grad.zero_()
        return loss.cpu(), m.cpu()

    loss_original, m_original = train_model()
    model.load_state_dict(state_dict)
    ht.hpu.ModuleCacher()(model=model, inplace=True)
    loss_cached, m_cached = train_model()
    assert loss_original == loss_cached
    assert m_original == m_cached


class ModelPropNet(torch.nn.Module):
    def __init__(self):
        super(ModelPropNet, self).__init__()
        self.Linear1 = torch.nn.Linear(20, 25)
        self.relu = torch.nn.ReLU()
        self.Linear2 = torch.nn.Linear(25, 3)

    def forward(self, inp):
        res = self.Linear1(inp)
        res2 = self.relu(res)
        res3 = self.Linear2(res2)
        res4 = self.relu(res3)
        return res4


def test_module_cacher_propnet():
    torch.manual_seed(123)
    module1_cpu = ModelPropNet()
    torch.manual_seed(123)
    module1_hpu = ModelPropNet().to("hpu")
    inputs_cpu = []
    inputs_hpu = []
    outputs_target_cpu = []
    outputs_target_hpu = []

    for i in range(6):
        t = torch.rand(i + 1, 20)
        inputs_cpu.append(t)
        inputs_hpu.append(t.to("hpu"))
        o = torch.tensor(1.0)
        outputs_target_cpu.append(o)
        outputs_target_hpu.append(o.to("hpu"))

    module1_hpu = ht.hpu.ModuleCacher()(have_grad_accumulation=True, model=module1_hpu, inplace=True)
    loss_fn = torch.nn.MSELoss()
    optim_y_cpu = torch.optim.SGD(module1_cpu.parameters(), lr=0.1)
    optim_y_hpu = torch.optim.SGD(module1_hpu.parameters(), lr=0.1)

    optim_y_cpu.zero_grad()
    optim_y_hpu.zero_grad()

    for i in range(6):
        module1_hpu.set_iteration_count(i)
        out_cpu = module1_cpu(inputs_cpu[i])
        out_hpu = module1_hpu(inputs_hpu[i])
        loss_cpu = loss_fn(out_cpu.sum(), outputs_target_cpu[i])
        loss_hpu = loss_fn(out_hpu.sum(), outputs_target_hpu[i])
        loss_cpu.backward()
        loss_hpu.backward()
        ht.core.mark_step()

        with torch.no_grad():
            for p in module1_cpu.parameters():
                if p is not None:
                    p.mul_(0.5)

            for p in module1_hpu.parameters():
                if p is not None:
                    p.mul_(0.5)

        torch.nn.utils.clip_grad_norm_(module1_cpu.parameters(), 0.1)
        torch.nn.utils.clip_grad_norm_(module1_hpu.parameters(), 0.1)
        optim_y_cpu.step()
        optim_y_hpu.step()
        ht.core.mark_step()
        assert torch.allclose(out_cpu, out_hpu.to("cpu"))


def test_module_cacher_propnet_views():
    torch.manual_seed(123)
    module1_cpu = ModelPropNet()
    torch.manual_seed(123)
    module1_hpu = ModelPropNet().to("hpu")
    inputs_cpu = []
    inputs_hpu = []
    outputs_target_cpu = []
    outputs_target_hpu = []

    for i in range(6):
        t = torch.rand(i + 1, 20)
        inputs_cpu.extend(t)
        inputs_hpu.extend(t.to("hpu"))
        o = torch.tensor(1.0)
        outputs_target_cpu.append(o)
        outputs_target_hpu.append(o.to("hpu"))

    module1_hpu = ht.hpu.ModuleCacher()(have_grad_accumulation=True, model=module1_hpu, inplace=True)
    loss_fn = torch.nn.MSELoss()
    optim_y_cpu = torch.optim.SGD(module1_cpu.parameters(), lr=0.1)
    optim_y_hpu = torch.optim.SGD(module1_hpu.parameters(), lr=0.1)

    optim_y_cpu.zero_grad()
    optim_y_hpu.zero_grad()

    for i in range(6):
        module1_hpu.set_iteration_count(i)
        out_cpu = module1_cpu(inputs_cpu[i])
        out_hpu = module1_hpu(inputs_hpu[i])
        loss_cpu = loss_fn(out_cpu.sum(), outputs_target_cpu[i])
        loss_hpu = loss_fn(out_hpu.sum(), outputs_target_hpu[i])
        loss_cpu.backward()
        loss_hpu.backward()
        ht.core.mark_step()

        with torch.no_grad():
            for p in module1_cpu.parameters():
                if p is not None:
                    p.mul_(0.5)

            for p in module1_hpu.parameters():
                if p is not None:
                    p.mul_(0.5)

        torch.nn.utils.clip_grad_norm_(module1_cpu.parameters(), 0.1)
        torch.nn.utils.clip_grad_norm_(module1_hpu.parameters(), 0.1)
        optim_y_cpu.step()
        optim_y_hpu.step()
        ht.core.mark_step()
        assert torch.allclose(out_cpu, out_hpu.to("cpu"))


class ModelPropRandNet(torch.nn.Module):
    def __init__(self):
        super(ModelPropRandNet, self).__init__()
        self.Linear1 = torch.nn.Linear(20, 25)
        self.relu = torch.nn.ReLU()
        self.Linear2 = torch.nn.Linear(25, 3)

    def forward(self, inp):
        inp.normal_()
        res = self.Linear1(inp)
        res2 = self.relu(res)
        res3 = self.Linear2(res2)
        res4 = self.relu(res3)
        return res4


def test_module_cacher_propnet_rand():
    torch.manual_seed(123)
    module1_hpu_ref = ModelPropRandNet().to("hpu")
    torch.manual_seed(123)
    module1_hpu = ModelPropRandNet().to("hpu")
    inputs_hpu_ref = []
    inputs_hpu = []
    outputs_target_hpu_ref = []
    outputs_target_hpu = []
    outputs_hpu_ref = []
    outputs_hpu = []

    for i in range(6):
        t = torch.rand(1, 20)
        inputs_hpu_ref.append(t.to("hpu"))
        inputs_hpu.append(t.to("hpu"))
        o = torch.tensor(1.0)
        outputs_target_hpu_ref.append(o.to("hpu"))
        outputs_target_hpu.append(o.to("hpu"))

    module1_hpu = ht.hpu.ModuleCacher()(
        have_grad_accumulation=True, model=module1_hpu, inplace=True, allow_unused_input=True, dry_run=True
    )
    loss_fn = torch.nn.MSELoss()
    optim_y_hpu_ref = torch.optim.SGD(module1_hpu_ref.parameters(), lr=0.1)
    optim_y_hpu = torch.optim.SGD(module1_hpu.parameters(), lr=0.1)

    optim_y_hpu_ref.zero_grad()
    optim_y_hpu.zero_grad()
    torch.manual_seed(0)
    for i in range(6):
        module1_hpu.set_iteration_count(i)
        out_hpu = module1_hpu(inputs_hpu[i])
        loss_hpu = loss_fn(out_hpu.sum(), outputs_target_hpu[i])
        loss_hpu.backward()
        ht.core.mark_step()

        with torch.no_grad():
            for p in module1_hpu.parameters():
                if p is not None:
                    p.mul_(0.5)

        torch.nn.utils.clip_grad_norm_(module1_hpu.parameters(), 0.1)
        optim_y_hpu.step()
        ht.core.mark_step()
        outputs_hpu.append(out_hpu.to("cpu"))

    torch.manual_seed(0)
    for i in range(6):
        out_hpu = module1_hpu_ref(inputs_hpu_ref[i])
        loss_hpu = loss_fn(out_hpu.sum(), outputs_target_hpu_ref[i])
        loss_hpu.backward()
        ht.core.mark_step()

        with torch.no_grad():
            for p in module1_hpu_ref.parameters():
                if p is not None:
                    p.mul_(0.5)

        torch.nn.utils.clip_grad_norm_(module1_hpu_ref.parameters(), 0.1)
        optim_y_hpu_ref.step()
        ht.core.mark_step()
        outputs_hpu_ref.append(out_hpu.to("cpu"))

    for i in range(6):
        assert torch.allclose(outputs_hpu[i], outputs_hpu_ref[i])
