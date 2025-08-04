###############################################################################
# Copyright (c) 2021-2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################

import habana_frameworks.torch as ht
import habana_frameworks.torch.core as htcore
import torch
import torch.nn.functional as F
from test_utils import _kernel_copy_to_device, compare_tensors
from torch import nn, optim


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def testCaptureWithMultipleStreams():
    device = torch.device("hpu")
    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=0.1)
    batchsize = 200

    # Placeholders used for capture
    static_input = torch.randn(batchsize, 1, 28, 28, device="hpu")
    static_target = torch.randint(0, 10, (batchsize,), device="hpu")

    # First we warmup
    s = htcore.hpu.Stream()
    s.wait_stream(htcore.hpu.current_stream())
    with htcore.hpu.stream(s):
        for _i in range(3):
            optimizer.zero_grad(set_to_none=True)
            y_pred = model(static_input)
            loss = F.nll_loss(y_pred, static_target)
            loss.backward()
            optimizer.step()
    htcore.hpu.current_stream().wait_stream(s)

    # Then we capture
    g = htcore.hpu.HPUGraph()
    optimizer.zero_grad(set_to_none=True)

    with htcore.hpu.graph(g):
        static_y_pred = model(static_input)
        static_loss = F.nll_loss(static_y_pred, static_target)
        static_loss.backward()
        optimizer.step()
    g.replay()


def testCaptureWithSingleStream():
    device = torch.device("hpu")
    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=0.1)
    batchsize = 200

    static_input = torch.randn(batchsize, 1, 28, 28, device="hpu")
    static_target = torch.randint(0, 10, (batchsize,), device="hpu")

    s = htcore.hpu.Stream()
    g = htcore.hpu.HPUGraph()
    optimizer.zero_grad(set_to_none=True)
    # capture
    with htcore.hpu.graph(g, stream=s):
        static_y_pred = model(static_input)
        static_loss = F.nll_loss(static_y_pred, static_target)
        static_loss.backward()
        optimizer.step()
    g.replay()


class ModelPropNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.Linear1 = torch.nn.Linear(20, 25)
        self.relu = torch.nn.ReLU()
        self.Linear2 = torch.nn.Linear(25, 3)

    def forward(self, inp):
        s = htcore.hpu.Stream()
        with htcore.hpu.stream(s):
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


class ModelHpu(torch.nn.Module):
    def __init__(self, inp_size, out_size):
        super().__init__()
        self.Linear1 = torch.nn.Linear(inp_size, out_size)

    def forward(self, inp, m):
        s = htcore.hpu.Stream()
        with htcore.hpu.stream(s):
            res = self.Linear1(inp)
        return res - m


def test_graph_capture_scalar(asynchronous=False, disable_tensor_cache=False):
    N, D_in, H, D_out = 2, 2, 2, 2
    module1_cpu = ModelHpu(D_in, H).to("cpu")
    module1_hpu = _kernel_copy_to_device(module1_cpu, "hpu")
    loss_fn = torch.nn.MSELoss()
    module1_hpu = ht.hpu.wrap_in_hpu_graph(
        module1_hpu,
        asynchronous=asynchronous,
        disable_tensor_cache=disable_tensor_cache,
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

    def wrapped_func_scalar(data, data2, target, module1, loss_fn):
        tmp = module1(data, data2)
        loss = loss_fn(tmp[:], target)
        return loss

    for data, data2, target in zip(real_inputs_hpu, real_inputs_hpu_scalar, real_targets_hpu, strict=False):
        loss_hpu = wrapped_func_scalar(data, data2, target, module1_hpu, loss_fn)
        loss_hpu_vec.append(loss_hpu)
        ht.core.mark_step()

    for data, data2, target in zip(real_inputs_cpu, real_inputs_cpu_scalar, real_targets_cpu, strict=False):
        loss_cpu = wrapped_func_scalar(data, data2, target, module1_cpu, loss_fn)
        loss_cpu_vec.append(loss_cpu)

    compare_tensors(loss_hpu_vec, loss_cpu_vec, atol=0.001, rtol=1.0e-3)
