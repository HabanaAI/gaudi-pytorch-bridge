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

import copy
import torch
import habana_frameworks.torch as ht
from test_utils import compare_tensors, _kernel_copy_to_device

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

def test_graph_training():
    #N, D_in, H, D_out = 640, 4096, 2048, 1024
    N, D_in, H, D_out = 2, 2, 2, 2
    module1_cpu = torch.nn.Linear(D_in, H).to('cpu')
    module1_hpu = _kernel_copy_to_device(module1_cpu,"hpu")
    loss_fn = torch.nn.MSELoss()
    optimizer_cpu = torch.optim.SGD(module1_cpu.parameters(),lr=0.1)
    optimizer_hpu = torch.optim.SGD(module1_hpu.parameters(),lr=0.1)
    x_cpu = torch.randn(N, D_in, device='cpu')
    x_hpu = x_cpu.to('hpu')
    module1_hpu = ht.hpu.make_graphed_callables(module1_hpu, (x_hpu,))
    real_inputs_cpu = [torch.rand_like(x_cpu) for _ in range(100)]
    real_inputs_hpu = [input.to('hpu') for input in real_inputs_cpu]
    real_targets_cpu = [torch.randn(N, D_out, device="cpu") for _ in range(100)]
    real_targets_hpu = [target.to('hpu') for target in real_targets_cpu]

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
    compare_tensors(loss_hpu, loss_cpu, atol=0.001, rtol=1.e-3)


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

def test_multiple_graph_capture():
    #N, D_in, H, D_out = 640, 4096, 2048, 1024
    N, D_in, H, D_out, inner = 2, 2, 2, 2, 4
    module1_cpu = Model(D_in, H, inner).to('cpu')
    module1_hpu = _kernel_copy_to_device(module1_cpu,"hpu")
    loss_fn = torch.nn.MSELoss()
    module1_hpu = ht.hpu.wrap_in_hpu_graph(module1_hpu)
    x_cpu = torch.randn(N, D_in, device='cpu')
    real_inputs_cpu = [torch.rand_like(x_cpu) for _ in range(100)]
    real_inputs_hpu = [input.to('hpu') for input in real_inputs_cpu]
    real_targets_cpu = [torch.randn(N, D_out, device="cpu") for _ in range(100)]
    real_targets_hpu = [target.to('hpu') for target in real_targets_cpu]
    loss_hpu_vec = []
    loss_cpu_vec = []

    for data, target in zip(real_inputs_hpu, real_targets_hpu):
        loss_hpu = wrapped_func(data, target, module1_hpu, loss_fn)
        loss_hpu_vec.append(loss_hpu)
        ht.core.mark_step()

    for data, target in zip(real_inputs_cpu, real_targets_cpu):
        loss_cpu = wrapped_func(data, target, module1_cpu, loss_fn)
        loss_cpu_vec.append(loss_cpu)
    compare_tensors(loss_hpu_vec, loss_cpu_vec, atol=0.001, rtol=1.e-3)

def test_multiple_graph_capture_memoptimization(asynchronous=False, dry_run=False, release_memory_test=False):
    #N, D_in, H, D_out = 640, 4096, 2048, 1024
    N, D_in, H, D_out, inner = 20, 20, 20, 20, 40
    module1_cpu = Model(D_in, H, inner).to('cpu')
    module1_hpu = _kernel_copy_to_device(module1_cpu,"hpu")
    loss_fn = torch.nn.MSELoss()
    module1_hpu = ht.hpu.wrap_in_hpu_graph(module1_hpu, asynchronous=asynchronous, disable_tensor_cache=True, dry_run=dry_run)
    x_cpu = torch.randn(N, D_in, device='cpu')
    ITERATION=10
    real_inputs_cpu = [torch.rand_like(x_cpu) for _ in range(ITERATION)]
    real_inputs_hpu = [input.to('hpu') for input in real_inputs_cpu]
    real_targets_cpu = [torch.randn(N, D_out, device="cpu") for _ in range(ITERATION)]
    real_targets_hpu = [target.to('hpu') for target in real_targets_cpu]
    loss_hpu_vec = []
    loss_cpu_vec = []

    import habana_frameworks.torch as htx
    count = 0
    for data, target in zip(real_inputs_hpu, real_targets_hpu):
        loss_hpu = wrapped_func(data, target, module1_hpu, loss_fn)
        loss_hpu_vec.append(loss_hpu)
        ht.core.mark_step()
        if release_memory_test:
            module1_hpu.destroy()
        count = count+1

    for data, target in zip(real_inputs_cpu, real_targets_cpu):
        loss_cpu = wrapped_func(data, target, module1_cpu, loss_fn)
        loss_cpu_vec.append(loss_cpu)
    compare_tensors(loss_hpu_vec, loss_cpu_vec, atol=0.001, rtol=1.e-3)

def test_tensor_packer():
    x = torch.randn(3, 4).to('hpu')
    y = torch.randn(3, 4).to('hpu')
    z = torch.randn(3, 4).to('hpu')

    output = {'x' : x, 'y' : y}, z

    tensor_packer = ht.hpu.TensorPacker()
    tensors, metadata = tensor_packer.pack(output)
    output_unpacked = tensor_packer.unpack(tensors, metadata)

    metadata_expected = "({'x': #0, 'y': #1}, #2)"
    assert str(metadata) == metadata_expected, "Incorrect metadata:\nExpected {0},  but got {1}".format(metadata_expected, metadata)

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
        return {'x' : x, 'y' : y}, z

def test_cached_module_training(disable_tensor_cache=False):
    model = Net().to('hpu')
    state_dict = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.SGD(model.parameters(),lr=0.1)

    meta_args = [((3, 4), True), ((5, 4), False), ((11, 4), True)]

    net_input = []
    net_output = []
    for i in range(2):
        for item in meta_args:
            x = torch.randn(item[0]).to('hpu')
            x.requires_grad_()
            y = torch.randn(item[0]).to('hpu')
            net_input.append({'x' : x, 'y' : y, 'boolean_var' : item[1]})
            net_output.append(torch.randn(item[0][0]).to('hpu'))

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
            inp['x'].grad.add_(1.0)
            m = m + inp['x'].grad.sum()
            inp['x'].grad.zero_()
        return loss.cpu(), m.cpu()

    loss_original, m_original = train_model()
    model.load_state_dict(state_dict)
    ht.hpu.ModuleCacher()(model=model, inplace=True, disable_tensor_cache=disable_tensor_cache)
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
    module1_cpu = ModelHpu(D_in, H).to('cpu')
    module1_hpu = _kernel_copy_to_device(module1_cpu,"hpu")
    loss_fn = torch.nn.MSELoss()
    module1_hpu = ht.hpu.wrap_in_hpu_graph(module1_hpu, asynchronous=asynchronous, disable_tensor_cache=disable_tensor_cache)
    x_cpu = torch.randn(N, D_in, device='cpu')
    ITERATION=5
    real_inputs_cpu = [torch.rand_like(x_cpu) for _ in range(ITERATION)]
    real_inputs_cpu_scalar = [torch.tensor(i) for i in range(ITERATION)]

    real_inputs_hpu = [input.to('hpu') for input in real_inputs_cpu]
    real_inputs_hpu_scalar = [input.to('hpu') for input in real_inputs_cpu_scalar]

    real_targets_cpu = [torch.randn(N, D_out, device="cpu") for _ in range(ITERATION)]
    real_targets_hpu = [target.to('hpu') for target in real_targets_cpu]
    loss_hpu_vec = []
    loss_cpu_vec = []

    for data, data2, target in zip(real_inputs_hpu, real_inputs_hpu_scalar, real_targets_hpu):
        loss_hpu = wrapped_func_scalar(data, data2, target, module1_hpu, loss_fn)
        loss_hpu_vec.append(loss_hpu)
        ht.core.mark_step()

    for data, data2, target in zip(real_inputs_cpu, real_inputs_cpu_scalar, real_targets_cpu):
        loss_cpu = wrapped_func_scalar(data, data2, target, module1_cpu, loss_fn)
        loss_cpu_vec.append(loss_cpu)

    compare_tensors(loss_hpu_vec, loss_cpu_vec, atol=0.001, rtol=1.e-3)

def wrapped_func_scalar(data, data2, target, module1, loss_fn):
    tmp = module1(data, data2)
    loss = loss_fn(tmp[:], target)
    return loss

def test_multiple_graph_capture_with_views():
    N, D_in, H, D_out, inner = 2, 2, 2, 2, 2
    module1_cpu = Model(D_in, H, inner).to('cpu')
    module1_hpu = _kernel_copy_to_device(module1_cpu,"hpu")
    loss_fn = torch.nn.MSELoss()
    module1_hpu = ht.hpu.wrap_in_hpu_graph(module1_hpu, disable_tensor_cache=True)
    x_cpu = torch.randn(N, D_in, device='cpu')
    real_inputs_cpu = [torch.rand_like(x_cpu) for _ in range(2)]
    real_inputs_hpu = [input.to('hpu') for input in real_inputs_cpu]
    real_inputs_cpu1 = [torch.rand_like(x_cpu) for _ in range(2)]
    real_inputs_hpu1 = [input.to('hpu') for input in real_inputs_cpu1]
    real_targets_cpu = [torch.ones(N, D_out, device="cpu") for _ in range(2)]
    real_targets_hpu = [target.to('hpu') for target in real_targets_cpu]
    loss_hpu_vec = []
    loss_cpu_vec = []

    #Input as view first time while capture, second time not view
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

    compare_tensors(loss_hpu_vec, loss_cpu_vec, atol=0.001, rtol=1.e-3)
    loss_hpu_vec = []
    loss_cpu_vec = []

    #Input as not view first time while capture, view on second turn
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

    compare_tensors(loss_hpu_vec, loss_cpu_vec, atol=0.001, rtol=1.e-3)
    loss_hpu_vec = []
    loss_cpu_vec = []

    #Input as not view first time while capture, view on second turn
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

    compare_tensors(loss_hpu_vec, loss_cpu_vec, atol=0.001, rtol=1.e-3)
    loss_hpu_vec = []
    loss_cpu_vec = []

if __name__ == "__main__":
    test_multiple_graph_capture()
    test_multiple_graph_capture_memoptimization()
    # test_multiple_graph_capture_memoptimization(asynchronous=True)
    test_multiple_graph_capture_memoptimization(dry_run=True)
    test_multiple_graph_capture_memoptimization(release_memory_test=True)
    test_graph_capture_simple()
    test_graph_training()
    test_tensor_packer()
    test_cached_module_training()
    test_cached_module_training(disable_tensor_cache=True)
    test_graph_capture_scalar(disable_tensor_cache=True)
    test_multiple_graph_capture_with_views()
