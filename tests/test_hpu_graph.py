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
    loss = loss_fn(tmp, target)
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


if __name__ == "__main__":
    test_multiple_graph_capture()
    test_graph_capture_simple()
    test_graph_training()
