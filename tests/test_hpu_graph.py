import torch
import habana_frameworks.torch as ht

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
    module1 = torch.nn.Linear(D_in, H).to('hpu')
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(module1.parameters(),lr=0.1)
    x = torch.randn(N, D_in, device='hpu')
    module1 = ht.hpu.make_graphed_callables(module1, (x,))
    real_inputs = [torch.rand_like(x) for _ in range(10)]
    real_targets = [torch.randn(N, D_out, device="hpu") for _ in range(10)]
    for data, target in zip(real_inputs, real_targets):
        optimizer.zero_grad(set_to_none=True)
        tmp = module1(data)
        loss = loss_fn(tmp, target)
        loss.backward()
        optimizer.step()

if __name__ == "__main__":
    test_graph_capture_simple()
    test_graph_training()
