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
        g.replay()


def test_graph_capture_simple():
    tA_h = torch.zeros(10,2).to('hpu')
    for i in range(10):
        if i == 0:
            warp_func(True)
        else:
            warp_func(False)
    ht.hpu.synchronize()

if __name__ == "__main__":
    test_graph_capture_simple()
