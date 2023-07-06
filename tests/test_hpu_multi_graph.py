import habana_frameworks.torch as ht
import torch

device = "hpu"


class GraphTest(object):
    def __init__(self, size: int) -> None:
        self.g = ht.hpu.HPUGraph()
        self.s = ht.hpu.Stream()
        self.size = size

    def warp_func(self, first: bool) -> None:
        if first:
            with ht.hpu.stream(self.s):
                self.g.capture_begin()
                a = torch.full((self.size,), 1, device="hpu")
                b = a
                b = b + 1
                self.g.capture_end()
        else:
            self.g.replay()


def test_graph_capture_simple():
    gt1 = GraphTest(size=1000)
    gt2 = GraphTest(size=2000)
    gt3 = GraphTest(size=3000)
    for i in range(10):
        if i == 0:
            gt1.warp_func(True)
            gt2.warp_func(True)
            gt3.warp_func(True)
        else:
            gt3.warp_func(False)
            gt2.warp_func(False)
            gt1.warp_func(False)
    ht.hpu.synchronize()


if __name__ == "__main__":
    test_graph_capture_simple()
    print("test ran")
